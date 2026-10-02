/**
 * WI-37603 — run a pgvector query with HNSW ITERATIVE SCAN enabled.
 *
 * ## The bug this exists to fix
 *
 * With `hnsw.iterative_scan = off` (the pgvector default, and what this box
 * runs) an HNSW index scan stops after `hnsw.ef_search` candidates — 40 by
 * default — and returns AT MOST that many rows NO MATTER WHAT LIMIT IS ASKED.
 * It does not error and it does not warn; the caller simply gets fewer rows
 * than it requested and cannot tell that from "the corpus has no more".
 *
 * So the search engine's over-fetch, `candidateLimit = ctx.limit * 3`
 * (`hybrid.ts:401`), was a NO-OP for every semantic leg past 40 rows.
 *
 * MEASURED on the live 302,093-row embedded `session_turns` corpus (2026-08-10),
 * asking for 600 rows every time:
 *
 *   ef_search=40  -> 40 rows    ef_search=200 -> 212 rows    ef_search=800 -> 600 rows
 *
 * i.e. the request is satisfied only once ef_search exceeds it. With iterative
 * scanning ON instead, at the SAME ef_search=40: 600 rows. At the realistic page
 * size (`candidateLimit=90`, a limit=30 search) it is 40 -> 90 rows for
 * 15-17ms -> 28-29ms. ~12ms buys 2.25x the semantic candidates on a normal page,
 * against a search whose query embedding alone costs hundreds of ms.
 *
 * ## Why a transaction, and not a startup parameter
 *
 * The canonical client (`libs/papercusp/libs/db/src/connection.ts`) already
 * injects GUCs as libpq STARTUP parameters — that is how `statement_timeout` is
 * set (:1023). This GUC cannot go there, for two independent reasons:
 *
 *   1. pgvector registers its GUCs in `_PG_init`, i.e. only once the extension
 *      library is loaded into the session. At startup the parameter does not yet
 *      exist, so it would be rejected as unrecognized.
 *   2. PgBouncer is enabled here (`PAPERCUSP_PGBOUNCER=1`) and connection.ts:1003
 *      already documents `unsupported startup parameter` breakage on the pooled
 *      org-pool path.
 *
 * PgBouncer is also why a bare `SET` would be wrong: under transaction pooling
 * the connection goes back to the pool and the setting does not travel with the
 * next query. `SET LOCAL` inside an EXPLICIT transaction is scoped to exactly the
 * statements that need it and is safe under transaction pooling — which is why
 * the sibling implementation this is modelled on does the same.
 *
 * ## Modelled on the memory subsystem, which hit this first
 *
 * `libs/generic/memory/src/canonical-store.ts` fixed this exact bug on 2026-08-02
 * (EI-19386910150607131) after measuring a scoped `LIMIT 12` pull returning **0
 * rows against 18,510 eligible**: `probeIterativeScan()` (:321) capability-probes
 * the GUC once, `runVectorSearch()` (:375) wraps the query. Its docblock records
 * why `relaxed_order` beats `strict_order`: identical top-12 and identical worst
 * score at ~4x the speed (8.6ms vs 35.2ms), and every caller re-ranks downstream
 * anyway — which is true here too, since the engine RRF-fuses and then re-ranks.
 *
 * That code is NOT directly reusable: memory drives a `pg` Pool
 * (`conn.connect()` / `conn.query()`), search drives a postgres.js `PgHandle`.
 * This is the postgres.js equivalent, deliberately kept to the same shape.
 */
import type { PgHandle } from './types';

/**
 * Does this server understand `hnsw.iterative_scan` (pgvector >= 0.8)?
 *
 * Probed once per handle and cached when the result is stable. An older build
 * rejects the GUC with `unrecognized configuration parameter`, so that
 * permanent capability answer is cached. A transient connection failure must
 * not poison the process: failing open for that call (a narrower candidate
 * list, i.e. exactly today's behaviour) beats failing the search outright, and
 * the next call gets another chance to probe.
 *
 * Keyed by handle rather than module-global because a process legitimately holds
 * several handles (org-app, org-admin, a test harness against a fresh db) and
 * they need not be the same server. A WeakMap so a discarded handle's cached
 * answer is collectable with it.
 */
const supportByHandle = new WeakMap<PgHandle, Promise<boolean>>();
// Keep bounded probes separate so a concurrent legacy probe that is still
// waiting on pool acquisition cannot make an abort-aware caller wait on it.
const boundedSupportByHandle = new WeakMap<PgHandle, Promise<boolean>>();

type ReadOnlyTransactionRunner = <T>(body: (sql: PgHandle) => Promise<T>) => Promise<T>;

function isMissingIterativeScanParameter(error: unknown): boolean {
  const candidate = error as { code?: unknown; message?: unknown } | null;
  const code = candidate?.code;
  const message = typeof candidate?.message === 'string' ? candidate.message : '';
  // PostgreSQL's SQLSTATE for an unknown configuration parameter is
  // undefined_object (42704). The message fallback keeps test doubles and
  // drivers that omit SQLSTATE from taking the transient path.
  return code === '42704' || /unrecognized configuration parameter[\s\S]*hnsw\.iterative_scan/i.test(message);
}

function probeIterativeScan(sql: PgHandle, runReadOnlyTransaction?: ReadOnlyTransactionRunner): Promise<boolean> {
  const cache = runReadOnlyTransaction ? boundedSupportByHandle : supportByHandle;
  const cached = cache.get(sql);
  if (cached) return cached;

  let probe!: Promise<boolean>;
  probe = (async () => {
    try {
      // Inside a transaction so the SET is rolled back either way and can never
      // leak onto a pooled connection.
      if (runReadOnlyTransaction) {
        await runReadOnlyTransaction(async (tx) => {
          await tx`SET LOCAL hnsw.iterative_scan = relaxed_order`;
        });
      } else {
        await sql.begin(async (tx) => {
          await tx`SET LOCAL hnsw.iterative_scan = relaxed_order`;
        });
      }
      return true;
    } catch (error) {
      const permanentUnsupported = isMissingIterativeScanParameter(error);
      // Name the error: a "transient" verdict that recurs on every call is
      // indistinguishable from a deterministic one without it (WI-10002536).
      const candidate = error as { code?: unknown; message?: unknown } | null;
      const detail = `${String(candidate?.code ?? 'no-code')}: ${String(candidate?.message ?? error)}`;
      console.warn(
        permanentUnsupported
          ? '[hnsw-iterative-scan] server does not support hnsw.iterative_scan; using the legacy capped scan'
          : `[hnsw-iterative-scan] capability probe failed transiently (${detail}); using the legacy capped scan for this call and will retry`,
      );
      // An undefined GUC is a stable server capability result and remains
      // cached. Any other failure may be a pool hiccup or restart; remove only
      // THIS probe so a newer probe that raced with it is never deleted.
      if (!permanentUnsupported && cache.get(sql) === probe) {
        cache.delete(sql);
      }
      return false;
    }
  })();
  cache.set(sql, probe);
  return probe;
}

/** pgvector's accepted range for `hnsw.ef_search`. */
const EF_SEARCH_MIN = 1;
const EF_SEARCH_MAX = 1000;

export interface IterativeScanOptions {
  runReadOnlyTransaction?: ReadOnlyTransactionRunner;
  /**
   * `hnsw.ef_search` for the body's transaction (pgvector default 40): the size
   * of the candidate list each HNSW scan iteration keeps. Larger buys recall at
   * the cost of reading more of the graph. Measured for consult's ANN chunk leg
   * (generic-rag-chunking D-046): top-1 agreed with exact on 81/90 queries at
   * 40, 90/90 at 100 and 200. Set with set_config(..., true), so it is
   * transaction-local exactly like `SET LOCAL`, and applied only where iterative
   * scan is (a server without it runs the body unchanged, as before).
   */
  efSearch?: number;
}

function efSearchSetting(efSearch: number | undefined): string | null {
  if (efSearch === undefined) return null;
  if (!Number.isInteger(efSearch) || efSearch < EF_SEARCH_MIN || efSearch > EF_SEARCH_MAX) {
    throw new Error(`withIterativeScan: efSearch must be an integer in ${EF_SEARCH_MIN}..${EF_SEARCH_MAX}, got ${String(efSearch)}`);
  }
  return String(efSearch);
}

/**
 * Run `body` with HNSW iterative scanning enabled, so an ORDER BY `<=>` query
 * returns as many rows as it asked for instead of stopping at `ef_search`.
 *
 * `body` receives a TRANSACTION-SCOPED handle and MUST use it — including for
 * any nested `sql``...``` fragments it interpolates, which otherwise belong to a
 * different handle than the one executing them.
 *
 * Fails OPEN: on a server without the GUC the body runs unchanged on the
 * original handle, which is precisely the pre-WI-37603 behaviour. An error
 * raised by `body` itself is NOT swallowed and NOT retried — retrying would run
 * the caller's query a second time to report the same failure.
 */
export async function withIterativeScan<T>(
  sql: PgHandle,
  body: (sql: PgHandle) => Promise<T>,
  options: IterativeScanOptions = {},
): Promise<T> {
  const runReadOnlyTransaction = options.runReadOnlyTransaction;
  // Validated before anything runs: a bad value is a caller bug, not a degrade.
  const efSearch = efSearchSetting(options.efSearch);
  // A transaction-scoped handle (postgres.js TransactionSql, e.g. the one
  // sessions:search hands runHybridSearch) has no `begin`. Probing it threw
  // `sql.begin is not a function` on EVERY call and warned as if transient
  // (WI-10002536). Run the body on the caller's transaction with the legacy
  // capped scan — the same fallback that failed probe always produced.
  if (!runReadOnlyTransaction && typeof (sql as { begin?: unknown }).begin !== 'function') return body(sql);
  const supported = await probeIterativeScan(sql, runReadOnlyTransaction);
  if (runReadOnlyTransaction) {
    // The injected runner owns the READ ONLY transaction and its acquisition,
    // statement timeout, and abort handling. Run the body through it even on an
    // older server without iterative scan so the fallback query stays bounded.
    return runReadOnlyTransaction(async (tx) => {
      if (supported) {
        await tx`SET LOCAL hnsw.iterative_scan = relaxed_order`;
        // SET cannot take a bind parameter; set_config(..., is_local => true) can.
        if (efSearch !== null) await tx`SELECT set_config('hnsw.ef_search', ${efSearch}, true)`;
      }
      return body(tx);
    });
  }
  if (!supported) return body(sql);
  return sql.begin(async (tx) => {
    // READ ONLY keeps it honest: these are search reads, and the marker makes an
    // accidental write inside a search leg fail loudly rather than commit.
    await tx`SET TRANSACTION READ ONLY`;
    await tx`SET LOCAL hnsw.iterative_scan = relaxed_order`;
    if (efSearch !== null) await tx`SELECT set_config('hnsw.ef_search', ${efSearch}, true)`;
    return body(tx as unknown as PgHandle);
  }) as Promise<T>;
}

/** Test seam: forget the cached capability probe for a handle. */
export function resetIterativeScanProbe(sql: PgHandle): void {
  supportByHandle.delete(sql);
  boundedSupportByHandle.delete(sql);
}
