/**
 * Search orchestration over a set of pluggable `SearchSource`s.
 *
 * `runFullTextSearch` runs BM25 only (search:fulltext); `runHybridSearch`
 * runs BM25 + pgvector embeddings and fuses them via Reciprocal Rank
 * Fusion (search:semantic). Each source call is independently try/caught:
 * a source whose index/extension/embedding column is missing degrades to
 * "skipped" (logged) rather than failing the whole search — this is the
 * explicit form of the old optional-`ctx.tx` contract (P-013).
 */

import { rrfCombine, RRF_K_DEFAULT } from '@papercusp/rrf';
import type { SearchSource, SearchSourceParams, SearchHit, Listing, PgHandle, Embedder, SearchFilters } from './types';
import { applyRecencyRerank, type RecencyRank } from './recency';
import { applyMinScore, type MinScoreFloors } from './min-score';
import { pickTopGroups, countGroups, type GroupKeyOf } from './group';
import { resolveSearchDefaults, type AppliedDefaults } from './defaults';
import {
  newLegAccumulator,
  legOfRanker,
  finaliseLeg,
  summariseLegs,
  type SearchLegs,
} from './legs';

export interface SearchContext {
  sql: PgHandle;
  query: string;
  workspaceId: string;
  scopeFilter: string | null;
  /** Final result cap (sources may over-fetch internally). */
  limit: number;
  /** Optional structured filters threaded to every source (P-002; sources opt in). */
  filters?: SearchFilters;
  /** Optional abort signal (e.g. a route timeout watchdog, or a client that
   *  gave up). Checked before every awaited stage — embed, each source's
   *  lexical/vector call — so an aborted search stops STARTING work instead of
   *  grinding every source to completion after the caller stopped listening
   *  (2026-07-09 incident: timed-out searches burned 34–69s server-side past
   *  a 30s 408). Also threaded to sources (SearchSourceParams.signal) for
   *  opt-in statement-level cancellation. An abort REJECTS (AbortError) —
   *  never a silent partial result — and outranks the graceful skip-a-
   *  throwing-source degradation.  */
  signal?: AbortSignal;
  /** Optional wall-clock budget (ms) for the QUERY-EMBED step only. When the
   *  embedder takes longer, `runHybridSearch` degrades to BM25-only (queryVec
   *  stays null) instead of blocking up to the route watchdog. undefined / ≤0
   *  = unbounded (prior behavior; strictly opt-in per caller). Interactive
   *  callers (e.g. the desktop transcript-search pill) pass a short budget so a
   *  slow or COLD query-embedder — a local ONNX cold-start, or a degraded
   *  OpenAI path under exhausted quota — can't turn a ~20ms BM25 search into a
   *  30s 408 (2026-07-10: measured search:fulltext 20ms vs hybrid 30s, with all
   *  the time in the single unbounded `await embedder(query)` below). The
   *  orphaned embed still runs to completion in the background (warming the
   *  embedder's memoized pipeline for the next query); only the WAIT is bounded
   *  — the Embedder fn has no cancel seam. A route-abort via `signal` still
   *  REJECTS and outranks this timeout. */
  embedTimeoutMs?: number;
  /** Optional recency re-rank applied to the fused candidate list BEFORE the
   *  top-N cut (so a recent-but-lower-relevance candidate in the `limit*3`
   *  over-fetch pool can be rescued). See {@link RecencyRank}.
   *
   *  P-017: absent ⇒ the ENGINE DEFAULT applies (a host policy registered via
   *  `configureSearchDefaults`), not "off". Pass `false` to force pure-relevance
   *  ranking regardless of policy. */
  recency?: RecencyRank | false;
  /** Optional ROLLUP applied to the fused candidate list immediately before the
   *  top-N cut: when set, the page keeps only the best-ranked row per group, so
   *  `limit` counts DISTINCT GROUPS rather than raw rows (P-036).
   *
   *  For a caller that ranks rows but PRESENTS something coarser — the
   *  transcript search ranks session turns and renders session cards — several
   *  turns from one session otherwise consume several page slots while covering
   *  one card (measured 2026-08-09: the top-30 `theory` turns covered 24
   *  distinct sessions). This cannot be fixed by the caller after the fact,
   *  because the rows that would back-fill the freed slots are past `limit` and
   *  already discarded; hence a cut-time hook rather than a post-filter.
   *
   *  Absent ⇒ byte-identical row-level behaviour. Returning null/undefined for a
   *  row leaves it ungrouped (it keeps its own slot). NOT a scoring change: it
   *  cannot promote a group no member row reached the candidate pool with. See
   *  {@link pickTopGroups}. */
  groupBy?: GroupKeyOf;
  /** WI-4734: defer highlight computation to AFTER fusion, for the final top-N
   *  only. Applies per source and only to sources that implement
   *  `hydrateHighlights` (others keep inline highlights — safe mixed-source
   *  behavior). With `limit*3` over-fetch across two rankers, inline
   *  highlighting computes ~6× more highlight expressions than the caller
   *  displays; deferral turns that into ONE batched hydration of `limit` rows. */
  deferHighlight?: boolean;
  /** How the lexical leg combines the query's terms — see
   *  {@link SearchSourceParams.lexicalMode}. Threaded through to every source
   *  unchanged; absent ⇒ each source keeps its historical AND semantics, so
   *  this is byte-identical for every caller that does not pass it. A source
   *  that does not implement the mode is free to ignore it. */
  lexicalMode?: 'and' | 'coverage-graded';
  /** `coverage-graded` only: the anchor's cost budget in summed per-lexeme
   *  document frequency — see {@link SearchSourceParams.lexicalAnchorDfBudget}.
   *  Threaded through unchanged; absent ⇒ each source keeps its own default. */
  lexicalAnchorDfBudget?: number;
  /**
   * PER-RANKER minimum-score floors, applied to each ranker's list BEFORE
   * fusion (see {@link MinScoreFloors}).
   *
   * This is the only stage at which a weak match can be rejected: RRF fuses
   * by rank position and discards the native score, so after fusion a rank-1
   * noise hit is indistinguishable from a rank-1 excellent one. Floors are
   * per-ranker because `ts_rank_cd` and cosine similarity are different
   * scales — one global number cannot floor both.
   *
   * P-017: absent ⇒ the ENGINE DEFAULT applies (a host policy registered via
   * `configureSearchDefaults`), NOT "no floor" — that inversion is the whole
   * point of the seam. Pass `false` to force the unfloored engine regardless
   * of policy.
   */
  minScore?: MinScoreFloors | false;
  log?: (msg: string) => void;
}

export interface SearchResult {
  results: SearchHit[];
  totalHits: number;
  /** Distinct groups in the FULL candidate pool, present only when a
   *  `groupBy` rollup ran. `totalHits` counts rows and so cannot answer "30 of
   *  how many sessions?" on a grouped page; this can. Optional rather than
   *  required precisely so no existing caller or fixture is stranded. */
  totalGroups?: number;
  /** P-017/P-020: which ranking features actually ran. Stated, never assumed —
   *  a feature silently not applying is the exact failure this engine had. */
  applied: AppliedDefaults;
  /**
   * P-020: which RANKING LEGS actually ran and what each contributed to
   * fusion. Reported on every result so a silently single-legged search
   * cannot look like a healthy one — `legs.degraded` + `legs.warning` are the
   * ready-made verdict; the per-leg counts are there when you need the why.
   *
   * ⚠ Read this, not `embedderAvailable`, to answer "did semantic really
   * happen": that flag only says a query VECTOR was produced, which stays
   * true when every per-source embedding query fails and when a leg runs
   * perfectly but returns nothing (EI-19447237774252790).
   */
  legs: SearchLegs;
}

/** BM25-only full-text search across the given sources (search:fulltext). */
export async function runFullTextSearch(
  sources: SearchSource[],
  ctx: SearchContext,
): Promise<SearchResult> {
  // Resolve the ranking policy ONCE up front: caller value wins, `false` is a
  // hard off, absent falls through to the registered engine default (P-017).
  const { minScore, applied } = resolveSearchDefaults(
    { query: ctx.query, limit: ctx.limit, mode: 'fulltext', embedder: null },
    { minScore: ctx.minScore },
  );
  // P-020: a fulltext search has one leg by construction. Report it anyway —
  // a caller must be able to read the SAME shape from both entry points, and a
  // source that failed here is just as invisible as it is in hybrid.
  const lexicalLeg = newLegAccumulator();
  const semanticLeg = newLegAccumulator();
  const hits: SearchHit[] = [];
  for (const source of sources) {
    ctx.signal?.throwIfAborted();
    lexicalLeg.attempted = true;
    try {
      const listing = await source.lexical({
        sql: ctx.sql,
        query: ctx.query,
        workspaceId: ctx.workspaceId,
        scopeFilter: ctx.scopeFilter,
        limit: ctx.limit,
        filters: ctx.filters,
        signal: ctx.signal,
      });
      // Floor before collecting: fulltext has no fusion stage, but the floor
      // means the same thing here — reject what the ranker itself scored as
      // noise, in the ranker's own units.
      const { list, dropped, floor } = applyMinScore(listing, 'lexical', minScore);
      lexicalLeg.callsRun++;
      lexicalLeg.candidates += list.length;
      lexicalLeg.floored += dropped;
      if (dropped > 0) {
        ctx.log?.(
          `search:fulltext ${source.name} lexical minScore ${floor} dropped ${dropped}/${listing.length}`,
        );
      }
      // Carry the native score as provenance too, so a caller renders the same
      // shape whether it came through fulltext or hybrid.
      for (const item of list) hits.push({ ...item.row, rankerScores: { lexical: item.row.score } });
    } catch (err) {
      // Abort outranks graceful degradation: a cancelled search must reject,
      // not be "skipped" into a quiet partial result.
      if (ctx.signal?.aborted) throw err;
      lexicalLeg.callsFailed++;
      lexicalLeg.failures.push({
        source: source.name,
        ranker: 'lexical',
        error: (err as Error).message,
      });
      ctx.log?.(`search:fulltext ${source.name} skipped: ${(err as Error).message}`);
    }
  }
  // Global re-rank by raw score across all sources, then top-N.
  hits.sort((a, b) => b.score - a.score);
  return {
    results: hits.slice(0, ctx.limit),
    totalHits: hits.length,
    applied,
    legs: summariseLegs(finaliseLeg(lexicalLeg), finaliseLeg(semanticLeg)),
  };
}

export interface HybridResult extends SearchResult {
  /** True iff a query embedding was produced (else this was BM25-only). */
  embedderAvailable: boolean;
}

/** Thrown internally when the query-embed exceeds `ctx.embedTimeoutMs`. Caught
 *  by `runHybridSearch` → degrade to BM25-only. Exported so callers/tests can
 *  distinguish a budget trip from a genuine embedder error. */
export class EmbedTimeoutError extends Error {
  constructor(readonly budgetMs: number) {
    super(`query embed exceeded ${budgetMs}ms budget`);
    this.name = 'EmbedTimeoutError';
  }
}

/** Await `embedder(query)` but give up after `budgetMs` (→ EmbedTimeoutError)
 *  or when `signal` aborts (→ the abort reason). With neither bound this is a
 *  bare pass-through, byte-identical to `await embedder(query)`. The losing
 *  embed promise is NOT cancelled — the `Embedder` fn has no signal seam — but
 *  its eventual settle is swallowed so a slow embed that finishes after we've
 *  degraded never surfaces as an unhandled rejection. */
async function embedWithBudget(
  embedder: Embedder,
  query: string,
  budgetMs: number | undefined,
  signal: AbortSignal | undefined,
): Promise<number[]> {
  const p = embedder(query);
  const bounded = budgetMs !== undefined && budgetMs > 0;
  if (!bounded && !signal) return p;
  return await new Promise<number[]>((resolve, reject) => {
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const finish = (fn: () => void): void => {
      if (settled) return;
      settled = true;
      if (timer !== undefined) clearTimeout(timer);
      if (signal) signal.removeEventListener('abort', onAbort);
      fn();
    };
    function onAbort(): void {
      finish(() => reject(signal!.reason ?? new Error('aborted')));
    }
    if (signal) {
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener('abort', onAbort);
    }
    if (bounded) {
      timer = setTimeout(() => finish(() => reject(new EmbedTimeoutError(budgetMs!))), budgetMs);
    }
    // Swallow the orphaned settle after we've resolved/rejected (no cancel seam).
    p.then(
      (v) => finish(() => resolve(v)),
      (e) => finish(() => reject(e)),
    );
  });
}

/**
 * Hybrid (BM25 + pgvector) search fused via RRF (search:semantic).
 *
 * `mode='hybrid'` runs both rankers; `mode='embeddings'` runs only the
 * vector ranker. If no embedder is supplied, it throws, or it exceeds
 * `ctx.embedTimeoutMs`, `hybrid` falls back to BM25-only and `embeddings`
 * returns empty — mirroring the original tool's silent degradation.
 */
export async function runHybridSearch(
  sources: SearchSource[],
  ctx: SearchContext & { mode: 'embeddings' | 'hybrid'; embedder: Embedder | null },
): Promise<HybridResult> {
  ctx.signal?.throwIfAborted();
  // P-017: resolve the ranking policy ONCE, before any leg runs — the lexical legs
  // are floored long before the query vector settles, so this cannot wait on the
  // embed. Resolving off the embedder INSTANCE (not "was a vector produced") is
  // what makes that safe: a host keys its floor to the embedding space it
  // calibrated, and if the vector leg never runs there are no embedding hits for
  // an embedding floor to filter anyway.
  const { minScore: minScoreFloors, recency: recencyRank, applied } = resolveSearchDefaults(
    { query: ctx.query, limit: ctx.limit, mode: ctx.mode, embedder: ctx.embedder },
    { minScore: ctx.minScore, recency: ctx.recency },
  );
  // WI-4734: kick the query-embed off WITHOUT awaiting it — the BM25 legs don't
  // need the vector, so they run CONCURRENTLY with the embed instead of behind
  // it (the old serial shape put the whole embed latency on the critical path
  // even though BM25 was independent). The vector legs await it below.
  const embedP = ctx.embedder
    ? embedWithBudget(ctx.embedder, ctx.query, ctx.embedTimeoutMs, ctx.signal)
    : null;
  // Pre-attach a no-op catch so an embed failure that settles while BM25 is
  // still running never surfaces as an unhandled rejection (it is re-awaited —
  // and properly handled — below).
  embedP?.catch(() => {});

  // P-020: per-leg execution accounting. Fed at the two choke points every
  // candidate must pass through (`floorList` for what the floor removed,
  // `recordInput` for what actually reached fusion), so a new ranker leg
  // cannot be added later and quietly go unreported.
  const legs = { lexical: newLegAccumulator(), semantic: newLegAccumulator() };

  // Over-fetch per source so RRF has fusion headroom before the top-N cut.
  const candidateLimit = ctx.limit * 3;
  const inputs: Array<{ name: string; list: Listing }> = [];

  // Pre-fusion provenance. RRF's output score is an RRF value, not a native
  // one, and the native score is gone the moment fusion runs — so capture it
  // here, per ranker, keyed by the fusion key. `keyOfRow` lets the final hits
  // find their own entry without re-deriving rrfCombine's first-occurrence
  // rule: it keeps the row OBJECT that fusion carries through.
  const nativeByKey = new Map<string, Record<string, number>>();
  const keyOfRow = new Map<SearchHit, string>();
  /** Apply `ranker`'s floor (if any) to a source's list, logging what it cut. */
  const floorList = (list: Listing, ranker: string, sourceName: string): Listing => {
    const { list: kept, dropped, floor } = applyMinScore(list, ranker, minScoreFloors);
    legOfRanker(ranker, legs).floored += dropped;
    if (dropped > 0) {
      ctx.log?.(
        `${ranker} ${sourceName} minScore ${floor} dropped ${dropped}/${list.length}`,
      );
    }
    return kept;
  };
  /** Record a floored list's native scores, then enqueue it as an RRF input. */
  const recordInput = (ranker: string, list: Listing): void => {
    // Counted here rather than at the query: this is the post-floor,
    // post-dedupe list fusion genuinely sees, which is the only number that
    // distinguishes "the leg ran" from "the leg contributed".
    legOfRanker(ranker, legs).candidates += list.length;
    for (const entry of list) {
      keyOfRow.set(entry.row, entry.key);
      const existing = nativeByKey.get(entry.key);
      if (existing) {
        // First occurrence wins, mirroring rrfCombine's row-payload rule.
        if (!(ranker in existing)) existing[ranker] = entry.score;
      } else {
        nativeByKey.set(entry.key, { [ranker]: entry.score });
      }
    }
    inputs.push({ name: ranker, list });
  };
  const sourceParams = (source: SearchSource): SearchSourceParams => ({
    sql: ctx.sql,
    query: ctx.query,
    workspaceId: ctx.workspaceId,
    scopeFilter: ctx.scopeFilter,
    limit: candidateLimit,
    filters: ctx.filters,
    signal: ctx.signal,
    // Only defer for a source that can hydrate afterwards; others keep inline
    // highlights so mixed-source calls stay correct.
    ...(ctx.deferHighlight && source.hydrateHighlights ? { wantHighlight: false } : {}),
    // Passed through only when the caller asked for a non-default mode, so a
    // source cannot tell "caller said 'and'" from "caller said nothing" — the
    // two must behave identically and this makes that structural.
    ...(ctx.lexicalMode && ctx.lexicalMode !== 'and' ? { lexicalMode: ctx.lexicalMode } : {}),
    // Same rule as lexicalMode above: passed through ONLY when the caller set
    // it, so a source cannot tell "caller said nothing" from "caller said the
    // default". 0 is a MEANINGFUL value here (argmin-only), so test for
    // undefined rather than truthiness.
    ...(ctx.lexicalAnchorDfBudget === undefined
      ? {}
      : { lexicalAnchorDfBudget: ctx.lexicalAnchorDfBudget }),
  });

  // Fresh-candidate window (RecencyRank.freshWindowMs): when the recency
  // re-rank is active, ALSO fetch a BM25 candidate list restricted to the
  // recent window, as one more RRF input. Without it, the re-rank can only
  // reorder candidates that survived the relevance-only over-fetch cut — on a
  // large corpus old high-term-density rows can fill the whole pool and starve
  // every recent match out of eligibility (WI-5097: the agents-pill search
  // returned nothing from the current day). Skipped when the caller's own
  // `since` filter is already as narrow or narrower.
  const recencyWeight = recencyRank ? Math.max(0, Math.min(1, recencyRank.weight ?? 0.3)) : 0;
  const freshWindowMs = recencyRank?.freshWindowMs ?? 0;
  let freshSinceIso: string | null = null;
  if (ctx.mode === 'hybrid' && recencyWeight > 0 && freshWindowMs > 0) {
    const nowMs = recencyRank?.now ?? Date.now();
    const freshSinceMs = nowMs - freshWindowMs;
    const callerSinceMs = ctx.filters?.since ? Date.parse(ctx.filters.since) : NaN;
    if (!(Number.isFinite(callerSinceMs) && callerSinceMs >= freshSinceMs)) {
      freshSinceIso = new Date(freshSinceMs).toISOString();
    }
  }

  if (ctx.mode === 'hybrid') {
    // All BM25 legs in parallel (independent queries; postgres-js pools).
    // Promise.all preserves source order, so RRF input order — and therefore
    // tie-breaking — is deterministic and identical to the old serial loop.
    // The fresh legs join the SAME fan-out (one concurrent stage, no extra
    // latency step).
    const runLeg = async (source: SearchSource, since: string | null): Promise<Listing | null> => {
      ctx.signal?.throwIfAborted();
      const label = since ? 'lexical-fresh' : 'lexical';
      // P-020: EXECUTION accounting, recorded HERE rather than at the floor /
      // fusion choke points below. Those two see only candidates, so they can
      // never separate "the leg ran and found nothing" from "the leg never
      // ran" — which is precisely the question `status` exists to answer, and
      // the distinction the measured EI-19447237774252790 case turns on.
      const leg = legOfRanker(label, legs);
      leg.attempted = true;
      try {
        const params = sourceParams(source);
        const listing = await source.lexical(
          since ? { ...params, filters: { ...params.filters, since } } : params,
        );
        leg.callsRun++;
        return listing;
      } catch (err) {
        if (ctx.signal?.aborted) throw err;
        leg.callsFailed++;
        leg.failures.push({ source: source.name, ranker: label, error: (err as Error).message });
        ctx.log?.(`${label} ${source.name} skipped: ${(err as Error).message}`);
        return null;
      }
    };
    const [lexicalLists, freshLists] = await Promise.all([
      Promise.all(sources.map((source) => runLeg(source, null))),
      freshSinceIso
        ? Promise.all(sources.map((source) => runLeg(source, freshSinceIso)))
        : Promise.resolve(sources.map((): Listing | null => null)),
    ]);
    for (let i = 0; i < sources.length; i++) {
      const sourceName = sources[i]!.name;
      const raw = lexicalLists[i];
      // Floor BOTH legs before the identity check below, so the "same ranking
      // twice" verdict is made on the lists fusion will actually see.
      const list = raw ? floorList(raw, 'lexical', sourceName) : null;
      if (list) recordInput('lexical', list);
      const rawFresh = freshLists[i];
      const fresh = rawFresh ? floorList(rawFresh, 'lexical-fresh', sourceName) : null;
      // P-003: the fresh leg exists to guarantee recent rows a SEAT in the
      // candidate pool (WI-5097) — not to give them a second relevance vote.
      // rrfCombine SUMS contributions per key, so a row appearing in both legs
      // used to collect 1/(k+mainRank) + 1/(k+freshRank), i.e. recency counted
      // once here as inflated relevance and again in the recency blend. Only
      // rows the relevance-only cut actually MISSED need adding, so dedupe
      // against the main list. This also subsumes the old identical-list guard:
      // a source that ignores `since` returns the same rows, every one of which
      // is filtered out here, leaving nothing to push.
      const alreadyPooled = new Set((list ?? []).map((entry) => entry.key));
      const freshOnly = fresh ? fresh.filter((entry) => !alreadyPooled.has(entry.key)) : null;
      if (freshOnly && freshOnly.length > 0) recordInput('lexical-fresh', freshOnly);
    }
  }

  let queryVec: number[] | null = null;
  if (embedP) {
    try {
      queryVec = await embedP;
    } catch (err) {
      // A route-abort outranks the timeout: rethrow so a cancelled search
      // rejects. An EmbedTimeoutError or a genuine embedder failure is caught
      // here → queryVec stays null → BM25-only degrade (embedderAvailable=false).
      if (ctx.signal?.aborted) throw err;
      // P-020: the semantic leg WAS wanted (an embedder is configured) but
      // could not be attempted at all. That is `blocked` — distinct from a
      // source call failing, and it must not read as `not-run`, which means
      // "never wanted" and would make a dead embedder look like a plain
      // lexical-only search.
      legs.semantic.attempted = true;
      legs.semantic.blocked = `query embed failed: ${(err as Error).message}`;
      ctx.log?.(`search:semantic query embed failed: ${(err as Error).message}`);
    }
  }

  if (queryVec) {
    const qVec = `[${queryVec.join(',')}]`;
    const embedLists = await Promise.all(
      sources.map(async (source) => {
        // A source with no embedding query has no semantic leg to attempt.
        // Leaving `attempted` false here is what makes "no registered source
        // implements one" report `not-run` instead of a phantom failure.
        if (!source.embedding) return null;
        ctx.signal?.throwIfAborted();
        legs.semantic.attempted = true;
        try {
          const listing = await source.embedding({ ...sourceParams(source), qVec });
          legs.semantic.callsRun++;
          return listing;
        } catch (err) {
          if (ctx.signal?.aborted) throw err;
          legs.semantic.callsFailed++;
          legs.semantic.failures.push({
            source: source.name,
            ranker: 'embeddings',
            error: (err as Error).message,
          });
          ctx.log?.(`embed ${source.name} skipped: ${(err as Error).message}`);
          return null;
        }
      }),
    );
    for (let i = 0; i < sources.length; i++) {
      const list = embedLists[i];
      if (!list) continue;
      // The floor matters most here: on a sparsely-embedded corpus the vector
      // leg returns its k nearest rows whether or not any of them is a match,
      // and fusion would hand rank-1 noise the same weight as a real hit.
      recordInput('embeddings', floorList(list, 'embeddings', sources[i]!.name));
    }
  }

  const fusedRaw = rrfCombine(inputs, RRF_K_DEFAULT);
  // Optional recency re-rank over the FULL candidate pool, before the top-N cut.
  const fused = recencyRank ? applyRecencyRerank(fusedRaw, recencyRank) : fusedRaw;
  // P-036: the optional rollup sits BETWEEN the re-rank and the cut — the only
  // point where the full pool is in final order and still intact. Ordering
  // matters: rolling up before the recency re-rank would pick each group's
  // representative by relevance alone and then let the re-rank reorder a page
  // whose members were already chosen, which is not the same page.
  const page = ctx.groupBy
    ? pickTopGroups(fused, ctx.groupBy, ctx.limit)
    : fused.slice(0, ctx.limit);
  const results = page.map(({ row, score, rankers }) => {
    const key = keyOfRow.get(row);
    const native = key === undefined ? undefined : nativeByKey.get(key);
    // Copy: `nativeByKey` keeps accumulating per-key state and callers mutate
    // hits (highlight hydration below), so the hit must not alias it.
    return { ...row, score, rankers, ...(native ? { rankerScores: { ...native } } : {}) };
  });

  // WI-4734: hydrate highlights for the FINAL top-N only (per deferring source,
  // one batched call). Best-effort — a hydration failure degrades that source's
  // highlights to '' rather than failing a search that already has results.
  if (ctx.deferHighlight) {
    const bySource = new Map<string, SearchHit[]>();
    for (const hit of results) {
      const list = bySource.get(hit.source);
      if (list) list.push(hit);
      else bySource.set(hit.source, [hit]);
    }
    await Promise.all(
      sources.map(async (source) => {
        const hits = bySource.get(source.name);
        if (!hits?.length || !source.hydrateHighlights) return;
        ctx.signal?.throwIfAborted();
        try {
          const highlights = await source.hydrateHighlights(
            sourceParams(source),
            hits.map((h) => h.source_id),
          );
          for (const h of hits) h.highlight = highlights.get(h.source_id) ?? h.highlight;
        } catch (err) {
          if (ctx.signal?.aborted) throw err;
          ctx.log?.(`highlight ${source.name} skipped: ${(err as Error).message}`);
        }
      }),
    );
  }

  return {
    results,
    totalHits: fused.length,
    ...(ctx.groupBy ? { totalGroups: countGroups(fused, ctx.groupBy) } : {}),
    embedderAvailable: !!queryVec,
    applied,
    legs: summariseLegs(finaliseLeg(legs.lexical), finaliseLeg(legs.semantic)),
  };
}
