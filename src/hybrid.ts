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
import type { SearchSource, SearchHit, Listing, PgHandle, Embedder, SearchFilters } from './types';
import { applyRecencyRerank, type RecencyRank } from './recency';

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
   *  bm25/vector call — so an aborted search stops STARTING work instead of
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
   *  over-fetch pool can be rescued). Absent ⇒ ranking is byte-identical to
   *  pure relevance. See {@link RecencyRank}. */
  recency?: RecencyRank;
  /** WI-4734: defer highlight computation to AFTER fusion, for the final top-N
   *  only. Applies per source and only to sources that implement
   *  `hydrateHighlights` (others keep inline highlights — safe mixed-source
   *  behavior). With `limit*3` over-fetch across two rankers, inline
   *  highlighting computes ~6× more highlight expressions than the caller
   *  displays; deferral turns that into ONE batched hydration of `limit` rows. */
  deferHighlight?: boolean;
  log?: (msg: string) => void;
}

export interface SearchResult {
  results: SearchHit[];
  totalHits: number;
}

/** BM25-only full-text search across the given sources (search:fulltext). */
export async function runFullTextSearch(
  sources: SearchSource[],
  ctx: SearchContext,
): Promise<SearchResult> {
  const hits: SearchHit[] = [];
  for (const source of sources) {
    ctx.signal?.throwIfAborted();
    try {
      const listing = await source.bm25({
        sql: ctx.sql,
        query: ctx.query,
        workspaceId: ctx.workspaceId,
        scopeFilter: ctx.scopeFilter,
        limit: ctx.limit,
        filters: ctx.filters,
        signal: ctx.signal,
      });
      for (const item of listing) hits.push(item.row);
    } catch (err) {
      // Abort outranks graceful degradation: a cancelled search must reject,
      // not be "skipped" into a quiet partial result.
      if (ctx.signal?.aborted) throw err;
      ctx.log?.(`search:fulltext ${source.name} skipped: ${(err as Error).message}`);
    }
  }
  // Global re-rank by raw score across all sources, then top-N.
  hits.sort((a, b) => b.score - a.score);
  return { results: hits.slice(0, ctx.limit), totalHits: hits.length };
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
  let queryVec: number[] | null = null;
  ctx.signal?.throwIfAborted();
  if (ctx.embedder) {
    try {
      queryVec = await embedWithBudget(ctx.embedder, ctx.query, ctx.embedTimeoutMs, ctx.signal);
    } catch (err) {
      // A route-abort outranks the timeout: rethrow so a cancelled search
      // rejects. An EmbedTimeoutError or a genuine embedder failure is caught
      // here → queryVec stays null → BM25-only degrade (embedderAvailable=false).
      if (ctx.signal?.aborted) throw err;
      ctx.log?.(`search:semantic query embed failed: ${(err as Error).message}`);
    }
  }

  // Over-fetch per source so RRF has fusion headroom before the top-N cut.
  const candidateLimit = ctx.limit * 3;
  const inputs: Array<{ name: string; list: Listing }> = [];

  if (ctx.mode === 'hybrid') {
    for (const source of sources) {
      ctx.signal?.throwIfAborted();
      try {
        const list = await source.bm25({
          sql: ctx.sql,
          query: ctx.query,
          workspaceId: ctx.workspaceId,
          scopeFilter: ctx.scopeFilter,
          limit: candidateLimit,
          filters: ctx.filters,
          signal: ctx.signal,
        });
        inputs.push({ name: 'bm25', list });
      } catch (err) {
        if (ctx.signal?.aborted) throw err;
        ctx.log?.(`bm25 ${source.name} skipped: ${(err as Error).message}`);
      }
    }
  }

  if (queryVec) {
    const qVec = `[${queryVec.join(',')}]`;
    for (const source of sources) {
      if (!source.embedding) continue;
      ctx.signal?.throwIfAborted();
      try {
        const list = await source.embedding({
          sql: ctx.sql,
          query: ctx.query,
          workspaceId: ctx.workspaceId,
          scopeFilter: ctx.scopeFilter,
          limit: candidateLimit,
          qVec,
          filters: ctx.filters,
          signal: ctx.signal,
        });
        inputs.push({ name: 'embeddings', list });
      } catch (err) {
        if (ctx.signal?.aborted) throw err;
        ctx.log?.(`embed ${source.name} skipped: ${(err as Error).message}`);
      }
    }
  }

  const fusedRaw = rrfCombine(inputs, RRF_K_DEFAULT);
  // Optional recency re-rank over the FULL candidate pool, before the top-N cut.
  const fused = ctx.recency ? applyRecencyRerank(fusedRaw, ctx.recency) : fusedRaw;
  const results = fused
    .slice(0, ctx.limit)
    .map(({ row, score, rankers }) => ({ ...row, score, rankers }));
  return { results, totalHits: fused.length, embedderAvailable: !!queryVec };
}
