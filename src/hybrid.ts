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

/**
 * Hybrid (BM25 + pgvector) search fused via RRF (search:semantic).
 *
 * `mode='hybrid'` runs both rankers; `mode='embeddings'` runs only the
 * vector ranker. If no embedder is supplied or it throws, `hybrid` falls
 * back to BM25-only and `embeddings` returns empty — mirroring the
 * original tool's silent degradation.
 */
export async function runHybridSearch(
  sources: SearchSource[],
  ctx: SearchContext & { mode: 'embeddings' | 'hybrid'; embedder: Embedder | null },
): Promise<HybridResult> {
  let queryVec: number[] | null = null;
  ctx.signal?.throwIfAborted();
  if (ctx.embedder) {
    try {
      queryVec = await ctx.embedder(ctx.query);
    } catch (err) {
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

  const fused = rrfCombine(inputs, RRF_K_DEFAULT);
  const results = fused
    .slice(0, ctx.limit)
    .map(({ row, score, rankers }) => ({ ...row, score, rankers }));
  return { results, totalHits: fused.length, embedderAvailable: !!queryVec };
}
