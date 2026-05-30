/**
 * Conformance suite for the search engine orchestration.
 *
 * The engine is exercised with pure in-memory `SearchSource` doubles —
 * no Postgres, no embedder model. The fake `sql` handle is never touched
 * (the doubles return canned listings), which is exactly how a host would
 * unit-test its own sources' wiring. Covers fan-out + global re-rank,
 * RRF fusion across rankers, mode selection, and the graceful-degradation
 * contract (a source/embedder that throws is skipped, not fatal — P-013).
 */

import { describe, it, expect, vi } from 'vitest';
import type { RankedItem } from '@papercusp/rrf';
import { runFullTextSearch, runHybridSearch, type SearchContext } from './hybrid';
import type { SearchSource, SearchHit, Listing, PgHandle, Embedder } from './types';

// The doubles never call SQL — a structural placeholder is enough.
const sql = {} as PgHandle;

function hit(source: string, id: string, score: number): SearchHit {
  return { source, source_id: id, excerpt: `ex-${id}`, highlight: `hl-${id}`, score, rankers: [source] };
}
function ranked(h: SearchHit): RankedItem<SearchHit> {
  return { key: `${h.source}:${h.source_id}`, score: h.score, row: h };
}
function listing(...hits: SearchHit[]): Listing {
  return hits.map(ranked);
}

/** A source returning canned bm25 / embedding listings. */
function fakeSource(
  name: string,
  opts: { bm25?: SearchHit[]; embedding?: SearchHit[]; throwBm25?: boolean; throwEmbed?: boolean } = {},
): SearchSource {
  const src: SearchSource = {
    name,
    bm25: vi.fn(async () => {
      if (opts.throwBm25) throw new Error(`${name} bm25 boom`);
      return listing(...(opts.bm25 ?? []));
    }),
  };
  if (opts.embedding || opts.throwEmbed) {
    src.embedding = vi.fn(async () => {
      if (opts.throwEmbed) throw new Error(`${name} embed boom`);
      return listing(...(opts.embedding ?? []));
    });
  }
  return src;
}

const baseCtx: SearchContext = { sql, query: 'q', workspaceId: 'w', harnessFilter: null, limit: 5 };

describe('runFullTextSearch', () => {
  it('fans out across sources, re-ranks globally by score, reports totalHits', async () => {
    const a = fakeSource('A', { bm25: [hit('A', '1', 3), hit('A', '2', 1)] });
    const b = fakeSource('B', { bm25: [hit('B', '1', 2)] });
    const res = await runFullTextSearch([a, b], baseCtx);
    expect(res.results.map((h) => `${h.source}:${h.source_id}`)).toEqual(['A:1', 'B:1', 'A:2']);
    expect(res.totalHits).toBe(3);
  });

  it('caps results at ctx.limit (but totalHits counts all)', async () => {
    const a = fakeSource('A', { bm25: [hit('A', '1', 3), hit('A', '2', 2), hit('A', '3', 1)] });
    const res = await runFullTextSearch([a], { ...baseCtx, limit: 2 });
    expect(res.results).toHaveLength(2);
    expect(res.totalHits).toBe(3);
  });

  it('degrades gracefully: a throwing source is skipped + logged, others still return', async () => {
    const log = vi.fn();
    const bad = fakeSource('bad', { throwBm25: true });
    const good = fakeSource('good', { bm25: [hit('good', '1', 1)] });
    const res = await runFullTextSearch([bad, good], { ...baseCtx, log });
    expect(res.results.map((h) => h.source)).toEqual(['good']);
    expect(log).toHaveBeenCalledWith(expect.stringContaining('bad skipped'));
  });
});

describe('runHybridSearch', () => {
  const embedder: Embedder = async () => [0.1, 0.2, 0.3];

  it('hybrid mode fuses bm25 + embedding via RRF; embedderAvailable=true', async () => {
    // Same item top of both rankers → fused above an item in only one list.
    const a = fakeSource('A', { bm25: [hit('A', '1', 9), hit('A', '2', 1)], embedding: [hit('A', '1', 9)] });
    const res = await runHybridSearch([a], { ...baseCtx, mode: 'hybrid', embedder });
    expect(res.embedderAvailable).toBe(true);
    expect(res.results[0]?.source_id).toBe('1'); // present in both rankers → top
    expect(res.results.map((h) => h.rankers).flat()).toContain('bm25');
    expect(res.results.map((h) => h.rankers).flat()).toContain('embeddings');
  });

  it('hybrid mode with no embedder falls back to BM25-only; embedderAvailable=false', async () => {
    const a = fakeSource('A', { bm25: [hit('A', '1', 3)], embedding: [hit('A', '9', 9)] });
    const res = await runHybridSearch([a], { ...baseCtx, mode: 'hybrid', embedder: null });
    expect(res.embedderAvailable).toBe(false);
    expect(res.results.map((h) => h.source_id)).toEqual(['1']); // embedding list never consulted
    expect(a.embedding).not.toHaveBeenCalled();
  });

  it("embeddings mode runs ONLY the vector ranker (no bm25 calls)", async () => {
    const a = fakeSource('A', { bm25: [hit('A', 'bm', 5)], embedding: [hit('A', 'vec', 5)] });
    const res = await runHybridSearch([a], { ...baseCtx, mode: 'embeddings', embedder });
    expect(a.bm25).not.toHaveBeenCalled();
    expect(res.results.map((h) => h.source_id)).toEqual(['vec']);
    expect(res.embedderAvailable).toBe(true);
  });

  it('degrades when the embedder throws: hybrid → BM25-only, embedderAvailable=false', async () => {
    const log = vi.fn();
    const throwingEmbedder: Embedder = async () => { throw new Error('embed model down'); };
    const a = fakeSource('A', { bm25: [hit('A', '1', 3)], embedding: [hit('A', '9', 9)] });
    const res = await runHybridSearch([a], { ...baseCtx, mode: 'hybrid', embedder: throwingEmbedder, log });
    expect(res.embedderAvailable).toBe(false);
    expect(res.results.map((h) => h.source_id)).toEqual(['1']);
    expect(log).toHaveBeenCalledWith(expect.stringContaining('embed failed'));
  });

  it('skips sources without an embedding ranker on the vector pass (no crash)', async () => {
    const withVec = fakeSource('withVec', { bm25: [hit('withVec', '1', 1)], embedding: [hit('withVec', '1', 1)] });
    const noVec = fakeSource('noVec', { bm25: [hit('noVec', '2', 1)] }); // no embedding fn
    const res = await runHybridSearch([withVec, noVec], { ...baseCtx, mode: 'hybrid', embedder });
    expect(res.embedderAvailable).toBe(true);
    expect(res.results.map((h) => h.source_id).sort()).toEqual(['1', '2']);
  });

  it('a single throwing source does not sink the others in hybrid mode', async () => {
    const log = vi.fn();
    const bad = fakeSource('bad', { throwBm25: true, throwEmbed: true });
    const good = fakeSource('good', { bm25: [hit('good', '1', 1)], embedding: [hit('good', '1', 1)] });
    const res = await runHybridSearch([bad, good], { ...baseCtx, mode: 'hybrid', embedder, log });
    expect(res.results.map((h) => h.source)).toEqual(['good']);
  });
});
