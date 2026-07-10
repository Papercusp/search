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

const baseCtx: SearchContext = { sql, query: 'q', workspaceId: 'w', scopeFilter: null, limit: 5 };

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

  // ── P-002 (session-search-scope-2026-07-05) — the filter bag is threaded ──
  // The engine passes ctx.filters VERBATIM to every source call (bm25 AND
  // embedding legs); an absent filters stays undefined (no accidental {}).
  it('threads ctx.filters to every source call, verbatim', async () => {
    const a = fakeSource('A', { bm25: [hit('A', '1', 1)], embedding: [hit('A', '2', 1)] });
    const filters = { owners: ['su-1'], speaker: 'assistant', since: '2026-07-01T00:00:00Z' };
    await runFullTextSearch([a], { ...baseCtx, filters });
    expect(a.bm25).toHaveBeenCalledWith(expect.objectContaining({ filters }));
    const embedder: Embedder = async () => [0.1, 0.2];
    await runHybridSearch([a], { ...baseCtx, filters, mode: 'hybrid', embedder });
    expect(a.bm25).toHaveBeenLastCalledWith(expect.objectContaining({ filters }));
    expect(a.embedding).toHaveBeenLastCalledWith(expect.objectContaining({ filters }));
  });

  it('omits filters entirely when the ctx carries none (sources see undefined)', async () => {
    const a = fakeSource('A', { bm25: [hit('A', '1', 1)] });
    await runFullTextSearch([a], baseCtx);
    expect(a.bm25).toHaveBeenCalledWith(expect.objectContaining({ filters: undefined }));
  });

  // ── GAP 13 — cross-source re-rank contract ───────────────────────────────
  // `runFullTextSearch` (hybrid.ts:51) does ONE global sort by the raw,
  // ranker-native score (`b.score - a.score`) across heterogeneous sources.
  // These tests PIN that contract so a future change (e.g. a per-source score
  // normalization, or switching to RRF fusion like the hybrid path) is a
  // deliberate, test-visible decision rather than a silent regression.

  it('GAP 13: cross-source ordering is by raw score, descending, regardless of source', async () => {
    // Interleaved raw scores across two sources. The contract is that the raw
    // ts_rank_cd-style number is treated as directly comparable across sources
    // (NO per-source normalization) — so the merged order is purely score-desc.
    const a = fakeSource('A', { bm25: [hit('A', '1', 0.9), hit('A', '2', 0.3)] });
    const b = fakeSource('B', { bm25: [hit('B', '1', 0.5), hit('B', '2', 0.1)] });
    const res = await runFullTextSearch([a, b], baseCtx);
    expect(res.results.map((h) => `${h.source}:${h.source_id}`)).toEqual([
      'A:1', // 0.9
      'B:1', // 0.5
      'A:2', // 0.3
      'B:2', // 0.1
    ]);
  });

  it('GAP 13: equal scores keep a STABLE order — earlier source first, then within-source order', async () => {
    // Three hits all tied at 1.0 spanning two sources. A stable sort preserves
    // insertion order: source A is fanned out before B, and within A the
    // listing order is preserved. This pins determinism of the tie-break (no
    // accidental reliance on an unstable sort).
    const a = fakeSource('A', { bm25: [hit('A', '1', 1.0), hit('A', '2', 1.0)] });
    const b = fakeSource('B', { bm25: [hit('B', '1', 1.0)] });
    const res = await runFullTextSearch([a, b], baseCtx);
    expect(res.results.map((h) => `${h.source}:${h.source_id}`)).toEqual(['A:1', 'A:2', 'B:1']);

    // And the tie-break is insensitive to the fan-out order ONLY in that it
    // tracks it: swapping the source array swaps the tied order, proving the
    // order is the (stable) fan-out order, not some hidden key.
    const res2 = await runFullTextSearch([b, a], baseCtx);
    expect(res2.results.map((h) => `${h.source}:${h.source_id}`)).toEqual(['B:1', 'A:1', 'A:2']);
  });

  it('GAP 13: overlapping source_ids across DIFFERENT sources are NOT deduped/fused', async () => {
    // fulltext does NO fusion — two distinct sources can legitimately each have
    // a hit with the same source_id (e.g. a shared harness_slug key). They must
    // BOTH survive as separate rows (distinguished by `source`), not be merged.
    const a = fakeSource('A', { bm25: [hit('A', 'shared', 0.8)] });
    const b = fakeSource('B', { bm25: [hit('B', 'shared', 0.4)] });
    const res = await runFullTextSearch([a, b], baseCtx);
    expect(res.results).toHaveLength(2);
    expect(res.results.map((h) => `${h.source}:${h.source_id}`)).toEqual(['A:shared', 'B:shared']);
    expect(res.totalHits).toBe(2);
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

// ── Abort contract (2026-07-09 search-timeout hardening) ─────────────────────
// A caller-supplied `ctx.signal` (route timeout watchdog / client gave up)
// must stop the engine from STARTING further work and must REJECT — never a
// silent partial result, and never swallowed by the skip-a-throwing-source
// degradation path.
describe('abort (ctx.signal) contract', () => {
  const embedder: Embedder = async () => [0.1, 0.2, 0.3];

  it('a pre-aborted signal rejects before any source or embedder call', async () => {
    const a = fakeSource('A', { bm25: [hit('A', '1', 1)], embedding: [hit('A', '1', 1)] });
    const embedSpy = vi.fn(embedder);
    const ctl = new AbortController();
    ctl.abort();
    await expect(runFullTextSearch([a], { ...baseCtx, signal: ctl.signal })).rejects.toThrow();
    await expect(
      runHybridSearch([a], { ...baseCtx, signal: ctl.signal, mode: 'hybrid', embedder: embedSpy }),
    ).rejects.toThrow();
    expect(a.bm25).not.toHaveBeenCalled();
    expect(a.embedding).not.toHaveBeenCalled();
    expect(embedSpy).not.toHaveBeenCalled();
  });

  it('aborting mid-pipeline stops later sources from starting', async () => {
    const ctl = new AbortController();
    const first: SearchSource = {
      name: 'first',
      bm25: vi.fn(async () => {
        ctl.abort(); // abort fires while the first source is in flight
        return listing(hit('first', '1', 1));
      }),
    };
    const second = fakeSource('second', { bm25: [hit('second', '1', 1)] });
    await expect(
      runHybridSearch([first, second], { ...baseCtx, signal: ctl.signal, mode: 'hybrid', embedder: null }),
    ).rejects.toThrow();
    expect(second.bm25).not.toHaveBeenCalled();
  });

  it('an abort surfacing AS a source throw is rethrown, not skipped-and-degraded', async () => {
    const ctl = new AbortController();
    const src: SearchSource = {
      name: 'aborty',
      bm25: vi.fn(async () => {
        ctl.abort();
        throw new Error('query cancelled');
      }),
    };
    const log = vi.fn();
    await expect(runFullTextSearch([src], { ...baseCtx, signal: ctl.signal, log })).rejects.toThrow(
      'query cancelled',
    );
    expect(log).not.toHaveBeenCalled(); // never logged as a "skipped" degradation
  });

  it('threads ctx.signal to every source call (bm25 + embedding legs)', async () => {
    const a = fakeSource('A', { bm25: [hit('A', '1', 1)], embedding: [hit('A', '1', 1)] });
    const ctl = new AbortController();
    await runHybridSearch([a], { ...baseCtx, signal: ctl.signal, mode: 'hybrid', embedder });
    expect(a.bm25).toHaveBeenCalledWith(expect.objectContaining({ signal: ctl.signal }));
    expect(a.embedding).toHaveBeenCalledWith(expect.objectContaining({ signal: ctl.signal }));
    await runFullTextSearch([a], { ...baseCtx, signal: ctl.signal });
    expect(a.bm25).toHaveBeenLastCalledWith(expect.objectContaining({ signal: ctl.signal }));
  });

  it('no signal (absent) keeps the legacy behavior — sources see signal: undefined', async () => {
    const a = fakeSource('A', { bm25: [hit('A', '1', 1)] });
    const res = await runFullTextSearch([a], baseCtx);
    expect(res.totalHits).toBe(1);
    expect(a.bm25).toHaveBeenCalledWith(expect.objectContaining({ signal: undefined }));
  });
});
