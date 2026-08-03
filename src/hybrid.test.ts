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
import { EmbedTimeoutError, runFullTextSearch, runHybridSearch, type SearchContext } from './hybrid';
import type { SearchSource, SearchSourceParams, SearchHit, Listing, PgHandle, Embedder } from './types';

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

// ── Embed budget (embedTimeoutMs) — 2026-07-10 interactive-search hardening ──
// An interactive search (the desktop transcript pill) must not block on a slow
// or COLD query-embedder. When the embed exceeds `ctx.embedTimeoutMs` the hybrid
// search degrades to BM25-only (queryVec null) instead of hanging to the route
// watchdog and 408ing (measured: search:fulltext 20ms vs hybrid 30s, all the
// time in the single `await embedder(query)`). A real route-abort (ctx.signal)
// still REJECTS and outranks the timeout. undefined/≤0 budget = unbounded (the
// prior behavior — strictly opt-in per caller).
describe('embed budget (embedTimeoutMs)', () => {
  const neverEmbedder: Embedder = () => new Promise<number[]>(() => {}); // never settles

  it('a query-embed that exceeds the budget degrades to BM25-only (embedderAvailable=false)', async () => {
    const log = vi.fn();
    const a = fakeSource('A', { bm25: [hit('A', '1', 3)], embedding: [hit('A', '9', 9)] });
    const res = await runHybridSearch([a], {
      ...baseCtx, mode: 'hybrid', embedder: neverEmbedder, embedTimeoutMs: 10, log,
    });
    expect(res.embedderAvailable).toBe(false);
    expect(res.results.map((h) => h.source_id)).toEqual(['1']); // the BM25 hit, NOT the embed-only hit
    expect(a.embedding).not.toHaveBeenCalled(); // no queryVec ⇒ vector leg never runs
    expect(log).toHaveBeenCalledWith(expect.stringContaining('exceeded 10ms budget'));
  });

  it('a query-embed within the budget runs the semantic leg normally', async () => {
    const embedder: Embedder = async () => [0.1, 0.2, 0.3];
    const a = fakeSource('A', { bm25: [hit('A', '1', 3)], embedding: [hit('A', '1', 9)] });
    const res = await runHybridSearch([a], { ...baseCtx, mode: 'hybrid', embedder, embedTimeoutMs: 500 });
    expect(res.embedderAvailable).toBe(true);
    expect(a.embedding).toHaveBeenCalled();
  });

  it('an undefined budget stays unbounded — the embed is awaited to completion', async () => {
    const embedder: Embedder = async () => [0.1, 0.2, 0.3];
    const a = fakeSource('A', { bm25: [hit('A', '1', 3)], embedding: [hit('A', '1', 9)] });
    const res = await runHybridSearch([a], { ...baseCtx, mode: 'hybrid', embedder }); // no embedTimeoutMs
    expect(res.embedderAvailable).toBe(true);
  });

  it('a route-abort during the embed wait REJECTS and outranks the timeout budget', async () => {
    const a = fakeSource('A', { bm25: [hit('A', '1', 1)], embedding: [hit('A', '1', 1)] });
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 5);
    await expect(
      runHybridSearch([a], {
        ...baseCtx, mode: 'hybrid', embedder: neverEmbedder, embedTimeoutMs: 10_000, signal: ctl.signal,
      }),
    ).rejects.toThrow();
    clearTimeout(t);
  });

  it('EmbedTimeoutError carries the budget and is distinct from a plain embed failure', () => {
    const e = new EmbedTimeoutError(4000);
    expect(e).toBeInstanceOf(Error);
    expect(e.name).toBe('EmbedTimeoutError');
    expect(e.budgetMs).toBe(4000);
    expect(e.message).toContain('4000ms');
  });
});

describe('WI-4734 — embed ∥ BM25 concurrency', () => {
  it('runs the BM25 legs concurrently with the query embed (BM25 never waits behind it)', async () => {
    // The embedder only resolves once bm25 has ALREADY run: under the old
    // serial shape (await embed → then bm25) the embed would hit its 500ms
    // budget and degrade to BM25-only. Under the parallel shape, bm25 runs
    // inside the embed window, releases the gate, and the embedding leg
    // completes — so embedderAvailable === true is the concurrency proof.
    let releaseEmbed!: () => void;
    const embedGate = new Promise<void>((r) => { releaseEmbed = r; });
    const src: SearchSource = {
      name: 'A',
      bm25: vi.fn(async () => {
        releaseEmbed();
        return listing(hit('A', 'kw', 2));
      }),
      embedding: vi.fn(async () => listing(hit('A', 'vec', 1))),
    };
    const embedder: Embedder = vi.fn(async () => {
      await embedGate;
      return [0.1, 0.2];
    });
    const res = await runHybridSearch([src], {
      ...baseCtx,
      mode: 'hybrid',
      embedder,
      embedTimeoutMs: 500,
    });
    expect(res.embedderAvailable).toBe(true);
    expect(src.bm25).toHaveBeenCalledTimes(1);
    expect(src.embedding).toHaveBeenCalledTimes(1);
    const ids = res.results.map((h) => h.source_id);
    expect(ids).toContain('kw');
    expect(ids).toContain('vec');
  });
});

describe('WI-4734 — deferred highlight hydration (deferHighlight)', () => {
  /** A source that records the wantHighlight it was called with and supports
   *  batched post-fusion hydration. */
  function deferringSource(name: string, hits: SearchHit[]) {
    const seen: Array<boolean | undefined> = [];
    const hydrate = vi.fn(async (_p: unknown, ids: string[]) =>
      new Map(ids.map((id) => [id, `HYDRATED-${id}`])),
    );
    const src: SearchSource = {
      name,
      bm25: vi.fn(async (p) => {
        seen.push(p.wantHighlight);
        return listing(...hits.map((h) => ({ ...h, highlight: '' })));
      }),
      hydrateHighlights: hydrate,
    };
    return { src, seen, hydrate };
  }

  it('passes wantHighlight:false to a hydrating source and hydrates ONLY the final top-N', async () => {
    const { src, seen, hydrate } = deferringSource('A', [
      hit('A', '1', 3), hit('A', '2', 2), hit('A', '3', 1),
    ]);
    const res = await runHybridSearch([src], {
      ...baseCtx,
      limit: 2,
      mode: 'hybrid',
      embedder: null,
      deferHighlight: true,
    });
    expect(seen).toEqual([false]);
    expect(hydrate).toHaveBeenCalledTimes(1);
    const hydratedIds = hydrate.mock.calls[0][1] as string[];
    expect(hydratedIds).toHaveLength(2); // final top-N only, never the 3-candidate pool
    expect(res.results.map((h) => h.highlight)).toEqual(
      res.results.map((h) => `HYDRATED-${h.source_id}`),
    );
  });

  it('a source WITHOUT hydrateHighlights keeps inline highlights (wantHighlight never forced false)', async () => {
    const seen: Array<boolean | undefined> = [];
    const src: SearchSource = {
      name: 'A',
      bm25: vi.fn(async (p) => {
        seen.push(p.wantHighlight);
        return listing(hit('A', '1', 1));
      }),
    };
    const res = await runHybridSearch([src], {
      ...baseCtx,
      mode: 'hybrid',
      embedder: null,
      deferHighlight: true,
    });
    expect(seen).toEqual([undefined]);
    expect(res.results[0].highlight).toBe('hl-1'); // the inline highlight survived
  });

  it('a hydration failure degrades that source to its existing highlights — the search still returns', async () => {
    const src: SearchSource = {
      name: 'A',
      bm25: vi.fn(async () => listing({ ...hit('A', '1', 1), highlight: '' })),
      hydrateHighlights: vi.fn(async () => { throw new Error('hydrate boom'); }),
    };
    const log = vi.fn();
    const res = await runHybridSearch([src], {
      ...baseCtx,
      mode: 'hybrid',
      embedder: null,
      deferHighlight: true,
      log,
    });
    expect(res.results).toHaveLength(1);
    expect(res.results[0].highlight).toBe('');
    expect(log).toHaveBeenCalledWith(expect.stringContaining('highlight A skipped'));
  });

  it('without deferHighlight the seam is inert — no wantHighlight, no hydration call', async () => {
    const { src, seen, hydrate } = deferringSource('A', [hit('A', '1', 1)]);
    await runHybridSearch([src], { ...baseCtx, mode: 'hybrid', embedder: null });
    expect(seen).toEqual([undefined]);
    expect(hydrate).not.toHaveBeenCalled();
  });
});

// ── WI-5097 — the fresh-candidate leg (RecencyRank.freshWindowMs) ───────────
// The recency re-rank alone can only reorder candidates that survived the
// relevance-only over-fetch cut; on a large corpus, old high-term-density rows
// can fill the whole pool and starve every recent match out of eligibility.
// With freshWindowMs set, the engine adds a recent-window BM25 leg as one more
// RRF input so recent matches are guaranteed pool representation.
describe('WI-5097 — fresh-candidate leg (recency.freshWindowMs)', () => {
  const NOW = 1_800_000_000_000;
  const H = 3_600_000;
  const iso = (t: number) => new Date(t).toISOString();

  /** A source whose bm25 honors filters.since over a timed corpus — rows are
   *  score-desc, capped at p.limit, exactly like a real tsvector source. */
  function timedSource(name: string, rows: Array<{ id: string; ts: number; score: number }>): SearchSource {
    return {
      name,
      bm25: vi.fn(async (p: SearchSourceParams) => {
        const since = p.filters?.since ? Date.parse(p.filters.since) : null;
        return rows
          .filter((r) => since == null || r.ts >= since)
          .sort((a, b) => b.score - a.score)
          .slice(0, p.limit)
          .map((r) => ranked({ ...hit(name, r.id, r.score), ts: iso(r.ts) }));
      }),
    };
  }

  // 6 old spammy rows (30d) outscore 1 recent row (1h) — with limit 2 the
  // candidateLimit-6 main leg is 100% old rows; the recent one never makes it.
  const corpus = [
    ...Array.from({ length: 6 }, (_, i) => ({ id: `old-${i}`, ts: NOW - 30 * 24 * H, score: 10 - i })),
    { id: 'recent', ts: NOW - 1 * H, score: 1 },
  ];
  const recency = { halfLifeMs: 24 * H, weight: 0.3, freshWindowMs: 48 * H, now: NOW };

  it('rescues a recent match that the relevance-only pool cut would starve out', async () => {
    const src = timedSource('A', corpus);

    // CONTROL — no fresh window: the recent row is not even a candidate.
    const control = await runHybridSearch([src], {
      ...baseCtx, limit: 2, mode: 'hybrid', embedder: null,
      recency: { ...recency, freshWindowMs: 0 },
    });
    expect(control.results.map((h) => h.source_id)).not.toContain('recent');

    // WITH the fresh leg: the recent row enters via the since-window list and
    // the blended re-rank puts it FIRST (decay ~0.97 vs ~0 for 30d-old rows).
    const res = await runHybridSearch([src], {
      ...baseCtx, limit: 2, mode: 'hybrid', embedder: null, recency,
    });
    expect(res.results[0].source_id).toBe('recent');
    // The fresh leg queried the source with since = now - freshWindowMs.
    expect(src.bm25).toHaveBeenLastCalledWith(
      expect.objectContaining({ filters: expect.objectContaining({ since: iso(NOW - 48 * H) }) }),
    );
  });

  it('drops a fresh list identical to the main one (source ignoring `since` is never double-weighted)', async () => {
    const rows = [hit('A', '1', 3), hit('A', '2', 2)];
    const ignoring = fakeSource('A', { bm25: rows }); // canned list — ignores filters entirely
    const withFresh = await runHybridSearch([ignoring], { ...baseCtx, mode: 'hybrid', embedder: null, recency });
    const without = await runHybridSearch([ignoring], {
      ...baseCtx, mode: 'hybrid', embedder: null, recency: { ...recency, freshWindowMs: 0 },
    });
    // Same hits, same fused-then-blended scores — the redundant list added nothing.
    expect(withFresh.results.map((h) => [h.source_id, h.score])).toEqual(
      without.results.map((h) => [h.source_id, h.score]),
    );
  });

  it('runs no fresh leg when recency is absent, weight is 0, or the window is 0', async () => {
    for (const rec of [undefined, { ...recency, weight: 0 }, { ...recency, freshWindowMs: 0 }]) {
      const src = timedSource('A', corpus);
      await runHybridSearch([src], { ...baseCtx, mode: 'hybrid', embedder: null, recency: rec });
      expect(src.bm25).toHaveBeenCalledTimes(1);
    }
  });

  it('skips the fresh leg when the caller`s own since filter is already as narrow or narrower', async () => {
    const src = timedSource('A', corpus);
    await runHybridSearch([src], {
      ...baseCtx, mode: 'hybrid', embedder: null, recency,
      filters: { since: iso(NOW - 24 * H) }, // narrower than the 48h window
    });
    expect(src.bm25).toHaveBeenCalledTimes(1);

    // A WIDER caller window still gets the fresh leg (clamped to the window).
    const src2 = timedSource('B', corpus);
    await runHybridSearch([src2], {
      ...baseCtx, mode: 'hybrid', embedder: null, recency,
      filters: { since: iso(NOW - 30 * 24 * H) },
    });
    expect(src2.bm25).toHaveBeenCalledTimes(2);
    expect(src2.bm25).toHaveBeenLastCalledWith(
      expect.objectContaining({
        filters: expect.objectContaining({ since: iso(NOW - 48 * H) }),
      }),
    );
  });

  it('never runs a fresh leg in embeddings-only mode (it is a BM25 construct)', async () => {
    const src = timedSource('A', corpus);
    const embedder: Embedder = async () => [0.1];
    await runHybridSearch([src], { ...baseCtx, mode: 'embeddings', embedder, recency });
    expect(src.bm25).not.toHaveBeenCalled();
  });
});

describe('P-001 — per-ranker minScore floors (applied BEFORE fusion)', () => {
  const embedder: Embedder = async () => [0.1, 0.2, 0.3];

  it('no floors ⇒ byte-identical ranking to the pre-floor engine', async () => {
    const src = (): SearchSource =>
      fakeSource('A', {
        bm25: [hit('A', '1', 0.9), hit('A', '2', 0.01)],
        embedding: [hit('A', '3', 0.8), hit('A', '4', 0.02)],
      });
    const before = await runHybridSearch([src()], { ...baseCtx, mode: 'hybrid', embedder });
    const after = await runHybridSearch([src()], {
      ...baseCtx,
      mode: 'hybrid',
      embedder,
      minScore: {},
    });
    expect(after.results.map((h) => h.source_id)).toEqual(before.results.map((h) => h.source_id));
    expect(after.totalHits).toBe(before.totalHits);
  });

  it('drops sub-floor candidates from the VECTOR leg before fusion', async () => {
    const a = fakeSource('A', {
      bm25: [hit('A', 'lex', 0.5)],
      // Rank 1 in its own list, but a similarity nobody would call a match.
      embedding: [hit('A', 'noise', 0.03), hit('A', 'real', 0.72)],
    });
    const res = await runHybridSearch([a], {
      ...baseCtx,
      mode: 'hybrid',
      embedder,
      minScore: { embeddings: 0.35 },
    });
    const ids = res.results.map((h) => h.source_id);
    expect(ids).toContain('real');
    expect(ids).not.toContain('noise');
  });

  it('is what stops rank-1 noise outranking a real hit — the defect RRF cannot see', async () => {
    // 'noise' is rank 1 of the vector leg, so RRF hands it the SAME 1/(k+1)
    // contribution as a rank-1 hit from any other ranker. Only a pre-fusion
    // floor can tell the difference, because fusion discards the native score.
    const src = (): SearchSource =>
      fakeSource('A', {
        bm25: [hit('A', 'lex', 0.4)],
        embedding: [hit('A', 'noise', 0.02)],
      });
    const unfloored = await runHybridSearch([src()], { ...baseCtx, mode: 'hybrid', embedder });
    expect(unfloored.results.map((h) => h.source_id)).toContain('noise');
    expect(unfloored.results[0]?.score).toBeCloseTo(unfloored.results[1]?.score ?? -1);

    const floored = await runHybridSearch([src()], {
      ...baseCtx,
      mode: 'hybrid',
      embedder,
      minScore: { embeddings: 0.3 },
    });
    expect(floored.results.map((h) => h.source_id)).toEqual(['lex']);
  });

  it('floors each ranker in its OWN units — one number cannot serve both scales', async () => {
    // A lexical ts_rank_cd of 0.08 is a perfectly ordinary hit; a cosine of
    // 0.08 is noise. A single global floor either keeps both or kills both.
    const a = fakeSource('A', {
      bm25: [hit('A', 'lex', 0.08)],
      embedding: [hit('A', 'vec', 0.08)],
    });
    const res = await runHybridSearch([a], {
      ...baseCtx,
      mode: 'hybrid',
      embedder,
      minScore: { bm25: 0.02, embeddings: 0.35 },
    });
    expect(res.results.map((h) => h.source_id)).toEqual(['lex']);
  });

  it('a floor that empties a ranker leaves the other ranker intact, not an empty search', async () => {
    const a = fakeSource('A', {
      bm25: [hit('A', '1', 0.6), hit('A', '2', 0.5)],
      embedding: [hit('A', '9', 0.05)],
    });
    const res = await runHybridSearch([a], {
      ...baseCtx,
      mode: 'hybrid',
      embedder,
      minScore: { embeddings: 0.5 },
    });
    expect(res.results.map((h) => h.source_id)).toEqual(['1', '2']);
    expect(res.embedderAvailable).toBe(true); // the embedder worked; its hits were rejected
  });

  it('logs what a floor cut, per ranker + source', async () => {
    const log = vi.fn();
    const a = fakeSource('A', { bm25: [hit('A', '1', 0.6)], embedding: [hit('A', '9', 0.05)] });
    await runHybridSearch([a], {
      ...baseCtx,
      mode: 'hybrid',
      embedder,
      minScore: { embeddings: 0.5 },
      log,
    });
    expect(log).toHaveBeenCalledWith(
      expect.stringContaining('embeddings A minScore 0.5 dropped 1/1'),
    );
  });

  it('applies the bm25 family floor to the fresh leg too', async () => {
    const a = fakeSource('A', { bm25: [hit('A', '1', 0.01)] });
    const log = vi.fn();
    await runHybridSearch([a], {
      ...baseCtx,
      mode: 'hybrid',
      embedder: null,
      log,
      recency: { weight: 0.3, halfLifeMs: 24 * 3600_000, freshWindowMs: 48 * 3600_000 },
      minScore: { bm25: 0.1 },
    });
    expect(log).toHaveBeenCalledWith(expect.stringContaining('bm25-fresh A minScore 0.1'));
  });

  it('runFullTextSearch honours the bm25 floor as well', async () => {
    const a = fakeSource('A', { bm25: [hit('A', '1', 0.6), hit('A', '2', 0.004)] });
    const res = await runFullTextSearch([a], { ...baseCtx, minScore: { bm25: 0.01 } });
    expect(res.results.map((h) => h.source_id)).toEqual(['1']);
    expect(res.totalHits).toBe(1); // the floored-out row is not a "hit" at all
  });
});

describe('P-001 — rankerScores provenance (pre-fusion native scores)', () => {
  const embedder: Embedder = async () => [0.1, 0.2, 0.3];

  it('carries each ranker`s NATIVE score, keyed by ranker', async () => {
    const a = fakeSource('A', {
      bm25: [hit('A', '1', 0.0731)],
      embedding: [hit('A', '1', 0.6142)],
    });
    const res = await runHybridSearch([a], { ...baseCtx, mode: 'hybrid', embedder });
    expect(res.results[0]?.rankerScores).toEqual({ bm25: 0.0731, embeddings: 0.6142 });
    // ...and `score` is the RRF value, which is why the provenance is needed.
    expect(res.results[0]?.score).not.toBe(0.6142);
  });

  it('reports ONLY the rankers that actually returned the hit', async () => {
    const a = fakeSource('A', {
      bm25: [hit('A', 'lex-only', 0.5)],
      embedding: [hit('A', 'vec-only', 0.8)],
    });
    const res = await runHybridSearch([a], { ...baseCtx, mode: 'hybrid', embedder });
    const byId = Object.fromEntries(res.results.map((h) => [h.source_id, h.rankerScores]));
    expect(byId['lex-only']).toEqual({ bm25: 0.5 });
    expect(byId['vec-only']).toEqual({ embeddings: 0.8 });
  });

  it('reports the score that SURVIVED the floor, never a dropped one', async () => {
    const a = fakeSource('A', {
      bm25: [hit('A', '1', 0.5)],
      embedding: [hit('A', '1', 0.04)],
    });
    const res = await runHybridSearch([a], {
      ...baseCtx,
      mode: 'hybrid',
      embedder,
      minScore: { embeddings: 0.35 },
    });
    expect(res.results[0]?.rankerScores).toEqual({ bm25: 0.5 });
    expect(res.results[0]?.rankers).toEqual(['bm25']);
  });

  it('does not alias engine state — mutating a hit`s rankerScores cannot corrupt another', async () => {
    const a = fakeSource('A', { bm25: [hit('A', '1', 0.5)], embedding: [hit('A', '1', 0.9)] });
    const res = await runHybridSearch([a], { ...baseCtx, mode: 'hybrid', embedder });
    res.results[0]!.rankerScores!.bm25 = -999;
    const again = await runHybridSearch([a], { ...baseCtx, mode: 'hybrid', embedder });
    expect(again.results[0]?.rankerScores?.bm25).toBe(0.5);
  });

  it('runFullTextSearch carries the same shape (bm25 native score)', async () => {
    const a = fakeSource('A', { bm25: [hit('A', '1', 0.42)] });
    const res = await runFullTextSearch([a], baseCtx);
    expect(res.results[0]?.rankerScores).toEqual({ bm25: 0.42 });
  });
});
