/**
 * Conformance suite for the embed-latency sampler (EI-21491088289861649).
 *
 * Two layers, mirroring leg-health.test.ts's split:
 *   · PURE — summariseEmbedSamples over explicit sample lists: grouping,
 *     nearest-rank percentiles, window filtering, budget attribution, and the
 *     truncation-honesty flag. No global state.
 *   · ENGINE — the observation is actually WIRED into runHybridSearch's embed
 *     choke point: every graded terminal path records exactly one sample with
 *     the caller label + the budget it was graded against, and a caller-side
 *     abort records NOTHING (a cancellation says nothing about latency).
 */

import { beforeEach, describe, expect, it } from 'vitest';
import type { RankedItem } from '@papercusp/rrf';
import { runHybridSearch } from './hybrid';
import {
  EMBED_LATENCY_CAPACITY,
  UNATTRIBUTED_CALLER,
  embedLatencyObservedCount,
  observeEmbedLatency,
  readEmbedLatency,
  resetEmbedLatency,
  summariseEmbedSamples,
  type EmbedLatencySample,
} from './embed-latency';
import type { SearchHit, SearchSource, Listing, PgHandle, Embedder } from './types';

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
function fakeSource(name: string, lex: SearchHit[]): SearchSource {
  return { name, lexical: async () => listing(...lex) };
}

const T0 = 1_000_000;

function sample(over: Partial<EmbedLatencySample> = {}): EmbedLatencySample {
  return {
    atMs: T0,
    caller: 'search:semantic',
    budgetMs: 4000,
    durationMs: 100,
    outcome: 'ok',
    ...over,
  };
}

beforeEach(() => {
  resetEmbedLatency();
});

describe('summariseEmbedSamples — PURE arithmetic', () => {
  it('nearest-rank percentiles over the retained durations, all outcomes included', () => {
    const s = (durationMs: number, outcome: EmbedLatencySample['outcome'] = 'ok') =>
      sample({ durationMs, outcome });
    // 100 samples: 99 fast oks + one slow timeout that really consumed ~its
    // budget of wall time. Excluding it would understate exactly the tail
    // this module exists to see, so p99 must land ON it (rank ceil(.99·100)=99
    // of ascending → second-largest = 120; rank 99 holds 120).
    const samples = Array.from({ length: 99 }, (_, i) => s(10 + i));
    samples.push(s(4000, 'timeout'));
    const w = summariseEmbedSamples(samples, 60 * 60_000, T0 + 1, false);
    expect(w.embeds).toBe(100);
    const c = w.callers[0]!;
    expect(c.n).toBe(100);
    expect(c.ok).toBe(99);
    expect(c.timeout).toBe(1);
    expect(c.error).toBe(0);
    expect(c.p50Ms).toBeLessThan(80);
    expect(c.p99Ms).toBeGreaterThanOrEqual(107); // near the tail, not the median
    expect(c.maxMs).toBe(4000);
  });

  it('groups per caller and takes the MOST RECENT non-null budget', () => {
    const samples = [
      sample({ caller: 'a', budgetMs: 1200 }),
      sample({ caller: 'b', budgetMs: 4000 }),
      sample({ caller: 'a', budgetMs: null }), // an unbounded re-grade must NOT erase the known budget…
      sample({ caller: 'a', budgetMs: 1500 }), // …and a newer budget WINS (last non-null)
    ];
    const w = summariseEmbedSamples(samples, 60 * 60_000, T0 + 1, false);
    const byCaller = new Map(w.callers.map((c) => [c.caller, c]));
    expect(byCaller.get('a')).toMatchObject({ n: 3, budgetMs: 1500 });
    expect(byCaller.get('b')).toMatchObject({ n: 1, budgetMs: 4000 });
  });

  it('an empty window reports null percentiles and zero counts — never a fake clean bill', () => {
    const w = summariseEmbedSamples([], 60 * 60_000, T0, false);
    expect(w.embeds).toBe(0);
    expect(w.callers).toEqual([]);
    expect(w.oldestSampleMs).toBeNull();
    expect(w.newestSampleMs).toBeNull();
  });

  it('drops samples older than the window', () => {
    const samples = [sample({ atMs: T0 - 61 * 60_000 }), sample({ atMs: T0 - 59 * 60_000 })];
    const w = summariseEmbedSamples(samples, 60 * 60_000, T0, false);
    expect(w.embeds).toBe(1);
    expect(w.callers[0]?.n).toBe(1);
  });

  it('flags capacity truncation honestly: full ring fully inside the window ⇒ counts are floors', () => {
    const samples = Array.from({ length: EMBED_LATENCY_CAPACITY }, (_, i) => sample({ atMs: T0 - i }));
    // nowMs AFTER the oldest retained sample ⇒ the window asked for more than
    // the ring can hold ⇒ evictions happened.
    const w = summariseEmbedSamples(samples, 60 * 60_000, T0 + 10_000, true);
    expect(w.truncatedByCapacity).toBe(true);
    // A full ring whose retained data FULLY COVERS the window (the window
    // does not reach back past the oldest retained sample) is NOT truncated —
    // the read is a total, not a floor.
    const covered = Array.from({ length: EMBED_LATENCY_CAPACITY }, (_, i) =>
      sample({ atMs: T0 - i * 60_000 }),
    );
    const w2 = summariseEmbedSamples(covered, 60 * 60_000, T0 + 10_000, true);
    expect(w2.truncatedByCapacity).toBe(false);
  });
});

describe('observe/read round-trip', () => {
  it('records and reads through the process-local buffer; reset clears', () => {
    observeEmbedLatency(sample());
    expect(embedLatencyObservedCount()).toBe(1);
    expect(readEmbedLatency({ windowMs: 60_000, nowMs: T0 + 1 }).embeds).toBe(1);
    resetEmbedLatency();
    expect(embedLatencyObservedCount()).toBe(0);
    expect(readEmbedLatency({ windowMs: 60_000, nowMs: T0 + 1 }).embeds).toBe(0);
  });

  it('observe NEVER throws on a malformed sample — a counter is never worth a failed search', () => {
    expect(() =>
      observeEmbedLatency(undefined as unknown as EmbedLatencySample),
    ).not.toThrow();
  });

  it('unlabeled traffic lands under the visible unattributed label, not someone else\u2019s numbers', () => {
    observeEmbedLatency(sample({ caller: UNATTRIBUTED_CALLER }));
    const w = readEmbedLatency({ windowMs: 60_000, nowMs: T0 + 1 });
    expect(w.callers.map((c) => c.caller)).toEqual([UNATTRIBUTED_CALLER]);
  });
});

describe('engine wiring — runHybridSearch records what the sampler needs', () => {
  it('a succeeding embed records one ok sample carrying the caller label AND its budget', async () => {
    const src = fakeSource('A', [hit('A', '1', 1)]);
    const embedder: Embedder = async () => [0.1, 0.2];
    await runHybridSearch([src], {
      sql,
      query: 'q',
      workspaceId: 'w',
      scopeFilter: null,
      limit: 5,
      mode: 'hybrid',
      embedder,
      embedTimeoutMs: 4000,
      caller: 'search:semantic',
    });
    const w = readEmbedLatency({ windowMs: 60_000 });
    expect(w.embeds).toBe(1);
    expect(w.callers[0]).toMatchObject({ caller: 'search:semantic', n: 1, ok: 1, budgetMs: 4000 });
  });

  it('a budget-blown embed degrades to BM25-only AND records a timeout sample with the budget attached', async () => {
    const src = fakeSource('A', [hit('A', '1', 1)]);
    const never: Embedder = () => new Promise<number[]>(() => {});
    const res = await runHybridSearch([src], {
      sql,
      query: 'q',
      workspaceId: 'w',
      scopeFilter: null,
      limit: 5,
      mode: 'hybrid',
      embedder: never,
      embedTimeoutMs: 20,
      caller: 'midturn:related-context',
    });
    expect(res.embedderAvailable).toBe(false); // degraded, not hung
    expect(res.results.length).toBeGreaterThan(0); // BM25 still answered
    const w = readEmbedLatency({ windowMs: 60_000 });
    expect(w.callers[0]).toMatchObject({
      caller: 'midturn:related-context',
      timeout: 1,
      budgetMs: 20, // THE point: the sample knows which budget was blown
    });
  });

  it('a throwing embedder records error, not timeout', async () => {
    const src = fakeSource('A', [hit('A', '1', 1)]);
    const boom: Embedder = async () => {
      throw new Error('embedder boom');
    };
    await runHybridSearch([src], {
      sql,
      query: 'q',
      workspaceId: 'w',
      scopeFilter: null,
      limit: 5,
      mode: 'hybrid',
      embedder: boom,
      embedTimeoutMs: 5000,
      caller: 'docs:search',
    });
    const c = readEmbedLatency({ windowMs: 60_000 }).callers[0]!;
    expect(c.error).toBe(1);
    expect(c.timeout).toBe(0);
  });

  it('a pre-aborted search records NOTHING — a caller-side cancellation says nothing about latency', async () => {
    const src = fakeSource('A', [hit('A', '1', 1)]);
    const ctl = new AbortController();
    ctl.abort();
    await expect(
      runHybridSearch([src], {
        sql,
        query: 'q',
        workspaceId: 'w',
        scopeFilter: null,
        limit: 5,
        mode: 'hybrid',
        embedder: async () => [0.1],
        embedTimeoutMs: 5000,
        signal: ctl.signal,
        caller: 'sessions:search',
      }),
    ).rejects.toThrow();
    expect(readEmbedLatency({ windowMs: 60_000 }).embeds).toBe(0);
  });

  it('an unlabeled call site shows up AS unattributed instead of vanishing', async () => {
    const src = fakeSource('A', [hit('A', '1', 1)]);
    await runHybridSearch([src], {
      sql,
      query: 'q',
      workspaceId: 'w',
      scopeFilter: null,
      limit: 5,
      mode: 'hybrid',
      embedder: async () => [0.1],
      embedTimeoutMs: 4000,
    });
    expect(readEmbedLatency({ windowMs: 60_000 }).callers[0]?.caller).toBe(UNATTRIBUTED_CALLER);
  });

  it('the two incident budgets are distinguishable in ONE window — the whole point (EI-21491088289861649)', async () => {
    const src = fakeSource('A', [hit('A', '1', 1)]);
    // One cold-ish embed (~1900ms simulated by the recorded duration below)
    // graded against TWO different callers' budgets at the same instant.
    const now = Date.now();
    observeEmbedLatency(
      sample({ atMs: now, caller: 'search:semantic', budgetMs: 4000, durationMs: 1924, outcome: 'ok' }),
    );
    observeEmbedLatency(
      sample({ atMs: now, caller: 'midturn:related-context', budgetMs: 1200, durationMs: 1924, outcome: 'timeout' }),
    );
    const w = readEmbedLatency({ windowMs: 60_000 });
    const semantic = w.callers.find((c) => c.caller === 'search:semantic')!;
    const midturn = w.callers.find((c) => c.caller === 'midturn:related-context')!;
    // Same sidecar health, opposite verdicts — each against ITS OWN budget.
    expect(semantic.ok).toBe(1);
    expect(midturn.timeout).toBe(1);
  });
});
