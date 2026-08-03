/**
 * Conformance suite for the ENGINE-LEVEL DEFAULTS seam (P-017).
 *
 * The property under test is an INVERSION, so it is tested as one: absent
 * used to mean "feature off" and now means "apply the registered policy".
 * Every case below therefore pins one cell of the three-way table —
 * value / `false` / absent — because the bug this seam exists to prevent is
 * a feature silently not applying, which no amount of "it returned results"
 * can detect.
 *
 * The end-to-end cases run through `runHybridSearch` with in-memory source
 * doubles rather than asserting on `resolveSearchDefaults` alone: a unit test
 * of the resolver would still pass if the engine forgot to USE the resolved
 * value, which is exactly the wiring mistake most likely to be made.
 */

import { describe, it, expect, afterEach, vi } from 'vitest';
import type { RankedItem } from '@papercusp/rrf';
import { runFullTextSearch, runHybridSearch, type SearchContext } from './hybrid';
import { configureSearchDefaults, resetSearchDefaults, resolveSearchDefaults } from './defaults';
import type { SearchSource, SearchHit, Listing, PgHandle, Embedder } from './types';

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
function fakeSource(name: string, opts: { bm25?: SearchHit[]; embedding?: SearchHit[] } = {}): SearchSource {
  const src: SearchSource = { name, bm25: vi.fn(async () => listing(...(opts.bm25 ?? []))) };
  if (opts.embedding) src.embedding = vi.fn(async () => listing(...(opts.embedding ?? [])));
  return src;
}

const baseCtx: SearchContext = { sql, query: 'q', workspaceId: 'w', scopeFilter: null, limit: 5 };
const embedder: Embedder = async () => [0.1, 0.2, 0.3];

/** A source whose vector leg returns one strong and one sub-floor candidate —
 *  the sparsely-embedded-corpus shape the floor exists to reject. */
function vectorSource(): SearchSource {
  return fakeSource('V', {
    bm25: [],
    embedding: [hit('V', 'strong', 0.9), hit('V', 'noise', 0.1)],
  });
}

afterEach(() => {
  resetSearchDefaults();
});

describe('configureSearchDefaults — the three-way contract', () => {
  it('with NO policy registered, an absent option filters nothing (the pre-P-017 engine)', async () => {
    const res = await runHybridSearch([vectorSource()], { ...baseCtx, mode: 'hybrid', embedder });
    expect(res.results.map((h) => h.source_id)).toEqual(['strong', 'noise']);
    expect(res.applied.minScore).toBeNull();
    expect(res.applied.policyRegistered).toBe(false);
  });

  it('APPLIES the registered default when the caller passes nothing — the inversion', async () => {
    configureSearchDefaults({ minScore: () => ({ embeddings: 0.5 }) });
    const res = await runHybridSearch([vectorSource()], { ...baseCtx, mode: 'hybrid', embedder });
    // The sub-floor candidate is gone WITHOUT the call site asking for it.
    expect(res.results.map((h) => h.source_id)).toEqual(['strong']);
    expect(res.applied.minScore).toEqual({ embeddings: 0.5 });
    expect(res.applied.policyRegistered).toBe(true);
  });

  it('lets an EXPLICIT caller value win over the registered default', async () => {
    configureSearchDefaults({ minScore: () => ({ embeddings: 0.5 }) });
    const res = await runHybridSearch([vectorSource()], {
      ...baseCtx,
      mode: 'hybrid',
      embedder,
      // Stricter than the policy: drops BOTH candidates.
      minScore: { embeddings: 0.95 },
    });
    expect(res.results).toEqual([]);
    expect(res.applied.minScore).toEqual({ embeddings: 0.95 });
  });

  it('treats `false` as a HARD OFF that outranks the registered default', async () => {
    configureSearchDefaults({ minScore: () => ({ embeddings: 0.5 }) });
    const res = await runHybridSearch([vectorSource()], {
      ...baseCtx,
      mode: 'hybrid',
      embedder,
      minScore: false,
    });
    expect(res.results.map((h) => h.source_id)).toEqual(['strong', 'noise']);
    expect(res.applied.minScore).toBeNull();
    // The distinction that makes `false` necessary at all: a policy IS
    // registered, and the caller still got the unfloored engine.
    expect(res.applied.policyRegistered).toBe(true);
  });

  it('resolves the policy from the EMBEDDER INSTANCE, so a foreign space is not floored', async () => {
    const calibrated: Embedder = async () => [1];
    const foreign: Embedder = async () => [1];
    // The shape of the real papercusp policy: recognise your own embedder,
    // return undefined (⇒ floor nothing) for anyone else's.
    configureSearchDefaults({
      minScore: (ctx) => (ctx.embedder === calibrated ? { embeddings: 0.5 } : undefined),
    });

    const floored = await runHybridSearch([vectorSource()], {
      ...baseCtx,
      mode: 'hybrid',
      embedder: calibrated,
    });
    expect(floored.results.map((h) => h.source_id)).toEqual(['strong']);

    const unfloored = await runHybridSearch([vectorSource()], {
      ...baseCtx,
      mode: 'hybrid',
      embedder: foreign,
    });
    // Degrades to today's behavior rather than importing a foreign threshold.
    expect(unfloored.results.map((h) => h.source_id)).toEqual(['strong', 'noise']);
    expect(unfloored.applied.minScore).toBeNull();
  });

  it('applies the default recency re-rank, INCLUDING its fresh-candidate leg', async () => {
    const now = Date.now();
    // 'old' outranks 'new' on pure relevance; recency must rescue 'new'.
    // `RecencyRank.getTime` defaults to reading `hit.ts`.
    const src = fakeSource('R', {
      bm25: [
        { ...hit('R', 'old', 0.9), ts: new Date(now - 90 * 86_400_000).toISOString() },
        { ...hit('R', 'new', 0.4), ts: new Date(now - 60_000).toISOString() },
      ] as SearchHit[],
    });
    configureSearchDefaults({
      recency: () => ({ weight: 1, halfLifeMs: 86_400_000 }),
    });
    const res = await runHybridSearch([src], { ...baseCtx, mode: 'hybrid', embedder: null });
    expect(res.applied.recency).toBe(true);
    expect(res.results[0]?.source_id).toBe('new');
  });

  it('a THROWING policy resolver degrades to no default instead of failing the search', async () => {
    configureSearchDefaults({
      minScore: () => {
        throw new Error('policy boom');
      },
    });
    const res = await runHybridSearch([vectorSource()], { ...baseCtx, mode: 'hybrid', embedder });
    // A misconfigured policy must never be able to take search down — the same
    // rule the engine already applies to a throwing source.
    expect(res.results.map((h) => h.source_id)).toEqual(['strong', 'noise']);
    expect(res.applied.minScore).toBeNull();
  });

  it('reaches runFullTextSearch too, and tells that path it has no vector leg', async () => {
    const seen: string[] = [];
    configureSearchDefaults({
      minScore: (ctx) => {
        seen.push(ctx.mode);
        return { bm25: 0.5 };
      },
    });
    const src = fakeSource('F', { bm25: [hit('F', 'strong', 0.9), hit('F', 'noise', 0.1)] });
    const res = await runFullTextSearch([src], baseCtx);
    expect(seen).toEqual(['fulltext']);
    expect(res.results.map((h) => h.source_id)).toEqual(['strong']);
    expect(res.applied.minScore).toEqual({ bm25: 0.5 });
  });

  it('replaces the whole policy on re-registration rather than merging it', () => {
    configureSearchDefaults({ minScore: () => ({ embeddings: 0.5 }) });
    configureSearchDefaults({ recency: () => ({ weight: 1, halfLifeMs: 86_400_000 }) });
    const ctx = { query: 'q', limit: 5, mode: 'hybrid' as const, embedder: null };
    const resolved = resolveSearchDefaults(ctx, {});
    // The first policy's floor must NOT survive into the second registration —
    // a half-applied policy is far harder to reason about than a replaced one.
    expect(resolved.minScore).toBeUndefined();
    expect(resolved.recency).toEqual({ weight: 1, halfLifeMs: 86_400_000 });
  });
});
