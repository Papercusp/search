/**
 * Unit suite for the per-ranker minimum-score floors (P-001).
 *
 * These are pure functions over ranked lists — no Postgres, no embedder.
 * The engine-level behaviour (floors applied BEFORE fusion, and the
 * `rankerScores` provenance they leave behind) is covered in hybrid.test.ts.
 */

import { describe, it, expect } from 'vitest';
import { applyMinScore, resolveMinScore, type MinScoreFloors } from './min-score';
import type { Listing, SearchHit } from './types';

function hit(id: string, score: number): SearchHit {
  return { source: 'S', source_id: id, excerpt: '', highlight: '', score, rankers: ['S'] };
}
function listing(...pairs: Array<[string, number]>): Listing {
  return pairs.map(([id, score]) => ({ key: `S:${id}`, score, row: hit(id, score) }));
}

describe('resolveMinScore', () => {
  it('returns undefined with no floors at all (the opt-in default)', () => {
    expect(resolveMinScore(undefined, 'lexical')).toBeUndefined();
    expect(resolveMinScore({}, 'lexical')).toBeUndefined();
  });

  it('returns a ranker its own floor', () => {
    expect(resolveMinScore({ lexical: 0.05, embeddings: 0.4 }, 'embeddings')).toBe(0.4);
  });

  it('leaves a ranker with no entry unfiltered even when others have floors', () => {
    expect(resolveMinScore({ embeddings: 0.4 }, 'lexical')).toBeUndefined();
  });

  it('falls back to the FAMILY floor for a hyphenated leg (lexical-fresh → lexical)', () => {
    expect(resolveMinScore({ lexical: 0.05 }, 'lexical-fresh')).toBe(0.05);
  });

  it('prefers an explicit leg floor over its family', () => {
    expect(resolveMinScore({ lexical: 0.05, 'lexical-fresh': 0.01 }, 'lexical-fresh')).toBe(0.01);
  });

  it('treats NaN as absent (a mis-parsed config must not silently drop everything)', () => {
    expect(resolveMinScore({ embeddings: Number.NaN }, 'embeddings')).toBeUndefined();
    // ...and it falls through to the family rather than short-circuiting.
    expect(resolveMinScore({ lexical: 0.05, 'lexical-fresh': Number.NaN }, 'lexical-fresh')).toBe(0.05);
  });

  it('honours an explicit Infinity — "drop this ranker" is a real setting', () => {
    expect(resolveMinScore({ embeddings: Infinity }, 'embeddings')).toBe(Infinity);
    expect(resolveMinScore({ embeddings: -Infinity }, 'embeddings')).toBe(-Infinity);
  });

  it('accepts 0 as a floor (not confused with absent)', () => {
    expect(resolveMinScore({ embeddings: 0 }, 'embeddings')).toBe(0);
  });
});

describe('applyMinScore', () => {
  const list = listing(['a', 0.9], ['b', 0.5], ['c', 0.1]);

  it('is a pass-through when the ranker has no floor — same object, nothing dropped', () => {
    const out = applyMinScore(list, 'lexical', undefined);
    expect(out.list).toBe(list);
    expect(out.dropped).toBe(0);
    expect(out.floor).toBeUndefined();
  });

  it('drops entries below the floor and keeps the rest in order', () => {
    const out = applyMinScore(list, 'embeddings', { embeddings: 0.5 });
    expect(out.list.map((e) => e.key)).toEqual(['S:a', 'S:b']);
    expect(out.dropped).toBe(1);
    expect(out.floor).toBe(0.5);
  });

  it('is inclusive at the floor (>=, not >)', () => {
    const out = applyMinScore(list, 'embeddings', { embeddings: 0.5 });
    expect(out.list.some((e) => e.key === 'S:b')).toBe(true);
  });

  it('can empty a list entirely — a ranker with nothing above the floor contributes nothing', () => {
    const out = applyMinScore(list, 'embeddings', { embeddings: 0.95 });
    expect(out.list).toEqual([]);
    expect(out.dropped).toBe(3);
  });

  it('returns the original list object when the floor cuts nothing', () => {
    const out = applyMinScore(list, 'embeddings', { embeddings: 0.05 });
    expect(out.list).toBe(list);
    expect(out.dropped).toBe(0);
    expect(out.floor).toBe(0.05);
  });

  it('drops a NaN score when a floor applies — an unrankable score cannot clear a bar', () => {
    const withNaN = listing(['a', 0.9], ['bad', Number.NaN]);
    const out = applyMinScore(withNaN, 'embeddings', { embeddings: 0 });
    expect(out.list.map((e) => e.key)).toEqual(['S:a']);
    expect(out.dropped).toBe(1);
  });

  it('keeps a NaN score when NO floor applies (pass-through changes nothing)', () => {
    const withNaN = listing(['a', 0.9], ['bad', Number.NaN]);
    expect(applyMinScore(withNaN, 'embeddings', {}).list).toHaveLength(2);
  });

  it('applies the family floor to a hyphenated leg', () => {
    const floors: MinScoreFloors = { lexical: 0.5 };
    expect(applyMinScore(list, 'lexical-fresh', floors).list.map((e) => e.key)).toEqual([
      'S:a',
      'S:b',
    ]);
  });

  it('never reorders — survivors keep their relative rank so ranks tighten up', () => {
    const jumbled = listing(['a', 0.9], ['b', 0.2], ['c', 0.8], ['d', 0.1]);
    const out = applyMinScore(jumbled, 'embeddings', { embeddings: 0.5 });
    expect(out.list.map((e) => e.key)).toEqual(['S:a', 'S:c']);
  });
});
