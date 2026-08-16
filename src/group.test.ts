/**
 * Unit suite for the optional final-page rollup (P-036).
 *
 * Pure functions over an already-ranked list — no Postgres, no embedder. The
 * engine-level wiring (rollup sits between the recency re-rank and the top-N
 * cut, and `totalGroups` reports the pool's distinct-group count) is covered in
 * hybrid.test.ts.
 */

import { describe, it, expect } from 'vitest';
import { pickTopGroups, countGroups, type GroupKeyOf } from './group';
import type { SearchHit } from './types';

function hit(id: string, score = 0): SearchHit {
  return { source: 'S', source_id: id, excerpt: '', highlight: '', score, rankers: ['S'] };
}
/** `rrfCombine`'s entry shape, minimally. */
function entry(id: string, score = 0) {
  return { row: hit(id, score), score, rankers: ['S'] };
}
/** Session key from `sess:turn` ids — the transcript-search shape. */
const bySession: GroupKeyOf = (h) => h.source_id.split(':')[0] ?? null;

describe('pickTopGroups', () => {
  it('keeps the FIRST row per group, so `limit` counts distinct groups not rows', () => {
    // a contributes 3 of the top 4 rows; without a rollup the page is 3/4 one session.
    const fused = [entry('a:1'), entry('a:2'), entry('b:1'), entry('a:3'), entry('c:1')];
    expect(pickTopGroups(fused, bySession, 3).map((e) => e.row.source_id)).toEqual([
      'a:1',
      'b:1',
      'c:1',
    ]);
  });

  it('back-fills the freed slots from DEEPER in the pool (the whole point)', () => {
    // The rows that fill the page are past the row-level cut of 3.
    const fused = [entry('a:1'), entry('a:2'), entry('a:3'), entry('d:1'), entry('e:1')];
    const rowLevel = fused.slice(0, 3).map((e) => e.row.source_id);
    expect(rowLevel).toEqual(['a:1', 'a:2', 'a:3']); // one session, three slots
    expect(pickTopGroups(fused, bySession, 3).map((e) => e.row.source_id)).toEqual([
      'a:1',
      'd:1',
      'e:1',
    ]);
  });

  it("preserves the CALLER's order and never re-sorts by score", () => {
    // A re-rank legitimately returns rows whose carried retrieval score does
    // NOT match their new position (the P-010 hazard). Sorting on score here
    // would silently undo it — so the input is deliberately score-ASCENDING.
    const fused = [entry('a:1', 0.1), entry('b:1', 0.9), entry('c:1', 0.5)];
    expect(pickTopGroups(fused, bySession, 3).map((e) => e.row.source_id)).toEqual([
      'a:1',
      'b:1',
      'c:1',
    ]);
  });

  it('CONTROL: a score-sorting implementation fails the order property above', () => {
    // Falsifiability: if `pickTopGroups` were written the obvious (wrong) way,
    // this is the test that would catch it. Keeping the wrong version here as a
    // control means the property is proven to discriminate, not merely to pass.
    const wrong = <T extends { row: SearchHit }>(f: readonly T[], k: GroupKeyOf, n: number) =>
      pickTopGroups([...f].sort((x, y) => y.row.score - x.row.score), k, n);
    const fused = [entry('a:1', 0.1), entry('b:1', 0.9), entry('c:1', 0.5)];
    expect(wrong(fused, bySession, 3).map((e) => e.row.source_id)).not.toEqual([
      'a:1',
      'b:1',
      'c:1',
    ]);
  });

  it('never collapses UNGROUPABLE rows together — each keeps its own slot', () => {
    // A null key means "no group", not "the null group". Bucketing these
    // together would silently delete real hits whose key merely failed to parse.
    const noKey: GroupKeyOf = () => null;
    const fused = [entry('a:1'), entry('b:1'), entry('c:1')];
    expect(pickTopGroups(fused, noKey, 3)).toHaveLength(3);
  });

  it('mixes groupable and ungroupable rows without either affecting the other', () => {
    const mixed: GroupKeyOf = (h) => (h.source_id.startsWith('x') ? null : bySession(h));
    const fused = [entry('a:1'), entry('x:1'), entry('a:2'), entry('x:2')];
    expect(pickTopGroups(fused, mixed, 4).map((e) => e.row.source_id)).toEqual([
      'a:1',
      'x:1',
      'x:2',
    ]);
  });

  it('is a no-op when every row is already its own group', () => {
    const fused = [entry('a:1'), entry('b:1'), entry('c:1')];
    expect(pickTopGroups(fused, bySession, 10).map((e) => e.row.source_id)).toEqual([
      'a:1',
      'b:1',
      'c:1',
    ]);
  });

  it('returns empty for a non-positive limit rather than the whole pool', () => {
    const fused = [entry('a:1'), entry('b:1')];
    expect(pickTopGroups(fused, bySession, 0)).toEqual([]);
    expect(pickTopGroups(fused, bySession, -1)).toEqual([]);
  });

  it('stops scanning once `limit` groups are held', () => {
    let calls = 0;
    const counting: GroupKeyOf = (h) => {
      calls++;
      return bySession(h);
    };
    const fused = Array.from({ length: 100 }, (_, i) => entry(`s${i}:1`));
    pickTopGroups(fused, counting, 3);
    expect(calls).toBe(3);
  });
});

describe('countGroups', () => {
  it('counts DISTINCT groups across the whole pool', () => {
    const fused = [entry('a:1'), entry('a:2'), entry('b:1'), entry('a:3'), entry('c:1')];
    expect(countGroups(fused, bySession)).toBe(3);
  });

  it('counts each ungroupable row as its own group', () => {
    const mixed: GroupKeyOf = (h) => (h.source_id.startsWith('x') ? null : bySession(h));
    const fused = [entry('a:1'), entry('a:2'), entry('x:1'), entry('x:2')];
    expect(countGroups(fused, mixed)).toBe(3); // {a} + two ungroupable
  });

  it('is 0 for an empty pool', () => {
    expect(countGroups([], bySession)).toBe(0);
  });
});
