import { describe, it, expect } from 'vitest';
import type { FusedItem } from '@papercusp/rrf';
import { applyRecencyRerank, toMillis } from './recency';
import type { SearchHit } from './types';

const NOW = Date.parse('2026-07-12T12:00:00Z');
const DAY = 86_400_000;

function fi(id: string, score: number, ts: string | number | null, rankers = ['bm25']): FusedItem<SearchHit> {
  return {
    row: { source: 'session_turn', source_id: id, excerpt: '', highlight: '', score, rankers, ts },
    score,
    rankers,
  };
}

const ids = (list: FusedItem<SearchHit>[]) => list.map((x) => x.row.source_id);

describe('toMillis', () => {
  it('parses epoch-ms, ISO, and Date; rejects garbage/nullish', () => {
    expect(toMillis(NOW)).toBe(NOW);
    expect(toMillis('2026-07-12T12:00:00Z')).toBe(NOW);
    expect(toMillis(new Date(NOW))).toBe(NOW);
    expect(toMillis(null)).toBeNull();
    expect(toMillis(undefined)).toBeNull();
    expect(toMillis('not-a-date')).toBeNull();
    expect(toMillis(Number.NaN)).toBeNull();
  });
});

describe('applyRecencyRerank', () => {
  it('floats a recent, lower-relevance hit above an old, higher-relevance one', () => {
    const input = [
      fi('old-strong', 1.0, NOW - 30 * DAY), // best relevance, 30d old
      fi('new-weak', 0.6, NOW),              // weaker, brand new
    ];
    const out = applyRecencyRerank(input, { halfLifeMs: DAY, weight: 0.5, now: NOW });
    expect(ids(out)).toEqual(['new-weak', 'old-strong']);
  });

  it('weight 0 ⇒ pure relevance order, no recency ranker tag', () => {
    const input = [fi('a', 1.0, NOW - 30 * DAY), fi('b', 0.6, NOW)];
    const out = applyRecencyRerank(input, { halfLifeMs: DAY, weight: 0, now: NOW });
    expect(ids(out)).toEqual(['a', 'b']);
    expect(out.every((x) => !x.rankers.includes('recency'))).toBe(true);
  });

  it('treats a missing timestamp as oldest (no boost)', () => {
    const input = [
      fi('recent-weak', 0.5, NOW),  // recent but lower relevance
      fi('untimed-strong', 1.0, null), // best relevance, no timestamp
    ];
    const out = applyRecencyRerank(input, { halfLifeMs: DAY, weight: 0.6, now: NOW });
    expect(ids(out)).toEqual(['recent-weak', 'untimed-strong']);
  });

  it('multiply mode scales relevance by decay', () => {
    const input = [fi('old', 1.0, NOW - 30 * DAY), fi('new', 1.0, NOW)];
    const out = applyRecencyRerank(input, { halfLifeMs: DAY, weight: 1, mode: 'multiply', now: NOW });
    expect(ids(out)).toEqual(['new', 'old']);
  });

  it('appends the recency ranker for timestamped hits when weight > 0', () => {
    const out = applyRecencyRerank([fi('a', 1.0, NOW)], { halfLifeMs: DAY, weight: 0.3, now: NOW });
    expect(out[0].rankers).toContain('recency');
    // an untimed hit does NOT get the tag
    const out2 = applyRecencyRerank([fi('b', 1.0, null)], { halfLifeMs: DAY, weight: 0.3, now: NOW });
    expect(out2[0].rankers).not.toContain('recency');
  });

  it('defaults getTime to hit.ts and weight to 0.3', () => {
    const input = [fi('old', 1.0, NOW - 10 * DAY), fi('new', 0.9, NOW)];
    // default weight 0.3 with a strong decay (10 half-lives) should still surface 'new'
    const out = applyRecencyRerank(input, { halfLifeMs: DAY, now: NOW });
    expect(ids(out)).toEqual(['new', 'old']);
  });

  it('supports a custom getTime extractor', () => {
    // stash the time somewhere other than .ts and read it via getTime
    const input = [
      { ...fi('old', 1.0, null), row: { source: 's', source_id: 'old', excerpt: `${NOW - 30 * DAY}`, highlight: '', score: 1, rankers: [] } },
      { ...fi('new', 0.6, null), row: { source: 's', source_id: 'new', excerpt: `${NOW}`, highlight: '', score: 0.6, rankers: [] } },
    ] as FusedItem<SearchHit>[];
    const out = applyRecencyRerank(input, {
      halfLifeMs: DAY,
      weight: 0.5,
      now: NOW,
      getTime: (h) => Number(h.excerpt),
    });
    expect(ids(out)).toEqual(['new', 'old']);
  });

  it('is pure — does not mutate the input array or items', () => {
    const input = [fi('a', 1.0, NOW - 30 * DAY), fi('b', 0.6, NOW)];
    const snapshotScores = input.map((x) => x.score);
    const snapshotOrder = ids(input);
    applyRecencyRerank(input, { halfLifeMs: DAY, weight: 0.5, now: NOW });
    expect(input.map((x) => x.score)).toEqual(snapshotScores);
    expect(ids(input)).toEqual(snapshotOrder);
  });

  it('breaks blended-score ties by input order (stable)', () => {
    // identical score + identical ts ⇒ identical blend ⇒ keep input order
    const input = [fi('first', 0.5, NOW), fi('second', 0.5, NOW)];
    const out = applyRecencyRerank(input, { halfLifeMs: DAY, weight: 0.5, now: NOW });
    expect(ids(out)).toEqual(['first', 'second']);
  });
});
