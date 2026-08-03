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
    // P-002: weight > 0.5 is now what "recency outranks relevance" REQUIRES.
    // This used to pass at 0.5 only because relevance was compressed into a
    // fraction of its nominal range; see the co-equality test below.
    const out = applyRecencyRerank(input, { halfLifeMs: DAY, weight: 0.7, now: NOW });
    expect(ids(out)).toEqual(['new-weak', 'old-strong']);
  });

  it('P-002: weight 0.5 is EXACTLY co-equal — best-relevance/oldest ties worst-relevance/newest', () => {
    const input = [
      fi('old-strong', 1.0, NOW - 30 * DAY), // relevance 1, decay ~0
      fi('new-weak', 0.6, NOW),              // relevance 0, decay 1
    ];
    const out = applyRecencyRerank(input, { halfLifeMs: DAY, weight: 0.5, now: NOW });
    // Both blend to 0.5. This is the contract `weight` now carries, and it is
    // only expressible because both terms span the same [0,1].
    expect(out[0].score).toBeCloseTo(0.5, 6);
    expect(out[1].score).toBeCloseTo(0.5, 6);
    // A tie preserves input order (documented stability guarantee).
    expect(ids(out)).toEqual(['old-strong', 'new-weak']);
  });

  it('P-002: the relevance/recency balance is INDEPENDENT of pool size', () => {
    // THE REGRESSION TEST FOR THE ACTUAL DEFECT. Under the old max-division,
    // normRelevance spanned [0.813,1] at n=15 but [0.407,1] at n=90, so the
    // same `weight` meant different things per query — recency reached parity
    // at weight .157 / .295 / .372 for n = 15 / 45 / 90. Here the SAME pool
    // shape is built at three sizes: the newest row is always ranked LAST on
    // relevance, and at weight 0.6 (recency-dominant) it must win every time.
    for (const n of [15, 45, 90]) {
      const input = Array.from({ length: n }, (_, i) =>
        // RRF-shaped scores: 1/(60+rank), oldest-first, newest row scores worst
        fi(`r${i}`, 1 / (60 + i + 1), i === n - 1 ? NOW : NOW - 30 * DAY),
      );
      const out = applyRecencyRerank(input, { halfLifeMs: DAY, weight: 0.6, now: NOW });
      expect(ids(out)[0], `n=${n}`).toBe(`r${n - 1}`);
      // ...and at 0.4 (relevance-dominant) it must lose, again at every size.
      const out2 = applyRecencyRerank(input, { halfLifeMs: DAY, weight: 0.4, now: NOW });
      expect(ids(out2)[0], `n=${n}`).toBe('r0');
    }
  });

  it('P-002: recency can move a hit at most w·(n-1)/(1-w) ranks', () => {
    // The stated, corpus-independent property that replaces "weight 0..1,
    // 1 = co-equal / dominant". n=21, w=0.25 ⇒ at most 0.25·20/0.75 = 6.67
    // ranks, so the freshest row starting at rank 20 lands no better than 13.
    const n = 21;
    const input = Array.from({ length: n }, (_, i) =>
      fi(`r${i}`, 1 / (60 + i + 1), i === n - 1 ? NOW : NOW - 365 * DAY),
    );
    const out = applyRecencyRerank(input, { halfLifeMs: DAY, weight: 0.25, now: NOW });
    const landed = ids(out).indexOf(`r${n - 1}`);
    const maxMove = Math.floor((0.25 * (n - 1)) / 0.75);
    expect(landed).toBeGreaterThanOrEqual(n - 1 - maxMove);
  });

  it('P-002: tied relevance scores get EQUAL relevance, so ordering falls to recency', () => {
    const input = [
      fi('tie-old', 0.5, NOW - 10 * DAY),
      fi('tie-new', 0.5, NOW),
      fi('tie-mid', 0.5, NOW - 5 * DAY),
    ];
    const out = applyRecencyRerank(input, { halfLifeMs: DAY, weight: 0.3, now: NOW });
    expect(ids(out)).toEqual(['tie-new', 'tie-mid', 'tie-old']);
  });

  it('P-002: a single candidate is maximally relevant (no divide-by-zero on n=1)', () => {
    const out = applyRecencyRerank([fi('only', 0.0164, NOW - 3 * DAY)], {
      halfLifeMs: DAY,
      weight: 0.25,
      now: NOW,
    });
    // normRel 1 ⇒ 0.75·1 + 0.25·0.5^3
    expect(out[0].score).toBeCloseTo(0.75 + 0.25 * 0.125, 6);
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

  it('defaults getTime to hit.ts and weight to 0.3 — and 0.3 now literally means 30%', () => {
    const input = [fi('old', 1.0, NOW - 10 * DAY), fi('new', 0.9, NOW)];
    const out = applyRecencyRerank(input, { halfLifeMs: DAY, now: NOW });
    // P-002: the default weight can no longer flip a MAXIMAL relevance gap,
    // however strong the decay. 'old' is rank 1 of 2 (relevance 1.0) and 'new'
    // is rank 2 (relevance 0.0), so 0.7 of relevance beats 0.3 of recency even
    // at 10 half-lives. This test previously asserted the opposite, which was
    // only reachable because relevance was compressed to a 0.1-wide band.
    expect(ids(out)).toEqual(['old', 'new']);
    expect(out[0].score).toBeCloseTo(0.7 + 0.3 * Math.pow(0.5, 10), 6);
    expect(out[1].score).toBeCloseTo(0.3, 6);
    // getTime still defaults to hit.ts — both hits were timestamped, so both
    // carry the recency tag.
    expect(out.every((x) => x.rankers.includes('recency'))).toBe(true);
  });

  it('supports a custom getTime extractor', () => {
    // stash the time somewhere other than .ts and read it via getTime
    const input = [
      { ...fi('old', 1.0, null), row: { source: 's', source_id: 'old', excerpt: `${NOW - 30 * DAY}`, highlight: '', score: 1, rankers: [] } },
      { ...fi('new', 0.6, null), row: { source: 's', source_id: 'new', excerpt: `${NOW}`, highlight: '', score: 0.6, rankers: [] } },
    ] as FusedItem<SearchHit>[];
    const out = applyRecencyRerank(input, {
      halfLifeMs: DAY,
      // 0.8, not 0.5: this test is about the EXTRACTOR being consulted, so the
      // weight must be unambiguously recency-dominant rather than the exact
      // co-equal point (where the two candidates legitimately tie — see the
      // co-equality test above).
      weight: 0.8,
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
