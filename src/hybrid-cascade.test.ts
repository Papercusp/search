/**
 * WI-37582 — the two-stage lexical cascade (`SearchContext.lexicalCascade`).
 *
 * WHAT IS BEING PINNED. `lexicalSql` builds `plainto_tsquery`, which ANDs every
 * lexeme, so a document must contain ALL query terms and recall falls with query
 * length by construction. Measured on the live corpus (D-056, 101 queries,
 * n=20/cell, reproduced exactly across two independent runs), the AND leg
 * returns ZERO rows for 30% of 1-2 term, 45-50% of 3-14 term and 70-90% of 15+
 * term real queries. The cascade re-queries in `coverage-graded` mode when
 * stage 1 under-fills and APPENDS what stage 1 missed.
 *
 * WHY THE APPEND-ONLY SHAPE IS THE WHOLE DESIGN. WI-8579 measured the simpler
 * single-query swap LOSING 180 admitted lines across 79 of 120 live turns (18 of
 * them the control's top-1) while the mean stayed flat — a SWAP, not an
 * addition, because a cap and char budget were applied over a REORDERED walk.
 * Every assertion here exists to make that specific regression impossible, so
 * `cascadeWrong` below reproduces it deliberately as an in-file control.
 *
 * WHY THE ORDER ASSERTIONS DRIVE **HYBRID** AND NOT FULLTEXT. `runFullTextSearch`
 * re-ranks globally by native score before the cut, so its page order does not
 * expose the per-source listing order the cascade actually guarantees. Hybrid
 * with a single source and no embedder fuses one list via RRF, which ranks by
 * list POSITION — so the page order IS the listing order, and the append-only
 * property becomes observable through the public API.
 *
 * `minScore: false` throughout: these tests isolate the cascade from the
 * registered floor policy (P-017), which would otherwise drop the small fake
 * scores and confound "the cascade did not fire" with "the floor ate it".
 */

import { describe, it, expect } from 'vitest';
import type { RankedItem } from '@papercusp/rrf';
import { runFullTextSearch, runHybridSearch, type SearchContext } from './hybrid';
import type { SearchSource, SearchSourceParams, SearchHit, Listing, PgHandle } from './types';

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

interface CascadeCall {
  mode: SearchSourceParams['lexicalMode'];
  since: string | undefined;
  limit: number;
}

/**
 * A source whose AND and coverage-graded responses differ, recording every call
 * so the test can assert not just the ROWS but whether a second query was paid
 * for at all — "did not fire" and "fired and found nothing" are different
 * outcomes and only the call log separates them.
 */
function cascadingSource(
  name: string,
  opts: { and: SearchHit[]; graded: SearchHit[] },
): SearchSource & { calls: CascadeCall[] } {
  const calls: CascadeCall[] = [];
  return {
    name,
    calls,
    async lexical(p: SearchSourceParams): Promise<Listing> {
      calls.push({ mode: p.lexicalMode, since: p.filters?.since, limit: p.limit });
      return listing(...(p.lexicalMode === 'coverage-graded' ? opts.graded : opts.and));
    },
  };
}

const baseCtx: SearchContext = {
  sql,
  query: 'q',
  workspaceId: 'w',
  scopeFilter: null,
  limit: 5,
  minScore: false,
};

const refs = (hits: SearchHit[]): string[] => hits.map((h) => `${h.source}:${h.source_id}`);

/** Page order for a single-source hybrid run = that source's lexical listing
 *  order (RRF over one list ranks by position). */
async function hybridPage(source: SearchSource, ctx: Partial<SearchContext> = {}): Promise<string[]> {
  const res = await runHybridSearch([source], {
    ...baseCtx,
    ...ctx,
    mode: 'hybrid',
    embedder: null,
    recency: false,
  });
  return refs(res.results);
}

// ── The property under test, as a reusable predicate ────────────────────────
// Stated once so the SAME check can be run against the real engine (must hold)
// and against the deliberately-wrong control (must fail). A property asserted
// only where it passes is not evidence that it can fail.

/** Append-only: `widened` starts with `baseline` verbatim — same rows, same
 *  order — and adds only after it. This is strictly stronger than a set
 *  superset, and it is the stronger claim that makes the widening safe: RRF
 *  fuses on position, so a preserved prefix means no existing hit was demoted. */
function isAppendOnly(baseline: string[], widened: string[]): boolean {
  return widened.slice(0, baseline.length).join('\0') === baseline.join('\0');
}

/** The WI-8579 regression SHAPE, reproduced on purpose: merge both stages, then
 *  re-rank the COMBINED walk by score and truncate to the cap. Every row is
 *  still "considered", which is exactly why it reads as safe — and it silently
 *  drops stage-1 rows off the end of the cut. */
function cascadeWrong(stage1: Listing, graded: Listing, limit: number): Listing {
  const seen = new Set(stage1.map((i) => i.row.source_id));
  const merged = [...stage1, ...graded.filter((i) => !seen.has(i.row.source_id))];
  return [...merged].sort((a, b) => b.score - a.score).slice(0, limit);
}

describe('lexical cascade — when it fires', () => {
  it('fires on under-fill: re-queries the SAME source in coverage-graded mode', async () => {
    const src = cascadingSource('S', {
      and: [hit('S', 'a', 9)],
      graded: [hit('S', 'a', 9), hit('S', 'g1', 5), hit('S', 'g2', 4)],
    });
    const res = await runFullTextSearch([src], baseCtx);

    expect(src.calls.map((c) => c.mode)).toEqual([undefined, 'coverage-graded']);
    expect(refs(res.results).sort()).toEqual(['S:a', 'S:g1', 'S:g2']);
  });

  it('does NOT fire when stage 1 already fills the limit — no second query is paid for', async () => {
    const src = cascadingSource('S', {
      and: [hit('S', 'a', 9), hit('S', 'b', 8), hit('S', 'c', 7)],
      graded: [hit('S', 'z', 99)],
    });
    const res = await runFullTextSearch([src], { ...baseCtx, limit: 3 });

    expect(src.calls).toHaveLength(1);
    expect(refs(res.results)).not.toContain('S:z');
  });

  it('does NOT fire when the CALLER set lexicalMode — the caller is driving the leg', async () => {
    // Protects the corpus-injection leg, which runs its own two-stage cascade
    // with a separately-derived stage-2 query and a hard latency budget.
    const src = cascadingSource('S', { and: [hit('S', 'a', 9)], graded: [hit('S', 'g1', 5)] });
    await runFullTextSearch([src], { ...baseCtx, lexicalMode: 'coverage-graded' });

    expect(src.calls).toHaveLength(1);
    expect(src.calls[0]!.mode).toBe('coverage-graded');
  });

  it('lexicalCascade:false opts out entirely', async () => {
    const src = cascadingSource('S', { and: [hit('S', 'a', 9)], graded: [hit('S', 'g1', 5)] });
    const res = await runFullTextSearch([src], { ...baseCtx, lexicalCascade: false });

    expect(src.calls).toHaveLength(1);
    expect(refs(res.results)).toEqual(['S:a']);
  });

  it('defaults ON — an untouched caller gets the widening', async () => {
    // The deliberate inversion of the usual opt-in reflex: every prior ranking
    // feature shipped opt-in and was never hand-propagated.
    const src = cascadingSource('S', { and: [], graded: [hit('S', 'g1', 5)] });
    const res = await runFullTextSearch([src], baseCtx);

    expect(src.calls).toHaveLength(2);
    expect(refs(res.results)).toEqual(['S:g1']);
  });

  it('rescues the measured worst case: an AND query that eliminates everything', async () => {
    // D-056: this is 70-90% of 15+ term queries. fulltext is lexical BY
    // CONTRACT, so before the cascade this was an EMPTY PAGE, not a degrade.
    const src = cascadingSource('S', { and: [], graded: [hit('S', 'g1', 5), hit('S', 'g2', 4)] });
    const before = await runFullTextSearch([src], { ...baseCtx, lexicalCascade: false });
    const after = await runFullTextSearch([src], baseCtx);

    expect(before.results).toHaveLength(0);
    expect(refs(after.results).sort()).toEqual(['S:g1', 'S:g2']);
  });
});

describe('lexical cascade — the append-only guarantee', () => {
  it('preserves stage-1 rows AND their order, appending only after them', async () => {
    // Stage-2 rows are given HIGHER native scores than stage 1 on purpose: a
    // merge-then-re-rank implementation would hoist them above the stage-1
    // rows, and that is the regression this assertion is here to catch.
    const src = cascadingSource('S', {
      and: [hit('S', 'a', 1), hit('S', 'b', 0.5)],
      graded: [hit('S', 'g1', 99), hit('S', 'a', 1), hit('S', 'g2', 98)],
    });
    const baseline = await hybridPage(cascadingSource('S', {
      and: [hit('S', 'a', 1), hit('S', 'b', 0.5)],
      graded: [],
    }), { lexicalCascade: false });
    const widened = await hybridPage(src);

    expect(baseline).toEqual(['S:a', 'S:b']);
    expect(widened).toEqual(['S:a', 'S:b', 'S:g1', 'S:g2']);
    expect(isAppendOnly(baseline, widened)).toBe(true);
  });

  it('de-duplicates by source_id — a row stage 1 already returned is not appended twice', async () => {
    const src = cascadingSource('S', {
      and: [hit('S', 'a', 9)],
      graded: [hit('S', 'a', 3), hit('S', 'a', 2), hit('S', 'g1', 1)],
    });
    const page = await hybridPage(src);

    expect(page).toEqual(['S:a', 'S:g1']);
  });

  it('appends at most the room stage 1 left — never more than the limit', async () => {
    const src = cascadingSource('S', {
      and: [hit('S', 'a', 9), hit('S', 'b', 8)],
      graded: [hit('S', 'g1', 7), hit('S', 'g2', 6), hit('S', 'g3', 5), hit('S', 'g4', 4)],
    });
    const page = await hybridPage(src, { limit: 3 });

    expect(page).toEqual(['S:a', 'S:b', 'S:g1']);
  });

  it('CONTROL: the WI-8579 merge-then-re-rank shape FAILS the same predicate', async () => {
    // Falsifiability. A guard that has never rejected anything is not evidence.
    // Same inputs, same cap — only the walk order differs.
    const stage1 = listing(hit('S', 'a', 1), hit('S', 'b', 0.5));
    const graded = listing(hit('S', 'g1', 99), hit('S', 'g2', 98));
    const baseline = ['S:a', 'S:b'];

    const wrong = refs(cascadeWrong(stage1, graded, 3).map((i) => i.row));
    expect(wrong).toEqual(['S:g1', 'S:g2', 'S:a']); // 'S:b' silently dropped
    expect(isAppendOnly(baseline, wrong)).toBe(false);

    // CALIBRATION: the predicate is not simply always-false — the real engine,
    // on the SAME inputs and cap, passes it.
    const real = await hybridPage(
      cascadingSource('S', {
        and: [hit('S', 'a', 1), hit('S', 'b', 0.5)],
        graded: [hit('S', 'g1', 99), hit('S', 'g2', 98)],
      }),
      { limit: 3 },
    );
    expect(isAppendOnly(baseline, real)).toBe(true);
  });
});

describe('lexical cascade — leg scoping', () => {
  const H = 3600_000;
  const NOW = Date.parse('2026-08-09T12:00:00.000Z');
  const recency = { halfLifeMs: 24 * H, weight: 0.3, freshWindowMs: 48 * H, now: NOW };

  it('cascades the MAIN lexical leg only, never lexical-fresh', async () => {
    // Under-fill is the fresh leg's NORMAL state — its window is genuinely
    // small — not the AND-elimination signal the trigger detects. Cascading it
    // would fire on nearly every search for a second query per source.
    const src = cascadingSource('S', { and: [hit('S', 'a', 9)], graded: [hit('S', 'g1', 5)] });
    await runHybridSearch([src], { ...baseCtx, mode: 'hybrid', embedder: null, recency });

    const fresh = src.calls.filter((c) => c.since !== undefined);
    const main = src.calls.filter((c) => c.since === undefined);
    expect(fresh).toHaveLength(1);
    expect(fresh[0]!.mode).toBeUndefined();
    expect(main.map((c) => c.mode)).toEqual([undefined, 'coverage-graded']);
  });

  it('fires per source, independently', async () => {
    const full = cascadingSource('A', {
      and: [hit('A', '1', 9), hit('A', '2', 8), hit('A', '3', 7)],
      graded: [hit('A', 'g', 6)],
    });
    const thin = cascadingSource('B', { and: [hit('B', '1', 9)], graded: [hit('B', 'g', 5)] });
    await runFullTextSearch([full, thin], { ...baseCtx, limit: 3 });

    expect(full.calls).toHaveLength(1);
    expect(thin.calls).toHaveLength(2);
  });
});

describe('lexical cascade — page-level scope of the guarantee', () => {
  it('append-only is per-SOURCE; a multi-source page is still cut globally by score', async () => {
    // CHARACTERISATION, not an endorsement. `runFullTextSearch` concatenates
    // every source's listing and re-ranks the whole pool by native score before
    // the cut, so widening a HIGH-scoring source can push a LOW-scoring source's
    // stage-1 row off the page. That global cut predates the cascade and is not
    // changed by it, but it does bound what "strict superset" may be claimed
    // over: the LISTING the cascade returns, never the multi-source PAGE.
    const weak = cascadingSource('A', { and: [hit('A', '1', 1)], graded: [] });
    const strong = cascadingSource('B', {
      and: [hit('B', '1', 3)],
      graded: [hit('B', 'g1', 5), hit('B', 'g2', 4), hit('B', 'g3', 3.5), hit('B', 'g4', 3.2)],
    });
    const before = await runFullTextSearch([weak, strong], { ...baseCtx, lexicalCascade: false });
    const after = await runFullTextSearch(
      [
        cascadingSource('A', { and: [hit('A', '1', 1)], graded: [] }),
        cascadingSource('B', {
          and: [hit('B', '1', 3)],
          graded: [hit('B', 'g1', 5), hit('B', 'g2', 4), hit('B', 'g3', 3.5), hit('B', 'g4', 3.2)],
        }),
      ],
      baseCtx,
    );

    expect(refs(before.results)).toContain('A:1');
    expect(refs(after.results)).not.toContain('A:1');
    expect(after.results).toHaveLength(5);
  });
});
