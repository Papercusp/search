import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { RankedItem } from '@papercusp/rrf';

import { runFullTextSearch, runHybridSearch, type SearchContext } from './hybrid';
import type { Embedder, Listing, PgHandle, SearchHit, SearchSource } from './types';

import {
  LEG_HEALTH_CAPACITY,
  legHealthObservedCount,
  observeLegs,
  readLegHealth,
  resetLegHealth,
  sampleOfLegs,
  summariseLegSamples,
  type LegSample,
} from './leg-health';
import { finaliseLeg, newLegAccumulator, summariseLegs, type LegReport } from './legs';

const NOW = Date.parse('2026-08-16T04:00:00Z');

/** Build a LegReport in a named state without hand-writing the shape. */
function leg(over: Partial<LegReport> = {}): LegReport {
  return {
    status: 'ran',
    candidates: 5,
    floored: 0,
    stage2Added: 0,
    callsRun: 1,
    callsFailed: 0,
    failures: [],
    blocked: null,
    ...over,
  };
}

/** The live failure this module was built for: embed budget blew, semantic leg dead. */
function semanticBlockedLegs() {
  return summariseLegs(
    leg(),
    leg({
      status: 'errored',
      candidates: 0,
      callsRun: 0,
      blocked: 'query embed failed: query embed exceeded 1200ms budget',
    }),
  );
}

function healthyLegs() {
  return summariseLegs(leg(), leg());
}

describe('sampleOfLegs', () => {
  it('records a semantic-blocked search as degraded and names which leg blocked', () => {
    const s = sampleOfLegs(semanticBlockedLegs(), NOW);
    expect(s.degraded).toBe(true);
    expect(s.semanticBlocked).toBe(true);
    expect(s.lexicalBlocked).toBe(false);
  });

  it('records a healthy search as not degraded', () => {
    const s = sampleOfLegs(healthyLegs(), NOW);
    expect(s.degraded).toBe(false);
    expect(s.semanticBlocked).toBe(false);
    expect(s.lexicalBlocked).toBe(false);
  });

  it('counts a leg as EMPTY only when the other leg worked', () => {
    // Semantic ran but returned nothing while lexical returned rows: real degradation.
    const oneSided = sampleOfLegs(
      summariseLegs(leg({ candidates: 5 }), leg({ candidates: 0 })),
      NOW,
    );
    expect(oneSided.semanticEmpty).toBe(true);

    // BOTH legs empty is a query with no matches, NOT a fault — mirrors
    // summariseLegs' asymmetric rule. Marking this degraded would make the
    // signal noise, since most zero-result searches are just misses.
    const bothEmpty = sampleOfLegs(
      summariseLegs(leg({ candidates: 0 }), leg({ candidates: 0 })),
      NOW,
    );
    expect(bothEmpty.semanticEmpty).toBe(false);
    expect(bothEmpty.lexicalEmpty).toBe(false);
  });

  it('never invents a verdict — it mirrors summariseLegs.degraded exactly', () => {
    // The authority is summariseLegs; this module only counts. A second opinion
    // about "degraded" would recreate the multi-surface divergence this codebase
    // already removed elsewhere.
    for (const legs of [healthyLegs(), semanticBlockedLegs()]) {
      expect(sampleOfLegs(legs, NOW).degraded).toBe(legs.degraded);
    }
  });
});

describe('summariseLegSamples', () => {
  const s = (atMs: number, over: Partial<LegSample> = {}): LegSample => ({
    atMs,
    degraded: false,
    semanticBlocked: false,
    lexicalBlocked: false,
    semanticEmpty: false,
    lexicalEmpty: false,
    ...over,
  });

  it('computes the degraded rate over the window', () => {
    const out = summariseLegSamples(
      [
        s(NOW - 1000, { degraded: true, semanticBlocked: true }),
        s(NOW - 2000, { degraded: true, semanticBlocked: true }),
        s(NOW - 3000),
        s(NOW - 4000),
      ],
      60_000,
      NOW,
      false,
    );
    expect(out.searches).toBe(4);
    expect(out.degraded).toBe(2);
    expect(out.degradedRate).toBe(0.5);
    expect(out.semanticBlocked).toBe(2);
  });

  it('excludes samples older than the window', () => {
    const out = summariseLegSamples(
      [s(NOW - 1000, { degraded: true }), s(NOW - 10 * 60_000)],
      60_000,
      NOW,
      false,
    );
    expect(out.searches).toBe(1);
    expect(out.degradedRate).toBe(1);
  });

  it('returns degradedRate null — NOT 0 — when nothing was recorded', () => {
    // A 0 rate reads as "retrieval is healthy". No data must not be able to
    // impersonate a clean bill of health; that is the same class of error as a
    // zeroed backlog reading as a drained one.
    const out = summariseLegSamples([], 60_000, NOW, false);
    expect(out.searches).toBe(0);
    expect(out.degradedRate).toBeNull();
  });

  it('marks the window truncated when the ring evicted samples inside it', () => {
    // Capacity reached, every retained sample sits inside the window, and the
    // window reaches back before the oldest one: older searches were evicted, so
    // the counts are FLOORS.
    const out = summariseLegSamples(
      [s(NOW - 1000), s(NOW - 2000)],
      60 * 60_000,
      NOW,
      true,
    );
    expect(out.truncatedByCapacity).toBe(true);
  });

  it('does not mark truncated when the ring never wrapped', () => {
    const out = summariseLegSamples([s(NOW - 1000)], 60 * 60_000, NOW, false);
    expect(out.truncatedByCapacity).toBe(false);
  });
});

describe('observeLegs / readLegHealth', () => {
  beforeEach(() => resetLegHealth());

  it('returns its argument unchanged so it can wrap a call in place', () => {
    const legs = healthyLegs();
    expect(observeLegs(legs, NOW)).toBe(legs);
  });

  it('accumulates the live failure signature across searches', () => {
    observeLegs(semanticBlockedLegs(), NOW - 1000);
    observeLegs(semanticBlockedLegs(), NOW - 2000);
    observeLegs(healthyLegs(), NOW - 3000);

    const h = readLegHealth({ windowMs: 60_000, nowMs: NOW });
    expect(h.searches).toBe(3);
    expect(h.degraded).toBe(2);
    expect(h.degradedRate).toBeCloseTo(2 / 3);
    expect(h.semanticBlocked).toBe(2);
  });

  it('counts every observed search monotonically', () => {
    observeLegs(healthyLegs(), NOW);
    observeLegs(healthyLegs(), NOW);
    expect(legHealthObservedCount()).toBe(2);
  });

  it('keeps the buffer bounded and reports the overflow honestly', () => {
    for (let i = 0; i < LEG_HEALTH_CAPACITY + 50; i++) {
      observeLegs(healthyLegs(), NOW - i);
    }
    const h = readLegHealth({ windowMs: 24 * 60 * 60_000, nowMs: NOW });
    expect(h.searches).toBe(LEG_HEALTH_CAPACITY); // retained, not total
    expect(legHealthObservedCount()).toBe(LEG_HEALTH_CAPACITY + 50); // total
    expect(h.truncatedByCapacity).toBe(true);
  });

  // FALSIFIABILITY CONTROL, kept permanently in the file rather than mutating the
  // shared tree. This is the behaviour the module replaces: rendering the verdict
  // as prose and retaining nothing. It can never report a rate, which is exactly
  // why three degraded retrievals in one session went unnoticed.
  it('control: rendering the warning as prose retains no measurable signal', () => {
    const rendered: string[] = [];
    for (const legs of [semanticBlockedLegs(), semanticBlockedLegs(), healthyLegs()]) {
      if (legs.warning) rendered.push(legs.warning);
    }
    // The prose exists...
    expect(rendered).toHaveLength(2);
    expect(rendered[0]).toContain('semantic leg blocked');
    // ...but nothing accumulated it, so no rate is derivable from it.
    expect(readLegHealth({ windowMs: 60_000, nowMs: NOW }).searches).toBe(0);
  });

  // ── THE WIRING PROOF ──────────────────────────────────────────────────────
  // Everything above tests the AGGREGATOR. None of it proves the aggregator is
  // actually ON the search path — and a metric that silently never records is
  // strictly worse than no metric, because a permanent zero reads as health.
  // These drive the REAL engine entrypoints and assert the counter moved.
  describe('is wired into the real search path', () => {
    const sql = {} as PgHandle;
    const baseCtx: SearchContext = {
      sql,
      query: 'q',
      workspaceId: 'w',
      scopeFilter: null,
      limit: 5,
    };
    const h = (source: string, id: string, score: number): SearchHit => ({
      source,
      source_id: id,
      excerpt: `ex-${id}`,
      highlight: `hl-${id}`,
      score,
      rankers: [source],
    });
    const listing = (...hits: SearchHit[]): Listing =>
      hits.map((x): RankedItem<SearchHit> => ({
        key: `${x.source}:${x.source_id}`,
        score: x.score,
        row: x,
      }));
    const source = (name: string, withEmbedding = false): SearchSource => {
      const s: SearchSource = { name, lexical: vi.fn(async () => listing(h(name, '1', 0.9))) };
      if (withEmbedding) s.embedding = vi.fn(async () => listing(h(name, '2', 0.8)));
      return s;
    };
    const embedder: Embedder = vi.fn(async () => [0.1, 0.2, 0.3]) as unknown as Embedder;

    it('runFullTextSearch records a sample', async () => {
      expect(legHealthObservedCount()).toBe(0);
      await runFullTextSearch([source('a')], baseCtx);
      expect(legHealthObservedCount()).toBe(1);
    });

    it('runHybridSearch records a sample', async () => {
      expect(legHealthObservedCount()).toBe(0);
      await runHybridSearch([source('a', true)], { ...baseCtx, mode: 'hybrid', embedder });
      expect(legHealthObservedCount()).toBe(1);
    });

    it('records the DEGRADED verdict when the embedder dies mid-search', async () => {
      // The live 2026-08-16 signature: the query embedder blows its budget, the
      // semantic leg never runs, and the search silently returns lexical-only.
      const throwing = vi.fn(async () => {
        throw new Error('query embed exceeded 1200ms budget');
      }) as unknown as Embedder;
      const res = await runHybridSearch([source('a', true)], {
        ...baseCtx,
        mode: 'hybrid',
        embedder: throwing,
        log: () => {},
      });
      expect(res.legs.degraded).toBe(true);
      const health = readLegHealth({ windowMs: 60_000 });
      expect(health.searches).toBe(1);
      expect(health.degraded).toBe(1);
      expect(health.degradedRate).toBe(1);
    });
  });

  it('an instrumentation fault never fails the search', () => {
    const legs = summariseLegs(finaliseLeg(newLegAccumulator()), finaliseLeg(newLegAccumulator()));
    // Even for a degenerate all-not-run report, observing must be total.
    expect(() => observeLegs(legs, NOW)).not.toThrow();
    expect(observeLegs(legs, NOW)).toBe(legs);
  });
});
