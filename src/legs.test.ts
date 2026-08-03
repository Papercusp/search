/**
 * Unit suite for the per-leg execution report (P-020, plan
 * semantic-search-fingerprint-coverage-2026-08-03).
 *
 * `legs.ts` shipped claiming "unit-tested standalone" in its own header with
 * no test file at all (EI-19466920786445299). These are those tests.
 *
 * The module is pure — no PG, no host types — so everything here is exercised
 * directly against accumulators.
 */

import { describe, it, expect } from 'vitest';
import {
  newLegAccumulator,
  legOfRanker,
  finaliseLeg,
  summariseLegs,
  type LegAccumulator,
  type LegReport,
} from './legs';

/** An accumulator with the given fields overridden. */
function acc(over: Partial<LegAccumulator> = {}): LegAccumulator {
  return { ...newLegAccumulator(), ...over };
}
/** A finalised leg that RAN and contributed `candidates` rows. */
function ran(candidates: number, over: Partial<LegAccumulator> = {}): LegReport {
  return finaliseLeg(acc({ attempted: true, callsRun: 1, candidates, ...over }));
}
const notRun = (): LegReport => finaliseLeg(acc());

describe('finaliseLeg — status derivation', () => {
  it('a leg never attempted is not-run', () => {
    expect(finaliseLeg(acc()).status).toBe('not-run');
  });

  it('a leg that made a returning call is ran', () => {
    expect(finaliseLeg(acc({ attempted: true, callsRun: 1 })).status).toBe('ran');
  });

  it('blocked outranks call counts — nothing was even asked', () => {
    const leg = finaliseLeg(acc({ attempted: true, blocked: 'query embed failed: boom' }));
    expect(leg.status).toBe('errored');
    expect(leg.blocked).toBe('query embed failed: boom');
  });

  it('attempted with every call throwing is errored', () => {
    expect(finaliseLeg(acc({ attempted: true, callsFailed: 2 })).status).toBe('errored');
  });

  it('PARTIAL failure still reads ran — the failures are reported alongside, not folded in', () => {
    const leg = finaliseLeg(
      acc({
        attempted: true,
        callsRun: 1,
        callsFailed: 1,
        failures: [{ source: 'S', ranker: 'bm25', error: 'boom' }],
      }),
    );
    expect(leg.status).toBe('ran');
    expect(leg.callsFailed).toBe(1);
  });

  it('attempted but no call ever made is not-run, not errored', () => {
    // e.g. every source was filtered out before its query ran.
    expect(finaliseLeg(acc({ attempted: true })).status).toBe('not-run');
  });

  it('carries candidates and floored through separately', () => {
    // The header's opposite-diagnoses pair: 0 candidates with a big floored
    // count is a mis-set floor; 0 and 0 is a query that found nothing.
    const leg = finaliseLeg(acc({ attempted: true, callsRun: 1, candidates: 0, floored: 40 }));
    expect(leg).toMatchObject({ candidates: 0, floored: 40 });
  });
});

describe('legOfRanker — routing', () => {
  it('routes every lexical ranker to the lexical leg and vectors to semantic', () => {
    const legs = { lexical: newLegAccumulator(), semantic: newLegAccumulator() };
    expect(legOfRanker('bm25', legs)).toBe(legs.lexical);
    expect(legOfRanker('bm25-fresh', legs)).toBe(legs.lexical);
    expect(legOfRanker('embeddings', legs)).toBe(legs.semantic);
  });
});

describe('summariseLegs — the degraded verdict', () => {
  it('both legs contributing is healthy', () => {
    const out = summariseLegs(ran(5), ran(3));
    expect(out).toMatchObject({ degraded: false, warning: null });
  });

  // ── The load-bearing asymmetry (see the module header) ──
  it('BOTH legs empty is NOT degraded — that is a query with no matches', () => {
    // A false alarm here would train readers to ignore the signal entirely,
    // which is the failure mode the asymmetry exists to prevent.
    const out = summariseLegs(ran(0), ran(0));
    expect(out).toMatchObject({ degraded: false, warning: null });
  });

  it('a leg that was never run does not raise a warning (mode with one leg by design)', () => {
    // mode:'embeddings' has no lexical leg — that is configuration, not degradation.
    expect(summariseLegs(notRun(), ran(4))).toMatchObject({ degraded: false, warning: null });
  });

  // ── THE MEASURED CASE (EI-19447237774252790) ──
  it('flags a search that silently went semantic-only: lexical RAN but contributed nothing', () => {
    const out = summariseLegs(ran(0), ran(7));
    expect(out.degraded).toBe(true);
    expect(out.warning).toContain('lexical leg contributed 0 candidates');
    expect(out.warning).toContain('semantic-only');
  });

  it('names the score floor when the floor is what emptied the leg', () => {
    const out = summariseLegs(ran(0, { floored: 40 }), ran(7));
    expect(out.warning).toContain('the score floor removed all 40');
  });

  it('flags the mirror case: semantic ran but contributed nothing', () => {
    const out = summariseLegs(ran(7), ran(0));
    expect(out.degraded).toBe(true);
    expect(out.warning).toContain('lexical-only');
  });

  it('reports a blocked leg by its reason', () => {
    const blocked = finaliseLeg(acc({ attempted: true, blocked: 'query embed failed: timeout' }));
    const out = summariseLegs(ran(5), blocked);
    expect(out.degraded).toBe(true);
    expect(out.warning).toContain('semantic leg blocked: query embed failed: timeout');
  });

  it('a blocked semantic leg is reported ONCE, not also as "contributed 0"', () => {
    const blocked = finaliseLeg(acc({ attempted: true, blocked: 'query embed failed: timeout' }));
    const out = summariseLegs(ran(5), blocked);
    expect(out.warning).not.toContain('lexical-only');
  });

  it('reports a leg whose every source call failed', () => {
    const errored = finaliseLeg(acc({ attempted: true, callsFailed: 2 }));
    const out = summariseLegs(errored, ran(3));
    expect(out.degraded).toBe(true);
    expect(out.warning).toContain('every lexical source call failed');
  });

  it('degrades an otherwise-healthy leg on PARTIAL source failure, naming the source', () => {
    const partial = ran(5, {
      callsFailed: 1,
      failures: [{ source: 'memories', ranker: 'bm25', error: 'timeout' }],
    });
    const out = summariseLegs(partial, ran(3));
    expect(out.degraded).toBe(true);
    expect(out.warning).toContain('1 lexical source call(s) failed: memories (timeout)');
  });

  it('caps the named failures at 3 and counts the rest', () => {
    const many = ran(5, {
      callsFailed: 5,
      failures: Array.from({ length: 5 }, (_, i) => ({
        source: `s${i}`,
        ranker: 'bm25',
        error: 'boom',
      })),
    });
    expect(summariseLegs(many, ran(3)).warning).toContain('+2 more');
  });

  it('joins several independent reasons into one line', () => {
    const errored = finaliseLeg(acc({ attempted: true, callsFailed: 1 }));
    const blocked = finaliseLeg(acc({ attempted: true, blocked: 'embed died' }));
    const out = summariseLegs(errored, blocked);
    expect(out.warning).toContain('semantic leg blocked: embed died');
    expect(out.warning).toContain('every lexical source call failed');
    expect(out.warning).toContain(';');
  });

  it('passes both leg reports through untouched', () => {
    const lexical = ran(5);
    const semantic = ran(3);
    expect(summariseLegs(lexical, semantic)).toMatchObject({ lexical, semantic });
  });
});
