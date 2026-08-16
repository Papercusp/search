/**
 * leg-health.ts — turn per-search leg degradation from a PRINTED ASIDE into a
 * MEASURED RATE (system-notices-on-its-own-2026-08-16 P-002).
 *
 * WHY THIS EXISTS. `summariseLegs` already computes a precise verdict for every
 * single search: which leg blocked, which contributed nothing, whether the
 * ranking returned is the ranking the configuration promises. That verdict is
 * excellent — and it was thrown away the instant it was rendered. It reached a
 * caller as one line of prose appended to a result, which a reader skims past,
 * and nothing anywhere counted how often it said "degraded".
 *
 * The consequence, observed directly on 2026-08-16: three separate retrieval
 * calls inside a single agent session came back carrying
 * `semantic leg blocked: query embed failed: query embed exceeded 1200ms budget`
 * — meaning every pointer returned was lexical-only — and an earlier session saw
 * the lexical leg return zero candidates. Both halves of hybrid retrieval were
 * failing intermittently, and the only evidence was prose in individual replies.
 * Nobody could answer "is retrieval degrading, and since when?" because the
 * answer was never written down anywhere. That is the mechanism behind the
 * complaint that "the knowledge existed and it didn't reach me": retrieval is
 * how stored knowledge reaches the moment of need, and it was quietly running at
 * half strength.
 *
 * WHAT THIS IS NOT. It is not a second opinion about what counts as degraded —
 * `summariseLegs` remains the sole authority and this module only counts what it
 * already decided. Adding a second definition of "degraded" would recreate
 * exactly the multi-surface divergence this codebase has fixed elsewhere.
 *
 * DOMAIN-FREE by construction (libs/generic contract): it records booleans and
 * timestamps, knows nothing about work-items, agents or Papercusp, and the host
 * decides what to do with the rate.
 *
 * SINGLETON DISCIPLINE. The ring buffer is pinned with `pinModuleState`. This is
 * not ceremony: a recorder and a reader that end up in two module records would
 * make `readLegHealth()` answer a confident, perfectly-shaped ZERO while
 * retrieval burned — the precise failure mode ("a bounded measurement rendered
 * as a confident number") this module exists to end. Hand-rolling the
 * globalThis/Symbol.for pair would fix correctness but hide the split from
 * `listModuleDuplications()`, so it is deliberately not done.
 */
import { pinModuleState } from '@papercusp/module-singleton';

import type { SearchLegs } from './legs';

/** How many recent searches are retained. Bounded on purpose — this is a health
 *  signal, not an audit log, and an unbounded buffer in a hot path is a leak. */
export const LEG_HEALTH_CAPACITY = 512;

/** One recorded search outcome. Deliberately tiny — this is written on every search. */
interface LegSample {
  atMs: number;
  degraded: boolean;
  semanticBlocked: boolean;
  lexicalBlocked: boolean;
  /** Leg ran but put nothing into fusion while the other leg worked. */
  semanticEmpty: boolean;
  lexicalEmpty: boolean;
}

interface LegHealthState {
  /** Ring buffer; `count` total ever recorded (monotonic, for overflow detection). */
  ring: LegSample[];
  next: number;
  count: number;
}

const state = pinModuleState<LegHealthState>('@papercusp/search.leg-health', () => ({
  ring: new Array<LegSample>(LEG_HEALTH_CAPACITY),
  next: 0,
  count: 0,
}));

/**
 * The aggregate over a trailing window.
 *
 * `truncatedByCapacity` is the field that keeps this honest, and it is the whole
 * reason the shape is not just a number. The ring holds a bounded number of
 * samples, so a window longer than the buffer covers is measured over a FLOOR,
 * not a total — and a bounded measurement rendered as a confident rate is
 * indistinguishable from a real one. When this is true, `searches` is "at least
 * this many", and `degradedRate` describes only the retained sample.
 */
export interface LegHealthWindow {
  windowMs: number;
  /** Searches recorded inside the window (a FLOOR when truncatedByCapacity). */
  searches: number;
  degraded: number;
  /** null when `searches === 0` — never 0, which would read as "healthy". */
  degradedRate: number | null;
  semanticBlocked: number;
  lexicalBlocked: number;
  semanticEmpty: number;
  lexicalEmpty: number;
  oldestSampleMs: number | null;
  newestSampleMs: number | null;
  /** The retained sample was capacity-bound; counts are floors, not totals. */
  truncatedByCapacity: boolean;
}

/**
 * PURE: derive a window from an explicit sample list. Split from the buffer so
 * the arithmetic is testable with no global state.
 */
export function summariseLegSamples(
  samples: readonly LegSample[],
  windowMs: number,
  nowMs: number,
  capacityReached: boolean,
): LegHealthWindow {
  const since = nowMs - windowMs;
  const inWindow = samples.filter((s) => s.atMs >= since);
  const searches = inWindow.length;
  const degraded = inWindow.filter((s) => s.degraded).length;
  // A window that reaches back past the oldest sample we still hold, on a full
  // ring, means older searches were evicted — the caller must not read a total.
  const oldest = inWindow.length > 0 ? Math.min(...inWindow.map((s) => s.atMs)) : null;
  const truncatedByCapacity = capacityReached && searches === samples.length && since < (oldest ?? since);
  return {
    windowMs,
    searches,
    degraded,
    degradedRate: searches === 0 ? null : degraded / searches,
    semanticBlocked: inWindow.filter((s) => s.semanticBlocked).length,
    lexicalBlocked: inWindow.filter((s) => s.lexicalBlocked).length,
    semanticEmpty: inWindow.filter((s) => s.semanticEmpty).length,
    lexicalEmpty: inWindow.filter((s) => s.lexicalEmpty).length,
    oldestSampleMs: oldest,
    newestSampleMs: inWindow.length > 0 ? Math.max(...inWindow.map((s) => s.atMs)) : null,
    truncatedByCapacity,
  };
}

/** PURE: reduce a finished `SearchLegs` verdict to the sample we retain. */
export function sampleOfLegs(legs: SearchLegs, atMs: number): LegSample {
  const lexicalRan = legs.lexical.status === 'ran';
  const semanticRan = legs.semantic.status === 'ran';
  return {
    atMs,
    degraded: legs.degraded,
    semanticBlocked: legs.semantic.blocked !== null,
    lexicalBlocked: legs.lexical.blocked !== null,
    // "Empty" only counts when the OTHER leg worked — mirroring summariseLegs'
    // asymmetric rule. Both legs empty is a query with no matches, not a fault.
    semanticEmpty: semanticRan && legs.semantic.candidates === 0 && lexicalRan && legs.lexical.candidates > 0,
    lexicalEmpty: lexicalRan && legs.lexical.candidates === 0 && semanticRan && legs.semantic.candidates > 0,
  };
}

/**
 * Record a finished search and hand the verdict straight back.
 *
 * Returns its argument so it can wrap a `summariseLegs(...)` call in place
 * (`legs: observeLegs(summariseLegs(a, b))`) without restructuring the caller —
 * and so a NEW call site naturally copies an idiom that keeps the measurement
 * wired up, rather than quietly omitting it.
 *
 * Never throws: an instrumentation fault must not fail a search.
 */
export function observeLegs(legs: SearchLegs, nowMs: number = Date.now()): SearchLegs {
  try {
    state.ring[state.next] = sampleOfLegs(legs, nowMs);
    state.next = (state.next + 1) % LEG_HEALTH_CAPACITY;
    state.count += 1;
  } catch {
    // Deliberately silent: a health counter is never worth a failed search.
  }
  return legs;
}

/** Read the trailing-window aggregate. Default window: 1 hour. */
export function readLegHealth(
  opts: { windowMs?: number; nowMs?: number } = {},
): LegHealthWindow {
  const windowMs = opts.windowMs ?? 60 * 60_000;
  const nowMs = opts.nowMs ?? Date.now();
  const samples = state.ring.filter((s): s is LegSample => s != null);
  return summariseLegSamples(samples, windowMs, nowMs, state.count >= LEG_HEALTH_CAPACITY);
}

/** Total searches ever observed by this process (monotonic). */
export function legHealthObservedCount(): number {
  return state.count;
}

/** TEST SEAM: drop every retained sample. */
export function resetLegHealth(): void {
  state.ring = new Array<LegSample>(LEG_HEALTH_CAPACITY);
  state.next = 0;
  state.count = 0;
}
