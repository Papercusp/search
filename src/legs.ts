/**
 * legs.ts — WHICH LEG ACTUALLY RAN, AND DID IT CONTRIBUTE ANYTHING? (plan
 * semantic-search-fingerprint-coverage-2026-08-03 P-020)
 *
 * The failure this exists to end: a hybrid search whose lexical leg returned
 * NOTHING still reports plain success. Fusion just has one input, the results
 * look like results, and no field anywhere says the search was single-legged.
 *
 * ⚠ THE MEASURED CASE, and it is why a boolean cannot carry this contract
 * (EI-19447237774252790): `plainto_tsquery` ANDs every term, so past ~15 terms
 * no document in a real corpus satisfies the lexical leg — 33,662 hits at 2
 * terms decaying monotonically to 0 at 15+. The leg EXECUTED PERFECTLY. No
 * error was thrown, nothing was logged, and `embedderAvailable` stayed `true`
 * because a query vector really was produced. Every existing signal reported
 * health while the search was silently embeddings-only, and fused MRR@10 fell
 * from 0.7542 to 0.3492 on the gold set. "Did the leg run" and "did the leg
 * contribute" are DIFFERENT QUESTIONS, and only the second one detects this.
 *
 * So a leg is reported by its CANDIDATE COUNT, not by an execution flag — and
 * post-floor, because a floor that ate every row is a leg that contributed
 * nothing no matter how healthy the query was. `floored` is kept separate so
 * the two are distinguishable: `candidates: 0, floored: 40` means the floor
 * is too high for this embedding space, while `candidates: 0, floored: 0`
 * means the query itself found nothing — opposite diagnoses, opposite fixes.
 *
 * WHY THE `degraded` PREDICATE LIVES HERE rather than at each caller: this is
 * the mistake `provenance.ts` already had to correct, where every UI
 * re-derived "is this a semantic match" by hard-coding ranker names in a
 * different place. A caller must not have to know that "0 lexical candidates
 * in a hybrid search" is the tell. The engine knows; the engine says.
 *
 * ⚠ The rule is deliberately ASYMMETRIC, to avoid a false alarm that would
 * train readers to ignore it: a search where BOTH legs return nothing is not
 * degraded, it is a query with no matches. Degradation is one leg working
 * while another silently does not — plus any source that actually failed.
 *
 * Pure — no PG, no host types, unit-tested standalone.
 */
import { isLexicalRanker } from './provenance';

/**
 * How a leg's execution ended.
 *
 * - `not-run`   — the leg was never attempted (no lexical leg in
 *                 `mode: 'embeddings'`; no semantic leg without an embedder,
 *                 or when no registered source implements one).
 * - `errored`   — the leg was attempted and produced nothing usable: either it
 *                 was blocked outright (the query embed failed/timed out) or
 *                 every source call it made threw.
 * - `ran`       — at least one source call returned a list. PARTIAL failure
 *                 still reads `ran`; read `callsFailed`/`failures` for that,
 *                 which is why they are reported alongside rather than folded
 *                 into the status.
 */
export type LegStatus = 'ran' | 'errored' | 'not-run';

/** One source call that threw, kept verbatim so a caller can name the cause. */
export interface LegFailure {
  /** The `SearchSource.name` whose query failed. */
  source: string;
  /** The ranker label under which it was called (e.g. `lexical`, `lexical-fresh`). */
  ranker: string;
  /** The error's message. */
  error: string;
}

/** What one ranking leg actually did during a search. */
export interface LegReport {
  status: LegStatus;
  /**
   * Rows this leg contributed to fusion, AFTER any score floor and after the
   * fresh-leg dedupe — i.e. what the fusion stage genuinely saw from this leg,
   * never what its queries happened to return. This is the number that
   * detects a silently single-legged search.
   */
  candidates: number;
  /** Rows removed by the min-score floor before fusion. See the header note on
   *  why this is reported separately from `candidates`. */
  floored: number;
  /**
   * Rows APPENDED by the lexical cascade's second stage — the widening that
   * fires when the historical AND query under-fills (see
   * `SearchContext.lexicalCascade`). Always 0 for the semantic leg, and 0 on
   * the lexical leg whenever stage 2 did not fire OR fired and found nothing
   * new.
   *
   * Reported rather than merely returned because the two zeros above are the
   * whole question: `stage2Added: 0` on a query whose AND stage was SILENT
   * (`candidates: 0` with the cascade enabled) says the wider mode does not
   * reach that query either — which is evidence FOR a different lexical
   * engine, not for tuning this one. Without this number that distinction is
   * unobservable from outside, and the engine's own docblock cites six
   * surfaces silently discarding the leg report as the defect it exists to
   * prevent.
   */
  stage2Added: number;
  /** Source-leg calls that returned. A source can be called more than once in
   *  one leg (the recency `lexical-fresh` window), so this counts CALLS, not
   *  distinct sources. */
  callsRun: number;
  /** Source-leg calls that threw. */
  callsFailed: number;
  /** Individual call failures; empty when none. */
  failures: LegFailure[];
  /** Set when the leg could not be attempted AT ALL despite being wanted —
   *  e.g. `query embed failed: …`. Null otherwise. Distinct from a source
   *  failure: nothing was even asked. */
  blocked: string | null;
}

/** The per-leg execution report returned on every search result. */
export interface SearchLegs {
  /** The term/BM25 leg. */
  lexical: LegReport;
  /** The vector/embedding leg. */
  semantic: LegReport;
  /**
   * True when this search did NOT run at full strength — the ranking returned
   * is not the ranking the configuration promises. See the header note for the
   * asymmetric rule (both legs empty is "no matches", not degradation).
   */
  degraded: boolean;
  /** One human-readable line naming what degraded, or null when healthy. */
  warning: string | null;
}

/**
 * The report for a search that ran no legs at all — every leg `not-run`, and
 * therefore NOT degraded (nothing was attempted, so nothing silently failed).
 *
 * For a host that constructs a `SearchResult` without going through the
 * engine. Using this instead of hand-writing the shape keeps such a result
 * honest: it says "no leg information" rather than fabricating health, and it
 * cannot drift out of sync when a leg is added here.
 */
export function emptyLegs(): SearchLegs {
  return summariseLegs(finaliseLeg(newLegAccumulator()), finaliseLeg(newLegAccumulator()));
}

/** Mutable per-leg counters the engine feeds while a search runs. */
export interface LegAccumulator {
  attempted: boolean;
  candidates: number;
  floored: number;
  /** See {@link LegReport.stage2Added}. */
  stage2Added: number;
  callsRun: number;
  callsFailed: number;
  failures: LegFailure[];
  blocked: string | null;
}

/** A fresh accumulator for one leg of one search. */
export function newLegAccumulator(): LegAccumulator {
  return {
    attempted: false,
    candidates: 0,
    floored: 0,
    stage2Added: 0,
    callsRun: 0,
    callsFailed: 0,
    failures: [],
    blocked: null,
  };
}

/**
 * Route a ranker label to the leg it belongs to. Reuses `provenance.ts`'s
 * predicate rather than matching the label here, so leg attribution and
 * provenance can never disagree about what counts as lexical. That shared
 * predicate is what let the `bm25` → `lexical` rename (P-013, landed) happen
 * without silently mis-attributing every lexical candidate to the semantic
 * leg — and it still absorbs a host that names its own lexical source `bm25`.
 */
export function legOfRanker(
  ranker: string,
  legs: { lexical: LegAccumulator; semantic: LegAccumulator },
): LegAccumulator {
  return isLexicalRanker(ranker) ? legs.lexical : legs.semantic;
}

/** Freeze an accumulator into its reported shape. */
export function finaliseLeg(acc: LegAccumulator): LegReport {
  let status: LegStatus;
  if (!acc.attempted) status = 'not-run';
  else if (acc.blocked !== null) status = 'errored';
  else if (acc.callsRun === 0 && acc.callsFailed > 0) status = 'errored';
  else if (acc.callsRun === 0) status = 'not-run';
  else status = 'ran';
  return {
    status,
    candidates: acc.candidates,
    floored: acc.floored,
    stage2Added: acc.stage2Added,
    callsRun: acc.callsRun,
    callsFailed: acc.callsFailed,
    failures: acc.failures,
    blocked: acc.blocked,
  };
}

/** Did this leg genuinely put rows into fusion? */
function contributed(leg: LegReport): boolean {
  return leg.status === 'ran' && leg.candidates > 0;
}

function describeFailures(name: string, leg: LegReport): string | null {
  if (leg.failures.length === 0) return null;
  const shown = leg.failures
    .slice(0, 3)
    .map((f) => `${f.source} (${f.error})`)
    .join(', ');
  const more = leg.failures.length > 3 ? `, +${leg.failures.length - 3} more` : '';
  return `${leg.failures.length} ${name} source call(s) failed: ${shown}${more}`;
}

/**
 * Decide whether a search ran at full strength, and say what did not.
 *
 * The asymmetry documented in the header is load-bearing: an empty leg is only
 * a degradation when ANOTHER leg worked. Both empty is a query with no
 * matches, which must not raise a warning or the signal becomes noise nobody
 * reads.
 */
export function summariseLegs(lexical: LegReport, semantic: LegReport): SearchLegs {
  const reasons: string[] = [];

  // A leg that was wanted but could not start at all.
  if (lexical.blocked) reasons.push(`lexical leg blocked: ${lexical.blocked}`);
  if (semantic.blocked) reasons.push(`semantic leg blocked: ${semantic.blocked}`);

  // A leg that was attempted and every one of its calls threw.
  if (lexical.status === 'errored' && !lexical.blocked) reasons.push('every lexical source call failed');
  if (semantic.status === 'errored' && !semantic.blocked) reasons.push('every semantic source call failed');

  // THE MEASURED CASE: one leg worked, the other silently contributed nothing.
  const lexicalLive = contributed(lexical);
  const semanticLive = contributed(semantic);
  if (semanticLive && !lexicalLive && lexical.status !== 'not-run') {
    reasons.push(
      lexical.floored > 0
        ? `lexical leg contributed 0 candidates (the score floor removed all ${lexical.floored}) — this search was effectively semantic-only`
        : 'lexical leg contributed 0 candidates — this search was effectively semantic-only',
    );
  }
  if (lexicalLive && !semanticLive && semantic.status !== 'not-run' && !semantic.blocked) {
    reasons.push(
      semantic.floored > 0
        ? `semantic leg contributed 0 candidates (the score floor removed all ${semantic.floored}) — this search was effectively lexical-only`
        : 'semantic leg contributed 0 candidates — this search was effectively lexical-only',
    );
  }

  // Partial source failures degrade a leg that otherwise looks healthy.
  const lexFail = describeFailures('lexical', lexical);
  if (lexFail && lexical.status === 'ran') reasons.push(lexFail);
  const semFail = describeFailures('semantic', semantic);
  if (semFail && semantic.status === 'ran') reasons.push(semFail);

  return {
    lexical,
    semantic,
    degraded: reasons.length > 0,
    warning: reasons.length > 0 ? reasons.join('; ') : null,
  };
}
