/**
 * Per-ranker minimum-score floors, applied to each ranker's list BEFORE fusion.
 *
 * WHY THIS EXISTS — and why the floor cannot be global:
 *
 * RRF ({@link rrfCombine}) fuses purely by RANK POSITION (`1/(k+rank)`) and
 * discards the ranker-native score entirely. That is what makes it robust
 * across incomparable score scales — and it is also why it cannot tell a
 * strong match from a weak one. A list of pure noise still has a rank 1, and
 * that rank-1 noise receives exactly the same fusion weight as a rank-1
 * near-perfect hit from another ranker. On a sparsely-populated vector index
 * the vector leg is mostly noise, so fusion actively promotes it.
 *
 * The only place that judgement can be made is BEFORE fusion, while the
 * native score still exists. Hence: filter each ranked list by a floor
 * expressed in that ranker's OWN units, then fuse what survives.
 *
 * A single global floor would be meaningless, because the units differ:
 *   - a `ts_rank_cd` lexical score is unbounded-positive and typically
 *     ~0.0-1.0, with its magnitude driven by term density and document
 *     length;
 *   - a pgvector cosine similarity (`1 - (v <=> q)`) is bounded in [-1, 1],
 *     and its "this is noise" level depends on the embedding model.
 * A number that floors one of those either deletes everything or nothing in
 * the other.
 *
 * Floors are OPT-IN per host: absent (or non-finite) ⇒ no filtering at all,
 * and ranking is byte-identical to the pre-floor engine.
 */

import type { Listing } from './types';

/**
 * Per-ranker floors, keyed by the ranker name the engine fuses under.
 *
 * Well-known engine ranker names:
 *   - `bm25`        — the lexical (`ts_rank_cd`) leg
 *   - `bm25-fresh`  — the fresh-window lexical leg (same units as `bm25`)
 *   - `embeddings`  — the pgvector cosine-similarity leg
 *
 * A ranker with no entry is UNFILTERED. Family fallback: a hyphenated ranker
 * name inherits its family's floor (`bm25-fresh` → `bm25`) unless it sets its
 * own, so a host does not have to enumerate every leg variant the engine may
 * grow. An explicit entry always wins over the family fallback — including an
 * explicit `Infinity`/`-Infinity`, which are legitimate "drop everything" /
 * "keep everything" settings.
 */
export interface MinScoreFloors {
  /** Floor for the lexical `ts_rank_cd` legs (inherited by `bm25-fresh`). */
  bm25?: number;
  /** Floor for the pgvector cosine-similarity leg. */
  embeddings?: number;
  /** Any other ranker name the host's engine fuses under. */
  [ranker: string]: number | undefined;
}

/**
 * The floor that applies to `ranker`: its own entry, else its family's
 * (the segment before the first `-`), else `undefined` (= unfiltered).
 *
 * A non-finite `NaN` entry is treated as absent — it can only come from a
 * mis-parsed config value, and silently filtering everything (nothing is
 * `>= NaN`) would be a worse answer than not filtering. `Infinity` is
 * deliberately NOT treated that way: it is a meaningful, explicitly-authored
 * "drop this ranker entirely".
 */
export function resolveMinScore(
  floors: MinScoreFloors | undefined,
  ranker: string,
): number | undefined {
  if (!floors) return undefined;
  const own = floors[ranker];
  if (typeof own === 'number' && !Number.isNaN(own)) return own;
  const dash = ranker.indexOf('-');
  if (dash > 0) {
    const family = floors[ranker.slice(0, dash)];
    if (typeof family === 'number' && !Number.isNaN(family)) return family;
  }
  return undefined;
}

export interface MinScoreOutcome {
  /** The surviving entries, in their original order (so ranks tighten up). */
  list: Listing;
  /** How many entries the floor removed. 0 when no floor applied. */
  dropped: number;
  /** The floor that was applied, or undefined when the ranker is unfiltered. */
  floor: number | undefined;
}

/**
 * Drop every entry scoring below `ranker`'s floor.
 *
 * The input list is already ordered best-first, and filtering preserves that
 * order, so the survivors' ranks simply tighten up (the best survivor becomes
 * rank 1) — which is exactly what fusion should see. An entry with a `NaN`
 * score is dropped whenever a floor applies: an unrankable score cannot be
 * shown to clear the bar.
 *
 * Returns the ORIGINAL list object (not a copy) when no floor applies or
 * nothing is dropped, so the no-floor path allocates nothing.
 */
export function applyMinScore(
  list: Listing,
  ranker: string,
  floors: MinScoreFloors | undefined,
): MinScoreOutcome {
  const floor = resolveMinScore(floors, ranker);
  if (floor === undefined) return { list, dropped: 0, floor: undefined };
  const kept = list.filter((entry) => entry.score >= floor);
  if (kept.length === list.length) return { list, dropped: 0, floor };
  return { list: kept, dropped: list.length - kept.length, floor };
}
