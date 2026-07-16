/**
 * recency.ts — an OPTIONAL, configurable recency re-rank for `runHybridSearch`.
 *
 * Relevance (BM25 + embeddings, fused by RRF) says nothing about time. For some
 * corpora — session transcripts especially — how RECENT a match is matters as
 * much as how well it matches. This applies a tunable time-decay blend to the
 * already-fused candidate list, BEFORE the top-N cut, so a recent-but-slightly-
 * less-relevant candidate can float up (and, since `runHybridSearch` over-fetches
 * `limit*3`, be rescued past the cut).
 *
 * Same seam as `@papercusp/facets`: the engine owns the math; the caller owns
 * ONE thing — how to read a hit's timestamp (`getTime`, defaulting to `hit.ts`).
 * The engine stays schema-agnostic.
 *
 * Deliberately NOT generalized into an arbitrary "pluggable Nth ranker" DSL —
 * recency is a single well-understood signal that deserves a concrete named
 * option. A general extra-ranker seam is a separate, later decision.
 */
import type { FusedItem } from '@papercusp/rrf';
import type { SearchHit } from './types';

export interface RecencyRank {
  /** Read a hit's timestamp. Return epoch-ms / ISO string / Date, or null/undefined
   *  for "no time" (that hit gets no recency boost — treated as oldest, never an
   *  error). Defaults to reading `hit.ts`. */
  getTime?: (hit: SearchHit) => number | string | Date | null | undefined;
  /** Decay half-life in ms: a hit this old keeps half its recency weight. */
  halfLifeMs: number;
  /** How strongly recency competes with relevance, 0..1 (clamped). 0 = off
   *  (pure relevance), 1 = recency co-equal / dominant. Default 0.3. */
  weight?: number;
  /**
   * Blend mode (default 'linear'):
   *  - 'linear'   final = (1-weight)·normRelevance + weight·decay
   *  - 'multiply' final = normRelevance · decay^weight   (recency SCALES relevance)
   * `normRelevance` is the fused RRF score normalized to [0,1] within the
   * candidate set (RRF scores aren't on a fixed scale, so normalize before blend).
   */
  mode?: 'linear' | 'multiply';
  /**
   * Optional FRESH-CANDIDATE window (ms). The re-rank alone can only reorder
   * candidates that survived the relevance-only over-fetch cut — on a large
   * corpus, old high-term-density rows can monopolize the whole `limit*3`
   * pool so no recent row is even ELIGIBLE for a recency boost (the
   * agents-pill "no results from today" failure, WI-5097). When set (> 0)
   * and `weight` > 0, `runHybridSearch` additionally fetches a BM25
   * candidate list restricted to `filters.since = now - freshWindowMs` and
   * feeds it to RRF as one more input — guaranteeing recent matches
   * representation in the pool; the blended re-rank still owns the final
   * ordering (an irrelevant-but-recent row won't leapfrog a good match).
   * Sources that ignore `filters.since` return an identical list, which the
   * engine detects and drops (no double-weighting). Absent/0 = off.
   */
  freshWindowMs?: number;
  /** Injectable clock (ms) for deterministic tests. Default `Date.now()`. */
  now?: number;
}

function clamp01(n: number): number {
  return n < 0 ? 0 : n > 1 ? 1 : n;
}

/** Coerce a getTime result to epoch-ms, or null when absent/unparseable. */
export function toMillis(v: number | string | Date | null | undefined): number | null {
  if (v == null) return null;
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  if (v instanceof Date) {
    const t = v.getTime();
    return Number.isNaN(t) ? null : t;
  }
  const t = Date.parse(v);
  return Number.isNaN(t) ? null : t;
}

/**
 * Re-rank a fused candidate list by blending relevance with a recency decay.
 * Pure — returns a NEW array, never mutates the input. Each returned item's
 * `score` becomes the blended score (so downstream ordering — e.g. per-session
 * top-score grouping — also honors recency), and 'recency' is appended to
 * `rankers` for items that had a usable timestamp (observability). Ties (equal
 * blended score) preserve the input order for stability.
 */
export function applyRecencyRerank(
  fused: readonly FusedItem<SearchHit>[],
  recency: RecencyRank,
): FusedItem<SearchHit>[] {
  const now = recency.now ?? Date.now();
  const halfLife = recency.halfLifeMs;
  const weight = clamp01(recency.weight ?? 0.3);
  const mode = recency.mode ?? 'linear';
  const getTime = recency.getTime ?? ((h: SearchHit) => h.ts);

  let maxScore = 0;
  for (const it of fused) if (it.score > maxScore) maxScore = it.score;

  const scored = fused.map((it, idx) => {
    const t = toMillis(getTime(it.row));
    const hasTime = t != null;
    const decay = hasTime && halfLife > 0 ? Math.pow(0.5, Math.max(0, now - t) / halfLife) : 0;
    const normRel = maxScore > 0 ? it.score / maxScore : 0;
    const blended = mode === 'multiply'
      ? normRel * Math.pow(decay, weight) // weight 0 → ×1 (pure relevance); 1 → ×decay
      : (1 - weight) * normRel + weight * decay;
    const rankers = hasTime && weight > 0 ? [...it.rankers, 'recency'] : it.rankers;
    return { item: { row: it.row, score: blended, rankers }, idx };
  });

  scored.sort((a, b) => b.item.score - a.item.score || a.idx - b.idx);
  return scored.map((s) => s.item);
}
