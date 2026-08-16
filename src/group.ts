/**
 * group.ts — an OPTIONAL final-page rollup for `runHybridSearch` (P-036).
 *
 * ── WHY THIS EXISTS ─────────────────────────────────────────────────────────
 * The engine ranks and cuts ROWS, but several callers ultimately present
 * something COARSER than a row. The transcript search ranks session *turns*
 * and renders session *cards*; the eval that pins its quality retrieves 30
 * turn-rows and measures how many distinct SESSIONS they cover. When one
 * session contributes several turns to the page, those turns consume page
 * slots while covering only one card — measured on the live corpus
 * (2026-08-09) the top-30 turns for `theory` covered just 24 distinct
 * sessions, so ~20% of the page was spent re-showing sessions already on it.
 *
 * ── WHY IT MUST LIVE INSIDE THE ENGINE ──────────────────────────────────────
 * A caller cannot fix this after the fact. `runHybridSearch` returns only the
 * top `limit` rows, so by the time a caller could dedupe, the rows that would
 * have back-filled the freed slots are already discarded. The rollup has to
 * happen on the FULL fused candidate pool, before the cut — which is what this
 * module is applied to.
 *
 * ── FIRST-OCCURRENCE IS "BEST", BY CONSTRUCTION ─────────────────────────────
 * `pickTopGroups` keeps the FIRST row it sees per group and relies on its input
 * already being in final rank order (fusion, then the optional recency
 * re-rank). That makes it a MAX-rollup without ever comparing scores — and
 * deliberately so: post-fusion the `score` is an RRF value, and a re-rank may
 * legitimately return rows whose carried retrieval score does not match their
 * new position (the P-010 hazard `groupHitsBySession` documents). Re-deriving
 * "best" by sorting on score here would silently undo any such re-rank. The
 * caller's order is the authority; this module only thins it.
 *
 * ⚠ NOT a scoring change. It cannot promote a session that no member row
 * reached the candidate pool with — it only stops one session from spending
 * several slots. See the plan's D-043/D-044 for what that is and is not worth.
 */

import type { SearchHit } from './types';

/**
 * Extracts the rollup key for a hit. Returning `null`/`undefined` means "this
 * row is not groupable" — such rows are always kept and never collapse into
 * each other, so a caller whose key is only sometimes derivable (an unparseable
 * source_id, a source that has no session) degrades to today's row-level
 * behaviour for exactly those rows instead of silently bucketing them together
 * under a shared falsy key.
 */
export type GroupKeyOf = (hit: SearchHit) => string | null | undefined;

/**
 * Keep the best-ranked row per group, then take the first `limit` of them —
 * i.e. `limit` counts DISTINCT GROUPS rather than raw rows.
 *
 * Pure, order-preserving, and generic over the fused-entry shape so it can run
 * on `rrfCombine`'s `{ row, score, rankers }` without importing it.
 *
 * Stops as soon as `limit` groups are held: the candidate pool is `limit * 3`
 * per leg, so this is a short scan rather than a full pass.
 */
export function pickTopGroups<T extends { row: SearchHit }>(
  fused: readonly T[],
  groupKeyOf: GroupKeyOf,
  limit: number,
): T[] {
  if (limit <= 0) return [];
  const seen = new Set<string>();
  const out: T[] = [];
  for (const entry of fused) {
    const key = groupKeyOf(entry.row);
    if (key === null || key === undefined) {
      // Ungroupable: always its own slot.
      out.push(entry);
    } else {
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(entry);
    }
    if (out.length >= limit) break;
  }
  return out;
}

/**
 * How many DISTINCT groups the full candidate pool holds — the honest
 * denominator for a grouped page ("30 of 68 sessions"), which `totalHits`
 * cannot supply because it counts rows. Ungroupable rows each count as one.
 */
export function countGroups<T extends { row: SearchHit }>(
  fused: readonly T[],
  groupKeyOf: GroupKeyOf,
): number {
  const seen = new Set<string>();
  let ungrouped = 0;
  for (const entry of fused) {
    const key = groupKeyOf(entry.row);
    if (key === null || key === undefined) ungrouped++;
    else seen.add(key);
  }
  return seen.size + ungrouped;
}
