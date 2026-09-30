/**
 * The SQL predicates that decide which rows the backfill still owes work on.
 *
 * They are shared, not restated, by everything that asks the same question: the
 * sweep's SELECT, its write guard, and any coverage measure. A restated copy
 * drifts the moment a `bodySql` or a label column changes.
 *
 * Label columns follow the embedding-space convention: `<embedCol>_mode` names
 * the embedder mode, `<embedCol>_profile` the exact profile id, and
 * `<embedCol>_recipe` the text-recipe version. The two identity columns and the
 * recipe column are independent axes; a recipe bump is invisible to any query
 * that matches on mode or profile.
 */
import type { BackfillTarget } from './types';

/** The column recording which embedder mode produced a row's vector. */
export function modeColOf(target: Pick<BackfillTarget, 'embedCol'>): string {
  return `${target.embedCol}_mode`;
}

/** The column recording the exact embedding profile that produced a row's vector. */
export function profileColOf(target: Pick<BackfillTarget, 'embedCol'>): string {
  return `${target.embedCol}_profile`;
}

/** The column recording which text recipe produced a row's vector. A sibling
 * column, never a token folded into `_mode`: query-time readers equality-match
 * the mode, and a composite token would match nothing there. */
export function recipeColOf(target: Pick<BackfillTarget, 'embedCol'>): string {
  return `${target.embedCol}_recipe`;
}

/**
 * The recipe version a row with a NULL `<embedCol>_recipe` holds. A row embedded
 * before the column existed was made by the recipe in force then, which is
 * version 1 by definition, so adding the column re-embeds nothing.
 */
export const BASELINE_RECIPE_VERSION = 1;

/**
 * The recipe version to enforce for a target, or null for "no recipe term":
 * the target declares no `recipeVersion`, or its table has no recipe column.
 */
export function activeRecipeVersion(target: BackfillTarget, recipeColPresent: boolean): number | null {
  if (!recipeColPresent) return null;
  return target.recipeVersion ?? null;
}

/** The rows a target can ever embed: those whose text is non-empty. */
export function eligiblePredicateSql(target: BackfillTarget): string {
  return `length(${target.bodySql}) > 0`;
}

/**
 * The rows the backfill still owes work on: no vector, or a vector in a
 * different embedding space, or (when `recipeExpr` is given) made from a
 * different text recipe.
 *
 * `modeExpr`, `recipeExpr` and `profile.profileExpr` are SQL expressions: a bind
 * placeholder such as `$2`, or a quoted literal. With `spaceAware` false (no mode
 * column) only a missing vector is stale.
 *
 * With `profile`, the exact profile column is authoritative. A NULL profile is
 * accepted as current only when `legacyModeCompatible` (the requested profile is
 * the current profile of the requested mode) and the row's mode matches.
 *
 * `IS DISTINCT FROM`, never `<>`: a NULL label is unknown and therefore stale.
 *
 * ⚠ A caller that ACTS on this predicate must re-assert the same expression in
 * its write guard. A narrower guard selects a row as stale, matches nothing on
 * write, and re-selects and re-embeds it forever.
 */
export function stalePredicateSql(
  target: BackfillTarget,
  spaceAware: boolean,
  modeExpr: string,
  recipeExpr: string | null = null,
  profile?: { profileExpr: string; legacyModeCompatible: boolean },
): string {
  const spaceTerm = profile
    ? profile.legacyModeCompatible && spaceAware
      ? `(${target.embedCol} IS NULL OR NOT (` +
        `${profileColOf(target)} IS NOT DISTINCT FROM ${profile.profileExpr} OR ` +
        `(${profileColOf(target)} IS NULL AND ${modeColOf(target)} IS NOT DISTINCT FROM ${modeExpr})))`
      : `(${target.embedCol} IS NULL OR ${profileColOf(target)} IS DISTINCT FROM ${profile.profileExpr})`
    : spaceAware
      ? `(${target.embedCol} IS NULL OR ${modeColOf(target)} IS DISTINCT FROM ${modeExpr})`
      : `${target.embedCol} IS NULL`;
  if (recipeExpr === null) return spaceTerm;
  return (
    `(${spaceTerm} OR coalesce(${recipeColOf(target)}, ${BASELINE_RECIPE_VERSION})` +
    ` IS DISTINCT FROM ${recipeExpr})`
  );
}

/**
 * Rows written within `hoursExpr` hours of now, or null when the target has no
 * write-time column. `hoursExpr` is a SQL expression (normally a bind placeholder).
 */
export function recentPredicateSql(target: BackfillTarget, hoursExpr: string): string | null {
  if (!target.recencyCol) return null;
  return target.recencyColKind === 'epochMs'
    ? `${target.recencyCol} >= (extract(epoch from now()) * 1000)::bigint - (${hoursExpr}::bigint * 3600000)`
    : `${target.recencyCol} >= now() - make_interval(hours => ${hoursExpr}::int)`;
}

/**
 * Rows written inside a SETTLED window: older than `graceMinExpr` minutes (the
 * sweep has had time to reach them) and newer than `hoursExpr` hours. Null when
 * the target has no write-time column.
 *
 * Both bounds matter. A single recent window averages the healthy write path
 * with an old, still-draining backlog and describes neither; the young bound
 * excludes rows that are legitimately unembedded because the sweep has not run
 * since they were written.
 */
export function settledPredicateSql(target: BackfillTarget, graceMinExpr: string, hoursExpr: string): string | null {
  if (!target.recencyCol) return null;
  const col = target.recencyCol;
  return target.recencyColKind === 'epochMs'
    ? `(${col} <= (extract(epoch from now()) * 1000)::bigint - (${graceMinExpr}::bigint * 60000)` +
        ` AND ${col} >= (extract(epoch from now()) * 1000)::bigint - (${hoursExpr}::bigint * 3600000))`
    : `(${col} <= now() - make_interval(mins => ${graceMinExpr}::int)` +
        ` AND ${col} >= now() - make_interval(hours => ${hoursExpr}::int))`;
}
