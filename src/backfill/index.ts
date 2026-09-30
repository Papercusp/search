/**
 * Embedding backfill engine: keeps a table's vector column filled in the active
 * embedding space. See sweeper.ts for the sweep and backfill-table.ts for one
 * target's pass.
 */
export type {
  BackfillEmbedFn,
  BackfillEmbedManyFn,
  BackfillLogger,
  BackfillProfile,
  BackfillProfileSelection,
  BackfillSql,
  BackfillStats,
  BackfillTarget,
  BackfillWidthSkew,
} from './types';
export {
  BASELINE_RECIPE_VERSION,
  activeRecipeVersion,
  eligiblePredicateSql,
  modeColOf,
  profileColOf,
  recentPredicateSql,
  recipeColOf,
  settledPredicateSql,
  stalePredicateSql,
} from './predicates';
export { truncateToChars, withDeadline } from './deadline';
export {
  DEFAULT_BATCH_CHUNK_SIZE,
  DEFAULT_ROW_TIMEOUT_MS,
  backfillTable,
  type BackfillTableOptions,
} from './backfill-table';
export { inspectBackfillTarget, type BackfillTargetInspection, type BackfillWidthSkewFn } from './inspect';
export {
  DEFAULT_LATCH_STALE_MS,
  DEFAULT_RESOLVE_TIMEOUT_MS,
  DEFAULT_SWEEP_BUDGET_MS,
  createBackfillSweepState,
  createBackfillSweeper,
  type BackfillSweepOptions,
  type BackfillSweepResult,
  type BackfillSweepState,
  type BackfillSweeper,
  type BackfillSweeperConfig,
  type ResolvedBackfillEmbedder,
} from './sweeper';
