/**
 * The backfill sweep: resolve the embedder once, inspect every target once, then
 * drain the live targets round-robin (one batch per target per round) until they
 * are all drained or the time budget runs out.
 *
 * Fairness comes from the rotation, not a row ceiling: no target waits more than
 * one round for a turn, and the target that leads rotates every sweep. A target
 * that embeds nothing in its turn is drained for this sweep, which covers both
 * "nothing left" and "only persistent failures left".
 *
 * A sweeper is single-flight. Its latch lives in a {@link BackfillSweepState} the
 * host may pass in (and pin, when a module can load more than once); the library
 * holds no module state. A latch held past `latchStaleMs` is presumed hung and
 * taken over.
 */
import { backfillTable, type BackfillTableOptions } from './backfill-table';
import { withDeadline } from './deadline';
import { inspectBackfillTarget, type BackfillWidthSkewFn } from './inspect';
import type {
  BackfillEmbedFn,
  BackfillEmbedManyFn,
  BackfillLogger,
  BackfillProfile,
  BackfillSql,
  BackfillStats,
  BackfillTarget,
} from './types';

/** Default wall-clock budget for one sweep. The deadline is checked between
 * targets, never mid-batch, so a sweep can overrun by up to one batch. */
export const DEFAULT_SWEEP_BUDGET_MS = 200_000;
/** A latch held this long is presumed dead. Keep it well above the slowest
 * healthy sweep, or a "recovery" runs two sweeps at once. */
export const DEFAULT_LATCH_STALE_MS = 60 * 60 * 1000;
/** Deadline for resolving the embedder, which happens before any row is read. */
export const DEFAULT_RESOLVE_TIMEOUT_MS = 60_000;

/** What `resolveEmbedder` returns: disabled, or an embedder and what it emits. */
export type ResolvedBackfillEmbedder<P extends BackfillProfile = BackfillProfile> =
  | { readonly mode: 'disabled' }
  | {
      readonly mode: string;
      readonly dims: number;
      readonly profile?: P;
      readonly embed: BackfillEmbedFn;
      readonly embedMany?: BackfillEmbedManyFn | null;
    };

export interface BackfillSweepState {
  running: boolean;
  lastResult: BackfillStats[] | null;
  /** Which target leads the next sweep (rotation offset). */
  sweepCounter: number;
  /** When the in-flight sweep took the latch, or null when idle. */
  startedAt: number | null;
}

export function createBackfillSweepState(): BackfillSweepState {
  return { running: false, lastResult: null, sweepCounter: 0, startedAt: null };
}

export interface BackfillSweepOptions {
  /** Per-target row ceiling for this sweep. Default: none (the budget bounds it). */
  maxRowsPerTarget?: number;
  /** Wall-clock budget for this sweep (default: the sweeper's `budgetMs`). */
  budgetMs?: number;
}

export type BackfillSweepResult = BackfillStats[] | { skipped: 'already_running' };

export interface BackfillSweeperConfig<P extends BackfillProfile = BackfillProfile>
  extends Pick<
    BackfillTableOptions<P>,
    | 'acceptsWidth'
    | 'maxInputChars'
    | 'batchSize'
    | 'batchChunkSize'
    | 'rowTimeoutMs'
    | 'batchTimeoutMs'
    | 'resolveProfileSelection'
    | 'logLabel'
    | 'storageLabel'
  > {
  /** The targets, in canonical order. Read at the start of every sweep. */
  getTargets: () => readonly BackfillTarget[];
  /** The SQL client, read after the embedder resolves. */
  getSql: () => BackfillSql;
  resolveEmbedder: () => Promise<ResolvedBackfillEmbedder<P>>;
  /** A batch embedder for a mode, or null to embed per row. Consulted only when
   * the resolved embedder carries no `embedMany`, and only once it is known to
   * be eligible. */
  resolveBatchEmbed?: (mode: string) => Promise<BackfillEmbedManyFn | null>;
  /** The host's verdict on a vector column's live width. */
  widthSkew?: BackfillWidthSkewFn;
  /** Appended to the width-skew refusal: what to fix in this host. */
  widthSkewHint?: string;
  budgetMs?: number;
  latchStaleMs?: number;
  resolveTimeoutMs?: number;
  logger?: BackfillLogger;
  /** Latch state. Pass a pinned object when the host module can load twice. */
  state?: BackfillSweepState;
  /** Clock, for tests. */
  now?: () => number;
}

export interface BackfillSweeper {
  run(opts?: BackfillSweepOptions): Promise<BackfillSweepResult>;
  lastResult(): BackfillStats[] | null;
  resetForTest(): void;
  readonly state: BackfillSweepState;
}

const zeroStats = (table: string, errors = 0): BackfillStats => ({
  table,
  scanned: 0,
  embedded: 0,
  errors,
  durationMs: 0,
});

export function createBackfillSweeper<P extends BackfillProfile = BackfillProfile>(
  config: BackfillSweeperConfig<P>,
): BackfillSweeper {
  const state = config.state ?? createBackfillSweepState();
  const logger = config.logger ?? console;
  const label = config.logLabel ?? 'backfill';
  const now = config.now ?? Date.now;
  const latchStaleMs = config.latchStaleMs ?? DEFAULT_LATCH_STALE_MS;
  const resolveTimeoutMs = config.resolveTimeoutMs ?? DEFAULT_RESOLVE_TIMEOUT_MS;

  async function run(opts: BackfillSweepOptions = {}): Promise<BackfillSweepResult> {
    const maxRowsPerTarget = opts.maxRowsPerTarget ?? Number.MAX_SAFE_INTEGER;
    const budgetMs = opts.budgetMs ?? config.budgetMs ?? DEFAULT_SWEEP_BUDGET_MS;
    // Every refusal is loud: a healthy sweep runs for minutes, so "a tick found one
    // in flight" and "the latch is wedged" must be distinguishable in the log.
    if (state.running) {
      const heldMs = state.startedAt === null ? 0 : now() - state.startedAt;
      if (heldMs < latchStaleMs) {
        logger.log(
          `[${label}] skipped: a sweep has held the latch for ${Math.round(heldMs / 1000)}s ` +
            `(stale at ${latchStaleMs / 1000}s)`,
        );
        return { skipped: 'already_running' };
      }
      logger.warn(
        `[${label}] STALE LATCH: previous sweep has held it for ${Math.round(heldMs / 1000)}s ` +
          `(> ${latchStaleMs / 1000}s) and is presumed hung — taking over. ` +
          `If this repeats, an embed call is hanging.`,
      );
    }
    state.running = true;
    state.startedAt = now();
    try {
      const targets = config.getTargets();
      // Say we started before the first thing that can hang.
      logger.log(`[${label}] sweep starting: resolving embedder…`);
      const resolved = await withDeadline(config.resolveEmbedder(), resolveTimeoutMs, 'embedder_resolve');
      logger.log(
        `[${label}] embedder resolved: mode=${resolved.mode} dims=${'dims' in resolved ? resolved.dims : 'n/a'}`,
      );
      // Disabled, or emitting a width the columns cannot hold: skip the whole sweep
      // rather than embed a batch per target only to count every row as an error.
      if (!('embed' in resolved) || resolved.mode === 'disabled' || !config.acceptsWidth(resolved.dims)) {
        state.lastResult = targets.map((t) => zeroStats(t.table));
        return state.lastResult;
      }
      const { mode, embed, profile } = resolved;
      const embedMany =
        resolved.embedMany ?? (config.resolveBatchEmbed ? await config.resolveBatchEmbed(mode) : null);
      if (embedMany) logger.log(`[${label}] batch-embed enabled (mode=${mode})`);
      const sql = config.getSql();
      const ext = await sql.unsafe<Array<{ extname: string }>>(
        `SELECT extname FROM pg_extension WHERE extname = 'vector'`,
      );
      if (!ext.length) {
        state.lastResult = targets.map((t) => zeroStats(t.table));
        return state.lastResult;
      }

      // Rotate which target leads this sweep; results return in canonical order.
      const offset = targets.length === 0 ? 0 : state.sweepCounter % targets.length;
      state.sweepCounter += 1;
      const rotated = [...targets.slice(offset), ...targets.slice(0, offset)];

      const byTable = new Map<string, BackfillStats>();
      // Inspect each target once; schema facts do not change mid-sweep.
      const live: Array<{
        target: BackfillTarget;
        spaceAware: boolean;
        profileColPresent: boolean;
        recipeColPresent: boolean;
      }> = [];
      for (const target of rotated) {
        try {
          const inspection = await inspectBackfillTarget(sql, target, config.widthSkew);
          if (!inspection.exists) {
            byTable.set(target.table, zeroStats(target.table));
            continue;
          }
          // The column exists but at the wrong width: every write would be rejected
          // and swallowed as a per-row error, a sweep that reports zero work and logs
          // nothing. Refuse the target loudly instead.
          if (inspection.widthSkew.length > 0) {
            const s = inspection.widthSkew[0]!;
            logger.error(
              `[${label}] SCHEMA WIDTH SKEW — refusing ${s.table}.${s.column}: it is ` +
                `vector(${s.liveDims}) but this code emits ${s.declaredDims}. Every write would be ` +
                `rejected and counted as a per-row error, so the sweep would report zero work and ` +
                `log nothing.` +
                (config.widthSkewHint ? ` ${config.widthSkewHint}` : ''),
            );
            byTable.set(target.table, zeroStats(target.table));
            continue;
          }
          live.push({
            target,
            spaceAware: inspection.spaceAware,
            profileColPresent: inspection.profileColPresent,
            recipeColPresent: inspection.recipeColPresent,
          });
          byTable.set(target.table, zeroStats(target.table));
        } catch (err) {
          byTable.set(target.table, zeroStats(target.table, 1));
          logger.warn(`[${label}] ${target.table} failed: ${(err as Error).message}`);
        }
      }

      const startedDrain = now();
      const deadline = startedDrain + budgetMs;
      const drained = new Set<string>();
      const consumed = new Map<string, number>();
      let rounds = 0;
      while (live.length > drained.size && now() < deadline) {
        rounds += 1;
        for (const { target, spaceAware, profileColPresent, recipeColPresent } of live) {
          if (drained.has(target.table)) continue;
          if (now() >= deadline) break;
          const used = consumed.get(target.table) ?? 0;
          if (used >= maxRowsPerTarget) {
            drained.add(target.table);
            continue;
          }
          try {
            const s = await backfillTable(sql, target, {
              embed,
              embedMany: embedMany ?? undefined,
              mode,
              batchSize: config.batchSize,
              maxRows: Math.min(config.batchSize, maxRowsPerTarget - used),
              spaceAware,
              recipeColPresent,
              profileColPresent,
              profile,
              resolveProfileSelection: config.resolveProfileSelection,
              acceptsWidth: config.acceptsWidth,
              maxInputChars: config.maxInputChars,
              batchChunkSize: config.batchChunkSize,
              rowTimeoutMs: config.rowTimeoutMs,
              batchTimeoutMs: config.batchTimeoutMs,
              logger,
              logLabel: label,
              storageLabel: config.storageLabel,
            });
            consumed.set(target.table, used + s.scanned);
            const acc = byTable.get(target.table)!;
            acc.scanned += s.scanned;
            acc.embedded += s.embedded;
            acc.errors += s.errors;
            acc.durationMs += s.durationMs;
            if (s.embedded === 0) drained.add(target.table);
          } catch (err) {
            byTable.get(target.table)!.errors += 1;
            drained.add(target.table);
            logger.warn(`[${label}] ${target.table} failed: ${(err as Error).message}`);
          }
        }
      }
      if (rounds > 0) {
        const elapsed = Math.round((now() - startedDrain) / 1000);
        logger.log(
          `[${label}] drain: ${rounds} round(s) in ${elapsed}s ` +
            `(budget ${Math.round(budgetMs / 1000)}s) · drained ${drained.size}/${live.length} target(s)` +
            (drained.size < live.length ? ' · budget expired with work remaining' : ''),
        );
      }
      const results = targets.map((t) => byTable.get(t.table)!);
      state.lastResult = results;
      return results;
    } finally {
      state.running = false;
      state.startedAt = null;
    }
  }

  return {
    run,
    lastResult: () => state.lastResult,
    resetForTest() {
      state.running = false;
      state.lastResult = null;
      state.sweepCounter = 0;
      state.startedAt = null;
    },
    state,
  };
}
