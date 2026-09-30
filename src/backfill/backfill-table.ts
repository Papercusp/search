/**
 * Fill one target's vector column: select stale, eligible rows in the target's
 * order, embed them (in batches when the host can, per row otherwise), and write
 * each vector with its labels under the same staleness guard the SELECT used.
 */
import { truncateToChars, withDeadline } from './deadline';
import {
  activeRecipeVersion,
  eligiblePredicateSql,
  modeColOf,
  profileColOf,
  recipeColOf,
  stalePredicateSql,
} from './predicates';
import type {
  BackfillEmbedFn,
  BackfillEmbedManyFn,
  BackfillLogger,
  BackfillProfile,
  BackfillProfileSelection,
  BackfillSql,
  BackfillStats,
  BackfillTarget,
} from './types';

/** Texts per batch-embed call. Small on purpose: a timed-out chunk loses only
 * its own rows, which then take the per-row path; earlier chunks are kept. */
export const DEFAULT_BATCH_CHUNK_SIZE = 16;
/** Per-row embed deadline. Fires only on a call that is never coming back. */
export const DEFAULT_ROW_TIMEOUT_MS = 90_000;

export interface BackfillTableOptions<P extends BackfillProfile = BackfillProfile> {
  /** Embed one text (the per-row path, and the fallback for a failed batch). */
  embed: BackfillEmbedFn;
  /** Embed many texts in one call. Omit to embed every row individually. */
  embedMany?: BackfillEmbedManyFn;
  /** The embedder mode `embed` runs as; written to `<embedCol>_mode`. */
  mode: string;
  /** Rows per SELECT. */
  batchSize: number;
  /** Row ceiling for this call. Defaults to 4 × `batchSize`. */
  maxRows?: number;
  /** The table has a `<embedCol>_mode` column (default true). Without it only
   * missing vectors are stale and no mode label is written. */
  spaceAware?: boolean;
  /** The table has a `<embedCol>_recipe` column. */
  recipeColPresent?: boolean;
  /** The table has a `<embedCol>_profile` column. Then `profile` and
   * `resolveProfileSelection` are required, and a profile the storage refuses
   * throws rather than writing an unplaceable vector. */
  profileColPresent?: boolean;
  profile?: P;
  resolveProfileSelection?: (mode: string, profile: P) => BackfillProfileSelection | null;
  /** Does a vector of this width fit the column? A vector that does not is
   * counted as an error and not written. */
  acceptsWidth: (dims: number) => boolean;
  /** Each body is truncated to this many characters before it is embedded, so an
   * oversized row cannot fail (and be retried) forever. */
  maxInputChars: number;
  /** Texts per batch-embed call (default {@link DEFAULT_BATCH_CHUNK_SIZE}). */
  batchChunkSize?: number;
  /** Per-row embed deadline (default {@link DEFAULT_ROW_TIMEOUT_MS}). */
  rowTimeoutMs?: number;
  /** Deadline for one batch-embed call of `n` texts (default `rowTimeoutMs`).
   * Keep it looser than the embedder's own, so its specific error surfaces. */
  batchTimeoutMs?: (n: number) => number;
  logger?: BackfillLogger;
  /** Log line prefix, printed as `[label]` (default `backfill`). */
  logLabel?: string;
  /** How errors name the storage (default `the vector storage`). */
  storageLabel?: string;
}

/**
 * Backfill one target. Stops at `maxRows`, when nothing stale remains, or when a
 * whole batch makes no progress (every row failed), so a persistently failing
 * row cannot spin the loop.
 */
export async function backfillTable<P extends BackfillProfile = BackfillProfile>(
  sql: BackfillSql,
  target: BackfillTarget,
  options: BackfillTableOptions<P>,
): Promise<BackfillStats> {
  const started = Date.now();
  const stats: BackfillStats = { table: target.table, scanned: 0, embedded: 0, errors: 0, durationMs: 0 };
  const {
    embed,
    embedMany,
    mode,
    batchSize,
    acceptsWidth,
    maxInputChars,
    spaceAware = true,
    batchChunkSize = DEFAULT_BATCH_CHUNK_SIZE,
    rowTimeoutMs = DEFAULT_ROW_TIMEOUT_MS,
    logger = console,
    logLabel = 'backfill',
    storageLabel = 'the vector storage',
  } = options;
  const maxRows = options.maxRows ?? batchSize * 4;
  const batchTimeoutMs = options.batchTimeoutMs ?? (() => rowTimeoutMs);
  const keyCols = target.keyCols;
  // Select each key column as k0, k1, … so the UPDATE targets the exact row by key.
  const keySelect = keyCols.map((c, i) => `${c} AS k${i}`).join(', ');

  let profileSelection: BackfillProfileSelection | null = null;
  if (options.profileColPresent) {
    if (!options.profile || !options.resolveProfileSelection) {
      throw new Error(
        `profile-aware target ${target.table}.${target.embedCol} requires an exact embedding profile`,
      );
    }
    profileSelection = options.resolveProfileSelection(mode, options.profile);
    if (!profileSelection) {
      throw new Error(
        `profile ${options.profile.profileId} is incompatible with ${storageLabel}; refusing ${target.table}.${target.embedCol}`,
      );
    }
  }
  const profileAware = profileSelection !== null;
  // Resolved once: the SELECT, the SET and the write guard must agree on it.
  const recipeVersion = activeRecipeVersion(target, options.recipeColPresent === true);
  const recipeAware = recipeVersion !== null;

  // PostgreSQL rejects a bound parameter the statement never references, so each
  // optional label gets a placeholder only when it is live.
  // SELECT: $1 batch size · [mode] · [profile] · [recipe] · failure offset.
  let sn = 1;
  const selModeExpr = spaceAware ? `$${++sn}` : 'NULL';
  const selProfileExpr = profileAware ? `$${++sn}` : null;
  const selRecipeExpr = recipeAware ? `$${++sn}` : null;
  const selOffsetExpr = `$${++sn}`;
  // UPDATE: $1 vector · [mode] · [profile] · [recipe] · key columns.
  let un = 1;
  const updModeExpr = spaceAware ? `$${++un}` : 'NULL';
  const updProfileExpr = profileAware ? `$${++un}` : null;
  const updRecipeExpr = recipeAware ? `$${++un}` : null;
  const updateKeyOffset = un + 1;
  const whereKeys = keyCols.map((c, i) => `${c} = $${i + updateKeyOffset}`).join(' AND ');
  const setClauses = [`${target.embedCol} = $1::vector`];
  if (spaceAware) setClauses.push(`${modeColOf(target)} = ${updModeExpr}`);
  if (profileAware) setClauses.push(`${profileColOf(target)} = ${updProfileExpr}`);
  if (recipeAware) setClauses.push(`${recipeColOf(target)} = ${updRecipeExpr}`);
  const profileTerm = (profileExpr: string | null) =>
    profileAware ? { profileExpr: profileExpr!, legacyModeCompatible: profileSelection!.legacyMode === mode } : undefined;
  // The write guard is BUILT FROM the staleness predicate, never restated.
  const updateSql =
    `UPDATE ${target.table}` +
    `\n   SET ${setClauses.join(', ')}` +
    `\n WHERE ${whereKeys}` +
    `\n   AND ${stalePredicateSql(target, spaceAware, updModeExpr, updRecipeExpr, profileTerm(updProfileExpr))}`;
  const labelParams = [
    ...(spaceAware ? [mode] : []),
    ...(profileAware ? [profileSelection!.profileId] : []),
    ...(recipeAware ? [recipeVersion] : []),
  ];
  const stale = stalePredicateSql(target, spaceAware, selModeExpr, selRecipeExpr, profileTerm(selProfileExpr));

  // Rows that FAIL stay stale, so with a stable order they would head every later
  // pull. OFFSET by the running failure count skips exactly them. A value cursor on
  // the order column would instead skip every row tied with the boundary value.
  let failedSoFar = 0;
  let warnedBatchFallback = false;
  const orderBy = target.orderBySql ? ` ORDER BY ${target.orderBySql}` : '';
  while (stats.scanned < maxRows) {
    const wantRows = Math.min(batchSize, maxRows - stats.scanned);
    const batch = await sql.unsafe<Array<Record<string, string>>>(
      `SELECT ${keySelect}, (${target.bodySql}) AS body
         FROM ${target.table}
        WHERE ${stale}
          AND ${eligiblePredicateSql(target)}${orderBy}
        LIMIT $1 OFFSET ${selOffsetExpr}`,
      [wantRows, ...labelParams, failedSoFar],
    );
    if (!batch.length) break;
    const texts = batch.map((r) => truncateToChars(r.body, maxInputChars));

    // Batch path, in chunks. A null entry sends that one row down the per-row path.
    let batchVectors: Array<number[] | null> | null = null;
    if (embedMany && batch.length > 1) {
      const acc: Array<number[] | null> = new Array<number[] | null>(texts.length).fill(null);
      let anyChunkOk = false;
      for (let off = 0; off < texts.length; off += batchChunkSize) {
        const slice = texts.slice(off, off + batchChunkSize);
        // A lone trailing row IS the per-row path.
        if (slice.length < 2) break;
        try {
          const vecs = await withDeadline(embedMany(slice), batchTimeoutMs(slice.length), 'embed_batch');
          if (vecs.length !== slice.length) {
            // A contract breach, not a transient failure: say so.
            logger.warn(
              `[${logLabel}] batch embed returned ${vecs.length} vectors for ` +
                `${slice.length} texts (${target.table}) — falling back to per-row`,
            );
            break;
          }
          for (let j = 0; j < vecs.length; j++) acc[off + j] = vecs[j]!;
          anyChunkOk = true;
        } catch (err) {
          // Never silent: an always-failing batch path looks exactly like a working
          // one from outside, only slower.
          if (!warnedBatchFallback) {
            warnedBatchFallback = true;
            logger.warn(
              `[${logLabel}] batch embed FAILED for ${target.table}, using the ` +
                `per-row path (slower). First error: ${(err as Error).message}`,
            );
          }
          // The embedder is slower than the budget now; the next chunk would pay the
          // same timeout. The rest of this batch takes the per-row path.
          break;
        }
      }
      batchVectors = anyChunkOk ? acc : null;
    }

    let embeddedThisBatch = 0;
    let failedThisBatch = 0;
    for (const [i, row] of batch.entries()) {
      stats.scanned += 1;
      try {
        const vec = batchVectors?.[i] ?? (await withDeadline(embed(texts[i]!), rowTimeoutMs, 'embed'));
        if (!acceptsWidth(vec.length)) {
          stats.errors += 1;
          failedThisBatch += 1;
          continue;
        }
        const keyVals = keyCols.map((_, k) => row[`k${k}`]);
        // By key, under the staleness guard: a row another sweep already brought
        // current is left alone. Vector and labels are written together.
        await sql.unsafe(updateSql, [`[${vec.join(',')}]`, ...labelParams, ...keyVals]);
        stats.embedded += 1;
        embeddedThisBatch += 1;
      } catch {
        stats.errors += 1;
        failedThisBatch += 1;
      }
    }
    failedSoFar += failedThisBatch;
    // Nothing in this batch embedded: what remains is persistent failure. Stop.
    if (embeddedThisBatch === 0) break;
  }

  stats.durationMs = Date.now() - started;
  return stats;
}
