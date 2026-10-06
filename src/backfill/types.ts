/**
 * Shared shapes for the embedding backfill engine.
 *
 * The engine fills a vector column for rows whose vector is missing or was made
 * in a different embedding space. The host supplies the targets (which tables,
 * which text), the embedder and the storage-width rules; this module supplies
 * the selection, the batching, the guarded writes and the round-robin sweep.
 *
 * Extracted from papercusp's embed-backfill sweep
 * (shared-vector-search-libraries-2026-09-29, P-006 per decision D-004).
 */

import type { PgHandle } from '../types';

/**
 * The part of the SQL client the engine uses: postgres.js's `unsafe(text, params)`.
 * Every table and column name the engine interpolates comes from the host's own
 * {@link BackfillTarget}; every row value is a bound parameter.
 */
export type BackfillSql = Pick<PgHandle, 'unsafe'>;

/** One table whose vector column the engine keeps filled. */
export interface BackfillTarget {
  /** Schema-qualified table name, e.g. `app.notes`. */
  table: string;
  /** The vector column. Its label columns are `<embedCol>_mode`, `_profile`, `_recipe`. */
  embedCol: string;
  /**
   * SQL expression for the text to embed. An empty result means the row is
   * ineligible. Keep free-text fields capped here (`left(col, n)`) as part of the
   * text recipe; the engine also truncates to `maxInputChars` before embedding.
   */
  bodySql: string;
  /** PRIMARY KEY column(s). The write targets the row by these, never by ctid. */
  keyCols: string[];
  /** `ORDER BY` body (without the keyword): the drain order, e.g. `created_at DESC`.
   * Name an indexed column so the pull is an index scan. Omitted ⇒ unordered. */
  orderBySql?: string;
  /** The column recording when a row was WRITTEN, for recent-coverage reads. */
  recencyCol?: string;
  /** How `recencyCol` compares to now(): a timestamptz, or bigint epoch milliseconds. */
  recencyColKind?: 'timestamptz' | 'epochMs';
  /**
   * Version of the text recipe (`bodySql`). Bump it in the same edit that
   * changes `bodySql`; rows recorded with an older recipe become stale. Omitted
   * (or 1) marks nothing stale.
   */
  recipeVersion?: number;
}

/** What one target's backfill did. */
export interface BackfillStats {
  table: string;
  scanned: number;
  /** Rows written. */
  embedded: number;
  errors: number;
  /** Rows embedded but not written: the row's lock was held by another writer, so
   * the write skipped it instead of waiting, or another writer had already brought
   * it current. A held row stays stale and a later pull retries it. */
  writeSkipped: number;
  durationMs: number;
}

/** Embed one text. */
export type BackfillEmbedFn = (text: string) => Promise<number[]>;

/** Embed several texts in one call; returns one vector per input, in order. */
export type BackfillEmbedManyFn = (texts: string[]) => Promise<number[][]>;

/** The exact profile an embedder runs as. Hosts may carry more fields. */
export interface BackfillProfile {
  readonly profileId: string;
}

/** A storage selection for a profile: which `<embedCol>_profile` value to write,
 * and whether rows recorded only by mode under `legacyMode` already count as
 * current. Same shape as the embedding-space module's selection. */
export interface BackfillProfileSelection {
  readonly profileId: string;
  readonly legacyMode: string | null;
}

/** The engine's log sink. Defaults to `console`. */
export interface BackfillLogger {
  log(message: string): void;
  warn(message: string): void;
  error(message: string): void;
}

/** A vector column whose live width disagrees with what the host stores. */
export interface BackfillWidthSkew {
  table: string;
  column: string;
  liveDims: number;
  declaredDims: number;
}
