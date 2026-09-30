/**
 * chunks/types — the registry contract for chunked search collections.
 *
 * A host registers one `ChunkSurface` per collection whose rows can be longer
 * than its embedder's input window. The sync engine (engine.ts) keeps a chunk
 * store in step with the parent rows: it splits new and edited parents, reuses
 * the embedding of any chunk whose text did not change, and prunes the chunks
 * of parents that were deleted or no longer qualify. Adding a collection is a
 * registry entry; no per-collection code or migration.
 *
 * SQL FRAGMENTS ARE HOST-AUTHORED CODE. `textSql`, `headerSql`, `eligibleSql`
 * and `versionSql` are SQL expressions evaluated with the parent table aliased
 * `p` (e.g. `p.content`, `p.updated_at`). They are interpolated into queries,
 * so they must come from the host's source code, never from user input.
 * Table and key identifiers are validated (validateChunkSurface).
 */

import type { Sql } from 'postgres';

/** Fixed-width windows with overlap (splitWindows). */
export interface WindowChunkSplitter {
  kind: 'window';
  /** Window width in UTF-16 code units. */
  size: number;
  /** Code units shared by consecutive windows; must be smaller than size. */
  overlap: number;
}

/** One chunk per markdown heading section (splitMarkdown). */
export interface MarkdownChunkSplitter {
  kind: 'markdown';
  /** Per-part cap; a longer section is split on line boundaries. */
  maxChars: number;
  /** Sections shorter than this (trimmed) are dropped. Default 20. */
  minChars?: number;
  /** Deepest heading level that starts a section. Default 3. */
  headingDepth?: number;
}

export type ChunkSplitter = WindowChunkSplitter | MarkdownChunkSplitter;

/**
 * One parent key column. A bare string is a text column; `{ column, type }`
 * names the column's SQL type so a chunk key (always text) can be cast back to
 * it and the parent's primary-key index stays usable in the prune anti-join.
 */
export type ChunkKeyColumn = string | { column: string; type: string };

export interface ChunkSurface {
  /** Stable collection name, stored in every chunk row (e.g. 'plans'). */
  surface: string;
  parent: {
    /** Schema-qualified parent table, e.g. 'harness_shared.harness_plans'. */
    table: string;
    /** Primary-key columns, in order. Stored as the chunk's parent_key text[]. */
    key: readonly ChunkKeyColumn[];
  };
  /** SQL expression (over alias `p`) for the text to chunk. */
  textSql: string;
  /** SQL expression (over `p`) for a context line embedded before every chunk. */
  headerSql?: string;
  /** SQL boolean (over `p`); rows where it is false are not chunked and are pruned. */
  eligibleSql?: string;
  /**
   * SQL timestamptz (over `p`) that advances whenever the text may have
   * changed (e.g. `p.updated_at`). With it, detection compares the chunk's
   * stored version instead of hashing parent text in SQL. Strongly
   * recommended for large tables: hashing every long row on every tick is the
   * trap turn-chunk-sync measured at ~5.5s per tick.
   */
  versionSql?: string;
  /** Rows no longer than this (characters) are not chunked. Default 2000, the parent vector's window. */
  minChars?: number;
  splitter: ChunkSplitter;
  /** Cap on chunks per parent. Text past the last chunk is dropped, logged and counted. */
  maxChunks: number;
  /**
   * Query-time only: subtracted from every chunk similarity before best-match
   * pooling with the parent vector (P-001 measured it per collection). The
   * sync engine ignores it.
   */
  chunkMargin?: number;
  /** Where chunks are stored. Default: the shared store (sharedChunkStore()). */
  store?: ChunkStore;
  /**
   * The parent's own embedding columns, read by the chunk-aware vector leg
   * (chunkAwareVectorLeg). Optional because sync never reads them.
   */
  parentVector?: { column: string; profileColumn?: string; modeColumn?: string };
}

/**
 * Where a store's chunk vectors can be READ from, for the chunk-aware vector leg.
 * 'shared' = one table for every surface keyed (surface, parent_key text[]);
 * 'typed' = a per-surface table carrying the parent key columns by name.
 * Column names default to embedding / embedding_profile / embedding_mode, and the
 * anchor to 'anchor' for shared keying (none for typed). null = the column is absent.
 */
export interface ChunkVectorTable {
  table: string;
  keying: 'shared' | 'typed';
  anchorColumn?: string | null;
  embeddingColumn?: string;
  profileColumn?: string | null;
  modeColumn?: string | null;
}

/** A surface after validation, with its defaults applied. */
export interface ResolvedChunkSurface extends ChunkSurface {
  minChars: number;
  keyColumns: readonly { column: string; type: string | null }[];
  /** Changes whenever splitter output for the same text could change, so a registry edit re-cuts. */
  splitterVersion: string;
}

/** A parent row the store selected for (re)chunking. */
export interface StaleParent {
  key: string[];
  text: string;
  header: string | null;
  /**
   * The parent's versionSql value as read, in Postgres text form; null when
   * the surface has none. Opaque to the engine and written back verbatim:
   * text keeps Postgres's microsecond precision, which a JS Date truncates to
   * milliseconds, and a truncated write-back reads as older than its parent,
   * so the parent would be re-selected forever.
   */
  version: string | null;
}

/** A stored chunk, as read back before a parent is rewritten. */
export interface ExistingChunk {
  chunkIdx: number;
  chunkSha: string;
  parentSha: string;
  splitterVersion: string;
  /** The stored vector in pgvector text form ('[0.1,0.2,…]'), or null when unembedded. */
  embedding: string | null;
  embeddingMode: string | null;
  embeddingProfile: string | null;
}

/** One chunk the planner cut from a parent. */
export interface PlannedChunk {
  chunkIdx: number;
  /** Markdown heading anchor; null for window chunks. */
  anchor: string | null;
  /** Context line embedded before the content; null when there is none. */
  header: string | null;
  content: string;
  /** hash(embeddedText) — the change key an embedding is reused by. */
  chunkSha: string;
}

/** A chunk row to write: the planned chunk plus any reused embedding. */
export interface ChunkRowToWrite extends PlannedChunk {
  embedding: string | null;
  embeddingMode: string | null;
  embeddingProfile: string | null;
}

export interface ChunkWriteMeta {
  parentSha: string;
  splitterVersion: string;
  /** StaleParent.version, verbatim. */
  version: string | null;
}

/**
 * Where a surface's chunks live and how its parents are found. The engine is
 * store-agnostic: the shared store (text_chunks) and a host's dedicated table
 * implement the same five operations.
 */
export interface ChunkStore {
  /** Human-readable name for logs, e.g. 'harness_shared.text_chunks'. */
  readonly name: string;
  /** Run fn in one transaction (a parent's delete + insert must be atomic). */
  transaction<T>(sql: Sql, fn: (tx: Sql) => Promise<T>): Promise<T>;
  /**
   * Parents longer than minChars, eligible, whose stored chunks are missing,
   * cut by another splitterVersion, or older than the parent (versionSql) /
   * cut from different text (no versionSql). Newest first when versioned.
   */
  selectStale(
    sql: Sql,
    surface: ResolvedChunkSurface,
    opts: { limit: number; excludeKeys: readonly string[] },
  ): Promise<StaleParent[]>;
  readExisting(sql: Sql, surface: ResolvedChunkSurface, key: readonly string[]): Promise<ExistingChunk[]>;
  /** Replace ALL of a parent's chunks with rows (rows may be empty). */
  replace(
    sql: Sql,
    surface: ResolvedChunkSurface,
    key: readonly string[],
    rows: readonly ChunkRowToWrite[],
    meta: ChunkWriteMeta,
  ): Promise<void>;
  /** The text did not change: advance the stored version only. */
  touch(sql: Sql, surface: ResolvedChunkSurface, key: readonly string[], version: string | null): Promise<void>;
  /**
   * Delete the chunks of up to `limit` parents that were deleted, became
   * ineligible, or shrank to minChars or less. Returns the chunk rows deleted.
   */
  prune(sql: Sql, surface: ResolvedChunkSurface, opts: { limit: number }): Promise<{ parents: number; chunks: number }>;
}

/** Injected hash. MUST be lowercase sha256 hex when any surface lacks versionSql (the store compares it with SQL sha256). */
export type ChunkHash = (text: string) => string;

export interface ChunkSyncLogger {
  warn(message: string, meta?: Record<string, unknown>): void;
  error(message: string, meta?: Record<string, unknown>): void;
}

export interface ChunkSurfaceSyncStats {
  surface: string;
  parentsSynced: number;
  parentsUnchanged: number;
  parentsEmpty: number;
  chunksWritten: number;
  embeddingsReused: number;
  pruned: number;
  parentsTruncatedByMaxChunks: number;
  errors: number;
  /** A full batch was selected, so more parents may be waiting. */
  more: boolean;
}

export interface ChunkSyncResult {
  surfaces: ChunkSurfaceSyncStats[];
  /** Where the next call should start, so a time budget cannot starve the later surfaces. */
  nextIndex: number;
  /** Surfaces not reached before the time budget ran out. */
  skippedForBudget: number;
}

export interface ChunkSyncOptions {
  hash: ChunkHash;
  /** Parents re-chunked per surface per call. Default 50. */
  batchPerSurface?: number;
  /** Parents pruned per surface per call. Default 500. Pass 0 to skip pruning. */
  pruneBatch?: number;
  /** Stop starting new surfaces (and new parents) after this many ms. Default: no budget. */
  timeBudgetMs?: number;
  /** Round-robin start position (the previous call's nextIndex). */
  startIndex?: number;
  /** Default: console. */
  logger?: ChunkSyncLogger;
  /**
   * Keys (surface + '\u001f' + key parts joined by '\u001f') whose text cut to
   * zero chunks. Caller-owned so it can persist across calls; the engine adds
   * to it and the store skips those parents, so they cannot hold the head of
   * the queue forever.
   */
  emptyKeys?: Set<string>;
  /** Clock, for tests. Default Date.now. */
  now?: () => number;
}
