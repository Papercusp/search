/**
 * The chunk-aware vector leg (generic-rag-chunking P-006, D-009/D-016).
 *
 * A long parent's single embedding describes its opening, so a question about
 * its tail misses. Chunking gives the tail its own vectors; this helper is the
 * one query that reads them back. It answers "which PARENTS are nearest to this
 * query vector", looking at both the parent's own vector and its chunks'.
 *
 *   mode 'retrieve'  parent-vector leg UNION ALL chunk leg, each joined to the
 *                    parent and filtered BEFORE its own LIMIT (so the caller's
 *                    filter cannot empty a leg after the fact), then one row
 *                    per parent at its SMALLEST distance (D-016 pooling). A
 *                    chunk's distance carries the surface's chunk margin, so a
 *                    tie between a parent and its own chunk goes to the parent.
 *   mode 'gist'      the parent-vector leg only: "what is this about".
 *
 * Rows come back as the parent key columns (by their own names), `distance`
 * (cosine distance, plus the margin on a chunk hit) and `matched_anchor` (the
 * chunk's anchor on a chunk hit when the store records one, else NULL),
 * ordered by distance and then key. The caller joins that back to its parent
 * for display, which is why the leg is exposed as a FRAGMENT too.
 *
 * Nothing here knows a host's schema or its embedding-space rules: the space
 * filter is injected (D-011) and every identifier is validated, never quoted
 * from caller text.
 */
import type { Fragment } from 'postgres';
import { withIterativeScan } from '../hnsw-iterative-scan';
import type { PgHandle } from '../types';
import { sharedChunkStore } from './shared-store';
import type { ChunkKeyColumn, ChunkStore, ChunkSurface, ChunkVectorTable } from './types';

const IDENT = /^[a-z_][a-z0-9_]*$/;
const QUALIFIED_IDENT = /^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)?$/;
const SQL_TYPE = /^[a-z_][a-z0-9_ ]*(\[\])?$/;
const CHUNK_ALIAS = 'c';

export type ChunkLegMode = 'retrieve' | 'gist';

/**
 * How each leg ranks. 'ann' orders by the vector operator itself, so an HNSW
 * index can serve the ORDER BY. 'exact' materialises the parents `parentFilter`
 * keeps, then ranks that slice and its chunks exhaustively, reaching the chunks
 * through the chunk table's key index instead of the vector index.
 */
export type ChunkLegScan = 'ann' | 'exact';

/** The CTE an 'exact' leg materialises the filtered parents into. */
const SLICE_CTE = 'chunk_leg_slice';

/** The parent's own embedding columns. */
export interface ParentVectorColumns {
  column: string;
  /** Embedding-profile column, handed to the space filter. */
  profileColumn?: string;
  /** Embedding-mode column, handed to the space filter. */
  modeColumn?: string;
}

/** Qualified column references the injected space filter should constrain. */
export interface SpaceFilterColumns {
  profileColumn: string | null;
  modeColumn: string | null;
}

export interface ChunkAwareVectorLegOptions {
  /** The registry entry (name, parent table + key, chunk margin, store). */
  surface: Pick<ChunkSurface, 'surface' | 'parent' | 'chunkMargin' | 'store' | 'parentVector'>;
  /** The query embedding in pgvector text form, e.g. '[0.1,0.2,...]'. */
  qVec: string;
  /** Rows wanted. Each leg fetches this many before pooling. */
  limit: number;
  mode: ChunkLegMode;
  /** Alias the parent table is bound to in both legs. Default 'p'. */
  parentAlias?: string;
  /** A predicate over `parentAlias`, applied inside BOTH legs before their LIMIT. Default TRUE. */
  parentFilter?: Fragment;
  /** Overrides `surface.parentVector`. One of the two is required. */
  parentVector?: ParentVectorColumns;
  /** Overrides the chunk table the surface's store declares (`store.queryTable`). */
  chunks?: ChunkVectorTable;
  /** Overrides `surface.chunkMargin`. */
  chunkMargin?: number;
  /** The active embedding-space predicate (D-011). Default: no constraint. */
  spaceFilter?: (cols: SpaceFilterColumns) => Fragment;
  /**
   * Default 'ann'. Choose 'exact' when `parentFilter` keeps a small fraction of
   * the table: an HNSW scan under a selective filter discards most of what it
   * reads before it fills the LIMIT, and exhaustive ranking of the slice is
   * cheaper and returns the true nearest rows. Measured on work_items:search's
   * feature family (about 2% of rows and of the surface's chunks): 131 ms ANN,
   * about 20k rows discarded, against about 30 ms per leg exact.
   */
  scan?: ChunkLegScan;
}

/** One pooled row. Key columns are present under their own names. */
export type ChunkAwareLegRow = Record<string, unknown> & {
  distance: number;
  matched_anchor: string | null;
};

interface KeyCol {
  column: string;
  type: string | null;
}

function ident(value: string, what: string): string {
  if (!IDENT.test(value)) throw new Error(`chunk vector leg: ${what} '${value}' is not a plain identifier`);
  return value;
}

function qualified(value: string, what: string): string {
  if (!QUALIFIED_IDENT.test(value)) {
    throw new Error(`chunk vector leg: ${what} '${value}' is not a plain [schema.]table identifier`);
  }
  return value;
}

function keyColumnsOf(key: readonly ChunkKeyColumn[], surface: string): KeyCol[] {
  if (key.length === 0) throw new Error(`chunk vector leg: surface '${surface}' has no parent key`);
  return key.map((k) => {
    const column = ident(typeof k === 'string' ? k : k.column, 'key column');
    const type = typeof k === 'string' ? null : k.type.trim().toLowerCase();
    if (type !== null && !SQL_TYPE.test(type)) throw new Error(`chunk vector leg: key type '${type}' is not a plain SQL type name`);
    return { column, type };
  });
}

/**
 * The chunk table to read: the override, else the surface's store, else the
 * shared store — the same default sync writes to when a surface names no store.
 */
export function chunkVectorTableOf(
  surface: Pick<ChunkSurface, 'surface' | 'store'>,
  override?: ChunkVectorTable,
): ChunkVectorTable {
  if (override) return override;
  const store: Pick<ChunkStore, 'name' | 'queryTable'> = surface.store ?? sharedChunkStore();
  if (store.queryTable) return store.queryTable;
  throw new Error(
    `chunk vector leg: surface '${surface.surface}' has no readable chunk table ` +
      `(store '${store.name}' declares no queryTable); pass \`chunks\` or give the store a queryTable`,
  );
}

/**
 * The leg as a fragment, for callers that wrap it in their own query (a display
 * join, highlighting). Execute it with the handle `withIterativeScan` hands you
 * when HNSW is involved — or call `chunkAwareVectorLeg`, which does that.
 */
export function chunkAwareVectorLegSql(sql: PgHandle, opts: ChunkAwareVectorLegOptions): Fragment {
  const surfaceName = opts.surface.surface;
  if (!Number.isInteger(opts.limit) || opts.limit <= 0) throw new Error('chunk vector leg: limit must be a positive integer');
  if (opts.mode !== 'retrieve' && opts.mode !== 'gist') throw new Error(`chunk vector leg: unknown mode '${String(opts.mode)}'`);
  const alias = ident(opts.parentAlias ?? 'p', 'parent alias');
  if (alias === CHUNK_ALIAS) throw new Error(`chunk vector leg: parent alias '${CHUNK_ALIAS}' is reserved for the chunk table`);
  const parentTable = qualified(opts.surface.parent.table, 'parent table');
  const keys = keyColumnsOf(opts.surface.parent.key, surfaceName);
  const pv = opts.parentVector ?? opts.surface.parentVector;
  if (!pv) throw new Error(`chunk vector leg: surface '${surfaceName}' names no parent vector column`);
  const parentVec = `${alias}.${ident(pv.column, 'parent vector column')}`;
  const margin = opts.chunkMargin ?? opts.surface.chunkMargin ?? 0;
  if (!Number.isFinite(margin) || margin < 0) throw new Error('chunk vector leg: chunk margin must be a finite number >= 0');

  const filter = opts.parentFilter ?? sql`TRUE`;
  const space = opts.spaceFilter ?? (() => sql`TRUE`);
  const q = opts.qVec;
  const keySelect = sql.unsafe(keys.map((k) => `${alias}.${k.column}`).join(', '));
  const keyNames = sql.unsafe(keys.map((k) => k.column).join(', '));

  const parentLeg = sql`
    (SELECT ${keySelect}, (${sql.unsafe(parentVec)} <=> ${q}::vector) AS distance,
            NULL::text AS matched_anchor, 0 AS leg
       FROM ${sql.unsafe(parentTable)} ${sql.unsafe(alias)}
      WHERE (${filter})
        AND ${sql.unsafe(parentVec)} IS NOT NULL
        AND (${space({
          profileColumn: pv.profileColumn ? `${alias}.${ident(pv.profileColumn, 'parent profile column')}` : null,
          modeColumn: pv.modeColumn ? `${alias}.${ident(pv.modeColumn, 'parent mode column')}` : null,
        })})
   ORDER BY ${sql.unsafe(parentVec)} <=> ${q}::vector
      LIMIT ${opts.limit})`;

  let legs = parentLeg;
  if (opts.mode === 'retrieve') {
    const chunks = chunkVectorTableOf(opts.surface, opts.chunks);
    const table = qualified(chunks.table, 'chunk table');
    const emb = `${CHUNK_ALIAS}.${ident(chunks.embeddingColumn ?? 'embedding', 'chunk embedding column')}`;
    const anchorCol = chunks.anchorColumn === undefined ? (chunks.keying === 'shared' ? 'anchor' : null) : chunks.anchorColumn;
    const anchor = anchorCol ? `${CHUNK_ALIAS}.${ident(anchorCol, 'chunk anchor column')}` : 'NULL::text';
    const profile = chunks.profileColumn === null ? null : `${CHUNK_ALIAS}.${ident(chunks.profileColumn ?? 'embedding_profile', 'chunk profile column')}`;
    const modeCol = chunks.modeColumn === null ? null : `${CHUNK_ALIAS}.${ident(chunks.modeColumn ?? 'embedding_mode', 'chunk mode column')}`;
    const join = chunks.keying === 'shared'
      ? keys
          .map((k, i) => (k.type
            ? `${alias}.${k.column} = ${CHUNK_ALIAS}.parent_key[${i + 1}]::${k.type}`
            : `${alias}.${k.column}::text = ${CHUNK_ALIAS}.parent_key[${i + 1}]`))
          .join(' AND ')
      : keys.map((k) => `${alias}.${k.column} = ${CHUNK_ALIAS}.${k.column}`).join(' AND ');
    const surfaceMatch = chunks.keying === 'shared' ? sql`${sql.unsafe(CHUNK_ALIAS)}.surface = ${surfaceName} AND` : sql``;
    legs = sql`${parentLeg}
    UNION ALL
    (SELECT ${keySelect}, (${sql.unsafe(emb)} <=> ${q}::vector) + ${margin}::float8 AS distance,
            ${sql.unsafe(anchor)} AS matched_anchor, 1 AS leg
       FROM ${sql.unsafe(table)} ${sql.unsafe(CHUNK_ALIAS)}
       JOIN ${sql.unsafe(parentTable)} ${sql.unsafe(alias)} ON ${sql.unsafe(join)}
      WHERE ${surfaceMatch} (${filter})
        AND ${sql.unsafe(emb)} IS NOT NULL
        AND (${space({ profileColumn: profile, modeColumn: modeCol })})
   ORDER BY ${sql.unsafe(emb)} <=> ${q}::vector
      LIMIT ${opts.limit})`;
  }

  // Parenthesised legs keep each ORDER BY/LIMIT on its own branch; without them
  // the LIMIT binds to the whole UNION and one leg can starve the other out.
  return sql`
    SELECT ${keyNames}, distance, matched_anchor FROM (
      SELECT DISTINCT ON (${keyNames}) ${keyNames}, distance, matched_anchor
        FROM (${legs}) scored
    ORDER BY ${keyNames}, distance, leg
    ) pooled
    ORDER BY distance, ${keyNames}
    LIMIT ${opts.limit}`;
}

/** Run the leg with HNSW iterative scan on and return the pooled rows. */
export async function chunkAwareVectorLeg(sql: PgHandle, opts: ChunkAwareVectorLegOptions): Promise<ChunkAwareLegRow[]> {
  return (await withIterativeScan(sql, (tx) => chunkAwareVectorLegSql(tx, opts) as unknown as Promise<ChunkAwareLegRow[]>));
}
