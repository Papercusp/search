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
 * Each leg ranks exactly over a materialised slice of the filtered parents, or
 * by ANN over an HNSW index (`scan`); `chunkScan` lets the chunk leg stay ANN
 * while the parent leg is exact (D-046).
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
/** A surface name the ANN chunk leg may inline as a SQL literal: no quote can appear. */
const SURFACE_LITERAL = /^[a-z0-9_][a-z0-9_.:-]*$/;
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
  /**
   * How the CHUNK leg ranks. Default: the same as `scan`. The one mixed form is
   * `scan: 'exact'` with `chunkScan: 'ann'` (generic-rag-chunking D-046): the
   * parent leg stays exact over the materialised slice, while the chunk leg
   * reads the chunk table alone, ordered by the vector operator, so an HNSW
   * index on the chunk table can serve it. Use it when the slice's chunks are
   * many and each one's vector is TOASTed, so the exact chunk leg's per-parent
   * key probes and detoasting cost more than an index walk.
   *
   * The chunk leg then names the surface as a SQL literal, so a partial HNSW
   * index `WHERE surface = '<name>'` matches even under a generic plan, and
   * tests slice membership as a filter over `to_jsonb(parent_key)` rather than
   * a join, so the planner has no cheaper join plan to prefer over the index.
   * Rows are approximate (ANN): pair it with `withIterativeScan`'s `efSearch`.
   * Requires a shared-keying chunk table and mode 'retrieve' to matter.
   */
  chunkScan?: ChunkLegScan;
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

  const scan = opts.scan ?? 'ann';
  if (scan !== 'ann' && scan !== 'exact') throw new Error(`chunk vector leg: unknown scan '${String(scan)}'`);
  const exact = scan === 'exact';
  const chunkScan = opts.chunkScan ?? scan;
  if (chunkScan !== 'ann' && chunkScan !== 'exact') throw new Error(`chunk vector leg: unknown chunkScan '${String(chunkScan)}'`);
  if (chunkScan === 'exact' && !exact) {
    throw new Error("chunk vector leg: chunkScan 'exact' needs scan 'exact' (the exact chunk leg reads the materialised slice)");
  }
  // The mixed form (D-046): exact parent leg over the slice, ANN chunk leg.
  const chunkAnnOverSlice = exact && chunkScan === 'ann';

  const filter = opts.parentFilter ?? sql`TRUE`;
  const space = opts.spaceFilter ?? (() => sql`TRUE`);
  const q = opts.qVec;
  const keySelect = sql.unsafe(keys.map((k) => `${alias}.${k.column}`).join(', '));
  const keyNames = sql.unsafe(keys.map((k) => k.column).join(', '));

  // 'exact': the filter runs once, into a materialised slice that keeps the
  // parent's own column names, so both legs read `alias.col` unchanged and no
  // vector index can serve either ORDER BY. 'ann': both legs read the table.
  const profileCol = pv.profileColumn ? ident(pv.profileColumn, 'parent profile column') : null;
  const modeCol = pv.modeColumn ? ident(pv.modeColumn, 'parent mode column') : null;
  const sliceCols = [...new Set([...keys.map((k) => k.column), ident(pv.column, 'parent vector column'), profileCol, modeCol])]
    .filter((c): c is string => c !== null)
    .map((c) => `${alias}.${c}`)
    .join(', ');
  const slice = exact
    ? sql`WITH ${sql.unsafe(SLICE_CTE)} AS MATERIALIZED (
        SELECT ${sql.unsafe(sliceCols)} FROM ${sql.unsafe(parentTable)} ${sql.unsafe(alias)} WHERE (${filter}))`
    : sql``;
  const parentSource = sql.unsafe(`${exact ? SLICE_CTE : parentTable} ${alias}`);
  const parentWhere = exact ? sql`TRUE` : filter;

  const parentLeg = sql`
    (SELECT ${keySelect}, (${sql.unsafe(parentVec)} <=> ${q}::vector) AS distance,
            NULL::text AS matched_anchor, 0 AS leg
       FROM ${parentSource}
      WHERE (${parentWhere})
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
    // 'exact' over the shared table compares the whole key array, in the same
    // `ARRAY[(p.col)::text, ...]` form the shared store writes it, so the chunk
    // table's (surface, parent_key, chunk_idx) key index serves each lookup.
    const join = chunks.keying === 'shared'
      ? exact
        ? `${CHUNK_ALIAS}.parent_key = ARRAY[${keys.map((k) => `(${alias}.${k.column})::text`).join(', ')}]`
        : keys
            .map((k, i) => (k.type
              ? `${alias}.${k.column} = ${CHUNK_ALIAS}.parent_key[${i + 1}]::${k.type}`
              : `${alias}.${k.column}::text = ${CHUNK_ALIAS}.parent_key[${i + 1}]`))
            .join(' AND ')
      : keys.map((k) => `${alias}.${k.column} = ${CHUNK_ALIAS}.${k.column}`).join(' AND ');
    const surfaceMatch = chunks.keying === 'shared' ? sql`${sql.unsafe(CHUNK_ALIAS)}.surface = ${surfaceName} AND` : sql``;
    if (chunkAnnOverSlice) {
      if (chunks.keying !== 'shared') {
        throw new Error(
          `chunk vector leg: chunkScan 'ann' with scan 'exact' needs a shared-keying chunk table; ` +
            `surface '${surfaceName}' reads '${table}' (keying '${chunks.keying}')`,
        );
      }
      if (!SURFACE_LITERAL.test(surfaceName)) {
        throw new Error(`chunk vector leg: surface '${surfaceName}' cannot be inlined as a SQL literal`);
      }
      // D-046. The inner query reads the chunk table ALONE and orders by the
      // vector operator, so an HNSW index can serve it; three choices keep the
      // planner on that index:
      //   - the surface is a literal, so a partial index `WHERE surface = '<name>'`
      //     matches under a generic plan too (a bind parameter never does);
      //   - membership in the slice is a filter over to_jsonb(parent_key) against
      //     a jsonb array built once from the slice, not a join: given a join, the
      //     planner prefers per-parent key probes, which is the cost this avoids
      //     (jsonb, because ANY over an array of text arrays flattens it to text);
      //   - nothing else in the inner query is ordered or joined.
      // The LIMIT stays inside, so the join back to the slice (for the parent's
      // typed key columns) reads at most `limit` rows.
      const sliceKey = `ARRAY[${keys.map((k) => `(${alias}.${k.column})::text`).join(', ')}]`;
      legs = sql`${parentLeg}
    UNION ALL
    (SELECT ${keySelect}, hit.distance, hit.matched_anchor, 1 AS leg
       FROM (SELECT ${sql.unsafe(CHUNK_ALIAS)}.parent_key AS parent_key,
                    (${sql.unsafe(emb)} <=> ${q}::vector) + ${margin}::float8 AS distance,
                    ${sql.unsafe(anchor)} AS matched_anchor
               FROM ${sql.unsafe(table)} ${sql.unsafe(CHUNK_ALIAS)}
              WHERE ${sql.unsafe(`${CHUNK_ALIAS}.surface = '${surfaceName}'`)}
                AND ${sql.unsafe(emb)} IS NOT NULL
                AND (${space({ profileColumn: profile, modeColumn: modeCol })})
                AND to_jsonb(${sql.unsafe(CHUNK_ALIAS)}.parent_key) = ANY (ARRAY(
                      SELECT to_jsonb(${sql.unsafe(sliceKey)}) FROM ${sql.unsafe(`${SLICE_CTE} ${alias}`)}))
           ORDER BY ${sql.unsafe(emb)} <=> ${q}::vector
              LIMIT ${opts.limit}) hit
       JOIN ${sql.unsafe(`${SLICE_CTE} ${alias}`)} ON hit.parent_key = ${sql.unsafe(sliceKey)})`;
    } else {
      // 'exact' orders by the output distance, which adds the margin: an HNSW
      // index can only serve `ORDER BY <column> <=> <query>` itself, so this sort is
      // exhaustive over the slice's chunks. Same order, since the margin is constant.
      const chunkOrder = exact ? sql`distance` : sql`${sql.unsafe(emb)} <=> ${q}::vector`;
      legs = sql`${parentLeg}
    UNION ALL
    (SELECT ${keySelect}, (${sql.unsafe(emb)} <=> ${q}::vector) + ${margin}::float8 AS distance,
            ${sql.unsafe(anchor)} AS matched_anchor, 1 AS leg
       FROM ${sql.unsafe(table)} ${sql.unsafe(CHUNK_ALIAS)}
       JOIN ${parentSource} ON ${sql.unsafe(join)}
      WHERE ${surfaceMatch} (${parentWhere})
        AND ${sql.unsafe(emb)} IS NOT NULL
        AND (${space({ profileColumn: profile, modeColumn: modeCol })})
   ORDER BY ${chunkOrder}
      LIMIT ${opts.limit})`;
    }
  }

  // Parenthesised legs keep each ORDER BY/LIMIT on its own branch; without them
  // the LIMIT binds to the whole UNION and one leg can starve the other out.
  return sql`${slice}
    SELECT ${keyNames}, distance, matched_anchor FROM (
      SELECT DISTINCT ON (${keyNames}) ${keyNames}, distance, matched_anchor
        FROM (${legs}) scored
    ORDER BY ${keyNames}, distance, leg
    ) pooled
    ORDER BY distance, ${keyNames}
    LIMIT ${opts.limit}`;
}

/**
 * Introspection only: the (profile, mode) column pairs the leg hands its
 * `spaceFilter`, in leg order (parent leg, then the chunk leg when `mode` is
 * 'retrieve'). It builds the leg against `sql` and DISCARDS the fragment, so
 * nothing executes and no iterative-scan cap applies. Callers that only need
 * the column census (embedding-space parity) use this rather than calling the
 * builder outside withIterativeScan.
 */
export function chunkAwareVectorLegSpaceColumns(
  sql: PgHandle,
  opts: Omit<ChunkAwareVectorLegOptions, 'spaceFilter' | 'qVec' | 'limit'>,
): SpaceFilterColumns[] {
  const seen: SpaceFilterColumns[] = [];
  chunkAwareVectorLegSql(sql, {
    ...opts,
    qVec: '[0]',
    limit: 1,
    spaceFilter: (cols) => {
      seen.push(cols);
      return sql`TRUE`;
    },
  } as ChunkAwareVectorLegOptions);
  return seen;
}

/** Run the leg with HNSW iterative scan on and return the pooled rows. */
export async function chunkAwareVectorLeg(sql: PgHandle, opts: ChunkAwareVectorLegOptions): Promise<ChunkAwareLegRow[]> {
  return (await withIterativeScan(sql, (tx) => chunkAwareVectorLegSql(tx, opts) as unknown as Promise<ChunkAwareLegRow[]>));
}
