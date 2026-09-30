/**
 * chunks/embed — the embed side of chunked collections, derived from the registry
 * (generic-rag-chunking P-008).
 *
 * The sync engine writes chunk rows with a NULL embedding, or with a vector
 * copied from an old chunk whose text did not change. Something must then embed
 * the NULL rows. Each store declares where its chunks live and which text to
 * embed (ChunkStore.embedTarget), so a host's embed sweep is a function of the
 * registry: chunkEmbedTargets(REGISTRY) lists the chunk tables to sweep, one
 * entry per table, however many collections write into it. Registering a new
 * collection in an existing store therefore adds no embed configuration at all.
 *
 * embedPendingChunks is a complete, host-agnostic sweep for hosts that have
 * none. A host with its own sweep (papercusp's embed-backfill) can read the
 * same targets instead.
 *
 * The write re-checks, per row, that the vector is still missing AND that the
 * row's embedded text still equals the text that was embedded. A parent re-split
 * between the read and the write replaces its rows, so a stale vector can never
 * land on a chunk whose text changed underneath it.
 */

import type { Sql } from 'postgres';
import { sharedChunkStore } from './shared-store';
import type { ChunkEmbedTarget, ChunkStore, ChunkSurface } from './types';

const IDENT = /^[a-z_][a-z0-9_]*$/;
const QUALIFIED_IDENT = /^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)?$/;
const DEFAULT_LIMIT = 64;

/** Embeds a batch of texts; returns one vector per text, in order. */
export type ChunkEmbedder = (texts: readonly string[]) => Promise<readonly (readonly number[])[]>;

/** A chunk row waiting for its vector. `key` maps each key column to its value. */
export interface PendingChunk {
  key: Record<string, unknown>;
  /** The exact text to embed (the target's embeddedTextSql). */
  text: string;
}

export interface EmbedPendingChunksOptions {
  embed: ChunkEmbedder;
  /** Rows embedded per call. Default 64. */
  limit?: number;
  /** Written to the target's modeColumn, when it has one. */
  mode?: string | null;
  /** Written to the target's profileColumn, when it has one. */
  profile?: string | null;
}

export interface EmbedPendingChunksResult {
  /** Rows read with a NULL vector. */
  selected: number;
  /** Rows that received a vector (fewer than selected when a row changed in between). */
  written: number;
}

function ident(value: string, what: string): string {
  if (!IDENT.test(value)) throw new Error(`chunk embed target: ${what} '${value}' is not a plain identifier`);
  return value;
}

/** Throws unless every identifier in the target is plain and its text SQL is one expression. */
export function validateChunkEmbedTarget(target: ChunkEmbedTarget): ChunkEmbedTarget {
  if (!QUALIFIED_IDENT.test(target.table)) {
    throw new Error(`chunk embed target: table '${target.table}' is not a plain [schema.]table identifier`);
  }
  if (target.keyColumns.length === 0) throw new Error(`chunk embed target '${target.table}': keyColumns is empty`);
  for (const k of target.keyColumns) ident(k, 'key column');
  ident(target.embeddingColumn, 'embedding column');
  if (target.modeColumn !== null) ident(target.modeColumn, 'mode column');
  if (target.profileColumn !== null) ident(target.profileColumn, 'profile column');
  const text = target.embeddedTextSql;
  if (typeof text !== 'string' || text.trim() === '') {
    throw new Error(`chunk embed target '${target.table}': embeddedTextSql must be a non-empty SQL expression`);
  }
  if (text.includes(';')) throw new Error(`chunk embed target '${target.table}': embeddedTextSql must be a single expression (no ';')`);
  return target;
}

/**
 * The embed targets a registry needs: one per distinct chunk table, in first-seen
 * order. A surface with no `store` writes to `defaultStore`, exactly as
 * syncChunkSurfaces and chunkAwareVectorLeg resolve it. Throws when a surface's
 * store declares no embedTarget, or when two stores declare the same table
 * differently: both are registry mistakes, not runtime conditions.
 */
export function chunkEmbedTargets(
  surfaces: readonly Pick<ChunkSurface, 'surface' | 'store'>[],
  defaultStore: ChunkStore = sharedChunkStore(),
): ChunkEmbedTarget[] {
  const byTable = new Map<string, ChunkEmbedTarget>();
  for (const surface of surfaces) {
    const store = surface.store ?? defaultStore;
    const target = store.embedTarget;
    if (!target) {
      throw new Error(
        `chunkEmbedTargets: surface '${surface.surface}' writes to store '${store.name}', which declares no embedTarget`,
      );
    }
    validateChunkEmbedTarget(target);
    const seen = byTable.get(target.table);
    if (!seen) {
      byTable.set(target.table, target);
    } else if (JSON.stringify(seen) !== JSON.stringify(target)) {
      throw new Error(`chunkEmbedTargets: two stores declare table '${target.table}' with different embed targets`);
    }
  }
  return [...byTable.values()];
}

/** Up to `limit` chunk rows with no vector yet, in key order. */
export async function selectPendingChunks(sql: Sql, target: ChunkEmbedTarget, limit: number): Promise<PendingChunk[]> {
  validateChunkEmbedTarget(target);
  if (!Number.isInteger(limit) || limit <= 0) throw new Error('selectPendingChunks: limit must be a positive integer');
  const keyObject = target.keyColumns.map((k) => `'${k}', ${k}`).join(', ');
  const rows = await sql.unsafe<{ key: Record<string, unknown>; text: string }[]>(
    `SELECT jsonb_build_object(${keyObject}) AS key, (${target.embeddedTextSql}) AS text
       FROM ${target.table}
      WHERE ${target.embeddingColumn} IS NULL
      ORDER BY ${target.keyColumns.join(', ')}
      LIMIT $1`,
    [limit] as never[],
  );
  return rows.map((r) => ({ key: r.key, text: r.text }));
}

function vectorText(v: readonly number[], i: number): string {
  if (v.length === 0) throw new Error(`embedPendingChunks: vector ${i} is empty`);
  for (const x of v) if (!Number.isFinite(x)) throw new Error(`embedPendingChunks: vector ${i} has a non-finite component`);
  return `[${v.join(',')}]`;
}

/**
 * Embed up to `limit` chunk rows of one target whose vector is NULL. Returns how
 * many were read and how many were written. Call it repeatedly (e.g. on a tick,
 * once per chunkEmbedTargets entry) until `selected` is 0.
 */
export async function embedPendingChunks(
  sql: Sql,
  target: ChunkEmbedTarget,
  opts: EmbedPendingChunksOptions,
): Promise<EmbedPendingChunksResult> {
  const pending = await selectPendingChunks(sql, target, opts.limit ?? DEFAULT_LIMIT);
  if (pending.length === 0) return { selected: 0, written: 0 };
  const vectors = await opts.embed(pending.map((p) => p.text));
  if (vectors.length !== pending.length) {
    throw new Error(`embedPendingChunks: embedder returned ${vectors.length} vectors for ${pending.length} texts`);
  }
  const payload = pending.map((p, i) => ({ key: p.key, vec: vectorText(vectors[i]!, i), txt: p.text }));

  const params: unknown[] = [JSON.stringify(payload)];
  const sets = [`${target.embeddingColumn} = u._embed_vec::vector`];
  if (target.modeColumn !== null) {
    params.push(opts.mode ?? null);
    sets.push(`${target.modeColumn} = $${params.length}::text`);
  }
  if (target.profileColumn !== null) {
    params.push(opts.profile ?? null);
    sets.push(`${target.profileColumn} = $${params.length}::text`);
  }
  // jsonb_populate_record against the chunk table's own row type turns each
  // key value back into its column type (text[] stays text[], int stays int),
  // so the key comparison below can use the primary-key index.
  const keySelect = target.keyColumns.map((k, i) => `r.${k} AS _embed_k${i}`).join(', ');
  const keyMatch = target.keyColumns.map((k, i) => `t.${k} = u._embed_k${i}`).join(' AND ');
  // The payload is bound as text and cast: a JSON string bound to a jsonb
  // parameter is encoded a second time by postgres-js.
  const result = await sql.unsafe(
    `UPDATE ${target.table} t
        SET ${sets.join(', ')}
       FROM (SELECT ${keySelect}, x.vec AS _embed_vec, x.txt AS _embed_txt
               FROM jsonb_to_recordset($1::text::jsonb) AS x(key jsonb, vec text, txt text)
              CROSS JOIN LATERAL jsonb_populate_record(NULL::${target.table}, x.key) AS r) u
      WHERE ${keyMatch}
        AND t.${target.embeddingColumn} IS NULL
        AND (${target.embeddedTextSql}) = u._embed_txt`,
    params as never[],
  );
  return { selected: pending.length, written: result.count };
}
