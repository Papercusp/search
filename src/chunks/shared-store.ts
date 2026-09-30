/**
 * chunks/shared-store — the ChunkStore for ONE shared chunk table holding
 * every surface's chunks, keyed (surface, parent_key text[], chunk_idx).
 *
 * Expected columns (the DDL is sql/text-chunks.reference.sql; papercusp's copy is
 * harness_shared.text_chunks, migration 1242):
 *   surface text, parent_key text[], chunk_idx int, anchor text, header text,
 *   content text, parent_sha text, chunk_sha text, splitter_version text,
 *   embedding vector, embedding_mode text, embedding_profile text,
 *   updated_at timestamptz, PRIMARY KEY (surface, parent_key, chunk_idx).
 *
 * There is no foreign key (parents live in tables with different key shapes),
 * so prune() is what removes the chunks of deleted parents, and readers must
 * join back to the parent.
 *
 * `updated_at` stores the parent's versionSql value AS READ at sync time (not
 * now()), so a parent edited while its chunks were being written stays
 * "newer than its chunks" and is picked up again: no commit race.
 *
 * The version travels as PG text end to end and is bound as `$n::text::timestamptz`.
 * postgres-js describes a statement's parameter types and serializes a
 * timestamptz parameter through `new Date(x)`, even when x is a string, which
 * truncates microseconds; the stored copy would then compare older than its
 * parent forever and the parent would be re-selected on every tick.
 */

import type { Sql } from 'postgres';
import type { ChunkStore, ExistingChunk, ResolvedChunkSurface, StaleParent } from './types';

const QUALIFIED_IDENT = /^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)?$/;

export interface SharedChunkStoreOptions {
  /** Schema-qualified chunk table. Default 'harness_shared.text_chunks'. */
  table?: string;
}

class Params {
  readonly values: unknown[] = [];
  add(v: unknown): string {
    this.values.push(v);
    return `$${this.values.length}`;
  }
}

function keyTextArray(surface: ResolvedChunkSurface): string {
  return `ARRAY[${surface.keyColumns.map((k) => `(p.${k.column})::text`).join(', ')}]`;
}

/**
 * Parent-key equality. A typed key casts the chunk side, so the parent's
 * primary-key index stays usable; an untyped key compares the parent column as
 * text, the same form the chunk-aware vector leg uses (vector-leg.ts). For a
 * text or varchar column that cast is a no-op and the index is still used; for
 * any other column type it is still correct, where a bare `p.col = parent_key[i]`
 * fails with "operator does not exist: uuid = text" (measured on
 * harness_shared.operator_turns, 2026-09-30).
 */
function keyMatchesChunk(surface: ResolvedChunkSurface, chunkAlias: string): string {
  return surface.keyColumns
    .map((k, i) => {
      const part = `${chunkAlias}.parent_key[${i + 1}]`;
      return k.type === null ? `(p.${k.column})::text = ${part}` : `p.${k.column} = (${part})::${k.type}`;
    })
    .join(' AND ');
}

/**
 * SQL form of engine.embeddedChunkText over a shared chunk row: `header\ncontent`,
 * or content alone when the header is NULL or empty. It must stay byte-identical
 * to embeddedChunkText, since chunk_sha hashes that text and a re-split reuses a
 * vector by chunk_sha. A host may wrap it (e.g. left(…, n)) to fit its embedder.
 */
export const SHARED_CHUNK_EMBEDDED_TEXT_SQL = `concat_ws(E'\\n', nullif(header, ''), content)`;

/** SQL form of engine.parentShaOf — must stay byte-identical to it. */
export function parentShaSql(textExpr: string, headerExpr: string): string {
  return `encode(sha256(convert_to(CASE WHEN nullif(${headerExpr}, '') IS NULL THEN ${textExpr} ELSE ${headerExpr} || chr(31) || ${textExpr} END, 'UTF8')), 'hex')`;
}

export function sharedChunkStore(opts: SharedChunkStoreOptions = {}): ChunkStore {
  const table = opts.table ?? 'harness_shared.text_chunks';
  if (!QUALIFIED_IDENT.test(table)) throw new Error(`sharedChunkStore: table '${table}' is not a plain [schema.]table identifier`);

  return {
    name: table,
    queryTable: { table, keying: 'shared' },
    embedTarget: {
      table,
      keyColumns: ['surface', 'parent_key', 'chunk_idx'],
      embeddingColumn: 'embedding',
      embeddedTextSql: SHARED_CHUNK_EMBEDDED_TEXT_SQL,
      modeColumn: 'embedding_mode',
      profileColumn: 'embedding_profile',
    },

    async transaction(sql, fn) {
      return (await sql.begin((tx) => fn(tx as unknown as Sql))) as Awaited<ReturnType<typeof fn>>;
    },

    async selectStale(sql, surface, { limit, excludeKeys }): Promise<StaleParent[]> {
      const p = new Params();
      const text = `(${surface.textSql})`;
      const header = surface.headerSql ? `(${surface.headerSql})` : 'NULL::text';
      const version = surface.versionSql ? `(${surface.versionSql})` : null;
      const keyArr = keyTextArray(surface);
      const fresh = version
        ? `c.updated_at >= coalesce(${version}, '-infinity'::timestamptz)`
        : `c.parent_sha = ${parentShaSql(text, header)}`;
      const query = `
        SELECT ${keyArr} AS parent_key, ${text} AS text, ${header} AS header,
               (${version ?? 'NULL::timestamptz'})::text AS version
          FROM ${surface.parent.table} p
         WHERE length(${text}) > ${p.add(surface.minChars)}
           ${surface.eligibleSql ? `AND (${surface.eligibleSql})` : ''}
           ${excludeKeys.length > 0 ? `AND array_to_string(${keyArr}, chr(31)) <> ALL(${p.add([...excludeKeys])}::text[])` : ''}
           AND NOT EXISTS (
             SELECT 1 FROM ${table} c
              WHERE c.surface = ${p.add(surface.surface)}
                AND c.parent_key = ${keyArr}
                AND c.chunk_idx = 0
                AND c.splitter_version = ${p.add(surface.splitterVersion)}
                AND ${fresh})
         ${version ? `ORDER BY ${version} DESC NULLS LAST` : ''}
         LIMIT ${p.add(limit)}`;
      const rows = await sql.unsafe<{ parent_key: string[]; text: string; header: string | null; version: string | null }[]>(
        query,
        p.values as never[],
      );
      return rows.map((r) => ({ key: r.parent_key, text: r.text, header: r.header, version: r.version }));
    },

    async readExisting(sql, surface, key): Promise<ExistingChunk[]> {
      const rows = await sql.unsafe<
        {
          chunk_idx: number;
          chunk_sha: string;
          parent_sha: string;
          splitter_version: string;
          embedding: string | null;
          embedding_mode: string | null;
          embedding_profile: string | null;
        }[]
      >(
        `SELECT chunk_idx, chunk_sha, parent_sha, splitter_version, embedding::text AS embedding,
                embedding_mode, embedding_profile
           FROM ${table}
          WHERE surface = $1 AND parent_key = $2::text[]
          ORDER BY chunk_idx
          FOR UPDATE`,
        [surface.surface, [...key]] as never[],
      );
      return rows.map((r) => ({
        chunkIdx: r.chunk_idx,
        chunkSha: r.chunk_sha,
        parentSha: r.parent_sha,
        splitterVersion: r.splitter_version,
        embedding: r.embedding,
        embeddingMode: r.embedding_mode,
        embeddingProfile: r.embedding_profile,
      }));
    },

    async replace(sql, surface, key, rows, meta) {
      await sql.unsafe(`DELETE FROM ${table} WHERE surface = $1 AND parent_key = $2::text[]`, [
        surface.surface,
        [...key],
      ] as never[]);
      if (rows.length === 0) return;
      // Parallel text arrays through unnest: postgres-js serializes a JS array
      // as a PG array for a ::text[] parameter, whereas a pre-stringified JSON
      // parameter it infers as jsonb gets double-encoded.
      await sql.unsafe(
        `INSERT INTO ${table}
           (surface, parent_key, chunk_idx, anchor, header, content, parent_sha, chunk_sha,
            splitter_version, embedding, embedding_mode, embedding_profile, updated_at)
         SELECT $1, $2::text[], u.idx::int, u.anchor, u.header, u.content, $3, u.chunk_sha,
                $4, u.embedding::vector, u.embedding_mode, u.embedding_profile, coalesce($5::text::timestamptz, now())
           FROM unnest($6::text[], $7::text[], $8::text[], $9::text[], $10::text[], $11::text[], $12::text[], $13::text[])
                AS u(idx, anchor, header, content, chunk_sha, embedding, embedding_mode, embedding_profile)`,
        [
          surface.surface,
          [...key],
          meta.parentSha,
          meta.splitterVersion,
          meta.version,
          rows.map((r) => String(r.chunkIdx)),
          rows.map((r) => r.anchor),
          rows.map((r) => r.header),
          rows.map((r) => r.content),
          rows.map((r) => r.chunkSha),
          rows.map((r) => r.embedding),
          rows.map((r) => r.embeddingMode),
          rows.map((r) => r.embeddingProfile),
        ] as never[],
      );
    },

    async touch(sql, surface, key, version) {
      await sql.unsafe(
        `UPDATE ${table} SET updated_at = coalesce($3::text::timestamptz, now())
          WHERE surface = $1 AND parent_key = $2::text[]`,
        [surface.surface, [...key], version] as never[],
      );
    },

    async prune(sql, surface, { limit }) {
      const p = new Params();
      const sSurface = p.add(surface.surface);
      const text = `(${surface.textSql})`;
      const version = surface.versionSql ? `(${surface.versionSql})` : null;
      const minChars = p.add(surface.minChars);
      // A parent keeps its chunks while it exists, is eligible, and is still
      // longer than the cut. With a version, a parent unchanged since its
      // chunks were written was long enough then, so its text is not re-read.
      const stillLong = version
        ? `(coalesce(${version}, '-infinity'::timestamptz) <= c.updated_at OR length(${text}) > ${minChars})`
        : `length(${text}) > ${minChars}`;
      const rows = await sql.unsafe<{ parents: number; chunks: number }[]>(
        `WITH doomed AS (
           SELECT c.parent_key
             FROM ${table} c
            WHERE c.surface = ${sSurface}
              AND c.chunk_idx = 0
              AND NOT EXISTS (
                SELECT 1 FROM ${surface.parent.table} p
                 WHERE ${keyMatchesChunk(surface, 'c')}
                   ${surface.eligibleSql ? `AND (${surface.eligibleSql})` : ''}
                   AND ${stillLong})
            LIMIT ${p.add(limit)}
         ), gone AS (
           DELETE FROM ${table} t USING doomed d
            WHERE t.surface = ${sSurface} AND t.parent_key = d.parent_key
           RETURNING 1
         )
         SELECT (SELECT count(*) FROM doomed)::int AS parents, (SELECT count(*) FROM gone)::int AS chunks`,
        p.values as never[],
      );
      return { parents: rows[0]?.parents ?? 0, chunks: rows[0]?.chunks ?? 0 };
    },
  };
}
