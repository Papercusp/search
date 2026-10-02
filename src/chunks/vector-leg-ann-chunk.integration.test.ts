/**
 * chunkAwareVectorLegSql `scan: 'exact', chunkScan: 'ann'` against real
 * Postgres + pgvector (generic-rag-chunking-2026-09-29 decision D-046).
 *
 * The mixed form keeps the parent leg exact over the filtered slice and lets a
 * per-surface PARTIAL HNSW index serve the chunk leg. This file checks:
 *   1. It returns the same pooled parents as the exact leg and as a brute-force
 *      ranking computed here, with ef_search high enough that the HNSW walk on
 *      this small fixture is exhaustive (so a mismatch is a SQL defect — wrong
 *      membership, key mapping or margin — never ANN approximation). Recall at
 *      the production ef_search on the real corpus is D-046's measurement.
 *   2. The partial index `WHERE surface = 'items'` is usable by the chunk leg:
 *      with sequential scans and sorts priced out, the plan walks it. The
 *      fixture drops the whole-table HNSW index, so only the partial one can
 *      serve an ordered read, and it carries another surface's chunks too, so
 *      the partial predicate is doing work.
 *   3. withIterativeScan's efSearch reaches the session as hnsw.ef_search.
 */
import { readFileSync } from 'node:fs';
import postgres from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';
import { withIterativeScan } from '../hnsw-iterative-scan';
import { chunkAwareVectorLegSql, sharedChunkStore, type ChunkLegScan, type ChunkSurface } from './index';

const SCHEMA = `vleg_annchunk_${process.pid}_${Date.now()}`;
const REFERENCE_SQL = readFileSync(new URL('../../sql/text-chunks.reference.sql', import.meta.url), 'utf8');
const DIMS = 768;
const MARGIN = 0.02;
const PARENTS = 240;
const RARE_EVERY = 12; // 20 of 240 parents are 'rare' (the selective slice)
const OTHER_SURFACE_CHUNKS = 200;
/** Above the fixture's chunk count, so the HNSW walk visits every node. */
const EXHAUSTIVE_EF = 1000;

/** Deterministic PRNG (mulberry32) so every run ranks the same fixture. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function unit(seed: number): number[] {
  const r = rng(seed);
  const v = Array.from({ length: DIMS }, () => r() * 2 - 1);
  const n = Math.hypot(...v);
  return v.map((x) => x / n);
}

const vecText = (v: readonly number[]) => `[${v.join(',')}]`;
const cosDist = (a: readonly number[], b: readonly number[]) => 1 - a.reduce((s, x, i) => s + x * b[i]!, 0);

const isRare = (id: number) => id % RARE_EVERY === 0;
const parentVec = (id: number) => unit(1000 + id);
const chunkVecs = (id: number) => [unit(5000 + id * 3), unit(5001 + id * 3)];
const QUERY = unit(42);

const ITEMS: ChunkSurface = {
  surface: 'items',
  parent: { table: `${SCHEMA}.items`, key: [{ column: 'id', type: 'int' }] },
  textSql: 'p.body',
  splitter: { kind: 'window', size: 1500, overlap: 250 },
  maxChunks: 4,
  chunkMargin: MARGIN,
  parentVector: { column: 'embedding' },
  store: sharedChunkStore({ table: `${SCHEMA}.text_chunks` }),
};

/** Brute force over the rare slice: each parent at min(parent, chunk + margin). */
function expectedTop(limit: number): Array<{ id: number; distance: number }> {
  const out: Array<{ id: number; distance: number }> = [];
  for (let id = 1; id <= PARENTS; id++) {
    if (!isRare(id)) continue;
    let d = cosDist(parentVec(id), QUERY);
    for (const c of chunkVecs(id)) d = Math.min(d, cosDist(c, QUERY) + MARGIN);
    out.push({ id, distance: d });
  }
  return out.sort((a, b) => a.distance - b.distance || a.id - b.id).slice(0, limit);
}

type Row = { id: number; distance: number; matched_anchor: string | null };

describe("chunkAwareVectorLegSql chunkScan:'ann' over an exact slice (D-046)", () => {
  let sql: postgres.Sql;

  const legSql = (tx: postgres.Sql, chunkScan: ChunkLegScan | undefined, limit: number) =>
    chunkAwareVectorLegSql(tx as never, {
      surface: ITEMS,
      qVec: vecText(QUERY),
      limit,
      mode: 'retrieve',
      scan: 'exact',
      chunkScan,
      parentFilter: tx`p.kind = 'rare'` as never,
    });

  const run = (chunkScan: ChunkLegScan | undefined, limit: number) =>
    withIterativeScan(sql as never, (tx) => legSql(tx as never, chunkScan, limit) as never, {
      efSearch: EXHAUSTIVE_EF,
    }) as unknown as Promise<Row[]>;

  beforeAll(async () => {
    sql = postgres(inject('searchPgUrl'), { max: 4, onnotice: () => {} });
    await sql.unsafe('CREATE EXTENSION IF NOT EXISTS vector');
    await sql.unsafe(`CREATE SCHEMA ${SCHEMA}`);
    await sql.begin(async (tx) => {
      await tx.unsafe(`SET LOCAL search_path TO ${SCHEMA}, public`);
      await tx.unsafe(REFERENCE_SQL);
    });
    // Only the partial index may serve an ordered chunk read (point 2 above).
    await sql.unsafe(`DROP INDEX ${SCHEMA}.text_chunks_embedding_hnsw_idx`);
    await sql.unsafe(`CREATE TABLE ${SCHEMA}.items (
      id int PRIMARY KEY, kind text NOT NULL, body text NOT NULL, embedding vector(${DIMS}))`);
    // One statement per table (unnest over parallel arrays), so the fixture costs
    // three round trips rather than ~900.
    const ids = Array.from({ length: PARENTS }, (_, i) => i + 1);
    await sql`INSERT INTO ${sql(`${SCHEMA}.items`)} (id, kind, body, embedding)
      SELECT id, kind, body, emb::vector
        FROM unnest(${ids}::int[], ${ids.map((id) => (isRare(id) ? 'rare' : 'common'))}::text[],
                    ${ids.map((id) => `item ${id}`)}::text[], ${ids.map((id) => vecText(parentVec(id)))}::text[])
          AS t(id, kind, body, emb)`;
    const chunkRows: Array<{ surface: string; key: string; idx: number; anchor: string | null; content: string; sha: string; emb: string }> = [];
    for (const id of ids) {
      chunkVecs(id).forEach((v, i) =>
        chunkRows.push({ surface: 'items', key: String(id), idx: i, anchor: `s${i}`, content: `chunk ${i} of ${id}`, sha: `c${id}-${i}`, emb: vecText(v) }),
      );
    }
    // Another surface's chunks share the table, keyed like rare items, and sit
    // nearer the query than any item chunk: a leg that read past its surface,
    // or matched membership by key alone, would return them.
    for (let i = 0; i < OTHER_SURFACE_CHUNKS; i++) {
      const near = QUERY.map((x, d) => x + 0.05 * (unit(9000 + i)[d] ?? 0));
      chunkRows.push({
        surface: 'other',
        key: String(RARE_EVERY * ((i % 20) + 1)),
        idx: Math.floor(i / 20),
        anchor: null,
        content: 'other',
        sha: `o${i}`,
        emb: vecText(near),
      });
    }
    await sql`INSERT INTO ${sql(`${SCHEMA}.text_chunks`)}
        (surface, parent_key, chunk_idx, anchor, content, parent_sha, chunk_sha, splitter_version, embedding)
      SELECT surface, ARRAY[k], idx, anchor, content, 'p', sha, 'v1', emb::vector
        FROM unnest(${chunkRows.map((r) => r.surface)}::text[], ${chunkRows.map((r) => r.key)}::text[],
                    ${chunkRows.map((r) => r.idx)}::int[], ${chunkRows.map((r) => r.anchor) as string[]}::text[],
                    ${chunkRows.map((r) => r.content)}::text[], ${chunkRows.map((r) => r.sha)}::text[],
                    ${chunkRows.map((r) => r.emb)}::text[])
          AS t(surface, k, idx, anchor, content, sha, emb)`;
    expect(Number((await sql`SELECT count(*) AS n FROM ${sql(`${SCHEMA}.text_chunks`)}`)[0]!.n)).toBe(
      PARENTS * 2 + OTHER_SURFACE_CHUNKS,
    );
    await sql.unsafe(`CREATE INDEX text_chunks_items_embedding_hnsw_idx ON ${SCHEMA}.text_chunks
      USING hnsw (embedding vector_cosine_ops) WHERE surface = 'items'`);
    await sql.unsafe(`ANALYZE ${SCHEMA}.items`);
    await sql.unsafe(`ANALYZE ${SCHEMA}.text_chunks`);
  });

  afterAll(async () => {
    if (!sql) return;
    await sql.unsafe(`DROP SCHEMA IF EXISTS ${SCHEMA} CASCADE`).catch(() => undefined);
    await sql.end({ timeout: 5 });
  });

  it('calibration: chunk wins exist in the slice, and the other surface is nearer than any item chunk', () => {
    const top = expectedTop(8);
    const byChunk = top.filter((r) => r.distance < cosDist(parentVec(r.id), QUERY) - 1e-12);
    expect(byChunk.length).toBeGreaterThan(0);
    expect(byChunk.length).toBeLessThan(top.length);
    const nearestItemChunk = Math.min(
      ...Array.from({ length: PARENTS }, (_, i) => Math.min(...chunkVecs(i + 1).map((c) => cosDist(c, QUERY)))),
    );
    const near = QUERY.map((x, d) => x + 0.05 * (unit(9000)[d] ?? 0));
    expect(cosDist(near, QUERY)).toBeLessThan(nearestItemChunk);
  });

  it('returns the brute-force nearest parents, the same as the exact chunk leg', async () => {
    for (const limit of [1, 5, 8]) {
      const ann = await run('ann', limit);
      const exact = await run(undefined, limit);
      const want = expectedTop(limit);
      expect(ann.map((r) => r.id)).toEqual(want.map((r) => r.id));
      expect(ann.map((r) => r.id)).toEqual(exact.map((r) => r.id));
      ann.forEach((r, i) => expect(Number(r.distance)).toBeCloseTo(want[i]!.distance, 5));
      ann.forEach((r, i) => expect(r.matched_anchor).toBe(exact[i]!.matched_anchor));
      expect(ann.every((r) => isRare(r.id))).toBe(true);
    }
  });

  it('the chunk leg can be served by the partial HNSW index', async () => {
    const plan = await sql.begin(async (tx) => {
      await tx.unsafe('SET LOCAL enable_seqscan = off');
      await tx.unsafe('SET LOCAL enable_sort = off');
      const rows = await tx`EXPLAIN ${legSql(tx as never, 'ann', 5) as never}`;
      return rows.map((r) => String((r as Record<string, unknown>)['QUERY PLAN'])).join('\n');
    });
    expect(plan).toMatch(/Index Scan using text_chunks_items_embedding_hnsw_idx on text_chunks/);
  });

  it('control: the exact chunk leg is not served by it under the same settings', async () => {
    const plan = await sql.begin(async (tx) => {
      await tx.unsafe('SET LOCAL enable_seqscan = off');
      await tx.unsafe('SET LOCAL enable_sort = off');
      const rows = await tx`EXPLAIN ${legSql(tx as never, undefined, 5) as never}`;
      return rows.map((r) => String((r as Record<string, unknown>)['QUERY PLAN'])).join('\n');
    });
    expect(plan).not.toMatch(/hnsw/i);
  });

  it('efSearch reaches the transaction as hnsw.ef_search, and only that transaction', async () => {
    // current_setting(..., true) reads NULL rather than erroring on a connection
    // that has not loaded the vector library yet.
    const read = (h: postgres.Sql) => h`SELECT current_setting('hnsw.ef_search', true) AS ef`;
    const inside = await withIterativeScan(
      sql as never,
      async (tx) => (await read(tx as unknown as postgres.Sql)) as unknown as Array<{ ef: string | null }>,
      { efSearch: 137 },
    );
    expect(inside[0]!.ef).toBe('137');
    for (let i = 0; i < 4; i++) {
      const after = (await read(sql)) as unknown as Array<{ ef: string | null }>;
      expect(after[0]!.ef).not.toBe('137');
    }
  });
});
