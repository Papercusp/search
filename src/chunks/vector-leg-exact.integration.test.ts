/**
 * chunkAwareVectorLegSql `scan: 'exact'` (generic-rag-chunking-2026-09-29 P-015,
 * decision D-032).
 *
 * Under a selective parent filter an HNSW scan discards most of what it reads
 * before it fills the LIMIT; work_items:search's feature family (about 2% of
 * rows) measured 131 ms that way. 'exact' materialises the filtered parents and
 * ranks that slice and its chunks exhaustively.
 *
 * This file checks three things against real Postgres + pgvector:
 *   1. 'exact' returns the true nearest parents of the filtered slice, pooled
 *      over the parent vector and every chunk (with the chunk margin), exactly as
 *      a brute-force ranking computed here in TypeScript says.
 *   2. No vector index serves either leg in 'exact'. The control: with sequential
 *      scans disabled, the same query in 'ann' DOES use the HNSW indexes, so the
 *      fixture's indexes are usable and the negative is not vacuous.
 *   3. 'gist' composes with 'exact', and an unknown scan value is refused.
 */
import { readFileSync } from 'node:fs';
import postgres from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';
import { withIterativeScan } from '../hnsw-iterative-scan';
import { chunkAwareVectorLegSql, sharedChunkStore, type ChunkLegMode, type ChunkLegScan, type ChunkSurface } from './index';

const SCHEMA = `vleg_exact_${process.pid}_${Date.now()}`;
const REFERENCE_SQL = readFileSync(new URL('../../sql/text-chunks.reference.sql', import.meta.url), 'utf8');
const DIMS = 768;
const MARGIN = 0.02;
const PARENTS = 240;
const RARE_EVERY = 12; // 20 of 240 parents are 'rare' (the selective slice)

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
function expectedTop(limit: number, mode: ChunkLegMode): Array<{ id: number; distance: number }> {
  const out: Array<{ id: number; distance: number }> = [];
  for (let id = 1; id <= PARENTS; id++) {
    if (!isRare(id)) continue;
    let d = cosDist(parentVec(id), QUERY);
    if (mode === 'retrieve') for (const c of chunkVecs(id)) d = Math.min(d, cosDist(c, QUERY) + MARGIN);
    out.push({ id, distance: d });
  }
  return out.sort((a, b) => a.distance - b.distance || a.id - b.id).slice(0, limit);
}

describe("chunkAwareVectorLegSql scan:'exact' (D-032)", () => {
  let sql: postgres.Sql;

  const legSql = (tx: postgres.Sql, mode: ChunkLegMode, scan: ChunkLegScan, limit: number) =>
    chunkAwareVectorLegSql(tx as never, {
      surface: ITEMS,
      qVec: vecText(QUERY),
      limit,
      mode,
      scan,
      parentFilter: tx`p.kind = 'rare'` as never,
    });

  const run = (mode: ChunkLegMode, scan: ChunkLegScan, limit: number) =>
    withIterativeScan(sql as never, (tx: never) => legSql(tx, mode, scan, limit) as never) as unknown as Promise<
      Array<{ id: number; distance: number; matched_anchor: string | null }>
    >;

  /** EXPLAIN text of the leg with sequential scans disabled (the planner's last resort). */
  const explain = (mode: ChunkLegMode, scan: ChunkLegScan) =>
    sql.begin(async (tx) => {
      await tx.unsafe('SET LOCAL enable_seqscan = off');
      const rows = await tx`EXPLAIN ${legSql(tx as never, mode, scan, 5) as never}`;
      return rows.map((r) => String((r as Record<string, unknown>)['QUERY PLAN'])).join('\n');
    }) as Promise<string>;

  beforeAll(async () => {
    sql = postgres(inject('searchPgUrl'), { max: 4, onnotice: () => {} });
    await sql.unsafe('CREATE EXTENSION IF NOT EXISTS vector');
    await sql.unsafe(`CREATE SCHEMA ${SCHEMA}`);
    await sql.begin(async (tx) => {
      await tx.unsafe(`SET LOCAL search_path TO ${SCHEMA}, public`);
      await tx.unsafe(REFERENCE_SQL);
    });
    await sql.unsafe(`CREATE TABLE ${SCHEMA}.items (
      id int PRIMARY KEY, kind text NOT NULL, body text NOT NULL, embedding vector(${DIMS}))`);
    await sql.unsafe(`CREATE INDEX ON ${SCHEMA}.items USING hnsw (embedding vector_cosine_ops)`);
    for (let id = 1; id <= PARENTS; id++) {
      await sql.unsafe(`INSERT INTO ${SCHEMA}.items (id, kind, body, embedding) VALUES ($1, $2, $3, $4::vector)`, [
        id,
        isRare(id) ? 'rare' : 'common',
        `item ${id}`,
        vecText(parentVec(id)),
      ] as never[]);
      const chunks = chunkVecs(id);
      for (let i = 0; i < chunks.length; i++) {
        await sql.unsafe(
          `INSERT INTO ${SCHEMA}.text_chunks (surface, parent_key, chunk_idx, anchor, content, parent_sha, chunk_sha, splitter_version, embedding)
           VALUES ('items', ARRAY[$1::text], $2, $3, $4, 'p', $5, 'v1', $6::vector)`,
          [String(id), i, `s${i}`, `chunk ${i} of ${id}`, `c${id}-${i}`, vecText(chunks[i]!)] as never[],
        );
      }
    }
    await sql.unsafe(`ANALYZE ${SCHEMA}.items`);
    await sql.unsafe(`ANALYZE ${SCHEMA}.text_chunks`);
  });

  afterAll(async () => {
    if (!sql) return;
    await sql.unsafe(`DROP SCHEMA IF EXISTS ${SCHEMA} CASCADE`).catch(() => undefined);
    await sql.end({ timeout: 5 });
  });

  it('calibration: the slice mixes parent-vector and chunk wins, so both legs matter', () => {
    const top = expectedTop(8, 'retrieve');
    const byChunk = top.filter((r) => r.distance < cosDist(parentVec(r.id), QUERY) - 1e-12);
    expect(byChunk.length).toBeGreaterThan(0);
    expect(byChunk.length).toBeLessThan(top.length);
  });

  it("'exact' returns the brute-force nearest parents of the filtered slice", async () => {
    for (const limit of [1, 5, 8]) {
      const got = await run('retrieve', 'exact', limit);
      const want = expectedTop(limit, 'retrieve');
      expect(got.map((r) => r.id)).toEqual(want.map((r) => r.id));
      got.forEach((r, i) => expect(Number(r.distance)).toBeCloseTo(want[i]!.distance, 5));
      expect(got.every((r) => isRare(r.id))).toBe(true);
    }
  });

  it("'gist' composes with 'exact': the parent vector alone, over the slice", async () => {
    const got = await run('gist', 'exact', 5);
    expect(got.map((r) => r.id)).toEqual(expectedTop(5, 'gist').map((r) => r.id));
    expect(got.every((r) => r.matched_anchor === null)).toBe(true);
  });

  it("no vector index serves 'exact'; both are usable, and 'ann' uses them (controls)", async () => {
    // Control 1: the same leg in 'ann' is served by the parent's HNSW index. (At
    // this fixture size the planner reaches the 480 chunks through the parent, so
    // the chunk index is shown usable by control 2 instead.)
    const ann = await explain('retrieve', 'ann');
    expect(ann).toMatch(/Index Scan using \S*items_embedding\S* on items/);
    // Control 2: the chunk table's HNSW index serves an ordered read on its own.
    const chunkIndex = await sql.begin(async (tx) => {
      await tx.unsafe('SET LOCAL enable_seqscan = off');
      const rows = await tx.unsafe(
        `EXPLAIN SELECT parent_key FROM ${SCHEMA}.text_chunks ORDER BY embedding <=> $1::vector LIMIT 5`,
        [vecText(QUERY)] as never[],
      );
      return rows.map((r) => String((r as Record<string, unknown>)['QUERY PLAN'])).join('\n');
    });
    expect(chunkIndex).toMatch(/Index Scan using text_chunks_embedding_hnsw_idx/);

    const exact = await explain('retrieve', 'exact');
    expect(exact).toMatch(/CTE Scan on chunk_leg_slice/);
    expect(exact).not.toMatch(/embedding\S*_idx|hnsw/i);
  });

  it('refuses an unknown scan value', () => {
    expect(() => legSql(sql, 'retrieve', 'approximate' as ChunkLegScan, 5)).toThrow(/unknown scan 'approximate'/);
  });
});
