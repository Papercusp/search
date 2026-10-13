/** BM25 D-057: halfvec candidate feasibility, not production recall/latency acceptance. */
import postgres from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';
import { withIterativeScan } from '../hnsw-iterative-scan';
import { chunkAwareVectorLegSql } from './vector-leg';
import { sharedChunkStore } from './shared-store';
import type { ChunkSurface } from './types';

const SCHEMA = `vleg_half_${process.pid}_${Date.now()}`;
const DIMS = 768;
const COUNT = 256;
const QUERY = `[1,${Array(DIMS - 1).fill(0).join(',')}]`;
const MARGIN = 0.02;
function vector(seed: number): string {
  let state = seed >>> 0;
  return `[${Array.from({ length: DIMS }, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32 - 0.5;
  }).join(',')}]`;
}
const near = (y: number) => `[1,${y},${Array(DIMS - 2).fill(0).join(',')}]`;
type Row = { id: number; distance: number; matched_anchor: string | null };

describe('halfvec candidates with original-vector reranking', () => {
  let sql: postgres.Sql;
  const surface: ChunkSurface = {
    surface: 'items', parent: { table: `${SCHEMA}.items`, key: [{ column: 'id', type: 'int' }] },
    textSql: 'p.body', splitter: { kind: 'window', size: 1500, overlap: 250 }, maxChunks: 4,
    chunkMargin: MARGIN,
    parentVector: { column: 'embedding', profileColumn: 'embedding_profile', modeColumn: 'embedding_mode' },
    store: sharedChunkStore({ table: `${SCHEMA}.chunks` }),
  };
  const leg = (tx: postgres.Sql, half: boolean, mode: 'gist' | 'retrieve' = 'retrieve', typed = false) =>
    chunkAwareVectorLegSql(tx as never, {
      surface, qVec: QUERY, limit: 8, mode, scan: 'ann',
      parentFilter: tx`p.eligible` as never,
      spaceFilter: cols => tx`${tx(cols.profileColumn!)} = 'profile_ok'
        AND ${tx(cols.modeColumn!)} = 'gemma'` as never,
      ...(typed ? { chunks: { table: `${SCHEMA}.typed_chunks`, keying: 'typed' as const, anchorColumn: 'anchor' } } : {}),
      ...(half ? { halfvecCandidates: { dimensions: DIMS, limit: COUNT * 2 } } : {}),
    });
  const run = (half: boolean, mode: 'gist' | 'retrieve' = 'retrieve') =>
    withIterativeScan(sql as never, tx => leg(tx as never, half, mode) as never,
      { efSearch: 1000 }) as unknown as Promise<Row[]>;

  beforeAll(async () => {
    sql = postgres(inject('searchPgUrl'), { max: 4, onnotice: () => {} });
    await sql.unsafe(`CREATE SCHEMA ${SCHEMA};
      CREATE TABLE ${SCHEMA}.items (
        id int PRIMARY KEY, body text, eligible boolean, embedding vector(${DIMS}),
        embedding_profile text, embedding_mode text);
      CREATE TABLE ${SCHEMA}.chunks (
        surface text, parent_key text[], chunk_idx int, anchor text, embedding vector(${DIMS}),
        embedding_profile text, embedding_mode text, PRIMARY KEY(surface, parent_key, chunk_idx));`);
    const parents = Array.from({ length: COUNT }, (_, i) => ({
      id: i + 1, body: `item ${i + 1}`, eligible: i < COUNT - 16,
      embedding: i === 0 ? near(.25002) : i === 1 ? near(.25) : i >= COUNT - 32 ? QUERY : vector(i + 1),
      embedding_profile: i >= COUNT - 32 && i < COUNT - 16 ? 'wrong' : 'profile_ok', embedding_mode: 'gemma',
    }));
    await sql`INSERT INTO ${sql(`${SCHEMA}.items`)} ${sql(parents)}`;
    const chunks = parents.flatMap(p => [0, 1].map(i => ({
      surface: 'items', parent_key: [String(p.id)], chunk_idx: i, anchor: `chunk-${p.id}-${i}`,
      embedding: p.id === 3 && i === 0 ? near(.1) : vector(1000 + p.id * 2 + i),
      embedding_profile: p.embedding_profile, embedding_mode: 'gemma',
    })));
    // Closer decoys exercise surface, parent and chunk profile exclusion.
    chunks.push({ surface: 'other', parent_key: ['4'], chunk_idx: 0, anchor: 'other',
      embedding: QUERY, embedding_profile: 'profile_ok', embedding_mode: 'gemma' });
    chunks.push({ surface: 'items', parent_key: ['4'], chunk_idx: 2, anchor: 'wrong-profile',
      embedding: QUERY, embedding_profile: 'wrong', embedding_mode: 'gemma' });
    await sql`INSERT INTO ${sql(`${SCHEMA}.chunks`)} ${sql(chunks)}`;
    await sql.unsafe(`
      CREATE INDEX parents_vector ON ${SCHEMA}.items USING hnsw (embedding vector_cosine_ops);
      CREATE INDEX parents_half ON ${SCHEMA}.items USING hnsw ((embedding::halfvec(${DIMS})) halfvec_cosine_ops);
      CREATE INDEX chunks_vector ON ${SCHEMA}.chunks USING hnsw (embedding vector_cosine_ops) WHERE surface='items';
      CREATE INDEX chunks_half ON ${SCHEMA}.chunks USING hnsw ((embedding::halfvec(${DIMS})) halfvec_cosine_ops) WHERE surface='items';
      CREATE TABLE ${SCHEMA}.typed_chunks AS
        SELECT parent_key[1]::int AS id, chunk_idx, anchor, embedding, embedding_profile, embedding_mode
          FROM ${SCHEMA}.chunks WHERE surface='items';
      CREATE INDEX typed_chunks_half ON ${SCHEMA}.typed_chunks
        USING hnsw ((embedding::halfvec(${DIMS})) halfvec_cosine_ops);
      ANALYZE ${SCHEMA}.items; ANALYZE ${SCHEMA}.chunks;`);
  });
  afterAll(async () => {
    if (!sql) return;
    try { await sql.unsafe(`DROP SCHEMA IF EXISTS ${SCHEMA} CASCADE`); }
    finally { await sql.end({ timeout: 5 }); }
  });

  it('reranks quantized ties using original distances and retains parent/chunk pooling and exclusions', async () => {
    const halfType = sql.unsafe(`halfvec(${DIMS})`);
    const calibration = await sql`
      SELECT (SELECT embedding::${halfType} <=> ${QUERY}::${halfType} FROM ${sql(`${SCHEMA}.items`)} WHERE id=1)
           = (SELECT embedding::${halfType} <=> ${QUERY}::${halfType} FROM ${sql(`${SCHEMA}.items`)} WHERE id=2) AS tied`;
    expect(calibration[0]!.tied).toBe(true);
    for (const mode of ['gist', 'retrieve'] as const) {
      const half = await run(true, mode);
      const reference = await sql<Row[]>`
        SELECT id, distance, matched_anchor FROM (
          SELECT DISTINCT ON (id) id, distance, matched_anchor FROM (
            SELECT p.id, p.embedding <=> ${QUERY}::vector AS distance, NULL::text AS matched_anchor, 0 AS leg
              FROM ${sql(`${SCHEMA}.items`)} p
             WHERE p.eligible AND p.embedding_profile='profile_ok' AND p.embedding_mode='gemma'
            UNION ALL
            SELECT p.id, (c.embedding <=> ${QUERY}::vector) + ${MARGIN}::float8, c.anchor, 1
              FROM ${sql(`${SCHEMA}.chunks`)} c JOIN ${sql(`${SCHEMA}.items`)} p ON c.parent_key=ARRAY[p.id::text]
             WHERE ${mode === 'retrieve'} AND p.eligible AND c.surface='items'
               AND c.embedding_profile='profile_ok' AND c.embedding_mode='gemma'
          ) hits ORDER BY id, distance, leg
        ) pooled ORDER BY distance, id LIMIT 8`;
      expect(half.map(r => r.id)).toEqual(reference.map(r => r.id));
      half.forEach((r, i) => {
        expect(Number(r.distance)).toBeCloseTo(Number(reference[i]!.distance), 10);
        expect(r.matched_anchor).toBe(reference[i]!.matched_anchor);
      });
      expect(half.findIndex(r => r.id === 2)).toBeLessThan(half.findIndex(r => r.id === 1));
      if (mode === 'retrieve') expect(half[0]!.matched_anchor).toBe('chunk-3-0');
      expect(half.map(r => r.id)).toEqual((await run(false, mode)).map(r => r.id));
    }
  });

  it('uses both expression indexes under a generic plan and stores fewer index pages', async () => {
    const plan = await sql.begin(async tx => {
      await tx.unsafe('SET LOCAL plan_cache_mode=force_generic_plan');
      await tx.unsafe('SET LOCAL enable_seqscan=off');
      await tx.unsafe('SET LOCAL enable_sort=off');
      const rows = await tx`EXPLAIN ${leg(tx as never, true) as never}`;
      return rows.map(r => String(r['QUERY PLAN'])).join('\n');
    });
    expect(plan).toContain('Index Scan using parents_half');
    expect(plan).toContain('Index Scan using chunks_half');
    for (const name of ['parents', 'chunks']) {
      const [size] = await sql`
        SELECT pg_relation_size(${`${SCHEMA}.${name}_half`}::regclass)::int AS half,
               pg_relation_size(${`${SCHEMA}.${name}_vector`}::regclass)::int AS original`;
      expect(Number(size!.half)).toBeLessThan(Number(size!.original));
      console.info('D057 isolated fixture index bytes', { name, ...size });
    }
  });

  it('preserves the same pooled rows and scores with typed chunk keys', async () => {
    const typed = await withIterativeScan(sql as never,
      tx => leg(tx as never, true, 'retrieve', true) as never, { efSearch: 1000 }) as unknown as Row[];
    expect(typed).toEqual(await run(true));
    const plan = await sql.begin(async tx => {
      await tx.unsafe('SET LOCAL plan_cache_mode=force_generic_plan');
      await tx.unsafe('SET LOCAL enable_seqscan=off');
      await tx.unsafe('SET LOCAL enable_sort=off');
      const rows = await tx`EXPLAIN ${leg(tx as never, true, 'retrieve', true) as never}`;
      return rows.map(r => String(r['QUERY PLAN'])).join('\n');
    });
    expect(plan).toContain('Index Scan using typed_chunks_half');
  });
});
