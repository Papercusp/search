/**
 * The embedding-space filter against a real pgvector table
 * (shared-vector-search-libraries-2026-09-29 P-001, AUTO-BAR-R-2-P-001):
 * a vector query filtered for space A returns every A row the query reaches
 * and no B row, for exact-profile rows and for mode-only (NULL profile) rows.
 */
import postgres from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';
import { createEmbeddingSpace } from './embedding-space';

const SCHEMA = `es_it_${process.pid}_${Date.now()}`;
const T = `${SCHEMA}.rows`;

// label: which space the row is truly in ('A' | 'B' | null for unplaceable).
const ROWS: Array<{ id: number; vec: string; profile: string | null; mode: string | null; label: 'A' | 'B' | null }> = [
  { id: 1, vec: '[1,0,0]', profile: 'a@v1', mode: 'a', label: 'A' },
  { id: 2, vec: '[0.9,0.1,0]', profile: 'a@v1', mode: null, label: 'A' },
  { id: 3, vec: '[0.8,0.2,0.1]', profile: null, mode: 'a', label: 'A' }, // mode-only, current profile
  { id: 4, vec: '[0.7,0.3,0]', profile: null, mode: 'a', label: 'A' },
  { id: 5, vec: '[1,0.01,0]', profile: 'b@v1', mode: 'b', label: 'B' }, // nearest to the query of all
  { id: 6, vec: '[0.95,0.05,0]', profile: 'b@v1', mode: 'a', label: 'B' }, // stale mode tag: profile wins
  { id: 7, vec: '[0.9,0,0.1]', profile: null, mode: 'b', label: 'B' },
  { id: 8, vec: '[0.85,0.1,0]', profile: 'a@v0', mode: 'a', label: null }, // retired profile of A's mode
  { id: 9, vec: '[0.6,0.4,0]', profile: null, mode: 'c', label: null }, // unknown mode
];
const QUERY = '[1,0,0]';

const space = createEmbeddingSpace({
  storage: { acceptedProfileIds: ['a@v1', 'b@v1'], dimensions: 3, distanceMetric: 'cosine', indexOperatorClass: 'vector_cosine_ops' },
  legacyModes: {
    a: { profileId: 'a@v1', dimensions: 3, distanceMetric: 'cosine' },
    b: { profileId: 'b@v1', dimensions: 3, distanceMetric: 'cosine' },
  },
});

let sql: postgres.Sql;

beforeAll(async () => {
  sql = postgres(inject('searchPgUrl'), { max: 1, onnotice: () => {} });
  await sql`CREATE EXTENSION IF NOT EXISTS vector`;
  await sql.unsafe(`CREATE SCHEMA ${SCHEMA}`);
  await sql.unsafe(`CREATE TABLE ${T} (id int PRIMARY KEY, embedding vector(3) NOT NULL, embedding_profile text, embedding_mode text)`);
  for (const r of ROWS) {
    await sql`INSERT INTO ${sql(SCHEMA)}.rows VALUES (${r.id}, ${r.vec}::vector, ${r.profile}, ${r.mode})`;
  }
});

afterAll(async () => {
  await sql?.unsafe(`DROP SCHEMA IF EXISTS ${SCHEMA} CASCADE`);
  await sql?.end();
});

const ids = (label: 'A' | 'B' | null) => ROWS.filter((r) => r.label === label).map((r) => r.id).sort((a, b) => a - b);

async function nearest(filter: postgres.Fragment | null): Promise<number[]> {
  const rows = await sql<Array<{ id: number }>>`
    SELECT id FROM ${sql(SCHEMA)}.rows
     WHERE ${filter ?? sql`TRUE`}
     ORDER BY embedding <=> ${QUERY}::vector
     LIMIT 100`;
  return rows.map((r) => r.id);
}

describe('embedding-space filter on a two-space pgvector table (R-2)', () => {
  it('the unfiltered query reaches every row, so the filter is what excludes', async () => {
    expect((await nearest(null)).sort((a, b) => a - b)).toEqual(ROWS.map((r) => r.id));
  });

  it('space A (current profile of mode a): every A row, no B row, nothing unplaceable', async () => {
    const selection = space.resolveCurrentSelection('a');
    expect(selection).toEqual({ profileId: 'a@v1', legacyMode: 'a' });
    const got = await nearest(space.predicateSql(sql, selection, 'embedding_profile', 'embedding_mode'));
    expect([...got].sort((a, b) => a - b)).toEqual(ids('A'));
    const unfiltered = await nearest(null);
    expect(unfiltered.filter((id) => ids('A').includes(id))).toEqual(got); // same rows, same order
  });

  it('space B through the query-side source filter: every B row, no A row', async () => {
    const got = await nearest(space.sourceFilterSql({ sql, embeddingProfile: { profileId: 'b@v1', legacyMode: 'b' } }, 'embedding_profile', 'embedding_mode'));
    expect([...got].sort((a, b) => a - b)).toEqual(ids('B'));
  });

  it('without a legacy mode only exact-profile rows match; unknown provenance matches nothing', async () => {
    const exact = await nearest(space.predicateSql(sql, space.resolveProfileIdSelection('a@v1', null), 'embedding_profile', 'embedding_mode'));
    expect([...exact].sort((a, b) => a - b)).toEqual([1, 2]);
    expect(await nearest(space.sourceFilterSql({ sql }, 'embedding_profile', 'embedding_mode'))).toEqual([]);
    expect(await nearest(space.sourceFilterSql({ sql, embeddingProfile: { profileId: 'a@v0', legacyMode: 'a' } }, 'embedding_profile', 'embedding_mode'))).toEqual([]);
  });

  it('row↔row comparability never pairs rows from different spaces', async () => {
    const pairs = await sql<Array<{ x: number; y: number }>>`
      SELECT x.id AS x, y.id AS y
        FROM ${sql(SCHEMA)}.rows x JOIN ${sql(SCHEMA)}.rows y ON x.id < y.id
       WHERE ${space.effectiveStoredProfileIdSql(sql, 'x.embedding_profile', 'x.embedding_mode')}
           = ${space.effectiveStoredProfileIdSql(sql, 'y.embedding_profile', 'y.embedding_mode')}`;
    const label = new Map(ROWS.map((r) => [r.id, r.label]));
    expect(pairs.length).toBeGreaterThan(0);
    for (const p of pairs) {
      expect(label.get(p.x), `${p.x}~${p.y}`).not.toBeNull();
      expect(label.get(p.x), `${p.x}~${p.y}`).toBe(label.get(p.y));
    }
    // Every same-space pair is comparable: A has 4 rows (6 pairs), B has 3 (3 pairs).
    expect(pairs).toHaveLength(6 + 3);
  });
});
