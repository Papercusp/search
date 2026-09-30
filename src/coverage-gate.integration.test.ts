/**
 * The coverage measure against a real pgvector table, with no host code
 * (shared-vector-search-libraries-2026-09-29 P-002 / P-007).
 *
 * coverage-gate.test.ts proves the verdict logic on hand-written samples. This
 * file proves the other half: that measureSurfaceCoverage counts what the
 * README says it counts on a real table, and that its samples drive the gate to
 * the right verdict end to end.
 *
 * The fixture is built so each count moves if exactly one rule breaks:
 *   - row 7 has a vector in a RETIRED profile  -> counts only when the label is ignored
 *   - row 8 has no vector                      -> never embedded
 *   - row 9 is ineligible (empty body) but has a vector -> never counted at all
 *   - rows 7, 8, 10 are inside the 24 h window, rows 1-6 and 9 are outside it
 */
import postgres from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';
import { createCoverageGate, measureSurfaceCoverage, type CoverageSurfaceSpec } from './coverage-gate';

const SCHEMA = `cg_it_${process.pid}_${Date.now()}`;
const TABLE = `${SCHEMA}.notes`;
const ACTIVE = 'a@v1';

// ageHours: how long ago the row was written.
const ROWS: Array<{ id: number; body: string; vec: string | null; profile: string | null; ageHours: number }> = [
  { id: 1, body: 'one', vec: '[1,0,0]', profile: ACTIVE, ageHours: 48 },
  { id: 2, body: 'two', vec: '[0,1,0]', profile: ACTIVE, ageHours: 48 },
  { id: 3, body: 'three', vec: '[0,0,1]', profile: ACTIVE, ageHours: 48 },
  { id: 4, body: 'four', vec: '[1,1,0]', profile: ACTIVE, ageHours: 48 },
  { id: 5, body: 'five', vec: '[0,1,1]', profile: ACTIVE, ageHours: 48 },
  { id: 6, body: 'six', vec: '[1,0,1]', profile: ACTIVE, ageHours: 48 },
  { id: 7, body: 'seven', vec: '[1,1,1]', profile: 'a@v0', ageHours: 2 }, // retired profile
  { id: 8, body: 'eight', vec: null, profile: null, ageHours: 2 }, // not embedded yet
  { id: 9, body: '', vec: '[0.5,0.5,0]', profile: ACTIVE, ageHours: 2 }, // ineligible
  { id: 10, body: 'ten', vec: '[0.2,0.3,0.4]', profile: ACTIVE, ageHours: 1 },
];

const FULL_SPEC: CoverageSurfaceSpec = {
  table: TABLE,
  vectorColumn: 'embedding',
  labelColumn: 'embedding_profile',
  eligibleSql: "body <> ''",
  recencyColumn: 'created_at',
};

let sql: postgres.Sql;

beforeAll(async () => {
  sql = postgres(inject('searchPgUrl'), { max: 1, onnotice: () => {} });
  await sql`CREATE EXTENSION IF NOT EXISTS vector`;
  await sql.unsafe(`CREATE SCHEMA ${SCHEMA}`);
  await sql.unsafe(
    `CREATE TABLE ${TABLE} (id int PRIMARY KEY, body text NOT NULL, embedding vector(3),
       embedding_profile text, created_at timestamptz NOT NULL)`,
  );
  for (const r of ROWS) {
    await sql`INSERT INTO ${sql(SCHEMA)}.notes
              VALUES (${r.id}, ${r.body}, ${r.vec}::vector, ${r.profile},
                      now() - make_interval(hours => ${r.ageHours}))`;
  }
});

afterAll(async () => {
  await sql?.unsafe(`DROP SCHEMA IF EXISTS ${SCHEMA} CASCADE`);
  await sql?.end();
});

describe('measureSurfaceCoverage on PostgreSQL', () => {
  it('counts eligible rows, rows embedded in the active space, and the recent window', async () => {
    const sample = await measureSurfaceCoverage(sql, FULL_SPEC, ACTIVE);
    expect(sample).toMatchObject({
      surface: `${TABLE}.embedding`,
      eligibleRows: 9, // every row but 9
      embeddedRows: 7, // 1-6 and 10: row 7's vector is in a retired profile, row 8 has none
      recentEligible: 3, // 7, 8, 10
      recentEmbedded: 1, // 10
    });
  });

  it('without a label column every stored vector counts, and no parameter is bound', async () => {
    // No label and no recency column means a statement with zero placeholders.
    // Postgres refuses a Bind whose parameter the statement never references,
    // so this also proves the measure binds only what it uses.
    const sample = await measureSurfaceCoverage(sql, { table: TABLE, vectorColumn: 'embedding', eligibleSql: "body <> ''" }, null);
    expect(sample).toMatchObject({ eligibleRows: 9, embeddedRows: 8, recentEligible: null, recentEmbedded: null });
  });

  it('a label column with no recency column binds only the label', async () => {
    const sample = await measureSurfaceCoverage(
      sql,
      { table: TABLE, vectorColumn: 'embedding', labelColumn: 'embedding_profile', eligibleSql: "body <> ''" },
      ACTIVE,
    );
    expect(sample).toMatchObject({ eligibleRows: 9, embeddedRows: 7, recentEligible: null });
  });

  it('with no eligibility predicate every row is eligible', async () => {
    const sample = await measureSurfaceCoverage(sql, { table: TABLE, vectorColumn: 'embedding' }, null);
    expect(sample).toMatchObject({ eligibleRows: 10, embeddedRows: 9 });
  });
});

describe('measured samples drive the gate end to end', () => {
  const gate = createCoverageGate({ sources: { notes: [`${TABLE}.embedding`], tags: [] } });

  it('a partially embedded table reads degraded, then healthy once the backlog is embedded', async () => {
    const before = await measureSurfaceCoverage(sql, FULL_SPEC, ACTIVE);
    const degraded = gate.assessScope(['notes', 'tags'], gate.snapshot([before]));
    expect(degraded.degraded).toBe(true);
    expect(degraded.degradedSources).toEqual(['notes']);
    const notes = degraded.perSource.find((s) => s.source === 'notes')!;
    expect(notes.verdict).toBe('degraded');
    expect(notes.coverage).toBeCloseTo(7 / 9, 6);
    expect(degraded.perSource.find((s) => s.source === 'tags')!.verdict).toBe('not-semantic');

    await sql.unsafe(
      `UPDATE ${TABLE} SET embedding = '[0.3,0.3,0.3]', embedding_profile = $1 WHERE id IN (7, 8)`,
      [ACTIVE],
    );
    const after = await measureSurfaceCoverage(sql, FULL_SPEC, ACTIVE);
    expect(after).toMatchObject({ eligibleRows: 9, embeddedRows: 9 });
    const healthy = gate.assessScope(['notes'], gate.snapshot([after]));
    expect(healthy.degraded).toBe(false);
    expect(healthy.warning).toBeNull();
    expect(healthy.perSource[0]!.verdict).toBe('healthy');
  });

  it('a real sample read after the age limit is unknown, not healthy', async () => {
    const sample = await measureSurfaceCoverage(sql, FULL_SPEC, ACTIVE);
    const later = new Date(new Date(sample.observedAt).getTime() + gate.thresholds.maxSampleAgeMs + 60_000);
    const report = gate.assessScope(['notes'], gate.snapshot([sample], later));
    expect(report.perSource[0]!.verdict).toBe('unknown');
    expect(report.unknownSources).toEqual(['notes']);
  });
});
