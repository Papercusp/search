/**
 * The backfill sweep against real PostgreSQL + pgvector
 * (shared-vector-search-libraries-2026-09-29 P-006, parity contract R-9 of D-004).
 *
 * This replays the scenario of papercusp's pre-move golden
 * (packages/operator-core/lib/search/embed-backfill-parity.integration.test.ts)
 * through the library sweeper alone: the same three tables, text recipes, key
 * columns and drain order, batch size 2, and an embedder that rejects one row.
 * The expected embed order, per-target stats and resulting rows are that
 * golden's, so the library reproduces the pre-move engine on the real SQL: the
 * SELECT (stale AND eligible, ORDER BY, LIMIT, OFFSET), the guarded UPDATE, the
 * round-robin, the failed-row retry and the drain rules.
 *
 * A second case covers the exact-profile and text-recipe labels, which the
 * golden's tables do not carry.
 */
import postgres from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';
import type { BackfillTarget } from './index';
import * as realSubject from './index';

// Copy-out mutation probes point this at a mirror of this directory holding one
// mutated file, so falsifiability is proven without dirtying the shared tree.
const subjectPath = process.env.PAPERCUSP_SEARCH_BACKFILL_SUBJECT;
const { createBackfillSweeper }: typeof realSubject = subjectPath
  ? ((await import(subjectPath)) as typeof realSubject)
  : realSubject;

const S = `bf_it_${process.pid}_${Date.now()}`;
const DIMS = 3;

function vectorFor(text: string): number[] {
  let seed = 7;
  for (const ch of text) seed = (seed * 31 + ch.charCodeAt(0)) % 100_003;
  return Array.from({ length: DIMS }, (_, i) => ((seed * (i + 1)) % 997) / 997);
}
const OLD = `[${new Array(DIMS).fill(0.5).join(',')}]`;

// The golden's three targets, in their TARGETS order, plus an absent one ahead of
// them (the golden's other targets are absent tables too).
const TARGETS: BackfillTarget[] = [
  { table: `${S}.absent`, embedCol: 'embedding', bodySql: 'body', keyCols: ['id'] },
  { table: `${S}.brainstorm`, embedCol: 'content_embedding', bodySql: 'left(content, 2000)', keyCols: ['harness_slug', 'phase'] },
  { table: `${S}.turns`, embedCol: 'text_embedding', bodySql: 'left(text, 2000)', keyCols: ['id'] },
  {
    table: `${S}.recipes`,
    embedCol: 'embedding',
    bodySql: `COALESCE(title, '') || E'\\n' || left(COALESCE(description, ''), 2000)`,
    keyCols: ['id'],
    orderBySql: 'created_at DESC',
  },
];

let sql: postgres.Sql;

beforeAll(async () => {
  sql = postgres(inject('searchPgUrl'), { max: 1, onnotice: () => {} });
  await sql`CREATE EXTENSION IF NOT EXISTS vector`;
  await sql.unsafe(`
    CREATE SCHEMA ${S};
    CREATE TABLE ${S}.brainstorm (
      harness_slug text NOT NULL, phase text NOT NULL, content text,
      content_embedding vector(${DIMS}), content_embedding_mode text,
      PRIMARY KEY (harness_slug, phase));
    INSERT INTO ${S}.brainstorm VALUES
      ('h', 'b1', 'brainstorm one (missing)', NULL, NULL),
      ('h', 'b2', 'brainstorm two (wrong space)', '${OLD}', 'local'),
      ('h', 'b3', 'brainstorm three (missing)', NULL, NULL),
      ('h', 'b4', 'brainstorm four (already correct)', '${OLD}', 'openai'),
      ('h', 'b5', '', NULL, NULL);
    CREATE TABLE ${S}.turns (id text PRIMARY KEY, text text, text_embedding vector(${DIMS}), text_embedding_mode text);
    INSERT INTO ${S}.turns VALUES
      ('t1', 'turn one (missing)', NULL, NULL),
      ('t2', 'turn two (wrong space)', '${OLD}', 'gemma'),
      ('tx', 'turn REJECTED by the embedder', NULL, NULL),
      ('t3', 'turn three (missing)', NULL, NULL);
    CREATE TABLE ${S}.recipes (
      id text PRIMARY KEY, title text, description text, created_at timestamptz NOT NULL,
      embedding vector(${DIMS}), embedding_mode text);
    INSERT INTO ${S}.recipes VALUES
      ('r1', 'recipe one', 'oldest, missing', '2026-01-01T00:00:00Z', NULL, NULL),
      ('r2', 'recipe two', 'middle, wrong space', '2026-01-02T00:00:00Z', '${OLD}', 'local'),
      ('r3', 'recipe three', 'newest, missing', '2026-01-03T00:00:00Z', NULL, NULL),
      ('r4', 'recipe four', 'already correct', '2026-01-04T00:00:00Z', '${OLD}', 'openai');

    CREATE TABLE ${S}.profiled (
      id text PRIMARY KEY, body text, embedding vector(${DIMS}),
      embedding_mode text, embedding_profile text, embedding_recipe int);
    INSERT INTO ${S}.profiled VALUES
      ('p1', 'current profile, current recipe', '${OLD}', 'm', 'm@v2', 2),
      ('p2', 'legacy mode-only row of the current profile', '${OLD}', 'm', NULL, 2),
      ('p3', 'retired profile of the same mode', '${OLD}', 'm', 'm@v1', 2),
      ('p4', 'current profile, old recipe (NULL reads as 1)', '${OLD}', 'm', 'm@v2', NULL),
      ('p5', 'missing', NULL, NULL, NULL, NULL);
  `);
});

afterAll(async () => {
  await sql?.unsafe(`DROP SCHEMA IF EXISTS ${S} CASCADE`);
  await sql?.end();
});

const logger = { log() {}, warn() {}, error() {} };

describe('backfill sweep on PostgreSQL', () => {
  it('reproduces the pre-move golden: embed order, per-target stats and rows', async () => {
    const attempts: string[] = [];
    const sweeper = createBackfillSweeper({
      getTargets: () => TARGETS,
      getSql: () => sql,
      resolveEmbedder: async () => ({
        mode: 'openai',
        dims: DIMS,
        embed: async (text: string) => {
          attempts.push(text);
          if (text.includes('REJECTED')) throw new Error('openai_embed_400');
          return vectorFor(text);
        },
      }),
      acceptsWidth: (d) => d === DIMS,
      widthSkew: (m) => m.filter((x) => x.dims !== DIMS).map((x) => ({ ...x, liveDims: x.dims, declaredDims: DIMS })),
      batchSize: 2,
      maxInputChars: 8000,
      logger,
    });

    const result = await sweeper.run();
    expect(Array.isArray(result)).toBe(true);

    expect(attempts).toEqual([
      'brainstorm one (missing)',
      'brainstorm two (wrong space)',
      'turn one (missing)',
      'turn two (wrong space)',
      'recipe three\nnewest, missing',
      'recipe two\nmiddle, wrong space',
      'brainstorm three (missing)',
      'turn REJECTED by the embedder',
      'turn three (missing)',
      'recipe one\noldest, missing',
      'turn REJECTED by the embedder',
    ]);

    const stats = (result as Array<{ table: string; scanned: number; embedded: number; errors: number }>).map(
      ({ table, scanned, embedded, errors }) => ({ table, scanned, embedded, errors }),
    );
    expect(stats).toEqual([
      { table: `${S}.absent`, scanned: 0, embedded: 0, errors: 0 },
      { table: `${S}.brainstorm`, scanned: 3, embedded: 3, errors: 0 },
      { table: `${S}.turns`, scanned: 5, embedded: 3, errors: 2 },
      { table: `${S}.recipes`, scanned: 3, embedded: 3, errors: 0 },
    ]);

    const rows = await sql.unsafe<Array<{ k: string; body: string; v: string | null; mode: string | null }>>(`
      SELECT 'b:' || phase AS k, content AS body, content_embedding::text AS v, content_embedding_mode AS mode FROM ${S}.brainstorm
      UNION ALL SELECT 't:' || id, text, text_embedding::text, text_embedding_mode FROM ${S}.turns
      UNION ALL SELECT 'r:' || id, title || E'\\n' || description, embedding::text, embedding_mode FROM ${S}.recipes
      ORDER BY 1`);
    const asVec = (v: string | null) => (v === null ? null : (JSON.parse(v) as number[]));
    const untouched = new Set(['b:b4', 'b:b5', 't:tx', 'r:r4']);
    for (const r of rows) {
      if (untouched.has(r.k)) {
        const before = r.k === 'b:b4' || r.k === 'r:r4' ? JSON.parse(OLD) : null;
        expect({ k: r.k, v: asVec(r.v) }).toEqual({ k: r.k, v: before });
        continue;
      }
      expect(r.mode, r.k).toBe('openai');
      const got = asVec(r.v)!;
      const want = vectorFor(r.body);
      expect(Math.max(...got.map((x, i) => Math.abs(x - want[i]!))), r.k).toBeLessThan(1e-6);
    }
    expect(rows.map((r) => r.k)).toEqual([
      'b:b1', 'b:b2', 'b:b3', 'b:b4', 'b:b5',
      'r:r1', 'r:r2', 'r:r3', 'r:r4',
      't:t1', 't:t2', 't:t3', 't:tx',
    ]);

    // Converged: a second sweep re-attempts only the persistently rejected row.
    attempts.length = 0;
    await sweeper.run();
    expect(attempts).toEqual(['turn REJECTED by the embedder']);
  });

  it('re-embeds exactly the rows in another profile or an older recipe, and writes all three labels', async () => {
    const attempts: string[] = [];
    const target: BackfillTarget = {
      table: `${S}.profiled`,
      embedCol: 'embedding',
      bodySql: 'body',
      keyCols: ['id'],
      orderBySql: 'id',
      recipeVersion: 2,
    };
    const sweeper = createBackfillSweeper({
      getTargets: () => [target],
      getSql: () => sql,
      resolveEmbedder: async () => ({
        mode: 'm',
        dims: DIMS,
        profile: { profileId: 'm@v2' },
        embed: async (text: string) => {
          attempts.push(text);
          return vectorFor(text);
        },
      }),
      // m@v2 is mode m's current profile, so m's mode-only rows count as current.
      resolveProfileSelection: (_mode, profile) => ({ profileId: profile.profileId, legacyMode: 'm' }),
      acceptsWidth: (d) => d === DIMS,
      batchSize: 10,
      maxInputChars: 8000,
      logger,
    });
    await sweeper.run();
    expect(attempts).toEqual([
      'retired profile of the same mode',
      'current profile, old recipe (NULL reads as 1)',
      'missing',
    ]);
    const rows = await sql.unsafe<Array<{ id: string; mode: string | null; profile: string | null; recipe: number | null }>>(
      `SELECT id, embedding_mode AS mode, embedding_profile AS profile, embedding_recipe AS recipe FROM ${S}.profiled ORDER BY id`,
    );
    expect(rows).toEqual([
      { id: 'p1', mode: 'm', profile: 'm@v2', recipe: 2 },
      { id: 'p2', mode: 'm', profile: null, recipe: 2 },
      { id: 'p3', mode: 'm', profile: 'm@v2', recipe: 2 },
      { id: 'p4', mode: 'm', profile: 'm@v2', recipe: 2 },
      { id: 'p5', mode: 'm', profile: 'm@v2', recipe: 2 },
    ]);
  });

  it('skips a row another transaction holds instead of waiting for it, and writes it once released (WI-10006735)', async () => {
    await sql.unsafe(`
      CREATE TABLE ${S}.held (
        id text PRIMARY KEY, body text, embedding vector(${DIMS}), embedding_mode text, created_at timestamptz NOT NULL);
      INSERT INTO ${S}.held VALUES
        ('h1', 'held one', NULL, NULL, '2026-01-01T00:00:00Z'),
        ('h2', 'held two (locked by another transaction)', NULL, NULL, '2026-01-02T00:00:00Z'),
        ('h3', 'held three', NULL, NULL, '2026-01-03T00:00:00Z');`);
    // A write that WAITED for h2 would end in this lock_timeout and count as an error.
    const sweepSql = postgres(inject('searchPgUrl'), { max: 1, onnotice: () => {}, connection: { lock_timeout: 3000 } });
    const holder = postgres(inject('searchPgUrl'), { max: 1, onnotice: () => {} });
    const sweeper = createBackfillSweeper({
      getTargets: () => [{ table: `${S}.held`, embedCol: 'embedding', bodySql: 'body', keyCols: ['id'], orderBySql: 'created_at' }],
      getSql: () => sweepSql,
      resolveEmbedder: async () => ({ mode: 'm', dims: DIMS, embed: async (text: string) => vectorFor(text) }),
      acceptsWidth: (d) => d === DIMS,
      batchSize: 10,
      maxInputChars: 8000,
      logger,
    });
    const pick = (r: unknown) =>
      (r as Array<{ table: string; scanned: number; embedded: number; errors: number; writeSkipped: number }>).map(
        ({ scanned, embedded, errors, writeSkipped }) => ({ scanned, embedded, errors, writeSkipped }),
      );
    const written = async () =>
      (await sql.unsafe<Array<{ id: string }>>(`SELECT id FROM ${S}.held WHERE embedding IS NOT NULL ORDER BY id`)).map((r) => r.id);

    let release!: () => void;
    const released = new Promise<void>((resolve) => (release = resolve));
    let signalLocked!: () => void;
    const locked = new Promise<void>((resolve) => (signalLocked = resolve));
    const tx = holder.begin(async (t) => {
      await t.unsafe(`SELECT 1 FROM ${S}.held WHERE id = 'h2' FOR UPDATE`);
      signalLocked();
      await released;
    });
    try {
      await locked;
      // Round 1 writes h1 and h3 and skips h2; round 2 finds only h2, still held, and drains.
      expect(pick(await sweeper.run())).toEqual([{ scanned: 4, embedded: 2, errors: 0, writeSkipped: 2 }]);
      expect(await written()).toEqual(['h1', 'h3']);
    } finally {
      release();
      await tx;
    }
    // Released: the next sweep writes it.
    expect(pick(await sweeper.run())).toEqual([{ scanned: 1, embedded: 1, errors: 0, writeSkipped: 0 }]);
    expect(await written()).toEqual(['h1', 'h2', 'h3']);
    await holder.end();
    await sweepSql.end();
  });

  it('refuses a column whose live width is not the stored width, measured from the catalog', async () => {
    await sql.unsafe(`CREATE TABLE ${S}.wide (id text PRIMARY KEY, body text, embedding vector(4), embedding_mode text)`);
    await sql.unsafe(`INSERT INTO ${S}.wide VALUES ('w1', 'wide', NULL, NULL)`);
    const errors: string[] = [];
    let embedded = 0;
    const sweeper = createBackfillSweeper({
      getTargets: () => [{ table: `${S}.wide`, embedCol: 'embedding', bodySql: 'body', keyCols: ['id'] }],
      getSql: () => sql,
      resolveEmbedder: async () => ({
        mode: 'm',
        dims: DIMS,
        embed: async () => {
          embedded += 1;
          return [1, 2, 3];
        },
      }),
      acceptsWidth: (d) => d === DIMS,
      widthSkew: (m) => m.filter((x) => x.dims !== DIMS).map((x) => ({ ...x, liveDims: x.dims, declaredDims: DIMS })),
      batchSize: 10,
      maxInputChars: 8000,
      logger: { ...logger, error: (m: string) => errors.push(m) },
    });
    await sweeper.run();
    expect(embedded).toBe(0);
    expect(errors[0]).toContain(`refusing ${S}.wide.embedding: it is vector(4) but this code emits 3`);
  });
});
