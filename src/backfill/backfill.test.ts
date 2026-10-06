/**
 * The backfill engine's mechanics against an in-memory fake SQL client
 * (shared-vector-search-libraries-2026-09-29 P-006, D-004). The real SQL
 * (predicates, guarded UPDATE, catalog probe) runs against PostgreSQL in
 * backfill.integration.test.ts.
 *
 * The fake enforces PostgreSQL's rule that a statement must reference every
 * parameter it is bound with (and bind every one it references), so a
 * placeholder layout that drifts from its parameter list fails here.
 */
import { describe, expect, it, vi } from 'vitest';
import type { BackfillLogger, BackfillSql, BackfillTarget } from './index';
import * as realSubject from './index';

// Copy-out mutation probes point this at a mirror of this directory holding one
// mutated file, so falsifiability is proven without dirtying the shared tree
// (same seam as backfill.integration.test.ts).
const subjectPath = process.env.PAPERCUSP_SEARCH_BACKFILL_SUBJECT;
const {
  backfillTable,
  createBackfillSweepState,
  createBackfillSweeper,
  eligiblePredicateSql,
  recentPredicateSql,
  settledPredicateSql,
  stalePredicateSql,
  truncateToChars,
}: typeof realSubject = subjectPath
  ? ((await import(subjectPath)) as typeof realSubject)
  : realSubject;

interface FakeRow {
  key: string;
  body: string;
  stale: boolean;
  vec?: string;
  labels?: unknown[];
  /** Another writer holds this row's lock: a SKIP LOCKED write matches nothing. */
  locked?: boolean;
}
interface FakeTable {
  rows: FakeRow[];
  /** Label columns present (the vector column always is, unless `absent`). */
  labels?: Array<'mode' | 'profile' | 'recipe'>;
  dims?: number;
  absent?: boolean;
}

function fakeDb(tables: Record<string, FakeTable>, opts: { vector?: boolean } = {}) {
  const calls: Array<{ query: string; params: unknown[] }> = [];
  const fake = {
    unsafe<T>(query: string, params: unknown[] = []): Promise<T> {
      calls.push({ query, params });
      const refs = [...query.matchAll(/\$(\d+)/g)].map((m) => Number(m[1]));
      const highest = refs.length ? Math.max(...refs) : 0;
      if (highest !== params.length) {
        return Promise.reject(new Error(`bind mismatch: statement references $${highest}, bound ${params.length}`));
      }
      if (query.includes('FROM pg_extension')) {
        return Promise.resolve((opts.vector === false ? [] : [{ extname: 'vector' }]) as T);
      }
      if (query.includes('information_schema.columns')) {
        const t = tables[String(params[0])];
        if (!t || t.absent) return Promise.resolve([] as T);
        const embedCol = String(params[1]);
        const out = [{ column_name: embedCol, dims: t.dims ?? 3 }];
        for (const l of t.labels ?? ['mode']) out.push({ column_name: `${embedCol}_${l}`, dims: null as never });
        return Promise.resolve(out as T);
      }
      const table = /(?:FROM|UPDATE) (\S+)/.exec(query)?.[1] ?? '';
      const t = tables[table];
      if (!t) return Promise.reject(new Error(`no such table ${table}`));
      if (query.trimStart().startsWith('SELECT')) {
        const limit = Number(params[0]);
        const offset = Number(params[params.length - 1]);
        const pending = t.rows.filter((r) => r.stale && r.body.length > 0);
        return Promise.resolve(pending.slice(offset, offset + limit).map((r) => ({ k0: r.key, body: r.body })) as T);
      }
      if (query.trimStart().startsWith('UPDATE')) {
        const row = t.rows.find((r) => r.key === params[params.length - 1]);
        // A held row is skipped only by a write that asks to skip it; any other
        // write would wait for it, which this fake reports as an error.
        if (row?.locked && !query.includes('SKIP LOCKED')) {
          return Promise.reject(new Error('canceling statement due to lock timeout'));
        }
        let count = 0;
        if (row && row.stale && !row.locked) {
          row.stale = false;
          row.vec = String(params[0]);
          row.labels = params.slice(1, -1);
          count = 1;
        }
        // postgres.js reports the affected-row count on the result.
        return Promise.resolve(Object.assign([], { count }) as T);
      }
      return Promise.reject(new Error(`unexpected statement: ${query.slice(0, 40)}`));
    },
  };
  // The engine only calls unsafe(text, params) and awaits the result.
  const sql = fake as unknown as BackfillSql;
  return { sql, calls };
}

const target = (table: string, extra: Partial<BackfillTarget> = {}): BackfillTarget => ({
  table,
  embedCol: 'embedding',
  bodySql: 'body',
  keyCols: ['id'],
  ...extra,
});
const rows = (...bodies: string[]): FakeRow[] => bodies.map((body, i) => ({ key: `r${i}`, body, stale: true }));
const vec3 = async (_text: string) => [0.1, 0.2, 0.3];
const quiet = (): BackfillLogger & { lines: string[] } => {
  const lines: string[] = [];
  return { lines, log: (m) => lines.push(m), warn: (m) => lines.push(m), error: (m) => lines.push(m) };
};
const base = { mode: 'm1', batchSize: 10, acceptsWidth: (d: number) => d === 3, maxInputChars: 1000 };

describe('predicates', () => {
  const t = target('s.t');
  it('stale: missing vector only, when the table has no mode column', () => {
    expect(stalePredicateSql(t, false, 'NULL')).toBe('embedding IS NULL');
  });
  it('stale: missing vector or another mode', () => {
    expect(stalePredicateSql(t, true, '$2')).toBe('(embedding IS NULL OR embedding_mode IS DISTINCT FROM $2)');
  });
  it('stale: exact profile, legacy mode-only rows count only when legacy-compatible', () => {
    expect(stalePredicateSql(t, true, '$2', null, { profileExpr: '$3', legacyModeCompatible: true })).toBe(
      '(embedding IS NULL OR NOT (embedding_profile IS NOT DISTINCT FROM $3 OR ' +
        '(embedding_profile IS NULL AND embedding_mode IS NOT DISTINCT FROM $2)))',
    );
    expect(stalePredicateSql(t, true, '$2', null, { profileExpr: '$3', legacyModeCompatible: false })).toBe(
      '(embedding IS NULL OR embedding_profile IS DISTINCT FROM $3)',
    );
  });
  it('stale: the recipe term reads a NULL recipe as the baseline version', () => {
    expect(stalePredicateSql(t, true, '$2', '$3')).toBe(
      '((embedding IS NULL OR embedding_mode IS DISTINCT FROM $2) OR coalesce(embedding_recipe, 1) IS DISTINCT FROM $3)',
    );
  });
  it('eligible, recent and settled', () => {
    expect(eligiblePredicateSql(t)).toBe('length(body) > 0');
    expect(recentPredicateSql(t, '$1')).toBeNull();
    expect(settledPredicateSql(t, '$1', '$2')).toBeNull();
    const ms = target('s.t', { recencyCol: 'created', recencyColKind: 'epochMs' });
    expect(recentPredicateSql(ms, '$1')).toBe(
      'created >= (extract(epoch from now()) * 1000)::bigint - ($1::bigint * 3600000)',
    );
    const ts = target('s.t', { recencyCol: 'created' });
    expect(settledPredicateSql(ts, '$1', '$2')).toBe(
      '(created <= now() - make_interval(mins => $1::int) AND created >= now() - make_interval(hours => $2::int))',
    );
  });
  it('truncateToChars counts code points, like left()', () => {
    expect(truncateToChars('abc', 5)).toBe('abc');
    expect(truncateToChars('abcdef', 3)).toBe('abc');
    expect(truncateToChars('a😀b😀c', 3)).toBe('a😀b');
  });
});

describe('backfillTable', () => {
  const profile = { profileId: 'p1' };
  const resolveProfileSelection = () => ({ profileId: 'p1', legacyMode: 'm1' });
  const combos = [false, true].flatMap((space) =>
    [false, true].flatMap((prof) => [false, true].map((recipe) => ({ space, prof, recipe }))),
  );
  it('calibration: the fake rejects a bind the statement does not reference, and one it lacks', async () => {
    const { sql } = fakeDb({});
    await expect(sql.unsafe('SELECT $1', [1, 2])).rejects.toThrow('bind mismatch');
    await expect(sql.unsafe('SELECT $1, $2', [1])).rejects.toThrow('bind mismatch');
  });

  it.each(combos)('binds exactly the parameters it references (%o)', async ({ space, prof, recipe }) => {
    const db = fakeDb({ 's.t': { rows: rows('a', 'b') } });
    const s = await backfillTable(db.sql, target('s.t', { recipeVersion: 2 }), {
      ...base,
      embed: vec3,
      spaceAware: space,
      profileColPresent: prof,
      recipeColPresent: recipe,
      profile,
      resolveProfileSelection,
    });
    expect(s).toMatchObject({ scanned: 2, embedded: 2, errors: 0 });
    const update = db.calls.find((c) => c.query.startsWith('UPDATE'))!;
    expect(update.params.slice(1, -1)).toEqual([...(space ? ['m1'] : []), ...(prof ? ['p1'] : []), ...(recipe ? [2] : [])]);
    expect(update.query.includes('embedding_mode = ')).toBe(space);
    expect(update.query.includes('embedding_profile = ')).toBe(prof);
    expect(update.query.includes('embedding_recipe = ')).toBe(recipe);
  });

  it('batches in chunks, keeps what earlier chunks made, and sends a lone row down the per-row path', async () => {
    const db = fakeDb({ 's.t': { rows: rows('a', 'b', 'c', 'd', 'e') } });
    const embedMany = vi.fn(async (texts: string[]) => texts.map(() => [1, 2, 3]));
    const embed = vi.fn(vec3);
    const s = await backfillTable(db.sql, target('s.t'), { ...base, embed, embedMany, batchChunkSize: 2 });
    expect(embedMany.mock.calls.map((c) => c[0])).toEqual([['a', 'b'], ['c', 'd']]);
    expect(embed.mock.calls.map((c) => c[0])).toEqual(['e']);
    expect(s).toMatchObject({ scanned: 5, embedded: 5, errors: 0 });
  });

  it('a batch failed by one bad row still embeds every other row of that batch (R-10)', async () => {
    const table: FakeTable = { rows: rows('a', 'BAD', 'c', 'd') };
    const db = fakeDb({ 's.t': table });
    const embedMany = vi.fn(async (texts: string[]) => {
      if (texts.includes('BAD')) throw new Error('bad input in batch');
      return texts.map(() => [1, 2, 3]);
    });
    const embed = vi.fn(async (text: string) => {
      if (text === 'BAD') throw new Error('bad input');
      return vec3(text);
    });
    const s = await backfillTable(db.sql, target('s.t'), { ...base, embed, embedMany, logger: quiet() });
    expect(embedMany).toHaveBeenCalledTimes(1);
    // The failed batch falls back to per-row: only the bad row fails, once.
    expect(embed.mock.calls.map((c) => c[0])).toEqual(['a', 'BAD', 'c', 'd']);
    expect(s).toMatchObject({ scanned: 4, embedded: 3, errors: 1 });
    expect(table.rows.filter((r) => !r.stale).map((r) => r.key)).toEqual(['r0', 'r2', 'r3']);
  });

  it('a failing chunk stops batching for that pull and warns once', async () => {
    const db = fakeDb({ 's.t': { rows: rows('a', 'b', 'c', 'd') } });
    const log = quiet();
    const embedMany = vi.fn(async () => {
      throw new Error('sidecar down');
    });
    const embed = vi.fn(vec3);
    const s = await backfillTable(db.sql, target('s.t'), {
      ...base,
      embed,
      embedMany,
      batchChunkSize: 2,
      logger: log,
      logLabel: 'bf',
    });
    expect(embedMany).toHaveBeenCalledTimes(1);
    expect(embed.mock.calls.map((c) => c[0])).toEqual(['a', 'b', 'c', 'd']);
    expect(s.embedded).toBe(4);
    expect(log.lines.filter((l) => l.startsWith('[bf] batch embed FAILED for s.t'))).toHaveLength(1);
  });

  it('a short batch result is a contract breach: warn and embed per row', async () => {
    const db = fakeDb({ 's.t': { rows: rows('a', 'b') } });
    const log = quiet();
    const embed = vi.fn(vec3);
    await backfillTable(db.sql, target('s.t'), {
      ...base,
      embed,
      embedMany: async () => [[1, 2, 3]],
      logger: log,
    });
    expect(embed).toHaveBeenCalledTimes(2);
    expect(log.lines[0]).toContain('batch embed returned 1 vectors for 2 texts (s.t)');
  });

  it('skips failed rows by OFFSET on the next pull and stops when a pull makes no progress', async () => {
    const db = fakeDb({ 's.t': { rows: rows('bad', 'ok1', 'ok2', 'bad2') } });
    const embed = vi.fn(async (t: string) => {
      if (t.startsWith('bad')) throw new Error('rejected');
      return [1, 2, 3];
    });
    const s = await backfillTable(db.sql, target('s.t'), { ...base, batchSize: 2, embed });
    const offsets = db.calls.filter((c) => c.query.trimStart().startsWith('SELECT')).map((c) => c.params.at(-1));
    // pull 1 (offset 0): bad, ok1 · pull 2 (offset 1): ok2, bad2 · pull 3 (offset 2): nothing left
    expect(offsets).toEqual([0, 1, 2]);
    expect(s).toMatchObject({ scanned: 4, embedded: 2, errors: 2 });

    const all = fakeDb({ 's.t': { rows: rows('bad', 'bad2', 'bad3') } });
    const s2 = await backfillTable(all.sql, target('s.t'), { ...base, batchSize: 2, embed });
    expect(all.calls.filter((c) => c.query.trimStart().startsWith('SELECT'))).toHaveLength(1);
    expect(s2).toMatchObject({ scanned: 2, embedded: 0, errors: 2 });
  });

  it('a row another writer holds is skipped without waiting, offset past, and not counted as written or failed', async () => {
    const table = { rows: rows('a', 'held', 'c', 'd') };
    table.rows[1]!.locked = true;
    const db = fakeDb({ 's.t': table });
    const s = await backfillTable(db.sql, target('s.t'), { ...base, batchSize: 2, embed: vec3 });
    // pull 1 (offset 0): a, held · pull 2 (offset 1, past the held row): c, d · pull 3 (offset 1): nothing
    const offsets = db.calls.filter((c) => c.query.trimStart().startsWith('SELECT')).map((c) => c.params.at(-1));
    expect(offsets).toEqual([0, 1, 1]);
    expect(s).toMatchObject({ scanned: 4, embedded: 3, errors: 0, writeSkipped: 1 });
    expect(table.rows.map((r) => r.stale)).toEqual([false, true, false, false]);

    // Every row held: one pull, nothing written, and the call stops instead of spinning.
    const allHeld = { rows: rows('x', 'y') };
    for (const r of allHeld.rows) r.locked = true;
    const db2 = fakeDb({ 's.t': allHeld });
    const s2 = await backfillTable(db2.sql, target('s.t'), { ...base, batchSize: 2, embed: vec3 });
    expect(db2.calls.filter((c) => c.query.trimStart().startsWith('SELECT'))).toHaveLength(1);
    expect(s2).toMatchObject({ scanned: 2, embedded: 0, errors: 0, writeSkipped: 2 });
  });

  it('the write locks its row with SKIP LOCKED and keeps the staleness guard as its last clause', async () => {
    const db = fakeDb({ 's.t': { rows: rows('a') } });
    await backfillTable(db.sql, target('s.t'), { ...base, embed: vec3 });
    const update = db.calls.find((c) => c.query.startsWith('UPDATE'))!.query.replace(/\s+/g, ' ');
    expect(update).toContain('AND (id) IN (SELECT id FROM s.t WHERE id = $3 FOR NO KEY UPDATE SKIP LOCKED)');
    expect(update.endsWith(stalePredicateSql(target('s.t'), true, '$2'))).toBe(true);
  });

  it('a client that reports no affected-row count is taken as having written the row', async () => {
    const calls: string[] = [];
    const sql = {
      unsafe: async (q: string) => {
        calls.push(q);
        return calls.length === 1 ? [{ k0: 'r0', body: 'a' }] : [];
      },
    } as unknown as BackfillSql;
    const s = await backfillTable(sql, target('s.t'), { ...base, embed: vec3 });
    expect(s).toMatchObject({ scanned: 1, embedded: 1, writeSkipped: 0 });
  });

  it('stops at maxRows (default 4 batches)', async () => {
    const db = fakeDb({ 's.t': { rows: rows(...'abcdefghij'.split('')) } });
    expect((await backfillTable(db.sql, target('s.t'), { ...base, batchSize: 2, embed: vec3 })).scanned).toBe(8);
    const db2 = fakeDb({ 's.t': { rows: rows(...'abcdefghij'.split('')) } });
    expect((await backfillTable(db2.sql, target('s.t'), { ...base, batchSize: 2, maxRows: 3, embed: vec3 })).scanned).toBe(3);
  });

  it('refuses a vector of the wrong width without writing it', async () => {
    const db = fakeDb({ 's.t': { rows: rows('a') } });
    const s = await backfillTable(db.sql, target('s.t'), { ...base, embed: async () => [1, 2] });
    expect(s).toMatchObject({ embedded: 0, errors: 1 });
    expect(db.calls.some((c) => c.query.startsWith('UPDATE'))).toBe(false);
  });

  it('truncates each body to maxInputChars before embedding, on both paths', async () => {
    const db = fakeDb({ 's.t': { rows: rows('abcdef', 'ghijkl', 'mnopqr') } });
    const embedMany = vi.fn(async (texts: string[]) => texts.map(() => [1, 2, 3]));
    const embed = vi.fn(vec3);
    await backfillTable(db.sql, target('s.t'), { ...base, embed, embedMany, batchChunkSize: 2, maxInputChars: 3 });
    expect(embedMany.mock.calls[0]![0]).toEqual(['abc', 'ghi']);
    expect(embed.mock.calls[0]![0]).toBe('mno');
  });

  it('a hung per-row embed times out and counts as an error', async () => {
    const db = fakeDb({ 's.t': { rows: rows('a') } });
    const s = await backfillTable(db.sql, target('s.t'), {
      ...base,
      embed: () => new Promise<number[]>(() => {}),
      rowTimeoutMs: 20,
    });
    expect(s).toMatchObject({ scanned: 1, embedded: 0, errors: 1 });
  });

  it('a profile-aware table needs a profile the storage accepts', async () => {
    const db = fakeDb({ 's.t': { rows: rows('a') } });
    await expect(backfillTable(db.sql, target('s.t'), { ...base, embed: vec3, profileColPresent: true })).rejects.toThrow(
      'requires an exact embedding profile',
    );
    await expect(
      backfillTable(db.sql, target('s.t'), {
        ...base,
        embed: vec3,
        profileColPresent: true,
        profile,
        resolveProfileSelection: () => null,
        storageLabel: 'prose storage',
      }),
    ).rejects.toThrow('profile p1 is incompatible with prose storage; refusing s.t.embedding');
  });
});

describe('createBackfillSweeper', () => {
  const sweeper = (tables: Record<string, FakeTable>, over: Record<string, unknown> = {}, dbOpts = {}) => {
    const db = fakeDb(tables, dbOpts);
    const log = quiet();
    const embed = vi.fn(vec3);
    const sw = createBackfillSweeper({
      getTargets: () => Object.keys(tables).map((t) => target(t)),
      getSql: () => db.sql,
      resolveEmbedder: async () => ({ mode: 'm1', dims: 3, embed }),
      acceptsWidth: (d) => d === 3,
      widthSkew: (m) => m.filter((x) => x.dims !== 3).map((x) => ({ ...x, liveDims: x.dims, declaredDims: 3 })),
      batchSize: 1,
      maxInputChars: 1000,
      logger: log,
      logLabel: 'bf',
      ...over,
    });
    return { sw, db, log, embed };
  };
  const zero = (table: string) => ({ table, scanned: 0, embedded: 0, errors: 0, writeSkipped: 0 });
  const plain = (r: unknown) => (r as Array<Record<string, unknown>>).map(({ durationMs: _d, ...s }) => s);

  it('drains live targets round-robin, one batch each per round, results in canonical order', async () => {
    const { sw, embed } = sweeper({ 's.a': { rows: rows('a1', 'a2') }, 's.b': { rows: rows('b1', 'b2', 'b3') } });
    const r = await sw.run();
    expect(embed.mock.calls.map((c) => c[0])).toEqual(['a1', 'b1', 'a2', 'b2', 'b3']);
    expect(plain(r)).toEqual([
      { table: 's.a', scanned: 2, embedded: 2, errors: 0, writeSkipped: 0 },
      { table: 's.b', scanned: 3, embedded: 3, errors: 0, writeSkipped: 0 },
    ]);
    expect(sw.lastResult()).toBe(r);
  });

  it('sums writeSkipped per target and drains a target whose only remaining row is held', async () => {
    const a = { rows: rows('a1', 'a2') };
    a.rows[0]!.locked = true;
    const { sw, embed } = sweeper({ 's.a': a }, { batchSize: 2 });
    // round 1: a1 held (skipped), a2 written · round 2: a1 still held, nothing written → drained
    expect(plain(await sw.run())).toEqual([{ table: 's.a', scanned: 3, embedded: 1, errors: 0, writeSkipped: 2 }]);
    expect(embed.mock.calls.map((c) => c[0])).toEqual(['a1', 'a2', 'a1']);
  });

  it('continues to the next target after a batch contains a failing row (R-10)', async () => {
    const tables = { 's.a': { rows: rows('a1', 'BAD') }, 's.b': { rows: rows('b1') } };
    const embed = vi.fn(async (text: string) => {
      if (text === 'BAD') throw new Error('bad input');
      return [1, 2, 3];
    });
    const embedMany = vi.fn(async (texts: string[]) => {
      if (texts.includes('BAD')) throw new Error('bad input in batch');
      return texts.map(() => [1, 2, 3]);
    });
    const { sw } = sweeper(tables, {
      batchSize: 2,
      resolveEmbedder: async () => ({ mode: 'm1', dims: 3, embed, embedMany }),
    });

    expect(plain(await sw.run({ maxRowsPerTarget: 2 }))).toEqual([
      { table: 's.a', scanned: 2, embedded: 1, errors: 1, writeSkipped: 0 },
      { table: 's.b', scanned: 1, embedded: 1, errors: 0, writeSkipped: 0 },
    ]);
    expect(embedMany).toHaveBeenCalledWith(['a1', 'BAD']);
    expect(embed.mock.calls.map((call) => call[0])).toEqual(['a1', 'BAD', 'b1']);
    expect(tables['s.a'].rows.map((row) => row.stale)).toEqual([false, true]);
    expect(tables['s.b'].rows[0]?.stale).toBe(false);
  });

  it('rotates which target leads each sweep', async () => {
    const tables = { 's.a': { rows: rows('a1', 'a2') }, 's.b': { rows: rows('b1', 'b2') } };
    const { sw, embed } = sweeper(tables);
    await sw.run({ maxRowsPerTarget: 1 });
    await sw.run({ maxRowsPerTarget: 1 });
    expect(embed.mock.calls.map((c) => c[0])).toEqual(['a1', 'b1', 'b2', 'a2']);
  });

  it('skips while a sweep holds the latch, and takes over a stale one', async () => {
    const state = createBackfillSweepState();
    const clock = { t: 1_000_000 };
    const { sw, log } = sweeper({ 's.a': { rows: rows('a1') } }, { state, now: () => clock.t, latchStaleMs: 1000 });
    state.running = true;
    state.startedAt = clock.t - 500;
    expect(await sw.run()).toEqual({ skipped: 'already_running' });
    expect(log.lines.at(-1)).toBe('[bf] skipped: a sweep has held the latch for 1s (stale at 1s)');
    state.startedAt = clock.t - 5000;
    expect(Array.isArray(await sw.run())).toBe(true);
    expect(log.lines.some((l) => l.startsWith('[bf] STALE LATCH'))).toBe(true);
    expect(state.running).toBe(false);
    expect(state.startedAt).toBeNull();
  });

  it('a disabled or ineligible embedder, or no vector extension, returns zero stats and embeds nothing', async () => {
    const tables = { 's.a': { rows: rows('a1') } };
    for (const [over, dbOpts] of [
      [{ resolveEmbedder: async () => ({ mode: 'disabled' as const }) }, {}],
      [{ resolveEmbedder: async () => ({ mode: 'm2', dims: 4, embed: vec3 }) }, {}],
      [{}, { vector: false }],
    ] as const) {
      const { sw, db } = sweeper(tables, over, dbOpts);
      expect(plain(await sw.run())).toEqual([zero('s.a')]);
      expect(db.calls.some((c) => c.query.includes(' AS body'))).toBe(false);
    }
  });

  it('refuses a width-skewed target loudly with the host hint, and zero-stats an absent one', async () => {
    const { sw, log, embed } = sweeper(
      { 's.a': { rows: rows('a1'), dims: 5 }, 's.b': { rows: rows('b1'), absent: true }, 's.c': { rows: rows('c1') } },
      { widthSkewHint: 'Run the width migration.' },
    );
    expect(plain(await sw.run())).toEqual([
      zero('s.a'),
      zero('s.b'),
      { table: 's.c', scanned: 1, embedded: 1, errors: 0, writeSkipped: 0 },
    ]);
    expect(embed.mock.calls.map((c) => c[0])).toEqual(['c1']);
    const skew = log.lines.find((l) => l.includes('SCHEMA WIDTH SKEW'))!;
    expect(skew).toContain('[bf] SCHEMA WIDTH SKEW — refusing s.a.embedding: it is vector(5) but this code emits 3.');
    expect(skew.endsWith(' Run the width migration.')).toBe(true);
  });

  it('stops at the budget, checked between targets', async () => {
    const clock = { t: 0 };
    const embed = vi.fn(async (_text: string) => {
      clock.t += 60;
      return [1, 2, 3];
    });
    const { sw, log } = sweeper(
      { 's.a': { rows: rows('a1', 'a2', 'a3') }, 's.b': { rows: rows('b1', 'b2', 'b3') } },
      { now: () => clock.t, budgetMs: 100, resolveEmbedder: async () => ({ mode: 'm1', dims: 3, embed }) },
    );
    await sw.run();
    expect(embed.mock.calls.map((c) => c[0])).toEqual(['a1', 'b1']);
    expect(log.lines.at(-1)).toContain('drained 0/2 target(s) · budget expired with work remaining');
  });

  it('batches only for an eligible embedder, preferring the embedder’s own embedMany', async () => {
    const resolveBatchEmbed = vi.fn(async () => async (texts: string[]) => texts.map(() => [1, 2, 3]));
    const { sw } = sweeper({ 's.a': { rows: rows('a1', 'a2') } }, { resolveBatchEmbed, batchSize: 2 });
    await sw.run();
    expect(resolveBatchEmbed).toHaveBeenCalledWith('m1');

    const own = vi.fn(async (texts: string[]) => texts.map(() => [1, 2, 3]));
    const second = sweeper(
      { 's.a': { rows: rows('a1', 'a2') } },
      { resolveBatchEmbed, batchSize: 2, resolveEmbedder: async () => ({ mode: 'm1', dims: 3, embed: vec3, embedMany: own }) },
    );
    await second.sw.run();
    expect(own).toHaveBeenCalledTimes(1);
    expect(resolveBatchEmbed).toHaveBeenCalledTimes(1);

    const disabled = sweeper({ 's.a': { rows: rows('a1') } }, {
      resolveBatchEmbed,
      resolveEmbedder: async () => ({ mode: 'disabled' as const }),
    });
    await disabled.sw.run();
    expect(resolveBatchEmbed).toHaveBeenCalledTimes(1);
  });

  it('a hung resolver times out and releases the latch', async () => {
    const { sw } = sweeper({ 's.a': { rows: rows('a1') } }, {
      resolveEmbedder: () => new Promise(() => {}),
      resolveTimeoutMs: 20,
    });
    await expect(sw.run()).rejects.toThrow('embedder_resolve_timeout_after_20ms');
    expect(sw.state.running).toBe(false);
    sw.resetForTest();
    expect(sw.state).toEqual(createBackfillSweepState());
  });
});
