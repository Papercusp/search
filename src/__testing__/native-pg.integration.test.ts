import postgres from 'postgres';
import { afterEach, describe, expect, inject, it, vi } from 'vitest';
import setup from './native-pg';

afterEach(() => vi.unstubAllEnvs());

async function privateDatabase(run: (url: string, sql: postgres.Sql) => Promise<void>) {
  const admin = postgres(inject('searchPgUrl'), { max: 1, onnotice: () => {} });
  const name = `search_setup_${process.pid}_${Date.now()}`;
  await admin.unsafe(`CREATE DATABASE ${name}`);
  const url = new URL(inject('searchPgUrl'));
  url.pathname = `/${name}`;
  const sql = postgres(url.toString(), { max: 1, onnotice: () => {} });
  try {
    await run(url.toString(), sql);
  } finally {
    await sql.end({ timeout: 5 });
    await admin.unsafe(`DROP DATABASE ${name} WITH (FORCE)`);
    await admin.end({ timeout: 5 });
  }
}

async function vectorReady(url: string) {
  const sql = postgres(url, { max: 1, onnotice: () => {} });
  try {
    const [row] = await sql<{ type: string | null; extensions: number }[]>`
      SELECT to_regtype('vector')::text AS type,
             (SELECT count(*)::int FROM pg_extension WHERE extname = 'vector') AS extensions`;
    return row;
  } finally {
    await sql.end({ timeout: 5 });
  }
}

describe('search Postgres global setup vector initialization', () => {
  it('installs vector before publishing an owned cluster URL', async () => {
    vi.stubEnv('SEARCH_TEST_PG_URL', '');
    let published: Promise<unknown> | undefined;
    const stop = await setup({ provide: (_key, url) => { published = vectorReady(url); } });
    try {
      expect(await published).toEqual({ type: 'vector', extensions: 1 });
    } finally {
      await stop?.();
    }
  });

  it('installs vector before publishing a supplied URL and reuses an existing installation', async () => {
    await privateDatabase(async (url) => {
      vi.stubEnv('SEARCH_TEST_PG_URL', url);
      for (let run = 0; run < 2; run++) {
        let published: Promise<unknown> | undefined;
        const stop = await setup({ provide: (_key, value) => { published = vectorReady(value); } });
        expect(stop).toBeUndefined(); // the supplied database belongs to the caller
        expect(await published).toEqual({ type: 'vector', extensions: 1 });
      }
    });
  });

  it('serializes independent setup connections behind an uncommitted vector installation', async () => {
    await privateDatabase(async (url, sql) => {
      vi.stubEnv('SEARCH_TEST_PG_URL', url);
      const holder = postgres(url, { max: 1, onnotice: () => {} });
      let release!: () => void;
      let installed!: (pid: number) => void;
      const ready = new Promise<number>((resolve) => { installed = resolve; });
      const gate = new Promise<void>((resolve) => { release = resolve; });
      const first = holder.begin(async (tx) => {
        await tx`SELECT pg_advisory_xact_lock(hashtext('@papercusp/search:vector-init'))`;
        await tx`CREATE EXTENSION vector`;
        const [row] = await tx<{ pid: number }[]>`SELECT pg_backend_pid() AS pid`;
        installed(row!.pid);
        await gate;
      });
      let attempts: Promise<unknown>[] = [];
      try {
        const pid = await ready;
        const published: Promise<unknown>[] = [];
        // Separate setup calls open separate PG sessions, as simultaneous test runs do.
        attempts = Array.from({ length: 2 }, () => setup({
          provide: (_key, value) => { published.push(vectorReady(value)); },
        }));
        const settled = Promise.allSettled(attempts);
        await expect.poll(async () => {
          const [row] = await sql<{ waiting: number }[]>`
            SELECT count(*)::int AS waiting FROM pg_locks waiting
            JOIN pg_locks held USING (locktype, database, classid, objid, objsubid)
            WHERE held.pid = ${pid} AND held.granted AND NOT waiting.granted
              AND waiting.locktype = 'advisory'`;
          return row!.waiting;
        }, { timeout: 10_000 }).toBe(2);
        expect(published).toHaveLength(0);
        release();
        await first;
        expect(await settled).toEqual([
          { status: 'fulfilled', value: undefined },
          { status: 'fulfilled', value: undefined },
        ]);
        expect(await Promise.all(published)).toEqual([
          { type: 'vector', extensions: 1 },
          { type: 'vector', extensions: 1 },
        ]);
      } finally {
        release();
        await first;
        await Promise.allSettled(attempts);
        await holder.end({ timeout: 5 });
      }
    });
  });
});
