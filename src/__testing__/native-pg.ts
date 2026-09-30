/**
 * Vitest globalSetup for @papercusp/search's integration suite: a Postgres with
 * pgvector that the suite owns.
 *
 * The library is published on its own, so it cannot use a host's test harness
 * (WI-4973). Two ways to get a database, in order:
 *
 *   1. SEARCH_TEST_PG_URL — a superuser URL of any Postgres with pgvector >= 0.8
 *      installed. Tests create and drop their own schema in it.
 *   2. Otherwise a throwaway cluster started from the host's Postgres binaries:
 *      initdb into a temp dir, trust auth on a free loopback port, stopped and
 *      deleted when the run ends. SEARCH_TEST_PG_BIN picks the bin directory;
 *      by default the newest /usr/lib/postgresql/<major>/bin whose share
 *      directory has the pgvector extension is used.
 *
 * With neither available the run FAILS with instructions. It never skips: a
 * suite that silently runs nothing reads exactly like a passing one.
 *
 * The cluster is a plain child process (not detached), so it dies with the run.
 */
import { spawn, execFile, type ChildProcess } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import postgres from 'postgres';

declare module 'vitest' {
  export interface ProvidedContext {
    /** Superuser URL of the suite's Postgres (pgvector installed, not yet created). */
    searchPgUrl: string;
  }
}

const execFileP = promisify(execFile);

/** The newest local Postgres bin directory whose installation ships pgvector. */
export function resolvePgBinDir(env: NodeJS.ProcessEnv = process.env): string {
  const override = env.SEARCH_TEST_PG_BIN;
  if (override) {
    if (!existsSync(join(override, 'initdb'))) throw new Error(`SEARCH_TEST_PG_BIN=${override} has no initdb`);
    return override;
  }
  for (let major = 30; major >= 13; major--) {
    const bin = `/usr/lib/postgresql/${major}/bin`;
    const vector = `/usr/share/postgresql/${major}/extension/vector.control`;
    if (existsSync(join(bin, 'initdb')) && existsSync(join(bin, 'postgres')) && existsSync(vector)) return bin;
  }
  throw new Error(
    '@papercusp/search integration tests need Postgres with pgvector: set SEARCH_TEST_PG_URL to a ' +
      'superuser URL, or SEARCH_TEST_PG_BIN to a Postgres bin directory whose installation has pgvector ' +
      '(none found under /usr/lib/postgresql/*/bin with /usr/share/postgresql/<major>/extension/vector.control).',
  );
}

async function freeLoopbackPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.once('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address();
      const port = typeof addr === 'object' && addr ? addr.port : 0;
      srv.close(() => (port ? resolve(port) : reject(new Error('no port assigned'))));
    });
  });
}

async function waitReady(url: string, budgetMs: number): Promise<string | null> {
  const deadline = Date.now() + budgetMs;
  let lastError = 'never tried';
  while (Date.now() < deadline) {
    const sql = postgres(url, { max: 1, connect_timeout: 2, onnotice: () => {} });
    try {
      await sql`SELECT 1`;
      return null;
    } catch (e) {
      lastError = e instanceof Error ? e.message : String(e);
    } finally {
      await sql.end({ timeout: 1 }).catch(() => undefined);
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  return lastError;
}

async function stopChild(child: ChildProcess, timeoutMs: number): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return;
  const exited = new Promise<void>((resolve) => child.once('exit', () => resolve()));
  child.kill('SIGINT'); // Postgres "fast" shutdown
  const timer = new Promise<'timeout'>((r) => setTimeout(() => r('timeout'), timeoutMs));
  if ((await Promise.race([exited, timer])) === 'timeout') {
    child.kill('SIGKILL');
    await exited;
  }
}

export default async function setup({ provide }: { provide: (key: 'searchPgUrl', value: string) => void }) {
  const given = process.env.SEARCH_TEST_PG_URL;
  if (given) {
    provide('searchPgUrl', given);
    return undefined;
  }

  const bin = resolvePgBinDir();
  const dataDir = await mkdtemp(join(tmpdir(), 'papercusp-search-pg-'));
  const removeDir = () => rm(dataDir, { recursive: true, force: true }).catch(() => undefined);
  let child: ChildProcess | undefined;
  try {
    await execFileP(
      join(bin, 'initdb'),
      ['-D', dataDir, '-U', 'postgres', '-A', 'trust', '-E', 'UTF8', '--locale=C', '--no-sync'],
      { maxBuffer: 16 * 1024 * 1024 },
    );
    const port = await freeLoopbackPort();
    const log: string[] = [];
    child = spawn(
      join(bin, 'postgres'),
      ['-D', dataDir, '-p', String(port), '-c', 'listen_addresses=127.0.0.1', '-c', `unix_socket_directories=${dataDir}`, '-c', 'fsync=off'],
      { stdio: ['ignore', 'pipe', 'pipe'] },
    );
    const keep = (b: Buffer) => {
      log.push(b.toString());
      if (log.length > 40) log.shift();
    };
    child.stdout?.on('data', keep);
    child.stderr?.on('data', keep);
    const url = `postgres://postgres@127.0.0.1:${port}/postgres`;
    const notReady = await waitReady(url, 60_000);
    if (notReady !== null) {
      throw new Error(`test Postgres never answered SELECT 1 (${notReady}); server log tail:\n${log.join('')}`);
    }
    provide('searchPgUrl', url);
    const proc = child;
    return async () => {
      await stopChild(proc, 10_000);
      await removeDir();
    };
  } catch (e) {
    if (child) await stopChild(child, 5_000);
    await removeDir();
    throw e;
  }
}
