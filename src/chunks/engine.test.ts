import { createHash } from 'node:crypto';
import type { Sql } from 'postgres';
import { describe, expect, it } from 'vitest';
// R-10: everything under test is imported through the package entry point.
import {
  chunkParentKeyId,
  parentShaOf,
  planChunks,
  resolveChunkSurface,
  splitterVersionOf,
  syncChunkSurfaces,
  type ChunkStore,
  type ChunkSurface,
  type ChunkSyncLogger,
  type ExistingChunk,
  type ResolvedChunkSurface,
  type StaleParent,
} from '../index';

const sha = (s: string) => createHash('sha256').update(s).digest('hex');
const NO_SQL = {} as Sql;

interface FakeParent {
  key: string[];
  text: string;
  header: string | null;
  version: Date | null;
  eligible: boolean;
}
interface FakeChunk extends ExistingChunk {
  anchor: string | null;
  header: string | null;
  content: string;
  updatedAt: Date;
}

/** In-memory ChunkStore with the shared store's selection and prune semantics. */
class FakeStore implements ChunkStore {
  readonly name = 'fake';
  parents = new Map<string, FakeParent>();
  chunks = new Map<string, FakeChunk[]>();
  failReplaceFor = new Set<string>();
  failSelect = false;
  touched = 0;

  constructor(readonly surfaceName: string) {}
  id(key: readonly string[]): string {
    return chunkParentKeyId(this.surfaceName, key);
  }
  put(p: Partial<FakeParent> & { key: string[]; text: string }): void {
    this.parents.set(this.id(p.key), { header: null, version: null, eligible: true, ...p });
  }
  embedAll(): void {
    for (const rows of this.chunks.values()) {
      for (const c of rows) if (c.embedding === null) {
        c.embedding = `[${c.chunkSha.slice(0, 6)}]`;
        c.embeddingMode = 'test';
        c.embeddingProfile = 'p1';
      }
    }
  }
  unembedded(): number {
    let n = 0;
    for (const rows of this.chunks.values()) for (const c of rows) if (c.embedding === null) n++;
    return n;
  }
  async transaction<T>(sql: Sql, fn: (tx: Sql) => Promise<T>): Promise<T> {
    return fn(sql);
  }
  async selectStale(_sql: Sql, s: ResolvedChunkSurface, opts: { limit: number; excludeKeys: readonly string[] }) {
    if (this.failSelect) throw new Error('boom: select');
    const exclude = new Set(opts.excludeKeys);
    const out: StaleParent[] = [];
    for (const p of this.parents.values()) {
      if (p.text.length <= s.minChars || !p.eligible || exclude.has(p.key.join('\u001f'))) continue;
      const c0 = this.chunks.get(this.id(p.key))?.find((c) => c.chunkIdx === 0);
      const fresh =
        c0 !== undefined &&
        c0.splitterVersion === s.splitterVersion &&
        (s.versionSql
          ? c0.updatedAt.getTime() >= (p.version?.getTime() ?? -Infinity)
          : c0.parentSha === parentShaOf(sha, p.text, p.header));
      if (!fresh) out.push({ key: p.key, text: p.text, header: p.header, version: p.version?.toISOString() ?? null });
    }
    const ms = (v: string | null) => (v === null ? 0 : Date.parse(v));
    if (s.versionSql) out.sort((a, b) => ms(b.version) - ms(a.version));
    return out.slice(0, opts.limit);
  }
  async readExisting(_sql: Sql, _s: ResolvedChunkSurface, key: readonly string[]) {
    return (this.chunks.get(this.id(key)) ?? []).map((c) => ({ ...c }));
  }
  async replace(_sql: Sql, _s: ResolvedChunkSurface, key: readonly string[], rows: readonly any[], meta: any) {
    if (this.failReplaceFor.has(key.join('/'))) throw new Error(`boom: replace ${key.join('/')}`);
    this.chunks.set(
      this.id(key),
      rows.map((r) => ({
        ...r,
        parentSha: meta.parentSha,
        splitterVersion: meta.splitterVersion,
        updatedAt: meta.version ? new Date(meta.version) : new Date(),
      })),
    );
    if (rows.length === 0) this.chunks.delete(this.id(key));
  }
  async touch(_sql: Sql, _s: ResolvedChunkSurface, key: readonly string[], version: string | null) {
    this.touched++;
    for (const c of this.chunks.get(this.id(key)) ?? []) c.updatedAt = version ? new Date(version) : new Date();
  }
  async prune(_sql: Sql, s: ResolvedChunkSurface, opts: { limit: number }) {
    let parents = 0;
    let chunks = 0;
    for (const [id, rows] of [...this.chunks]) {
      if (parents >= opts.limit) break;
      const p = this.parents.get(id);
      const keep = p !== undefined && p.eligible && p.text.length > s.minChars;
      if (!keep) {
        parents++;
        chunks += rows.length;
        this.chunks.delete(id);
      }
    }
    return { parents, chunks };
  }
}

function captureLogger() {
  const warns: { message: string; meta?: Record<string, unknown> }[] = [];
  const errors: { message: string; meta?: Record<string, unknown> }[] = [];
  const logger: ChunkSyncLogger = {
    warn: (message, meta) => warns.push({ message, meta }),
    error: (message, meta) => errors.push({ message, meta }),
  };
  return { logger, warns, errors };
}

const windowSurface = (over: Partial<ChunkSurface> = {}): ChunkSurface => ({
  surface: 'notes',
  parent: { table: 'app.notes', key: ['id'] },
  textSql: 'p.body',
  versionSql: 'p.updated_at',
  minChars: 100,
  splitter: { kind: 'window', size: 60, overlap: 10 },
  maxChunks: 8,
  ...over,
});

const mdSurface = (over: Partial<ChunkSurface> = {}): ChunkSurface => ({
  surface: 'docs',
  parent: { table: 'app.docs', key: ['id'] },
  textSql: 'p.body',
  headerSql: 'p.title',
  versionSql: 'p.updated_at',
  minChars: 100,
  splitter: { kind: 'markdown', maxChars: 400 },
  maxChunks: 16,
  ...over,
});

const section = (name: string, words: string) => `## ${name}\n\n${words}\n\n`;
const doc = (a: string, b: string, c: string) => `${section('Alpha', a)}${section('Beta', b)}${section('Gamma', c)}`;
const t = (n: number) => new Date(Date.UTC(2026, 8, 29, 12, n));

async function run(store: FakeStore, surface: ChunkSurface, extra: Record<string, unknown> = {}) {
  const log = captureLogger();
  const res = await syncChunkSurfaces(NO_SQL, [surface], store, { hash: sha, logger: log.logger, ...extra });
  return { stats: res.surfaces[0]!, res, ...log };
}

describe('planChunks', () => {
  it('complete-body windows retain tail content beyond a prefix cap', () => {
    const text = 'ordinary context '.repeat(8000) + 'TAIL-FINDING';
    const surface = { splitter: { kind: 'window' as const, size: 1500, overlap: 250 }, maxChunks: 32 };
    expect(planChunks(surface, text, 'Title', sha).truncated).toBe(true);
    const complete = planChunks({ ...surface, completeBody: true }, text, 'Title', sha);
    expect(complete.truncated).toBe(false);
    expect(complete.chunks.length).toBeGreaterThan(32);
    expect(complete.chunks.at(-1)?.content).toContain('TAIL-FINDING');
    expect(splitterVersionOf({ ...surface, completeBody: true })).not.toBe(splitterVersionOf(surface));
  });

  it('complete-body markdown retains late headings and continuation rows', () => {
    const text = Array.from({ length: 300 }, (_, i) => section(`Heading ${i}`, `content ${i} `.repeat(15))).join('');
    const result = planChunks({ splitter: { kind: 'markdown', maxChars: 80 }, maxChunks: 4, completeBody: true }, text, 'Doc', sha);
    expect(result.truncated).toBe(false);
    expect(result.chunks.at(-1)?.header).toContain('Heading 299');
    expect(result.chunks.at(-1)?.content).toContain('content 299');
  });

  it('cuts windows, embeds the header, and hashes header + content', () => {
    const text = 'x'.repeat(130);
    const plan = planChunks({ splitter: { kind: 'window', size: 60, overlap: 10 }, maxChunks: 8 }, text, 'Title', sha);
    expect(plan.chunks.map((c) => c.content.length)).toEqual([60, 60, 30]);
    expect(plan.truncated).toBe(false);
    expect(plan.chunks[0]!.header).toBe('Title');
    expect(plan.chunks[0]!.anchor).toBeNull();
    expect(plan.chunks[0]!.chunkSha).toBe(sha(`Title\n${'x'.repeat(60)}`));
  });

  it('gives markdown chunks their heading path as context', () => {
    const plan = planChunks({ splitter: { kind: 'markdown', maxChars: 400 }, maxChunks: 16 }, doc('a '.repeat(20), 'b '.repeat(20), 'c '.repeat(20)), 'Doc', sha);
    expect(plan.chunks.length).toBe(3);
    expect(plan.chunks.map((c) => c.header)).toEqual(['Doc › Alpha', 'Doc › Beta', 'Doc › Gamma']);
    expect(plan.chunks.every((c) => c.anchor !== null)).toBe(true);
  });

  it('keys the splitter version on every option that changes output', () => {
    const base = { splitter: { kind: 'window' as const, size: 60, overlap: 10 }, maxChunks: 8 };
    const v = splitterVersionOf(base);
    expect(splitterVersionOf({ ...base, maxChunks: 9 })).not.toBe(v);
    expect(splitterVersionOf({ ...base, splitter: { kind: 'window', size: 61, overlap: 10 } })).not.toBe(v);
    expect(splitterVersionOf({ ...base, splitter: { kind: 'window', size: 60, overlap: 11 } })).not.toBe(v);
  });

  it('parent sha is the plain text sha without a header, and changes with the header', () => {
    expect(parentShaOf(sha, 'abc', null)).toBe(sha('abc'));
    expect(parentShaOf(sha, 'abc', '')).toBe(sha('abc'));
    expect(parentShaOf(sha, 'abc', 'H')).toBe(sha('H\u001fabc'));
  });
});

describe('resolveChunkSurface', () => {
  it('applies defaults', () => {
    const r = resolveChunkSurface(windowSurface({ minChars: undefined }));
    expect(r.minChars).toBe(2000);
    expect(r.keyColumns).toEqual([{ column: 'id', type: null }]);
  });
  it.each([
    ['a bad table', { parent: { table: 'app.notes; drop', key: ['id'] } }],
    ['a bad key column', { parent: { table: 'app.notes', key: ['id x'] } }],
    ['an empty key', { parent: { table: 'app.notes', key: [] } }],
    ['overlap >= size', { splitter: { kind: 'window', size: 60, overlap: 60 } }],
    ['maxChunks 0', { maxChunks: 0 }],
    ['a multi-statement expression', { textSql: 'p.body; delete from x' }],
  ] as const)('rejects %s', (_label, over) => {
    expect(() => resolveChunkSurface(windowSurface(over as Partial<ChunkSurface>))).toThrow();
  });
});

describe('syncChunkSurfaces', () => {
  it('chunks long parents only and embeds nothing itself', async () => {
    const store = new FakeStore('notes');
    store.put({ key: ['1'], text: 'y'.repeat(150), version: t(1) });
    store.put({ key: ['2'], text: 'short', version: t(1) });
    const { stats } = await run(store, windowSurface());
    expect(stats).toMatchObject({ parentsSynced: 1, chunksWritten: 3, errors: 0, more: false });
    expect(store.chunks.has(store.id(['2']))).toBe(false);
    expect(store.unembedded()).toBe(3);
  });

  it('R-11: editing one section re-embeds only the chunk holding that section', async () => {
    const store = new FakeStore('docs');
    const a = 'alpha words '.repeat(12);
    const b = 'beta words '.repeat(12);
    const c = 'gamma words '.repeat(12);
    store.put({ key: ['d1'], text: doc(a, b, c), header: 'Doc', version: t(1) });
    await run(store, mdSurface());
    store.embedAll();
    const before = new Map(store.chunks.get(store.id(['d1']))!.map((x) => [x.chunkIdx, x.embedding]));
    expect(store.unembedded()).toBe(0);

    store.put({ key: ['d1'], text: doc(a, 'beta EDITED words '.repeat(12), c), header: 'Doc', version: t(2) });
    const { stats } = await run(store, mdSurface());

    expect(stats).toMatchObject({ parentsSynced: 1, chunksWritten: 3, embeddingsReused: 2, errors: 0 });
    const after = store.chunks.get(store.id(['d1']))!;
    expect(after.filter((x) => x.embedding === null).map((x) => x.chunkIdx)).toEqual([1]);
    expect(after[0]!.embedding).toBe(before.get(0));
    expect(after[2]!.embedding).toBe(before.get(2));
  });

  it("R-11: appending to a window-split parent keeps every unchanged window's embedding", async () => {
    // Varied, whitespace-free text so every window is distinct: a reused
    // embedding can only come from an identical window, never from a
    // look-alike one.
    const text = (n: number) => Array.from({ length: n }, (_, i) => String.fromCharCode(97 + ((i * 7 + (i >> 3)) % 26))).join('');
    const store = new FakeStore('notes');
    store.put({ key: ['1'], text: text(250), version: t(1) });
    await run(store, windowSurface());
    store.embedAll();
    const before = store.chunks.get(store.id(['1']))!.map((x) => ({ content: x.content, embedding: x.embedding }));
    expect(store.unembedded()).toBe(0);

    store.put({ key: ['1'], text: text(350), version: t(2) });
    const { stats } = await run(store, windowSurface());

    const after = store.chunks.get(store.id(['1']))!;
    expect(after.length).toBeGreaterThan(before.length);
    // Every window whose text the append did not touch keeps its embedding
    // object; every window it did touch (the old tail, re-cut, and the new
    // ones) is left for the embed sweep.
    const unchanged = after.filter((x) => before.some((b) => b.content === x.content));
    expect(unchanged.length).toBe(before.length - 1);
    for (const x of unchanged) expect(x.embedding).toBe(before.find((b) => b.content === x.content)!.embedding);
    expect(after.filter((x) => x.embedding === null).length).toBe(after.length - unchanged.length);
    expect(stats).toMatchObject({ parentsSynced: 1, chunksWritten: after.length, embeddingsReused: unchanged.length, errors: 0 });
  });

  it('a version bump with unchanged text only advances the stored version', async () => {
    const store = new FakeStore('notes');
    store.put({ key: ['1'], text: 'y'.repeat(150), version: t(1) });
    await run(store, windowSurface());
    store.embedAll();
    store.put({ key: ['1'], text: 'y'.repeat(150), version: t(5) });
    const { stats } = await run(store, windowSurface());
    expect(stats).toMatchObject({ parentsUnchanged: 1, parentsSynced: 0, chunksWritten: 0 });
    expect(store.touched).toBe(1);
    expect(store.unembedded()).toBe(0);
    const again = await run(store, windowSurface());
    expect(again.stats).toMatchObject({ parentsUnchanged: 0, parentsSynced: 0 });
  });

  it('a registry change to the splitter re-cuts, reusing embeddings of identical chunks', async () => {
    const store = new FakeStore('notes');
    store.put({ key: ['1'], text: 'y'.repeat(300), version: t(1) });
    await run(store, windowSurface({ maxChunks: 3 }));
    store.embedAll();
    const { stats } = await run(store, windowSurface({ maxChunks: 8 }));
    expect(stats.parentsSynced).toBe(1);
    expect(stats.chunksWritten).toBe(6);
    // Same window text => same chunk sha => the embedding is copied, not
    // recomputed. Five of the six new windows are the old 60-char text; the
    // 50-char last window is new.
    expect(stats.embeddingsReused).toBe(5);
    expect(store.unembedded()).toBe(1);
  });

  it('R-12: deleting a parent removes its chunks on the next sync', async () => {
    const store = new FakeStore('notes');
    store.put({ key: ['1'], text: 'y'.repeat(150), version: t(1) });
    store.put({ key: ['2'], text: 'z'.repeat(150), version: t(1) });
    await run(store, windowSurface());
    expect(store.chunks.size).toBe(2);
    store.parents.delete(store.id(['1']));
    const { stats } = await run(store, windowSurface());
    expect(stats.pruned).toBe(3);
    expect(store.chunks.has(store.id(['1']))).toBe(false);
    expect(store.chunks.has(store.id(['2']))).toBe(true);
  });

  it('R-12: a parent that shrinks below the cut or turns ineligible loses its chunks', async () => {
    const store = new FakeStore('notes');
    store.put({ key: ['1'], text: 'y'.repeat(150), version: t(1) });
    store.put({ key: ['2'], text: 'z'.repeat(150), version: t(1) });
    await run(store, windowSurface());
    store.put({ key: ['1'], text: 'tiny', version: t(2) });
    store.put({ key: ['2'], text: 'z'.repeat(150), version: t(2), eligible: false });
    const { stats } = await run(store, windowSurface());
    expect(stats.pruned).toBe(6);
    expect(store.chunks.size).toBe(0);
  });

  it('R-13: a failing parent is logged and counted, and the rest still sync', async () => {
    const store = new FakeStore('notes');
    store.put({ key: ['1'], text: 'y'.repeat(150), version: t(2) });
    store.put({ key: ['2'], text: 'z'.repeat(150), version: t(1) });
    store.failReplaceFor.add('1');
    const { stats, errors } = await run(store, windowSurface());
    expect(stats).toMatchObject({ errors: 1, parentsSynced: 1 });
    expect(errors).toHaveLength(1);
    expect(errors[0]!.meta).toMatchObject({ surface: 'notes', key: ['1'] });
    expect(String(errors[0]!.meta!.error)).toContain('boom: replace 1');
  });

  it('R-13: a failing collection is logged and counted, and the next collection still syncs', async () => {
    const broken = new FakeStore('notes');
    broken.failSelect = true;
    const ok = new FakeStore('docs');
    ok.put({ key: ['d1'], text: doc('a '.repeat(30), 'b '.repeat(30), 'c '.repeat(30)), header: 'Doc', version: t(1) });
    const log = captureLogger();
    const res = await syncChunkSurfaces(
      NO_SQL,
      [windowSurface({ store: broken }), mdSurface({ store: ok })],
      broken,
      { hash: sha, logger: log.logger },
    );
    expect(res.surfaces.map((s) => [s.surface, s.errors])).toEqual([
      ['notes', 1],
      ['docs', 0],
    ]);
    expect(res.surfaces[1]!.parentsSynced).toBe(1);
    expect(log.errors).toHaveLength(1);
    expect(log.errors[0]!.meta).toMatchObject({ surface: 'notes' });
    expect(String(log.errors[0]!.meta!.error)).toContain('boom: select');
  });

  it('R-13: an invalid registry entry is logged and counted instead of throwing', async () => {
    const store = new FakeStore('notes');
    const { stats, errors } = await run(store, windowSurface({ maxChunks: 0 }));
    expect(stats.errors).toBe(1);
    expect(errors[0]!.message).toContain('invalid chunk surface');
  });

  it('R-14: a parent cut off by maxChunks is logged and counted, and keeps exactly maxChunks chunks', async () => {
    const store = new FakeStore('notes');
    store.put({ key: ['long'], text: 'y'.repeat(60 + 50 * 20), version: t(1) });
    const { stats, warns } = await run(store, windowSurface({ maxChunks: 4 }));
    expect(stats).toMatchObject({ parentsTruncatedByMaxChunks: 1, chunksWritten: 4, parentsSynced: 1 });
    expect(store.chunks.get(store.id(['long']))).toHaveLength(4);
    expect(warns).toHaveLength(1);
    expect(warns[0]!.message).toContain('truncated');
    expect(warns[0]!.meta).toMatchObject({ surface: 'notes', key: ['long'], maxChunks: 4, producedChunks: 5 });
  });

  it('R-14: a parent that fits exactly is not reported as truncated', async () => {
    const store = new FakeStore('notes');
    store.put({ key: ['fit'], text: 'y'.repeat(60 + 50 * 3), version: t(1) });
    const { stats, warns } = await run(store, windowSurface({ maxChunks: 4 }));
    expect(stats).toMatchObject({ parentsTruncatedByMaxChunks: 0, chunksWritten: 4 });
    expect(warns).toHaveLength(0);
  });

  it('parents that cut to zero chunks are remembered and skipped, not re-selected forever', async () => {
    const store = new FakeStore('docs');
    store.put({ key: ['blank'], text: ' '.repeat(300), version: t(1) });
    const emptyKeys = new Set<string>();
    const first = await run(store, mdSurface(), { emptyKeys });
    expect(first.stats).toMatchObject({ parentsEmpty: 1, parentsSynced: 0 });
    expect(emptyKeys.has(chunkParentKeyId('docs', ['blank']))).toBe(true);
    const second = await run(store, mdSurface(), { emptyKeys });
    expect(second.stats).toMatchObject({ parentsEmpty: 0, parentsSynced: 0, more: false });
  });

  it('newest parents first, one batch per call, with more=true when a batch fills', async () => {
    const store = new FakeStore('notes');
    for (let i = 1; i <= 3; i++) store.put({ key: [String(i)], text: 'y'.repeat(150), version: t(i) });
    const { stats } = await run(store, windowSurface(), { batchPerSurface: 2 });
    expect(stats).toMatchObject({ parentsSynced: 2, more: true });
    expect(store.chunks.has(store.id(['1']))).toBe(false);
  });

  it('round-robins surfaces and stops at the time budget', async () => {
    const a = new FakeStore('notes');
    const b = new FakeStore('docs');
    const surfaces = [windowSurface({ store: a }), mdSurface({ store: b })];
    const full = await syncChunkSurfaces(NO_SQL, surfaces, a, { hash: sha, startIndex: 1, logger: captureLogger().logger });
    expect(full.surfaces.map((s) => s.surface)).toEqual(['docs', 'notes']);
    expect(full.nextIndex).toBe(0);

    let clock = 0;
    const budgeted = await syncChunkSurfaces(NO_SQL, surfaces, a, {
      hash: sha,
      timeBudgetMs: 10,
      now: () => (clock += 6),
      logger: captureLogger().logger,
    });
    expect(budgeted.surfaces.map((s) => s.surface)).toEqual(['notes']);
    expect(budgeted.skippedForBudget).toBe(1);
    expect(budgeted.nextIndex).toBe(1);
  });
});
