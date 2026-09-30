/**
 * "Simple addition": the definition of done for chunked collections
 * (generic-rag-chunking-2026-09-29 P-008, acceptance R-22, decision D-010).
 *
 * A fixture host with no papercusp code adds a collection, `notes`, by writing
 * ONE registry entry (NOTES below). Everything else the host runs is generic and
 * never names the collection:
 *
 *   - the shared chunk table, created by applying sql/text-chunks.reference.sql
 *     verbatim (the library's reference migration);
 *   - syncChunkSurfaces over the registry;
 *   - an embed sweep over chunkEmbedTargets(registry), the targets DERIVED from
 *     the registry rather than listed per table;
 *   - chunkAwareVectorLeg for search.
 *
 * End to end it checks that the sync writes the chunks, the derived embed target
 * picks them up, a query that matches only text past the 2,000-character cut
 * finds its parent in 'retrieve' mode but not in 'gist' mode, editing one
 * section re-embeds only that section's chunk, and deleting the parent removes
 * its chunks.
 *
 * The embedder is a deterministic bag of words: dimension 0 is a constant bias
 * (so no vector is ever zero) and each word adds to one hashed dimension. The
 * filler vocabulary is filtered so no filler word shares the query word's
 * dimension, which makes every expected ranking follow from word counts alone.
 */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import postgres from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';
import {
  DEFAULT_CHUNK_MIN_CHARS,
  chunkAwareVectorLeg,
  chunkEmbedTargets,
  embedPendingChunks,
  planChunks,
  resolveChunkSurface,
  selectPendingChunks,
  sharedChunkStore,
  syncChunkSurfaces,
  type ChunkLegMode,
  type ChunkSurface,
  type ChunkSurfaceSyncStats,
  type PendingChunk,
} from './index';

const SCHEMA = `simple_add_${process.pid}_${Date.now()}`;
const REFERENCE_SQL = readFileSync(new URL('../../sql/text-chunks.reference.sql', import.meta.url), 'utf8');

// ---- the fixture embedder -------------------------------------------------

const DIMS = 768; // the reference DDL's example width, applied unmodified
const QUERY_WORD = 'zebra';

function dimOf(word: string): number {
  let h = 0x811c9dc5; // FNV-1a
  for (let i = 0; i < word.length; i++) {
    h ^= word.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return 1 + (h % (DIMS - 1));
}

function embedText(text: string): number[] {
  const v = new Array<number>(DIMS).fill(0);
  v[0] = 1;
  for (const w of text.toLowerCase().match(/[a-z]+/g) ?? []) v[dimOf(w)]! += 1;
  const norm = Math.hypot(...v);
  return v.map((x) => x / norm);
}

const vecText = (v: readonly number[]) => `[${v.join(',')}]`;
const embedBatch = async (texts: readonly string[]) => texts.map(embedText);

// ---- the fixture parents --------------------------------------------------

const BASE_WORDS = (
  'amber basin canyon delta ember fjord granite harbor island jungle kelp lagoon meadow ' +
  'nectar orchard pebble quarry ridge savanna tundra upland valley willow yarrow alpine ' +
  'bramble cedar dune estuary forest glacier heath inlet juniper knoll marsh'
).split(' ');
/** Filler words that do not share the query word's embedding dimension. */
const VOCAB = BASE_WORDS.filter((w) => dimOf(w) !== dimOf(QUERY_WORD));

function filler(seed: number, words: number): string {
  const out: string[] = [];
  for (let i = 0; i < words; i++) out.push(VOCAB[(seed * 7 + i * 3 + Math.floor(i / VOCAB.length)) % VOCAB.length]!);
  const sentences: string[] = [];
  for (let i = 0; i < out.length; i += 12) sentences.push(`${out.slice(i, i + 12).join(' ')}.`);
  return sentences.join(' ');
}

const ZEBRA_SECTION =
  'The zebra herd crossed at dawn. Every zebra kept close; a young zebra lagged and an old zebra ' +
  'turned back for it. We counted zebra after zebra: zebra stripes, zebra foals, one zebra limping, ' +
  'and a lone zebra by the water.';

const LONG_ID = 1; // long; the query word appears only past the cut
const OTHER_ID = 2; // long; never mentions the query word
const SHORT_ID = 3; // short (not chunked); mentions the query word once

const LONG_TITLE = 'Expedition log';
const longBody = (rivers: string) =>
  [
    '# Field notes',
    filler(1, 40),
    '## Weather',
    filler(2, 150),
    '## Rivers',
    rivers,
    '## Markets',
    filler(4, 150),
    '## Zebras',
    ZEBRA_SECTION,
  ].join('\n\n');
const LONG_BODY = longBody(filler(3, 150));
const LONG_BODY_EDITED = longBody(filler(5, 150));

const OTHER_TITLE = 'Harbour survey';
const OTHER_BODY = ['# Harbour', filler(6, 150), '## Tides', filler(7, 150), '## Boats', filler(8, 150), '## Nets', filler(9, 150)].join('\n\n');

const SHORT_TITLE = 'Market day';
const SHORT_BODY = `${filler(10, 50)} Someone mentioned a zebra once.`;

// ---- THE addition: one registry entry --------------------------------------

/** The host's one shared chunk table (the reference DDL), used by every collection. */
const HOST_STORE = sharedChunkStore({ table: `${SCHEMA}.text_chunks` });

const NOTES: ChunkSurface = {
  surface: 'notes',
  parent: { table: `${SCHEMA}.notes`, key: [{ column: 'id', type: 'int' }] },
  textSql: 'p.body',
  headerSql: 'p.title',
  versionSql: 'p.updated_at',
  splitter: { kind: 'markdown', maxChars: 1500 },
  maxChunks: 16,
  parentVector: { column: 'embedding' },
  store: HOST_STORE,
};

const REGISTRY: readonly ChunkSurface[] = [NOTES];

// ---- the generic host (never names a collection) ---------------------------

const hash = (text: string) => createHash('sha256').update(text).digest('hex');

describe('simple addition: a collection registered with only a registry entry (P-008, R-22)', () => {
  let sql: postgres.Sql;
  const emptyKeys = new Set<string>();

  const sync = async (): Promise<ChunkSurfaceSyncStats> => {
    const result = await syncChunkSurfaces(sql, REGISTRY, HOST_STORE, { hash, emptyKeys });
    const stats = result.surfaces.find((s) => s.surface === NOTES.surface);
    if (!stats) throw new Error('the notes surface was not synced');
    return stats;
  };

  const pending = async (): Promise<PendingChunk[]> => {
    const out: PendingChunk[] = [];
    for (const target of chunkEmbedTargets(REGISTRY)) out.push(...(await selectPendingChunks(sql, target, 1000)));
    return out;
  };

  const embedAll = async (): Promise<number> => {
    let written = 0;
    for (const target of chunkEmbedTargets(REGISTRY)) {
      for (let round = 0; round < 10; round++) {
        const r = await embedPendingChunks(sql, target, { embed: embedBatch, limit: 50, mode: 'fixture', profile: 'bag-of-words' });
        written += r.written;
        if (r.selected === 0) break;
      }
    }
    return written;
  };

  const search = (mode: ChunkLegMode, limit: number) =>
    chunkAwareVectorLeg(sql, { surface: NOTES, qVec: vecText(embedText(QUERY_WORD)), limit, mode });

  const chunkRows = (id: number) =>
    sql.unsafe<{ chunk_idx: number; anchor: string | null; header: string | null; content: string; embedding: string | null }[]>(
      `SELECT chunk_idx, anchor, header, content, embedding::text AS embedding
         FROM ${SCHEMA}.text_chunks WHERE surface = $1 AND parent_key = $2::text[] ORDER BY chunk_idx`,
      [NOTES.surface, [String(id)]] as never[],
    );

  /** The host's pre-existing parent vector: left(title \n body, 2000). */
  const parentVectorText = (title: string, body: string) => `${title}\n${body}`.slice(0, DEFAULT_CHUNK_MIN_CHARS);

  beforeAll(async () => {
    sql = postgres(inject('searchPgUrl'), { max: 4, onnotice: () => {} });
    await sql.unsafe('CREATE EXTENSION IF NOT EXISTS vector');
    await sql.unsafe(`CREATE SCHEMA ${SCHEMA}`);
    // The reference migration, applied verbatim into the fixture schema.
    await sql.begin(async (tx) => {
      await tx.unsafe(`SET LOCAL search_path TO ${SCHEMA}, public`);
      await tx.unsafe(REFERENCE_SQL);
    });
    // The collection as it existed before chunking: rows plus a parent vector.
    await sql.unsafe(`CREATE TABLE ${SCHEMA}.notes (
      id int PRIMARY KEY, title text NOT NULL, body text NOT NULL,
      updated_at timestamptz NOT NULL DEFAULT clock_timestamp(), embedding vector(${DIMS}))`);
    for (const [id, title, body] of [
      [LONG_ID, LONG_TITLE, LONG_BODY],
      [OTHER_ID, OTHER_TITLE, OTHER_BODY],
      [SHORT_ID, SHORT_TITLE, SHORT_BODY],
    ] as const) {
      await sql.unsafe(`INSERT INTO ${SCHEMA}.notes (id, title, body, embedding) VALUES ($1, $2, $3, $4::vector)`, [
        id,
        title,
        body,
        vecText(embedText(parentVectorText(title, body))),
      ] as never[]);
    }
  });

  afterAll(async () => {
    if (!sql) return;
    await sql.unsafe(`DROP SCHEMA IF EXISTS ${SCHEMA} CASCADE`).catch(() => undefined);
    await sql.end({ timeout: 5 });
  });

  it('calibration: the fixture separates text past the cut from the parent vector', () => {
    expect(VOCAB.length).toBeGreaterThanOrEqual(30);
    expect(LONG_BODY.length).toBeGreaterThan(DEFAULT_CHUNK_MIN_CHARS);
    expect(OTHER_BODY.length).toBeGreaterThan(DEFAULT_CHUNK_MIN_CHARS);
    expect(SHORT_BODY.length).toBeLessThanOrEqual(DEFAULT_CHUNK_MIN_CHARS);
    // The query word is ONLY past the cut, so the long parent's own vector cannot see it.
    expect(LONG_BODY.indexOf(QUERY_WORD)).toBeGreaterThan(DEFAULT_CHUNK_MIN_CHARS);
    expect(parentVectorText(LONG_TITLE, LONG_BODY)).not.toContain(QUERY_WORD);
    expect(OTHER_BODY).not.toContain(QUERY_WORD);
    expect(parentVectorText(SHORT_TITLE, SHORT_BODY)).toContain(QUERY_WORD);
  });

  it('the sync writes the chunks the registry entry describes, and none for a short parent', async () => {
    const stats = await sync();
    const surface = resolveChunkSurface(NOTES);
    const planLong = planChunks(surface, LONG_BODY, LONG_TITLE, hash).chunks;
    const planOther = planChunks(surface, OTHER_BODY, OTHER_TITLE, hash).chunks;
    expect(planLong.length).toBe(5);
    expect(stats).toMatchObject({ errors: 0, parentsSynced: 2, chunksWritten: planLong.length + planOther.length, parentsTruncatedByMaxChunks: 0 });

    const rows = await chunkRows(LONG_ID);
    expect(rows.map((r) => ({ chunk_idx: r.chunk_idx, anchor: r.anchor, header: r.header, content: r.content }))).toEqual(
      planLong.map((c) => ({ chunk_idx: c.chunkIdx, anchor: c.anchor, header: c.header, content: c.content })),
    );
    expect(rows.find((r) => r.content.includes(QUERY_WORD))?.anchor).toBe('zebras');
    expect(await chunkRows(OTHER_ID)).toHaveLength(planOther.length);
    expect(await chunkRows(SHORT_ID)).toHaveLength(0);
  });

  it('the embed target derived from the registry picks up exactly the new chunks', async () => {
    const targets = chunkEmbedTargets(REGISTRY);
    expect(targets.map((t) => t.table)).toEqual([`${SCHEMA}.text_chunks`]);

    const before = await pending();
    const long = await chunkRows(LONG_ID);
    const other = await chunkRows(OTHER_ID);
    expect(before).toHaveLength(long.length + other.length);
    // The sweep embeds header + content, the text chunk_sha hashes.
    const expectedTexts = [...long, ...other].map((r) => (r.header ? `${r.header}\n${r.content}` : r.content));
    expect(before.map((p) => p.text).sort()).toEqual(expectedTexts.sort());

    expect(await embedAll()).toBe(before.length);
    expect(await pending()).toHaveLength(0);
    expect((await chunkRows(LONG_ID)).every((r) => r.embedding !== null)).toBe(true);
  });

  it("'retrieve' finds the parent by text past the cut; 'gist' does not", async () => {
    const retrieved = await search('retrieve', 1);
    expect(retrieved).toHaveLength(1);
    expect(retrieved[0]).toMatchObject({ id: LONG_ID, matched_anchor: 'zebras' });

    // Gist ranks by the parent vector alone: the short parent's one mention
    // wins, and the long parent (whose vector stops at the cut) is not first.
    const gist = await search('gist', 1);
    expect(gist).toHaveLength(1);
    expect(gist[0]).toMatchObject({ id: SHORT_ID, matched_anchor: null });
    const gistAll = await search('gist', 3);
    expect(gistAll.map((r) => r.id)).toContain(LONG_ID);
    expect(gistAll[0]?.id).not.toBe(LONG_ID);
  });

  it('editing one section re-embeds only that section’s chunk', async () => {
    const beforeRows = await chunkRows(LONG_ID);
    const riversIdx = beforeRows.find((r) => r.anchor === 'rivers')?.chunk_idx;
    expect(riversIdx).toBeDefined();

    await sql.unsafe(`UPDATE ${SCHEMA}.notes SET body = $1, updated_at = clock_timestamp() WHERE id = $2`, [
      LONG_BODY_EDITED,
      LONG_ID,
    ] as never[]);
    const stats = await sync();
    expect(stats).toMatchObject({ errors: 0, parentsSynced: 1, chunksWritten: beforeRows.length, embeddingsReused: beforeRows.length - 1 });

    const waiting = await pending();
    expect(waiting).toHaveLength(1);
    expect(waiting[0]!.key).toEqual({ surface: NOTES.surface, parent_key: [String(LONG_ID)], chunk_idx: riversIdx });

    // Every other chunk kept its vector byte for byte.
    const afterRows = await chunkRows(LONG_ID);
    for (const r of afterRows) {
      if (r.chunk_idx === riversIdx) expect(r.embedding).toBeNull();
      else expect(r.embedding).toBe(beforeRows.find((b) => b.chunk_idx === r.chunk_idx)?.embedding);
    }
    expect(await embedAll()).toBe(1);
    expect(await pending()).toHaveLength(0);
  });

  it("deleting the parent removes its chunks, and search stops returning it", async () => {
    const chunks = (await chunkRows(LONG_ID)).length;
    expect(chunks).toBeGreaterThan(0);
    await sql.unsafe(`DELETE FROM ${SCHEMA}.notes WHERE id = $1`, [LONG_ID] as never[]);
    const stats = await sync();
    expect(stats).toMatchObject({ errors: 0, pruned: chunks });
    expect(await chunkRows(LONG_ID)).toHaveLength(0);
    expect((await chunkRows(OTHER_ID)).length).toBeGreaterThan(0);
    expect((await search('retrieve', 3)).map((r) => r.id)).not.toContain(LONG_ID);
  });
});
