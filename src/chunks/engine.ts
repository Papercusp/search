/**
 * chunks/engine — the generic chunk sync engine (generic-rag-chunking P-004).
 *
 * Generalizes operator-core's turn-chunk-sync: for each registered surface,
 * pick the parents whose chunks are missing or stale, re-split each one, and
 * replace its chunks in one transaction. A re-split COPIES the embedding of
 * any old chunk with the same chunk_sha, so an edit re-embeds only the chunks
 * whose embedded text changed. Parents that were deleted, became ineligible,
 * or shrank below the cut lose their chunks through the store's prune.
 *
 * Failures are per parent and per surface: one bad row or one broken
 * collection never stops the rest, but every failure is LOGGED and COUNTED
 * (a silent catch {} is what hid the harness-doc sync defect
 * EI-24580496022910673). A parent cut off by maxChunks is logged and counted
 * too, so a too-small cap is visible instead of silently losing text.
 */

import type { Sql } from 'postgres';
import { splitMarkdown, splitWindows } from '../chunk';
import type {
  ChunkHash,
  ChunkRowToWrite,
  ChunkStore,
  ChunkSurface,
  ChunkSurfaceSyncStats,
  ChunkSyncLogger,
  ChunkSyncOptions,
  ChunkSyncResult,
  PlannedChunk,
  ResolvedChunkSurface,
} from './types';

/** The parent vector's input window: rows no longer than this are not chunked. */
export const DEFAULT_CHUNK_MIN_CHARS = 2000;
const DEFAULT_BATCH = 50;
const DEFAULT_PRUNE_BATCH = 500;
/** Separator between header and text in the parent sha, and between key parts. chr(31) is legal in PG text; chr(0) is not. */
export const CHUNK_KEY_SEP = '\u001f';

const IDENT = /^[a-z_][a-z0-9_]*$/;
const QUALIFIED_IDENT = /^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)?$/;
const SQL_TYPE = /^[a-z_][a-z0-9_ ]*(\[\])?$/;

const consoleLogger: ChunkSyncLogger = {
  warn: (message, meta) => console.warn(`[chunk-sync] ${message}`, meta ?? {}),
  error: (message, meta) => console.error(`[chunk-sync] ${message}`, meta ?? {}),
};

function positiveInt(n: unknown): boolean {
  return typeof n === 'number' && Number.isInteger(n) && n > 0;
}

/**
 * The version string stored with every chunk. It changes whenever the same
 * text could be cut differently, so editing a surface's splitter or cap in the
 * registry re-cuts its parents instead of trusting chunks cut the old way.
 */
export function splitterVersionOf(surface: Pick<ChunkSurface, 'splitter' | 'maxChunks'>): string {
  const s = surface.splitter;
  if (s.kind === 'window') return `window-v1:${s.size}/${s.overlap}@${surface.maxChunks}`;
  return `markdown-v1:${s.maxChars}/${s.minChars ?? 20}/${s.headingDepth ?? 3}@${surface.maxChunks}`;
}

/**
 * Validate a registry entry and apply its defaults. Throws on a malformed
 * entry: a registry mistake is a programming error, caught by the host's
 * registry test rather than discovered at sync time.
 */
export function resolveChunkSurface(surface: ChunkSurface): ResolvedChunkSurface {
  const where = `chunk surface '${surface.surface}'`;
  if (typeof surface.surface !== 'string' || surface.surface.trim() === '') {
    throw new Error('chunk surface: surface name must be a non-empty string');
  }
  if (!QUALIFIED_IDENT.test(surface.parent.table)) {
    throw new Error(`${where}: parent.table '${surface.parent.table}' is not a plain [schema.]table identifier`);
  }
  if (surface.parent.key.length === 0) throw new Error(`${where}: parent.key must name at least one column`);
  const keyColumns = surface.parent.key.map((k) => {
    const column = typeof k === 'string' ? k : k.column;
    const type = typeof k === 'string' ? null : k.type.trim().toLowerCase();
    if (!IDENT.test(column)) throw new Error(`${where}: key column '${column}' is not a plain identifier`);
    if (type !== null && !SQL_TYPE.test(type)) throw new Error(`${where}: key type '${type}' is not a plain SQL type name`);
    return { column, type };
  });
  for (const [field, value] of [
    ['textSql', surface.textSql],
    ['headerSql', surface.headerSql],
    ['eligibleSql', surface.eligibleSql],
    ['versionSql', surface.versionSql],
  ] as const) {
    if (value === undefined) continue;
    if (typeof value !== 'string' || value.trim() === '') throw new Error(`${where}: ${field} must be a non-empty SQL expression`);
    if (value.includes(';')) throw new Error(`${where}: ${field} must be a single expression (no ';')`);
  }
  if (!positiveInt(surface.maxChunks)) throw new Error(`${where}: maxChunks must be a positive integer`);
  const s = surface.splitter;
  if (s.kind === 'window') {
    if (!positiveInt(s.size)) throw new Error(`${where}: window size must be a positive integer`);
    if (!Number.isInteger(s.overlap) || s.overlap < 0 || s.overlap >= s.size) {
      throw new Error(`${where}: window overlap must be an integer in [0, size)`);
    }
  } else if (s.kind === 'markdown') {
    if (!positiveInt(s.maxChars)) throw new Error(`${where}: markdown maxChars must be a positive integer`);
  } else {
    throw new Error(`${where}: unknown splitter kind`);
  }
  const minChars = surface.minChars ?? DEFAULT_CHUNK_MIN_CHARS;
  if (!Number.isInteger(minChars) || minChars < 0) throw new Error(`${where}: minChars must be a non-negative integer`);
  return { ...surface, minChars, keyColumns, splitterVersion: splitterVersionOf(surface) };
}

/**
 * The parent sha: hash(text) when there is no header, else
 * hash(header + U+001F + text). A header edit therefore re-cuts too (every
 * chunk embeds it). With no header this equals a plain text sha, which is what
 * turn-chunk-sync stores as turn_sha. The shared store's SQL form computes the
 * same value, so `hash` must be sha256 hex for unversioned surfaces.
 */
export function parentShaOf(hash: ChunkHash, text: string, header: string | null): string {
  return header ? hash(`${header}${CHUNK_KEY_SEP}${text}`) : hash(text);
}

/** The text actually embedded for a chunk. */
export function embeddedChunkText(header: string | null, content: string): string {
  return header ? `${header}\n${content}` : content;
}

/** The emptyKeys entry for a parent. */
export function chunkParentKeyId(surface: string, key: readonly string[]): string {
  return [surface, ...key].join(CHUNK_KEY_SEP);
}

export interface ChunkPlan {
  chunks: PlannedChunk[];
  /** More chunks existed than maxChunks allowed; the tail was dropped. */
  truncated: boolean;
  /** Chunks the splitter produced before the cap. */
  producedChunks: number;
}

/** Split one parent into chunks. Pure. */
export function planChunks(
  surface: Pick<ChunkSurface, 'splitter' | 'maxChunks'>,
  text: string,
  header: string | null,
  hash: ChunkHash,
): ChunkPlan {
  const s = surface.splitter;
  const cap = surface.maxChunks;
  const headerLine = header && header.trim() !== '' ? header : null;
  let pieces: { anchor: string | null; header: string | null; content: string }[];
  let produced: number;
  if (s.kind === 'window') {
    // Ask for one more window than the cap: getting it back is how truncation is detected.
    const windows = splitWindows(text, { size: s.size, overlap: s.overlap, maxChunks: cap + 1 });
    produced = windows.length;
    pieces = windows.slice(0, cap).map((content) => ({ anchor: null, header: headerLine, content }));
  } else {
    const sections = splitMarkdown(text, {
      maxChars: s.maxChars,
      ...(s.minChars !== undefined ? { minChars: s.minChars } : {}),
      ...(s.headingDepth !== undefined ? { headingDepth: s.headingDepth } : {}),
    });
    produced = sections.length;
    pieces = sections.slice(0, cap).map((sec) => {
      const context = [headerLine, ...sec.headingPath].filter((x): x is string => !!x && x.trim() !== '');
      return { anchor: sec.anchor, header: context.length > 0 ? context.join(' › ') : null, content: sec.content };
    });
  }
  const chunks = pieces.map((p, chunkIdx) => ({
    chunkIdx,
    anchor: p.anchor,
    header: p.header,
    content: p.content,
    chunkSha: hash(embeddedChunkText(p.header, p.content)),
  }));
  return { chunks, truncated: produced > cap, producedChunks: produced };
}

function emptyStats(surface: string): ChunkSurfaceSyncStats {
  return {
    surface,
    parentsSynced: 0,
    parentsUnchanged: 0,
    parentsEmpty: 0,
    chunksWritten: 0,
    embeddingsReused: 0,
    pruned: 0,
    parentsTruncatedByMaxChunks: 0,
    errors: 0,
    more: false,
  };
}

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * Sync one surface: re-chunk up to `batch` stale parents, then prune.
 * Never throws; failures are logged and counted in the returned stats.
 */
export async function syncChunkSurface(
  sql: Sql,
  surfaceIn: ChunkSurface,
  store: ChunkStore,
  opts: ChunkSyncOptions & { deadline?: number },
): Promise<ChunkSurfaceSyncStats> {
  const logger = opts.logger ?? consoleLogger;
  const now = opts.now ?? Date.now;
  const stats = emptyStats(surfaceIn.surface);
  let surface: ResolvedChunkSurface;
  try {
    surface = resolveChunkSurface(surfaceIn);
  } catch (e) {
    stats.errors++;
    logger.error('invalid chunk surface', { surface: surfaceIn.surface, error: errText(e) });
    return stats;
  }
  const batch = opts.batchPerSurface ?? DEFAULT_BATCH;
  const emptyKeys = opts.emptyKeys ?? new Set<string>();
  const prefix = surface.surface + CHUNK_KEY_SEP;
  const excludeKeys = [...emptyKeys].filter((k) => k.startsWith(prefix)).map((k) => k.slice(prefix.length));

  let stale;
  try {
    stale = await store.selectStale(sql, surface, { limit: batch, excludeKeys });
  } catch (e) {
    stats.errors++;
    logger.error('chunk sync: selecting stale parents failed', { surface: surface.surface, store: store.name, error: errText(e) });
    return stats;
  }
  stats.more = stale.length >= batch;

  for (const parent of stale) {
    if (opts.deadline !== undefined && now() >= opts.deadline) {
      stats.more = true;
      break;
    }
    try {
      const parentSha = parentShaOf(opts.hash, parent.text, parent.header);
      await store.transaction(sql, async (tx) => {
        const existing = await store.readExisting(tx, surface, parent.key);
        if (
          existing.length > 0 &&
          existing.every((c) => c.parentSha === parentSha && c.splitterVersion === surface.splitterVersion)
        ) {
          await store.touch(tx, surface, parent.key, parent.version);
          stats.parentsUnchanged++;
          return;
        }
        const plan = planChunks(surface, parent.text, parent.header, opts.hash);
        if (plan.truncated) {
          stats.parentsTruncatedByMaxChunks++;
          logger.warn('chunk sync: parent truncated by maxChunks', {
            surface: surface.surface,
            key: parent.key,
            maxChunks: surface.maxChunks,
            producedChunks: plan.producedChunks,
            textChars: parent.text.length,
          });
        }
        const reuse = new Map<string, (typeof existing)[number]>();
        for (const c of existing) if (c.embedding !== null && !reuse.has(c.chunkSha)) reuse.set(c.chunkSha, c);
        const rows: ChunkRowToWrite[] = plan.chunks.map((c) => {
          const old = reuse.get(c.chunkSha);
          if (old) stats.embeddingsReused++;
          return {
            ...c,
            embedding: old?.embedding ?? null,
            embeddingMode: old?.embeddingMode ?? null,
            embeddingProfile: old?.embeddingProfile ?? null,
          };
        });
        await store.replace(tx, surface, parent.key, rows, {
          parentSha,
          splitterVersion: surface.splitterVersion,
          version: parent.version,
        });
        if (rows.length === 0) {
          stats.parentsEmpty++;
          emptyKeys.add(chunkParentKeyId(surface.surface, parent.key));
        } else {
          stats.parentsSynced++;
          stats.chunksWritten += rows.length;
        }
      });
    } catch (e) {
      stats.errors++;
      logger.error('chunk sync: parent failed', { surface: surface.surface, key: parent.key, error: errText(e) });
    }
  }

  const pruneBatch = opts.pruneBatch ?? DEFAULT_PRUNE_BATCH;
  if (pruneBatch > 0 && (opts.deadline === undefined || now() < opts.deadline)) {
    try {
      const pruned = await store.prune(sql, surface, { limit: pruneBatch });
      stats.pruned = pruned.chunks;
    } catch (e) {
      stats.errors++;
      logger.error('chunk sync: prune failed', { surface: surface.surface, store: store.name, error: errText(e) });
    }
  }
  return stats;
}

/**
 * Sync every surface once, round-robin from `startIndex`, within an optional
 * time budget. Never throws. The caller persists `nextIndex` (and emptyKeys)
 * between calls.
 */
export async function syncChunkSurfaces(
  sql: Sql,
  surfaces: readonly ChunkSurface[],
  defaultStore: ChunkStore,
  opts: ChunkSyncOptions,
): Promise<ChunkSyncResult> {
  const now = opts.now ?? Date.now;
  const n = surfaces.length;
  const result: ChunkSyncResult = { surfaces: [], nextIndex: 0, skippedForBudget: 0 };
  if (n === 0) return result;
  const start = (((opts.startIndex ?? 0) % n) + n) % n;
  const deadline = opts.timeBudgetMs !== undefined ? now() + opts.timeBudgetMs : undefined;
  for (let i = 0; i < n; i++) {
    const idx = (start + i) % n;
    if (deadline !== undefined && now() >= deadline) {
      result.skippedForBudget = n - i;
      result.nextIndex = idx;
      return result;
    }
    const surface = surfaces[idx]!;
    result.surfaces.push(await syncChunkSurface(sql, surface, surface.store ?? defaultStore, { ...opts, deadline }));
  }
  // Everyone ran: rotate by one so the same surface does not always go first.
  result.nextIndex = (start + 1) % n;
  return result;
}
