/**
 * Stored-row self-check: does the active embedder still reproduce what is on
 * disk?
 *
 * Take one real row already labelled as living in the active embedding space,
 * re-embed its own text with the active embedder, and compare the fresh vector
 * with the stored one. Identical text under an unchanged model lands at ~0
 * cosine distance; a stale label, a model or config drift, or a vector written
 * by the wrong embedder lands far away. Ranking against such vectors returns
 * noise without any error, so the check turns that silent failure into an
 * alert.
 *
 * A fixed canary string would only prove the embedder agrees with itself call
 * to call. Re-embedding a STORED row tests the invariant search relies on:
 * a vector, under its claimed label, is what the active embedder produces for
 * its text today.
 *
 * Everything host-specific is injected: the embedder, how a canary row is
 * found, and where an alert goes. The tick never throws; a failing dependency
 * becomes a `skipped` result and the next scheduled tick retries.
 *
 * Extracted from papercusp's embed-space-self-check
 * (shared-vector-search-libraries-2026-09-29, P-001).
 */

/** Below this cosine distance a re-embedding counts as the same vector. A
 * genuine desync (wrong space, foreign vector, width mismatch) lands far above
 * it; float noise and minor backend nondeterminism land far below. */
export const DEFAULT_DESYNC_DISTANCE_THRESHOLD = 0.05;

/** One already-embedded row in the active space. */
export interface SelfCheckCanary {
  /** The row's own text, exactly as it was embedded. */
  body: string;
  /** The stored vector, as numbers or as pgvector's `::text` literal. */
  storedVector: readonly number[] | string;
  /** Where the row lives, for messages (e.g. `schema.table#key`). */
  keyLabel: string;
}

/** A measured comparison, handed to the alert and clear sinks. */
export interface SelfCheckReading {
  canary: SelfCheckCanary;
  distance: number;
  threshold: number;
}

/**
 * How long a healthy pass stays trusted while none of its inputs change.
 * After this, the check re-embeds even when everything looks the same. That
 * covers drift no input can show, such as model files replaced under a running
 * process.
 */
export const DEFAULT_SELF_CHECK_MAX_AGE_MS = 6 * 60 * 60 * 1000;

/**
 * The last HEALTHY pass, kept by the host between ticks.
 *
 * Re-embedding the same text with the same embedder and comparing it with the
 * same stored vector can only repeat the previous answer. A scheduled tick that
 * does it anyway loads the embedding model on a host that is otherwise idle,
 * and that model is the largest piece of memory such a host holds. Measured on
 * a capacity VM (2026-10-02, WI-10005523): a 15-minute self-check tick
 * reloaded the model on every idle Server, so an idle unload could never stay
 * unloaded.
 */
export interface SelfCheckMemo {
  last: {
    embedderIdentity: string;
    keyLabel: string;
    body: string;
    storedVector: string;
    distance: number;
    atMs: number;
  } | null;
}

export function createSelfCheckMemo(): SelfCheckMemo {
  return { last: null };
}

export interface StoredRowSelfCheckDeps {
  /** The active embedder. */
  embed(text: string): Promise<readonly number[]>;
  /** One row in the active space, or null when none exists yet. */
  pickCanary(): Promise<SelfCheckCanary | null>;
  /** Called when the re-embedding is further than `threshold` from disk. */
  alert(reading: SelfCheckReading): Promise<unknown> | unknown;
  /** Called when it is within `threshold` (e.g. to close an earlier alert). */
  clear?(reading: SelfCheckReading): Promise<unknown> | unknown;
  /** Cosine-distance alert threshold. Default {@link DEFAULT_DESYNC_DISTANCE_THRESHOLD}. */
  threshold?: number;
  /**
   * Everything that decides the embedder's output space (backend, model,
   * revision, width). Must change whenever the vectors it produces could
   * change. With `memo`, it turns on the unchanged-inputs skip; without it,
   * every tick re-embeds.
   */
  embedderIdentity?: string;
  /** Last healthy pass, owned by the host. See {@link SelfCheckMemo}. */
  memo?: SelfCheckMemo;
  /** Default {@link DEFAULT_SELF_CHECK_MAX_AGE_MS}. */
  maxAgeMs?: number;
  /** Clock, for tests. */
  now?: () => number;
}

export type StoredRowSelfCheckResult =
  | { ok: true; status: 'healthy' | 'desync'; keyLabel: string; distance: number }
  | { ok: true; skipped: 'no_row_in_active_space' }
  | {
      ok: true;
      /** Same embedder, same row, same text, same stored vector as the last
       * healthy pass, within `maxAgeMs`. Nothing was embedded and no sink ran;
       * `distance` is the remembered reading. */
      skipped: 'unchanged_since_last_pass';
      keyLabel: string;
      distance: number;
      checkedAtMs: number;
    }
  | { ok: false; skipped: 'canary_read_failed' | 'embed_failed'; keyLabel?: string };

/** Parse a pgvector `::text` literal ("[0.1,0.2,...]") into numbers. */
export function parseVectorText(v: string): number[] {
  return v
    .slice(v.indexOf('[') + 1, v.lastIndexOf(']'))
    .split(',')
    .map(Number);
}

/**
 * Cosine DISTANCE (1 − cosine similarity), matching pgvector's `<=>`. A length
 * mismatch or a zero vector is itself a desync signal, so it returns the
 * maximal distance 1 instead of throwing.
 */
export function cosineDistance(a: readonly number[], b: readonly number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 1;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  if (na === 0 || nb === 0) return 1;
  return 1 - dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/** The alert rule: strictly further than the threshold. Equal is healthy. */
export function isEmbeddingDesync(distance: number, threshold: number = DEFAULT_DESYNC_DISTANCE_THRESHOLD): boolean {
  return !(distance <= threshold);
}

function storedVectorKey(v: SelfCheckCanary['storedVector']): string {
  return typeof v === 'string' ? v : JSON.stringify(v);
}

/** One self-check tick. Never throws. */
export async function runStoredRowSelfCheck(deps: StoredRowSelfCheckDeps): Promise<StoredRowSelfCheckResult> {
  const threshold = deps.threshold ?? DEFAULT_DESYNC_DISTANCE_THRESHOLD;
  const now = deps.now ?? Date.now;
  const memo = deps.embedderIdentity === undefined ? undefined : deps.memo;

  let canary: SelfCheckCanary | null;
  try {
    canary = await deps.pickCanary();
  } catch {
    return { ok: false, skipped: 'canary_read_failed' };
  }
  if (!canary) return { ok: true, skipped: 'no_row_in_active_space' };

  const storedKey = storedVectorKey(canary.storedVector);
  const last = memo?.last;
  if (
    last &&
    last.embedderIdentity === deps.embedderIdentity &&
    last.keyLabel === canary.keyLabel &&
    last.body === canary.body &&
    last.storedVector === storedKey &&
    now() - last.atMs < (deps.maxAgeMs ?? DEFAULT_SELF_CHECK_MAX_AGE_MS)
  ) {
    return {
      ok: true,
      skipped: 'unchanged_since_last_pass',
      keyLabel: canary.keyLabel,
      distance: last.distance,
      checkedAtMs: last.atMs,
    };
  }

  let fresh: readonly number[];
  try {
    fresh = await deps.embed(canary.body);
  } catch {
    if (memo) memo.last = null;
    return { ok: false, skipped: 'embed_failed', keyLabel: canary.keyLabel };
  }

  const stored = typeof canary.storedVector === 'string' ? parseVectorText(canary.storedVector) : canary.storedVector;
  const distance = cosineDistance(fresh, stored);
  const reading: SelfCheckReading = { canary, distance, threshold };
  const desync = isEmbeddingDesync(distance, threshold);
  try {
    await (desync ? deps.alert(reading) : deps.clear?.(reading));
  } catch {
    /* a sink failure must never fail the tick */
  }
  // Only a healthy pass is remembered: after a desync every tick re-checks, so
  // the clear sink runs as soon as the space is healthy again.
  if (memo) {
    memo.last = desync
      ? null
      : {
          embedderIdentity: deps.embedderIdentity!,
          keyLabel: canary.keyLabel,
          body: canary.body,
          storedVector: storedKey,
          distance,
          atMs: now(),
        };
  }
  return { ok: true, status: desync ? 'desync' : 'healthy', keyLabel: canary.keyLabel, distance };
}
