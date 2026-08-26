/**
 * embed-latency.ts — turn query-embed LATENCY from a per-call prose aside into
 * a first-class, per-caller p99-vs-budget measurement (EI-21491088289861649).
 *
 * WHY THIS EXISTS. The retrieval-degradation watchdog reads leg OUTCOMES: a
 * search either ran at full strength or degraded. That misses the failure shape
 * that actually occurs most often — an embed that is SLOW but still succeeds.
 * A cold sidecar answers in ~1.9s while perfectly healthy (measured on this
 * box: try1 1.92s cold, then ~1ms warm), which sails under one caller's 4000ms
 * budget and blows another caller's 1200ms budget at the same instant. Today
 * such a breach surfaces only INDIRECTLY, always dressed up as something else:
 * a wrong consult verdict ("no one knows more than you do"), a degraded
 * semantic leg ("query embed exceeded Nms budget"), a thin corpus. Every one of
 * those reports a fact about the SUBJECT; the truth is a fact about the
 * INSTRUMENT. This module is where the instrument finally measures itself.
 *
 * SHAPE. A bounded ring buffer written by the hybrid engine's embed choke point
 * (`embedWithBudget` in ./hybrid), read by the embed-latency watchdog in
 * operator-core — the same producer/consumer split as leg-health, for the same
 * reason: the writer must be synchronous, allocation-free-ish and never able to
 * fail a search, and the reader must be a separate concern that can be armed,
 * gated and escalated independently.
 *
 * SCOPE HONESTY — identical caveat to leg-health. This is IN-PROCESS state: it
 * measures the embeds THIS process ran. Quiet means "this process saw no
 * breach", never "the sidecar is healthy everywhere".
 */
import { pinModuleState } from '@papercusp/module-singleton';

/** How many recent embeds are retained. Bounded on purpose — same rationale as
 *  LEG_HEALTH_CAPACITY: a health signal, not an audit log. 512 covers roughly
 *  an hour of busy fleet traffic per process; the truncation flag keeps the
 *  read honest when traffic outruns it. */
export const EMBED_LATENCY_CAPACITY = 512;

/** How an observed embed ended. `timeout` means the CALLER's budget expired
 *  (EmbedTimeoutError) — the instrument was graded against its budget and lost;
 *  `error` means the embedder rejected on its own. A caller-side ABORT is
 *  deliberately NOT recorded: the caller stopped listening, which says nothing
 *  about embed latency, and letting cancellations into the window would poison
 *  p99 with noise that has no remedy. */
export type EmbedLatencyOutcome = 'ok' | 'timeout' | 'error';

/** One recorded embed. Written on every query embed — deliberately tiny. */
export interface EmbedLatencySample {
  atMs: number;
  /** Who issued the embed (e.g. 'search:semantic', 'midturn:related-context').
   *  'unattributed' when a caller did not label itself — real, counted, and
   *  visible as such, so unlabeled call sites show up AS unlabeled instead of
   *  silently vanishing into someone else's numbers. */
  caller: string;
  /** The budget this embed was graded against, null when unbounded. */
  budgetMs: number | null;
  durationMs: number;
  outcome: EmbedLatencyOutcome;
}

/** Per-caller trailing-window aggregate. Percentiles are NEAREST-RANK over the
 *  retained durations (all outcomes — a timed-out embed really consumed ~its
 *  budget of wall time, so excluding it would understate exactly the tail this
 *  exists to see). `budgetMs` is the most recent non-null budget the caller was
 *  seen grading against; null when every sample in the window was unbounded. */
export interface EmbedCallerLatency {
  caller: string;
  n: number;
  ok: number;
  timeout: number;
  error: number;
  budgetMs: number | null;
  p50Ms: number | null;
  p99Ms: number | null;
  maxMs: number | null;
}

/** The aggregate over a trailing window, across callers.
 *  `truncatedByCapacity`: the ring evicted older samples — counts are floors. */
export interface EmbedLatencyWindow {
  windowMs: number;
  embeds: number;
  callers: EmbedCallerLatency[];
  oldestSampleMs: number | null;
  newestSampleMs: number | null;
  truncatedByCapacity: boolean;
}

interface EmbedLatencyState {
  ring: (EmbedLatencySample | undefined)[];
  next: number;
  count: number;
}

const state = pinModuleState<EmbedLatencyState>('@papercusp/search.embed-latency', () => ({
  ring: new Array<EmbedLatencySample | undefined>(EMBED_LATENCY_CAPACITY),
  next: 0,
  count: 0,
}));

/**
 * Nearest-rank percentile over an ASCENDING numeric slice. rank = ceil(p/100·N)
 * clamped to [1, N]; empty ⇒ null (never 0, which would impersonate "fast").
 */
function percentileSorted(sortedAsc: readonly number[], p: number): number | null {
  const n = sortedAsc.length;
  if (n === 0) return null;
  const rank = Math.min(n, Math.max(1, Math.ceil((p / 100) * n)));
  return sortedAsc[rank - 1] ?? null;
}

/** Caller label used when a call site did not identify itself. Exported so the
 *  judge can treat unattributed traffic explicitly rather than guessing. */
export const UNATTRIBUTED_CALLER = 'unattributed';

/**
 * PURE: derive the per-caller window from an explicit sample list. Split from
 * the buffer so the arithmetic (grouping, budgets, percentiles) is testable
 * with no global state — mirrors summariseLegSamples.
 */
export function summariseEmbedSamples(
  samples: readonly EmbedLatencySample[],
  windowMs: number,
  nowMs: number,
  capacityReached: boolean,
): EmbedLatencyWindow {
  const since = nowMs - windowMs;
  const inWindow = samples.filter((s) => s.atMs >= since);

  // Group by caller, remembering time order inside each group (samples arrive
  // appended in time order, so filter() preserves it).
  const byCaller = new Map<string, EmbedLatencySample[]>();
  for (const s of inWindow) {
    const list = byCaller.get(s.caller);
    if (list) list.push(s);
    else byCaller.set(s.caller, [s]);
  }

  const callers: EmbedCallerLatency[] = [...byCaller.entries()]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([caller, list]) => {
      let budgetMs: number | null = null;
      for (const s of list) {
        if (s.budgetMs !== null) budgetMs = s.budgetMs; // last non-null wins
      }
      const sorted = list.map((s) => s.durationMs).sort((a, b) => a - b);
      return {
        caller,
        n: list.length,
        ok: list.filter((s) => s.outcome === 'ok').length,
        timeout: list.filter((s) => s.outcome === 'timeout').length,
        error: list.filter((s) => s.outcome === 'error').length,
        budgetMs,
        p50Ms: percentileSorted(sorted, 50),
        p99Ms: percentileSorted(sorted, 99),
        maxMs: sorted.length > 0 ? (sorted[sorted.length - 1] ?? null) : null,
      };
    });

  const oldest = inWindow.length > 0 ? Math.min(...inWindow.map((s) => s.atMs)) : null;
  // Same honesty rule as summariseLegSamples: a full ring whose entire retained
  // sample sits inside the window means older samples were EVICTED — the read
  // is a floor, not a total.
  const truncatedByCapacity =
    capacityReached && inWindow.length === samples.length && oldest !== null && since < oldest;

  return {
    windowMs,
    embeds: inWindow.length,
    callers,
    oldestSampleMs: oldest,
    newestSampleMs: inWindow.length > 0 ? Math.max(...inWindow.map((s) => s.atMs)) : null,
    truncatedByCapacity,
  };
}

/**
 * Record one finished embed. Never throws: an instrumentation fault must not
 * fail a search — same contract as observeLegs.
 */
export function observeEmbedLatency(sample: EmbedLatencySample): void {
  try {
    state.ring[state.next] = sample;
    state.next = (state.next + 1) % EMBED_LATENCY_CAPACITY;
    state.count += 1;
  } catch {
    // Deliberately silent: a health counter is never worth a failed embed.
  }
}

/** Read the trailing-window aggregate. Default window: 1 hour (matches the
 *  retrieval-degradation sweep's window so the two verdicts overlay). */
export function readEmbedLatency(opts: { windowMs?: number; nowMs?: number } = {}): EmbedLatencyWindow {
  const windowMs = opts.windowMs ?? 60 * 60_000;
  const nowMs = opts.nowMs ?? Date.now();
  const samples: EmbedLatencySample[] = [];
  for (const s of state.ring) {
    if (s != null) samples.push(s);
  }
  return summariseEmbedSamples(samples, windowMs, nowMs, state.count >= EMBED_LATENCY_CAPACITY);
}

/** Total embeds ever observed by this process (monotonic). */
export function embedLatencyObservedCount(): number {
  return state.count;
}

/** TEST SEAM: drop every retained sample. */
export function resetEmbedLatency(): void {
  state.ring = new Array<EmbedLatencySample | undefined>(EMBED_LATENCY_CAPACITY);
  state.next = 0;
  state.count = 0;
}
