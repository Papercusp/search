/**
 * ENGINE-LEVEL SEARCH DEFAULTS — the opt-OUT seam (P-017).
 *
 * The problem this exists to kill: every ranking feature this engine grew
 * (`minScore` floors, recency re-rank) landed as an OPTIONAL per-call field,
 * so a capability reached exactly the one surface whose bug prompted it and
 * had to be hand-propagated to every other call site — which never happened.
 * Measured 2026-08-03 across the 8 real `runHybridSearch` call sites: ONE
 * passed `minScore`, ONE passed `recency`. The other six inherited nothing,
 * including the fix for the owner-reported "search returns noise" bug.
 *
 * The inversion: a host registers ONE policy here, and every search inherits
 * it. A caller that genuinely needs different behavior still wins — it passes
 * an explicit value — and a caller that must have the feature OFF says so
 * explicitly with `false`. That three-way distinction is the whole point:
 *
 *   | the caller passes | what the engine does           |
 *   |-------------------|--------------------------------|
 *   | a value           | uses it (caller always wins)   |
 *   | `false`           | disables the feature outright  |
 *   | nothing           | applies the registered DEFAULT |
 *
 * `undefined` therefore had to stop meaning "off" and start meaning
 * "unspecified" — which is why `false` exists rather than this being a plain
 * optional. Without a distinct off-token there is no way to express "I really
 * do want no floor" once a default is registered.
 *
 * This module is deliberately domain-free. It holds no thresholds, no model
 * names and no notion of what a "prose" surface is: the host supplies all of
 * that. The resolvers are SYNCHRONOUS by design — they run on the search hot
 * path, and a host that needs async state must resolve it before registering
 * (see `embedderModeOf` on the papercusp side, which binds the mode to the
 * embedder instance so the resolver is a WeakMap lookup).
 */

import type { MinScoreFloors } from './min-score';
import type { RecencyRank } from './recency';
import type { Embedder } from './types';

/**
 * What a defaults resolver is told about the search being run.
 *
 * `embedder` is the INSTANCE the caller supplied, not a description of it.
 * That matters: a floor is only meaningful in the embedding space it was
 * measured in, so a host resolver must be able to tell "this is the embedder
 * I calibrated against" from "this is some other one". Passing the instance
 * lets the host answer that by identity, and a host that does not recognise
 * an embedder can safely return `undefined` (⇒ no floor) instead of applying
 * a threshold from a foreign space — which would silently delete real hits.
 */
export interface SearchDefaultsContext {
  query: string;
  /** Final page size the caller asked for. */
  limit: number;
  /** Which legs this search runs. `fulltext` has no vector leg at all. */
  mode: 'fulltext' | 'embeddings' | 'hybrid';
  /**
   * The embedder that will produce the query vector, or null when there is
   * none (a `fulltext` search, or a caller that resolved no embedder). This
   * is the instance itself — see the note above on why identity matters.
   *
   * NOTE this is the embedder the search was GIVEN, which is not yet proof
   * that a vector was produced: the embed can still fail or exceed its
   * budget and degrade the search to BM25-only. That is deliberately not
   * awaited here — defaults resolve once, up front, before the legs run.
   * It is also harmless: if the vector leg never runs there are no embedding
   * hits for an embedding floor to filter.
   */
  embedder: Embedder | null;
}

/**
 * The host's ranking policy. Every member is optional — a host registers only
 * the features it has actually measured a default for. Returning `undefined`
 * from a resolver means "no default for this search", which leaves the engine
 * byte-identical to having no policy registered at all.
 */
export interface SearchDefaultsHost {
  /**
   * Per-ranker score floors to apply when the caller did not specify any.
   * Return `undefined` to floor nothing — the correct answer whenever the
   * search is running in an embedding space you have not calibrated.
   */
  minScore?(ctx: SearchDefaultsContext): MinScoreFloors | undefined;
  /**
   * Recency re-rank to apply when the caller did not specify one. Return
   * `undefined` for pure-relevance ranking.
   */
  recency?(ctx: SearchDefaultsContext): RecencyRank | undefined;
}

let host: SearchDefaultsHost | null = null;

/**
 * Register the engine-wide ranking policy. Call once, at boot, before any
 * search runs. Registering again REPLACES the previous host outright (rather
 * than merging) so a partial re-registration can never leave a half-applied
 * policy that is hard to reason about.
 */
export function configureSearchDefaults(policy: SearchDefaultsHost): void {
  host = policy;
}

/** Drop any registered policy — restores the un-configured engine. For tests. */
export function resetSearchDefaults(): void {
  host = null;
}

/** The currently registered policy, or null. Exposed for assertions/debug. */
export function searchDefaultsHost(): SearchDefaultsHost | null {
  return host;
}

/**
 * Resolve one feature to the value the engine should actually use.
 *
 * `explicit` is exactly what the caller put in the `SearchContext`:
 * a value (wins), `false` (hard off), or `undefined` (fall through to the
 * registered default).
 *
 * A throwing host resolver degrades to "no default" rather than failing the
 * search — a misconfigured policy must not be able to take search down. This
 * mirrors the engine's existing rule that a broken source is skipped, not
 * fatal.
 */
export function resolveDefault<T>(
  explicit: T | false | undefined,
  resolver: ((ctx: SearchDefaultsContext) => T | undefined) | undefined,
  ctx: SearchDefaultsContext,
): T | undefined {
  if (explicit === false) return undefined;
  if (explicit !== undefined) return explicit;
  if (!resolver) return undefined;
  try {
    return resolver(ctx);
  } catch {
    return undefined;
  }
}

/**
 * What the engine actually applied for this search, after merging caller
 * intent with the registered defaults.
 *
 * Reported back on every result so a caller — or a conformance test — can see
 * which features really ran instead of assuming. A silently-absent feature is
 * precisely the failure mode P-017 exists to end, so the engine states it.
 */
export interface AppliedDefaults {
  /** The floors in force, or null when nothing was floored. */
  minScore: MinScoreFloors | null;
  /** True iff a recency re-rank was applied to the candidate pool. */
  recency: boolean;
  /** True iff a host policy was registered when this search ran. */
  policyRegistered: boolean;
}

/** Resolve the whole policy for one search in a single pass. */
export function resolveSearchDefaults(
  ctx: SearchDefaultsContext,
  explicit: { minScore?: MinScoreFloors | false; recency?: RecencyRank | false },
): { minScore: MinScoreFloors | undefined; recency: RecencyRank | undefined; applied: AppliedDefaults } {
  const policy = host;
  const minScore = resolveDefault(explicit.minScore, policy?.minScore, ctx);
  const recency = resolveDefault(explicit.recency, policy?.recency, ctx);
  return {
    minScore,
    recency,
    applied: {
      minScore: minScore ?? null,
      recency: !!recency,
      policyRegistered: policy !== null,
    },
  };
}
