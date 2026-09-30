/**
 * near-duplicate.ts — a corpus-calibrated near-duplicate check
 * (shared-vector-search-libraries-2026-09-29 P-004, moved out of papercusp's
 * plans/semantic-dedup.ts).
 *
 * The shape: a host already has a FIXED list of candidates it suspects are
 * duplicates of a new document (from a cheap token matcher, a title match, …)
 * and the cosine similarity of each candidate's stored vector to the new
 * document. This module partitions that list at a cut: candidates at or above
 * the cut are KEPT as likely duplicates, candidates below it are DROPPED as
 * merely topically adjacent.
 *
 * This is a THRESHOLD CLASSIFIER in the document-to-document pairing, not a
 * ranked search in the query-to-document pairing. A search floor (the minimum
 * score a query hit needs) does not license a duplicate cut, or the reverse:
 * the two compare different kinds of text.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * WHY THE CUT IS CORPUS-RELATIVE, NOT AN ABSOLUTE COSINE
 *
 * An absolute cosine cut on a dual-encoder is meaningful against ONE embedding
 * procedure only. When pooling, normalization or a prompt prefix changes, the
 * whole similarity scale shifts (typically upward) and a hand-picked constant
 * silently stops separating anything: in the corpus this was extracted from,
 * a 0.6 cut calibrated against 0.46–0.49 controls later sat below EVERY pair
 * in the corpus, so nothing could ever be dropped.
 *
 * The cut is therefore taken from the new document's OWN similarity
 * distribution against a background sample of the same corpus: a candidate is
 * kept only when it lands in the top `1 - quantile` tail of that distribution.
 * That is scale-free, so it survives re-embedding.
 * ────────────────────────────────────────────────────────────────────────────
 *
 * REFUSING TO CALIBRATE is a first-class outcome. Too few background samples,
 * or a background so compressed that its quantile is not a usable cut, returns
 * `null` from {@link calibrateNearDuplicateCut} and `verdict: false` from
 * {@link checkNearDuplicates}, with every candidate KEPT. A caller that treats
 * a non-verdict as "keep what the cheap matcher said" fails in the safe
 * direction.
 *
 * Errors thrown by the host's similarity reader or background sampler
 * propagate: whether a failure degrades to the cheap matcher is the host's
 * policy, not this library's.
 *
 * Domain-free by construction (libs/generic contract): no host tables, no host
 * constants, no I/O. The host supplies similarities and the background.
 */

/** The default quantile of the background distribution used as the cut. */
export const DEFAULT_NEAR_DUPLICATE_QUANTILE = 0.95;

/** Below this many usable background samples the cut is not calibrated. */
export const DEFAULT_MIN_BACKGROUND_SAMPLES = 32;

/** How many background similarities {@link checkNearDuplicates} asks the sampler for. */
export const DEFAULT_BACKGROUND_SAMPLE_LIMIT = 256;

/**
 * A calibrated cut above this is treated as degenerate: a background that is
 * nearly self-identical (a re-embedding collapse, a corpus of copies) cannot
 * separate anything, so it is refused rather than used.
 */
export const DEFAULT_MAX_CALIBRATED_CUT = 0.98;

/** Decimal places candidate similarities are rounded to in the outcome. */
export const DEFAULT_SIMILARITY_DECIMALS = 3;

/** Which rule produced a cut, and from how much evidence. */
export interface NearDuplicateCalibration {
  /** `corpus-relative`: a background quantile. `absolute-override`: a fixed cut the host forced. */
  basis: 'corpus-relative' | 'absolute-override';
  /** Candidates with similarity `>= cut` are kept as likely duplicates. */
  cut: number;
  /** Usable (finite) background samples the cut was taken from; 0 for an override. */
  backgroundSamples: number;
  /** The quantile used; absent for an absolute override. */
  quantile?: number;
}

export interface CalibrateCutOptions {
  /** Background quantile to cut at, in (0, 1]. Default {@link DEFAULT_NEAR_DUPLICATE_QUANTILE}. */
  quantile?: number;
  /** A fixed cut in (0, 1]. When set, the background is ignored. */
  absoluteOverride?: number | null;
  /** Minimum usable background samples. Default {@link DEFAULT_MIN_BACKGROUND_SAMPLES}. */
  minSamples?: number;
  /** Largest cut accepted as non-degenerate. Default {@link DEFAULT_MAX_CALIBRATED_CUT}. */
  maxCut?: number;
}

function assertUnitInterval(name: string, value: number): void {
  if (!(Number.isFinite(value) && value > 0 && value <= 1)) {
    throw new RangeError(`${name} must be a finite number in (0, 1]; got ${value}`);
  }
}

/**
 * Nearest-rank quantile over an ASCENDING array: the element at 0-based index
 * `floor(q * n)`, clamped to the array. For q = 0.95 and n = 256 that is index
 * 243. The array must be non-empty.
 */
export function nearestRankQuantile(sortedAsc: readonly number[], q: number): number {
  if (sortedAsc.length === 0) throw new RangeError('nearestRankQuantile needs a non-empty array');
  const i = Math.min(sortedAsc.length - 1, Math.max(0, Math.floor(q * sortedAsc.length)));
  return sortedAsc[i]!;
}

/**
 * The cut for a near-duplicate check, or `null` when the background cannot
 * support one (fewer than `minSamples` finite values, a quantile that is not
 * positive, or one above `maxCut`). Non-finite background values are ignored.
 */
export function calibrateNearDuplicateCut(
  background: readonly number[],
  opts: CalibrateCutOptions = {},
): NearDuplicateCalibration | null {
  const override = opts.absoluteOverride ?? null;
  if (override !== null) {
    assertUnitInterval('absoluteOverride', override);
    return { basis: 'absolute-override', cut: override, backgroundSamples: 0 };
  }
  const q = opts.quantile ?? DEFAULT_NEAR_DUPLICATE_QUANTILE;
  assertUnitInterval('quantile', q);
  const minSamples = opts.minSamples ?? DEFAULT_MIN_BACKGROUND_SAMPLES;
  const maxCut = opts.maxCut ?? DEFAULT_MAX_CALIBRATED_CUT;

  const usable = background.filter((n) => typeof n === 'number' && Number.isFinite(n));
  if (usable.length === 0 || usable.length < minSamples) return null;
  const cut = nearestRankQuantile([...usable].sort((a, b) => a - b), q);
  if (!(cut > 0) || cut > maxCut) return null;
  return { basis: 'corpus-relative', cut, backgroundSamples: usable.length, quantile: q };
}

export interface NearDuplicateCheck<T> extends CalibrateCutOptions {
  /** The candidates the host already suspects. Order is preserved in the outcome. */
  candidates: readonly T[];
  /** A candidate's stable key, used to look up its similarity. */
  keyOf: (candidate: T) => string;
  /**
   * Similarity of each candidate to the new document, keyed by `keyOf`. A key
   * absent from the map means "no stored vector": that candidate is KEPT
   * without a similarity, because there is no evidence to drop it.
   */
  similarities: (keys: readonly string[]) => Promise<ReadonlyMap<string, number>>;
  /**
   * The new document's similarity to up to `limit` background documents of the
   * same corpus, excluding the candidates themselves (so the sample is a
   * background, not the suspected duplicates). Not called when
   * `absoluteOverride` is set. Omitted means no background, hence no verdict
   * unless an override is set.
   */
  sampleBackground?: (excludeKeys: readonly string[], limit: number) => Promise<readonly number[]>;
  /** Background sample size requested. Default {@link DEFAULT_BACKGROUND_SAMPLE_LIMIT}. */
  backgroundLimit?: number;
  /** Round reported similarities to this many decimals; `null` reports them raw. Default 3. */
  similarityDecimals?: number | null;
}

/** A kept candidate: unchanged when it had no vector, otherwise carrying its similarity. */
export type NearDuplicateCandidate<T> = T | (T & { similarity: number });
export type DroppedNearDuplicateCandidate<T> = T & { similarity: number };

export type NearDuplicateOutcome<T> =
  | {
      verdict: true;
      /** Likely duplicates (similarity at or above the cut) and candidates with no vector. */
      kept: NearDuplicateCandidate<T>[];
      /** Candidates below the cut: topically adjacent, not duplicates. */
      dropped: DroppedNearDuplicateCandidate<T>[];
      calibration: NearDuplicateCalibration;
    }
  | {
      verdict: false;
      /** Every candidate, unchanged: no classification was made. */
      kept: T[];
      dropped: [];
      reason: 'no-candidates' | 'uncalibrated';
    };

/**
 * Partition `candidates` into likely duplicates (`kept`) and topically
 * adjacent documents (`dropped`) at a corpus-calibrated cut.
 *
 * Reads similarities first, then the background, so a host whose reads share
 * a connection sees them in a fixed order. With no candidates, neither read
 * is made.
 */
export async function checkNearDuplicates<T>(check: NearDuplicateCheck<T>): Promise<NearDuplicateOutcome<T>> {
  const candidates = [...check.candidates];
  if (candidates.length === 0) return { verdict: false, kept: [], dropped: [], reason: 'no-candidates' };

  const keys = candidates.map(check.keyOf);
  const sims = await check.similarities(keys);

  const override = check.absoluteOverride ?? null;
  const background =
    override !== null || !check.sampleBackground
      ? []
      : await check.sampleBackground(keys, check.backgroundLimit ?? DEFAULT_BACKGROUND_SAMPLE_LIMIT);

  const calibration = calibrateNearDuplicateCut(background, check);
  if (!calibration) return { verdict: false, kept: candidates, dropped: [], reason: 'uncalibrated' };

  const decimals = check.similarityDecimals === undefined ? DEFAULT_SIMILARITY_DECIMALS : check.similarityDecimals;
  const report = (sim: number): number => {
    if (decimals === null) return sim;
    const scale = 10 ** decimals;
    return Math.round(sim * scale) / scale;
  };

  const kept: NearDuplicateCandidate<T>[] = [];
  const dropped: DroppedNearDuplicateCandidate<T>[] = [];
  candidates.forEach((candidate, i) => {
    const sim = sims.get(keys[i]!);
    if (sim === undefined) kept.push(candidate);
    else if (sim >= calibration.cut) kept.push({ ...candidate, similarity: report(sim) });
    else dropped.push({ ...candidate, similarity: report(sim) });
  });
  return { verdict: true, kept, dropped, calibration };
}
