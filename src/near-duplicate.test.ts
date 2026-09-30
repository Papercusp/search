/**
 * Calibrated near-duplicate check, library half
 * (shared-vector-search-libraries-2026-09-29 P-004, R-7).
 *
 * Two layers:
 *   1. PARITY — every case in __fixtures__/near-duplicate-parity-cases.json
 *      carries the output papercusp's pre-move classifier produced for it
 *      (see the fixture's `provenance`). The library must return the same cut
 *      (within 1e-9), the same verdict and the same kept/dropped partition.
 *      The cases sit on the boundaries a drifted port breaks: a candidate AT
 *      the cut, 1e-9 either side of it, exactly the minimum sample count and
 *      one below it, non-finite background values, a degenerate background,
 *      an override that must skip the background read.
 *   2. CONTRACT — the option handling a host relies on that the fixture does
 *      not exercise (custom minimum/maximum, raw similarities, bad options,
 *      error propagation, candidate order).
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

import {
  DEFAULT_BACKGROUND_SAMPLE_LIMIT,
  DEFAULT_MIN_BACKGROUND_SAMPLES,
  DEFAULT_NEAR_DUPLICATE_QUANTILE,
  calibrateNearDuplicateCut,
  checkNearDuplicates,
  nearestRankQuantile,
} from './near-duplicate';

interface BackgroundSpec {
  seed?: number;
  n: number;
  lo?: number;
  hi?: number;
  constant?: number;
  extra?: string[];
}
interface ParityCase {
  name: string;
  background: BackgroundSpec;
  quantile?: number;
  absoluteOverride?: number;
  candidates: Array<{ key: string; similarity: number | null }>;
  expected: {
    verdict: boolean;
    calibration: { basis: string; cut: number; backgroundSamples: number; quantile?: number } | null;
    kept: Array<{ key: string; similarity?: number }>;
    dropped: Array<{ key: string; similarity: number }>;
    backgroundRead: boolean;
    backgroundLimit?: number;
    backgroundExclude?: string[];
  };
}

const FIXTURE = JSON.parse(
  readFileSync(new URL('./__fixtures__/near-duplicate-parity-cases.json', import.meta.url), 'utf8'),
) as { cases: ParityCase[] };

/** The fixture's documented generator: mulberry32(seed) → lo + (hi - lo) * r, rounded to 1e-6. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function backgroundOf(spec: BackgroundSpec): number[] {
  const out: number[] = [];
  if (spec.constant !== undefined) for (let i = 0; i < spec.n; i++) out.push(spec.constant);
  else {
    const r = mulberry32(spec.seed!);
    for (let i = 0; i < spec.n; i++) out.push(Math.round((spec.lo! + (spec.hi! - spec.lo!) * r()) * 1e6) / 1e6);
  }
  for (const x of spec.extra ?? []) out.push(Number(x));
  return out;
}

type Item = { key: string };

async function runLibrary(c: ParityCase) {
  const sims = new Map<string, number>();
  for (const cand of c.candidates) if (cand.similarity !== null) sims.set(cand.key, cand.similarity);
  const reads = { background: 0, limit: undefined as number | undefined, exclude: undefined as string[] | undefined };
  const outcome = await checkNearDuplicates<Item>({
    candidates: c.candidates.map((x) => ({ key: x.key })),
    keyOf: (x) => x.key,
    similarities: async () => sims,
    sampleBackground: async (exclude, limit) => {
      reads.background++;
      reads.limit = limit;
      reads.exclude = [...exclude];
      return backgroundOf(c.background);
    },
    ...(c.quantile !== undefined ? { quantile: c.quantile } : {}),
    ...(c.absoluteOverride !== undefined ? { absoluteOverride: c.absoluteOverride } : {}),
  });
  return { outcome, reads };
}

describe('checkNearDuplicates — parity with the pre-move papercusp classifier', () => {
  it('the fixture covers every boundary this suite claims', () => {
    const names = FIXTURE.cases.map((c) => c.name);
    expect(names).toEqual(
      expect.arrayContaining([
        'spread-256-q95-boundaries',
        'min-samples-exactly-32',
        'below-min-samples-31',
        'degenerate-background-cut-above-max',
        'absolute-override-0.7-skips-background',
        'no-candidates',
      ]),
    );
    // The generator must reproduce the documented draw, or every "parity" below
    // compares against a different background than the oracle saw.
    const spread = backgroundOf({ seed: 20260930, n: 256, lo: 0.635, hi: 0.837 });
    expect(spread).toHaveLength(256);
    expect(nearestRankQuantile([...spread].sort((a, b) => a - b), 0.95)).toBe(0.827627);
  });

  for (const c of FIXTURE.cases) {
    it(`${c.name}: same cut, verdict and partition`, async () => {
      const { outcome, reads } = await runLibrary(c);
      const e = c.expected;
      expect(outcome.verdict).toBe(e.verdict);

      if (outcome.verdict) {
        expect(e.calibration).not.toBeNull();
        expect(Math.abs(outcome.calibration.cut - e.calibration!.cut)).toBeLessThanOrEqual(1e-9);
        expect(outcome.calibration.basis).toBe(e.calibration!.basis);
        expect(outcome.calibration.backgroundSamples).toBe(e.calibration!.backgroundSamples);
        expect(outcome.calibration.quantile).toBe(e.calibration!.quantile);
      } else {
        expect(e.calibration).toBeNull();
      }

      expect(outcome.kept).toEqual(e.kept);
      expect(outcome.dropped).toEqual(e.dropped);
      expect(reads.background > 0).toBe(e.backgroundRead);
      if (e.backgroundRead) {
        expect(reads.limit).toBe(e.backgroundLimit);
        expect(reads.exclude).toEqual(e.backgroundExclude);
      }
    });
  }
});

describe('calibrateNearDuplicateCut', () => {
  const spread = Array.from({ length: 100 }, (_, i) => 0.5 + i * 0.004); // 0.500 … 0.896

  it('takes the nearest-rank quantile at index floor(q * n)', () => {
    const cal = calibrateNearDuplicateCut(spread);
    expect(cal).toEqual({
      basis: 'corpus-relative',
      cut: spread[95],
      backgroundSamples: 100,
      quantile: DEFAULT_NEAR_DUPLICATE_QUANTILE,
    });
    expect(calibrateNearDuplicateCut(spread, { quantile: 1 })!.cut).toBe(spread[99]);
  });

  it('honours a host minimum and maximum', () => {
    expect(calibrateNearDuplicateCut(spread.slice(0, 10))).toBeNull();
    expect(calibrateNearDuplicateCut(spread.slice(0, 10), { minSamples: 10 })!.backgroundSamples).toBe(10);
    expect(calibrateNearDuplicateCut(spread, { maxCut: 0.8 })).toBeNull();
    expect(DEFAULT_MIN_BACKGROUND_SAMPLES).toBe(32);
  });

  it('never calibrates from an empty background, whatever the minimum', () => {
    expect(calibrateNearDuplicateCut([], { minSamples: 0 })).toBeNull();
    expect(calibrateNearDuplicateCut([Number.NaN, Number.POSITIVE_INFINITY], { minSamples: 0 })).toBeNull();
  });

  it('refuses a quantile or override outside (0, 1] instead of producing a meaningless cut', () => {
    expect(() => calibrateNearDuplicateCut(spread, { quantile: 0 })).toThrow(RangeError);
    expect(() => calibrateNearDuplicateCut(spread, { quantile: 1.5 })).toThrow(RangeError);
    expect(() => calibrateNearDuplicateCut(spread, { quantile: Number.NaN })).toThrow(RangeError);
    expect(() => calibrateNearDuplicateCut(spread, { absoluteOverride: 0 })).toThrow(RangeError);
    expect(calibrateNearDuplicateCut([], { absoluteOverride: 0.4 })).toEqual({
      basis: 'absolute-override',
      cut: 0.4,
      backgroundSamples: 0,
    });
  });

  it('nearestRankQuantile rejects an empty array', () => {
    expect(() => nearestRankQuantile([], 0.5)).toThrow(RangeError);
  });
});

describe('checkNearDuplicates — host contract', () => {
  const background = Array.from({ length: 64 }, (_, i) => 0.6 + i * 0.005); // cut = index 60 = 0.9

  it('preserves candidate order and extra fields, and reports raw similarities on request', async () => {
    const candidates = [
      { id: 'b', title: 'second' },
      { id: 'a', title: 'first' },
      { id: 'c', title: 'third' },
    ];
    const out = await checkNearDuplicates({
      candidates,
      keyOf: (x) => x.id,
      similarities: async () => new Map([['a', 0.91234567], ['b', 0.2], ['c', 0.95]]),
      sampleBackground: async () => background,
      similarityDecimals: null,
    });
    expect(out.verdict).toBe(true);
    if (!out.verdict) return;
    expect(out.kept).toEqual([
      { id: 'a', title: 'first', similarity: 0.91234567 },
      { id: 'c', title: 'third', similarity: 0.95 },
    ]);
    expect(out.dropped).toEqual([{ id: 'b', title: 'second', similarity: 0.2 }]);
    // The caller's candidates are not mutated.
    expect(candidates[1]).toEqual({ id: 'a', title: 'first' });
  });

  it('requests DEFAULT_BACKGROUND_SAMPLE_LIMIT unless the host sets a limit', async () => {
    const limits: number[] = [];
    const run = (backgroundLimit?: number) =>
      checkNearDuplicates({
        candidates: ['x'],
        keyOf: (x) => x,
        similarities: async () => new Map([['x', 0.95]]),
        sampleBackground: async (_exclude, limit) => {
          limits.push(limit);
          return background;
        },
        ...(backgroundLimit !== undefined ? { backgroundLimit } : {}),
      });
    await run();
    await run(64);
    expect(limits).toEqual([DEFAULT_BACKGROUND_SAMPLE_LIMIT, 64]);
  });

  it('gives no verdict, keeping every candidate, when the host supplies no background sampler', async () => {
    const out = await checkNearDuplicates({
      candidates: ['x', 'y'],
      keyOf: (x) => x,
      similarities: async () => new Map([['x', 0.99], ['y', 0.01]]),
    });
    expect(out).toEqual({ verdict: false, kept: ['x', 'y'], dropped: [], reason: 'uncalibrated' });
  });

  it('propagates a failing reader instead of guessing a verdict', async () => {
    await expect(
      checkNearDuplicates({
        candidates: ['x'],
        keyOf: (x) => x,
        similarities: async () => {
          throw new Error('vector store down');
        },
        sampleBackground: async () => background,
      }),
    ).rejects.toThrow('vector store down');
    await expect(
      checkNearDuplicates({
        candidates: ['x'],
        keyOf: (x) => x,
        similarities: async () => new Map([['x', 0.99]]),
        sampleBackground: async () => {
          throw new Error('sample failed');
        },
      }),
    ).rejects.toThrow('sample failed');
  });
});
