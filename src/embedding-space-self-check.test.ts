/**
 * runStoredRowSelfCheck (shared-vector-search-libraries-2026-09-29 P-001,
 * AUTO-BAR-R-4-P-001): alerts exactly when the injected embedder's
 * re-embedding of the canary lies further than the threshold from the stored
 * vector. Equal to the threshold is healthy, as in papercusp's pre-move
 * embed-space-self-check.
 */
import { describe, expect, it, vi } from 'vitest';
import {
  DEFAULT_DESYNC_DISTANCE_THRESHOLD,
  cosineDistance,
  isEmbeddingDesync,
  parseVectorText,
  runStoredRowSelfCheck,
  type SelfCheckCanary,
} from './embedding-space-self-check';

const STORED = [1, 0];
/** A unit vector whose cosine distance from [1, 0] is exactly `d`. */
const atDistance = (d: number) => [1 - d, Math.sqrt(1 - (1 - d) ** 2)];
const canary: SelfCheckCanary = { body: 'row text', storedVector: '[1,0]', keyLabel: 'harness_shared.t#1' };

function run(fresh: readonly number[], threshold?: number) {
  const alert = vi.fn();
  const clear = vi.fn();
  const embed = vi.fn(async () => fresh);
  const result = runStoredRowSelfCheck({ embed, pickCanary: async () => canary, alert, clear, threshold });
  return { result, alert, clear, embed };
}

describe('runStoredRowSelfCheck — the alert boundary (R-4)', () => {
  it('a distance exactly equal to the threshold is healthy (exactly representable cases)', async () => {
    expect(await run(STORED, 0).result).toMatchObject({ status: 'healthy', distance: 0 });
    expect(await run([0, 1], 1).result).toMatchObject({ status: 'healthy', distance: 1 });
    expect(await run([0, 1], 0.999999).result).toMatchObject({ status: 'desync', distance: 1 });
  });

  it.each([0, 0.01, 0.049])('distance %s (< 0.05) is healthy and clears', async (d) => {
    const { result, alert, clear } = run(atDistance(d));
    const r = await result;
    expect(r).toMatchObject({ ok: true, status: 'healthy', keyLabel: canary.keyLabel });
    expect(alert).not.toHaveBeenCalled();
    expect(clear).toHaveBeenCalledTimes(1);
  });

  it.each([0.0501, 0.2, 1])('distance %s (> 0.05) alerts with the reading', async (d) => {
    const { result, alert, clear } = run(atDistance(d));
    const r = await result;
    expect(r).toMatchObject({ ok: true, status: 'desync' });
    expect(alert).toHaveBeenCalledTimes(1);
    expect(alert.mock.calls[0]![0]).toMatchObject({ canary, threshold: DEFAULT_DESYNC_DISTANCE_THRESHOLD });
    expect(alert.mock.calls[0]![0].distance).toBeCloseTo(d, 9);
    expect(clear).not.toHaveBeenCalled();
  });

  it('honours a caller threshold', async () => {
    expect((await run(atDistance(0.2), 0.3).result)).toMatchObject({ status: 'healthy' });
    expect((await run(atDistance(0.2), 0.1).result)).toMatchObject({ status: 'desync' });
  });

  it('a width mismatch is a desync, not an error', async () => {
    const { result, alert } = run([1, 0, 0]);
    expect(await result).toMatchObject({ status: 'desync', distance: 1 });
    expect(alert).toHaveBeenCalledTimes(1);
  });

  it('re-embeds the canary\'s own text with the injected embedder', async () => {
    const { result, embed } = run(STORED);
    await result;
    expect(embed).toHaveBeenCalledWith('row text');
  });
});

describe('runStoredRowSelfCheck — never throws', () => {
  it('no canary row is a healthy skip', async () => {
    const alert = vi.fn();
    expect(await runStoredRowSelfCheck({ embed: async () => STORED, pickCanary: async () => null, alert })).toEqual({
      ok: true,
      skipped: 'no_row_in_active_space',
    });
    expect(alert).not.toHaveBeenCalled();
  });
  it('a failing canary read or embedder is a skip', async () => {
    const alert = vi.fn();
    const boom = async () => { throw new Error('x'); };
    expect(await runStoredRowSelfCheck({ embed: async () => STORED, pickCanary: boom, alert })).toEqual({ ok: false, skipped: 'canary_read_failed' });
    expect(await runStoredRowSelfCheck({ embed: boom, pickCanary: async () => canary, alert })).toEqual({
      ok: false,
      skipped: 'embed_failed',
      keyLabel: canary.keyLabel,
    });
    expect(alert).not.toHaveBeenCalled();
  });
  it('a failing alert sink does not fail the tick', async () => {
    const r = await runStoredRowSelfCheck({
      embed: async () => [0, 1],
      pickCanary: async () => canary,
      alert: () => { throw new Error('sink down'); },
    });
    expect(r).toMatchObject({ ok: true, status: 'desync' });
  });
});

describe('helpers', () => {
  it('parseVectorText reads pgvector text', () => {
    expect(parseVectorText('[0.5,-1,2e-3]')).toEqual([0.5, -1, 0.002]);
  });
  it('cosineDistance matches pgvector <=> and treats degenerate input as maximal', () => {
    expect(cosineDistance([1, 0], [0, 1])).toBeCloseTo(1, 12);
    expect(cosineDistance([1, 1], [2, 2])).toBeCloseTo(0, 12);
    expect(cosineDistance([], [])).toBe(1);
    expect(cosineDistance([0, 0], [1, 0])).toBe(1);
  });
  it('isEmbeddingDesync is strict and treats NaN as a desync', () => {
    expect(isEmbeddingDesync(0.05, 0.05)).toBe(false);
    expect(isEmbeddingDesync(0.0500001, 0.05)).toBe(true);
    expect(isEmbeddingDesync(Number.NaN)).toBe(true);
  });
});
