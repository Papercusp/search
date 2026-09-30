/**
 * Coverage gate, library half (shared-vector-search-libraries-2026-09-29 P-002, R-5).
 *
 * Fixture host only: the sources, surfaces and thresholds below are not
 * papercusp's, so a pass here says the verdict logic works for any host.
 * The degraded verdict must hold EXACTLY below the floor — the boundary cases
 * are what a drifted comparison (`<=` for `<`, or a floor read from the wrong
 * threshold) would break.
 */
import { describe, expect, it } from 'vitest';

import {
  DEFAULT_COVERAGE_THRESHOLDS,
  assessSurfaceCoverage,
  buildCoverageCountQuery,
  buildCoverageSnapshot,
  createCoverageGate,
  measureSurfaceCoverage,
  type CoverageSample,
  type CoverageSqlHandle,
} from './coverage-gate';

const NOW = new Date('2026-01-10T12:00:00Z');
const minutesAgo = (m: number): Date => new Date(NOW.getTime() - m * 60_000);

const ARTICLES = 'library.articles.body_vec';
const ARTICLE_CHUNKS = 'library.article_chunks.vec';
const NOTES = 'library.notes.vec';

const gate = createCoverageGate({
  sources: {
    articles: [ARTICLES, ARTICLE_CHUNKS],
    notes: [NOTES],
    tags: [],
  },
  thresholds: { coverageFloor: 0.8, recentFloor: 0.9, minRecentSample: 5, maxSampleAgeMs: 30 * 60_000 },
});

const sample = (surface: string, embedded: number, eligible: number, extra: Partial<CoverageSample> = {}): CoverageSample => ({
  surface,
  observedAt: minutesAgo(5),
  eligibleRows: eligible,
  embeddedRows: embedded,
  ...extra,
});

describe('degraded exactly below the floor', () => {
  const cases: Array<[embedded: number, eligible: number, verdict: 'healthy' | 'degraded']> = [
    [0, 1000, 'degraded'],
    [799, 1000, 'degraded'],
    [7999, 10000, 'degraded'],
    [800, 1000, 'healthy'], // exactly at the floor is NOT degraded
    [801, 1000, 'healthy'],
    [1000, 1000, 'healthy'],
  ];
  it.each(cases)('%i/%i embedded -> %s', (embedded, eligible, verdict) => {
    const snap = gate.snapshot([sample(NOTES, embedded, eligible)], NOW);
    const a = gate.assessSource('notes', snap);
    expect(a.verdict).toBe(verdict);
    expect(a.coverage).toBeCloseTo(embedded / eligible, 12);
  });

  it('uses the configured floor, not the library default', () => {
    // 0.9 is healthy under this host's 0.8 floor but degraded under the 0.95 default.
    const snap = gate.snapshot([sample(NOTES, 90, 100)], NOW);
    expect(gate.assessSource('notes', snap).verdict).toBe('healthy');
    expect(assessSurfaceCoverage('notes', [NOTES], snap).verdict).toBe('degraded');
    expect(DEFAULT_COVERAGE_THRESHOLDS.coverageFloor).toBe(0.95);
  });
});

describe('absence of evidence is never healthy', () => {
  it('a source with no sample is unknown', () => {
    const a = gate.assessSource('notes', gate.snapshot([], NOW));
    expect(a.verdict).toBe('unknown');
    expect(a.unknownSurfaces).toEqual([NOTES]);
    expect(a.note).toMatch(/not the same as healthy/i);
  });

  it('a stale sample is unknown even when it read 100%', () => {
    const snap = gate.snapshot([sample(NOTES, 100, 100, { observedAt: minutesAgo(31) })], NOW);
    const a = gate.assessSource('notes', snap);
    expect(a.verdict).toBe('unknown');
    expect(a.surfaces[0]?.stale).toBe(true);
  });

  it('an unmapped source is unknown, with the host-overridable note', () => {
    expect(gate.assessSource('photos', new Map()).verdict).toBe('unknown');
    expect(gate.assessSource('photos', new Map()).note).toMatch(/no coverage mapping/);
    const custom = createCoverageGate({ sources: {}, describeUnmappedSource: (s) => `map ${s}` });
    expect(custom.assessSource('photos', new Map()).note).toBe('map photos');
  });

  it('an inherited object key is not a mapping', () => {
    expect(gate.assessSource('toString', new Map()).verdict).toBe('unknown');
  });

  it('a source with no vector leg is not-semantic, not unknown', () => {
    expect(gate.assessSource('tags', new Map()).verdict).toBe('not-semantic');
  });
});

describe('multi-surface sources', () => {
  it('take the best known leg', () => {
    const snap = gate.snapshot([sample(ARTICLES, 10, 100), sample(ARTICLE_CHUNKS, 95, 100)], NOW);
    const a = gate.assessSource('articles', snap);
    expect(a.verdict).toBe('healthy');
    expect(a.coverage).toBeCloseTo(0.95, 12);
  });

  it('report a partial view when one leg has no fresh sample', () => {
    const snap = gate.snapshot([sample(ARTICLES, 50, 100)], NOW);
    const a = gate.assessSource('articles', snap);
    expect(a.verdict).toBe('degraded');
    expect(a.unknownSurfaces).toEqual([ARTICLE_CHUNKS]);
    expect(a.note).toMatch(/PARTIAL index/);
    expect(a.note).toMatch(/Partial view/);
  });
});

describe('recent-window note', () => {
  it('calls a well-covered recent window historical backlog', () => {
    const snap = gate.snapshot([sample(NOTES, 50, 100, { recentEligible: 10, recentEmbedded: 10 })], NOW);
    expect(gate.assessSource('notes', snap).note).toMatch(/historical backlog/);
  });

  it('ignores a recent window smaller than minRecentSample', () => {
    const snap = gate.snapshot([sample(NOTES, 50, 100, { recentEligible: 4, recentEmbedded: 4 })], NOW);
    const a = gate.assessSource('notes', snap);
    expect(a.recentCoverage).toBeNull();
    expect(a.note).not.toMatch(/historical backlog/);
  });
});

describe('snapshot', () => {
  it('keeps the newest sample per surface regardless of input order', () => {
    const snap = buildCoverageSnapshot(
      [sample(NOTES, 100, 100, { observedAt: minutesAgo(1) }), sample(NOTES, 10, 100, { observedAt: minutesAgo(20) })],
      NOW,
    );
    expect(snap.get(NOTES)?.pct).toBe(1);
  });

  it('accepts bigint-as-string counts and reports n/a coverage for an empty table', () => {
    const snap = gate.snapshot([sample(NOTES, '0', '0')], NOW);
    expect(snap.get(NOTES)?.pct).toBeNull();
    expect(gate.assessSource('notes', snap).verdict).toBe('unknown');
  });
});

describe('roll-up', () => {
  it('flags degraded and unknown sources and stays quiet when all healthy', () => {
    const snap = gate.snapshot([sample(NOTES, 10, 100)], NOW);
    const r = gate.assessScope(['notes', 'articles', 'tags'], snap);
    expect(r.degradedSources).toEqual(['notes']);
    expect(r.unknownSources).toEqual(['articles']);
    expect(r.degraded).toBe(true);
    expect(r.warning).toMatch(/DEGRADED/);
    expect(r.warning).toMatch(/UNKNOWN/);

    const healthy = gate.assessScope(['notes'], gate.snapshot([sample(NOTES, 100, 100)], NOW));
    expect(healthy.degraded).toBe(false);
    expect(healthy.warning).toBeNull();
  });
});

describe('configuration', () => {
  it('refuses a floor above 1 or a negative threshold', () => {
    expect(() => createCoverageGate({ sources: {}, thresholds: { coverageFloor: 95 } })).toThrow(RangeError);
    expect(() => createCoverageGate({ sources: {}, thresholds: { maxSampleAgeMs: -1 } })).toThrow(RangeError);
  });
});

describe('measure', () => {
  it('binds the label only when the surface has a label column', () => {
    const plain = buildCoverageCountQuery({ table: 'library.notes', vectorColumn: 'vec' }, null);
    expect(plain.params).toEqual([]);
    expect(plain.sql).not.toMatch(/\$1/);
    expect(plain.surface).toBe('library.notes.vec');

    const labelled = buildCoverageCountQuery(
      { table: 'library.notes', vectorColumn: 'vec', labelColumn: 'model', recencyColumn: 'created_at', eligibleSql: "body <> ''" },
      'model-b',
      12,
    );
    expect(labelled.params).toEqual(['model-b', 12]);
    expect(labelled.sql).toContain('"model" = $1');
    expect(labelled.sql).toContain('make_interval(hours => $2)');
    expect(labelled.sql).toContain('FROM "library"."notes"');
    expect(labelled.hasRecent).toBe(true);
  });

  it('refuses identifiers that are not plain SQL names', () => {
    expect(() => buildCoverageCountQuery({ table: 'notes; drop table x', vectorColumn: 'vec' }, null)).toThrow(TypeError);
    expect(() => buildCoverageCountQuery({ table: 'notes', vectorColumn: 'vec"' }, null)).toThrow(TypeError);
    expect(() => buildCoverageCountQuery({ table: 'notes', vectorColumn: 'vec', labelColumn: 'm' }, null)).toThrow(TypeError);
  });

  it('turns the count row into a sample the gate can assess', async () => {
    const calls: Array<{ query: string; params?: unknown[] }> = [];
    const sql: CoverageSqlHandle = {
      unsafe: (query, params) => {
        calls.push({ query, params });
        return Promise.resolve([{ eligible_rows: '200', embedded_rows: '150', recent_eligible: '10', recent_embedded: '10' }]);
      },
    };
    const s = await measureSurfaceCoverage(
      sql,
      { table: 'library.notes', vectorColumn: 'vec', labelColumn: 'model', recencyColumn: 'created_at' },
      'model-b',
      { now: NOW },
    );
    expect(calls).toHaveLength(1);
    expect(s).toMatchObject({ surface: NOTES.replace('library.notes.vec', 'library.notes.vec'), eligibleRows: 200, embeddedRows: 150 });
    const a = gate.assessSource('notes', gate.snapshot([s], NOW));
    expect(a.verdict).toBe('degraded');
    expect(a.note).toMatch(/historical backlog/);
  });
});
