/**
 * A host that is not papercusp configures every extracted piece
 * (shared-vector-search-libraries-2026-09-29 acceptance R-12).
 *
 * The host is a herbarium catalogue: specimen sheets, their labels and loan
 * records in a `herbarium` schema, embedded by the catalogue's own models into
 * a 96-wide cosine column. Nothing below names a papercusp model, width, table
 * or source. Everything is configured through the library's public entry
 * (`./index`), the way an outside host would import it.
 *
 * The benchmark half of "coverage gate and benchmark" lives in
 * @papercusp/search-core, which has its own fixture-host spec.
 */
import { describe, expect, it } from 'vitest';
import {
  checkNearDuplicates,
  createBackfillSweeper,
  createCoverageGate,
  createEmbeddingSpace,
  type BackfillLogger,
  type BackfillSql,
  type CoverageSample,
  type EmbeddingProfileSpec,
} from './index';

// ─── the host's model list and storage ─────────────────────────────────────

const WIDTH = 96;

/** The catalogue's embedding models. Two fit its storage; two do not. */
const MODELS = {
  sheetsV1: { profileId: 'herbarium-sheets@1', dimensions: WIDTH, distanceMetric: 'cosine' },
  sheetsV2: { profileId: 'herbarium-sheets@2', dimensions: WIDTH, distanceMetric: 'cosine' },
  wide: { profileId: 'herbarium-wide@1', dimensions: 640, distanceMetric: 'cosine' },
  euclid: { profileId: 'herbarium-euclid@1', dimensions: WIDTH, distanceMetric: 'l2' },
} as const satisfies Record<string, EmbeddingProfileSpec>;

const space = createEmbeddingSpace({
  storage: {
    acceptedProfileIds: [MODELS.sheetsV1.profileId, MODELS.sheetsV2.profileId],
    dimensions: WIDTH,
    distanceMetric: 'cosine',
    indexOperatorClass: 'vector_cosine_ops',
  },
  // Rows written before the catalogue recorded profiles carry only the mode
  // label 'sheets', which today means sheets@2.
  legacyModes: { sheets: MODELS.sheetsV2 },
  storageLabel: 'herbarium vector storage',
});

/** The catalogue's embedder: hashed bag of words, WIDTH wide. */
function embedSheet(text: string): number[] {
  const v = new Array<number>(WIDTH).fill(0);
  for (const word of text.toLowerCase().split(/\W+/).filter(Boolean)) {
    let h = 0x811c9dc5;
    for (let i = 0; i < word.length; i++) h = Math.imul(h ^ word.charCodeAt(i), 0x01000193);
    v[(h >>> 0) % WIDTH] += 1;
  }
  return v;
}

function cosine(a: readonly number[], b: readonly number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return dot / Math.sqrt(na * nb);
}

// ─── embedding-space safety ────────────────────────────────────────────────

describe('fixture host: embedding-space safety', () => {
  it('accepts only the catalogue models its storage can hold, and says why it refuses the rest', () => {
    expect(space.validateCompatibility(MODELS.sheetsV1)).toEqual([]);
    expect(space.validateCompatibility(MODELS.sheetsV2)).toEqual([]);
    expect(space.validateCompatibility(MODELS.wide)).toEqual([
      'herbarium vector storage does not accept profile herbarium-wide@1; accepted=herbarium-sheets@1,herbarium-sheets@2',
      'herbarium vector storage has 96 dimensions; profile herbarium-wide@1 emits 640',
    ]);
    expect(space.validateCompatibility(MODELS.euclid)).toEqual([
      'herbarium vector storage does not accept profile herbarium-euclid@1; accepted=herbarium-sheets@1,herbarium-sheets@2',
      'herbarium vector storage uses cosine; profile herbarium-euclid@1 requires l2',
    ]);
    expect(space.fitsStorage(WIDTH)).toBe(true);
    expect(space.fitsStorage(MODELS.wide.dimensions)).toBe(false);
  });

  it('selects the exact model space, and credits mode-only rows to the current model alone', () => {
    const current = space.resolveCurrentSelection('sheets');
    expect(current).toEqual({ profileId: 'herbarium-sheets@2', legacyMode: 'sheets' });
    const previous = space.resolveSelection('sheets', MODELS.sheetsV1);
    expect(previous).toEqual({ profileId: 'herbarium-sheets@1', legacyMode: null });
    expect(space.resolveSelection('wide', MODELS.wide)).toBeNull();

    const modeOnlyRow = { profileId: null, mode: 'sheets' };
    expect(space.storedIdentityMatches(modeOnlyRow, current!)).toBe(true);
    expect(space.storedIdentityMatches(modeOnlyRow, previous!)).toBe(false);
    expect(space.storedIdentityMatches({ profileId: 'herbarium-sheets@1', mode: 'sheets' }, current!)).toBe(false);
  });

  it('reports the catalogue column that was created at the wrong width', () => {
    expect(
      space.computeColumnWidthSkew([
        { table: 'herbarium.sheets', column: 'note_vec', dims: WIDTH },
        { table: 'herbarium.labels', column: 'label_vec', dims: 640 },
        { table: 'herbarium.loans', column: 'terms_vec', dims: -1 },
      ]),
    ).toEqual([{ table: 'herbarium.labels', column: 'label_vec', liveDims: 640, declaredDims: WIDTH }]);
  });
});

// ─── coverage gate ─────────────────────────────────────────────────────────

describe('fixture host: coverage gate', () => {
  /** The catalogue measures coverage every few hours, so it tolerates 6h-old samples. */
  const gate = createCoverageGate({
    sources: {
      sheets: ['herbarium.sheets.note_vec'],
      labels: ['herbarium.labels.label_vec', 'herbarium.labels.ocr_vec'],
      loans: ['herbarium.loans.terms_vec'],
      collectors: [],
    },
    thresholds: { coverageFloor: 0.9, maxSampleAgeMs: 6 * 3600_000 },
  });
  const now = new Date('2026-05-01T12:00:00Z');
  const samples: CoverageSample[] = [
    { surface: 'herbarium.sheets.note_vec', observedAt: '2026-05-01T11:00:00Z', eligibleRows: '400', embeddedRows: '396' },
    // An older sample of the same surface is superseded, not averaged in.
    { surface: 'herbarium.sheets.note_vec', observedAt: '2026-04-30T23:00:00Z', eligibleRows: 400, embeddedRows: 100 },
    { surface: 'herbarium.labels.label_vec', observedAt: '2026-05-01T09:00:00Z', eligibleRows: 1000, embeddedRows: 700 },
    {
      surface: 'herbarium.labels.ocr_vec',
      observedAt: '2026-05-01T09:00:00Z',
      eligibleRows: 1000,
      embeddedRows: 850,
      recentEligible: 50,
      recentEmbedded: 50,
    },
    { surface: 'herbarium.loans.terms_vec', observedAt: '2026-04-30T12:00:00Z', eligibleRows: 10, embeddedRows: 10 },
  ];

  it('rolls the catalogue sources up into healthy, degraded, unknown and not-semantic', () => {
    const report = gate.assessScope(['sheets', 'labels', 'collectors', 'loans', 'exchanges'], gate.snapshot(samples, now));
    const verdicts = Object.fromEntries(report.perSource.map((a) => [a.source, a.verdict]));
    expect(verdicts).toEqual({
      sheets: 'healthy',
      labels: 'degraded',
      collectors: 'not-semantic',
      loans: 'unknown',
      exchanges: 'unknown',
    });
    const bySource = new Map(report.perSource.map((a) => [a.source, a]));
    expect(bySource.get('sheets')!.coverage).toBe(0.99);
    // The better of the two label legs decides, and its recent rows are all embedded.
    expect(bySource.get('labels')!.coverage).toBe(0.85);
    expect(bySource.get('labels')!.recentCoverage).toBe(1);
    expect(bySource.get('labels')!.note).toContain('historical backlog');
    expect(bySource.get('loans')!.unknownSurfaces).toEqual(['herbarium.loans.terms_vec']);
    expect(bySource.get('exchanges')!.note).toContain("source 'exchanges' has no coverage mapping");
    expect(report.degraded).toBe(true);
    expect(report.degradedSources).toEqual(['labels']);
    expect(report.unknownSources).toEqual(['loans', 'exchanges']);
  });

  it('the catalogue thresholds are the ones applied: the default 90-minute age would void its 3h-old samples', () => {
    const strict = createCoverageGate({ sources: { labels: ['herbarium.labels.label_vec', 'herbarium.labels.ocr_vec'] } });
    expect(strict.assessSource('labels', strict.snapshot(samples, now)).verdict).toBe('unknown');
    expect(gate.thresholds.coverageFloor).toBe(0.9);
  });
});

// ─── calibrated near-duplicate check ───────────────────────────────────────

describe('fixture host: calibrated near-duplicate check', () => {
  /**
   * A stored WIDTH-wide vector whose cosine with the new sheet is exactly `s`:
   * the new sheet is e0, and every other sheet leans on its own axis e_k.
   */
  function vectorAt(s: number, axis: number): number[] {
    const v = new Array<number>(WIDTH).fill(0);
    v[0] = s;
    v[axis] = Math.sqrt(1 - s * s);
    return v;
  }
  const newSheet = vectorAt(1, 1);
  const stored = new Map<string, number[]>([
    ['sheet-1142-rescan', vectorAt(0.97, 1)],
    ['sheet-0877', vectorAt(0.62, 2)],
    ['sheet-0310', vectorAt(0.31, 3)],
  ]);
  // Forty unrelated sheets with similarities 0.10 … 0.49.
  const background = Array.from({ length: 40 }, (_, i) => [`bg-${i}`, vectorAt(0.1 + i * 0.01, 10 + i)] as const);
  for (const [key, vec] of background) stored.set(key, vec);

  it('keeps candidates above the catalogue’s own background quantile and drops the rest', async () => {
    const excluded: string[][] = [];
    const outcome = await checkNearDuplicates({
      candidates: [{ id: 'sheet-1142-rescan' }, { id: 'sheet-0877' }, { id: 'sheet-0310' }, { id: 'sheet-0999-unembedded' }],
      keyOf: (c) => c.id,
      similarities: async (keys) =>
        new Map(keys.filter((k) => stored.has(k)).map((k) => [k, cosine(newSheet, stored.get(k)!)] as const)),
      sampleBackground: async (exclude, limit) => {
        excluded.push([...exclude]);
        return [...stored]
          .filter(([key]) => !exclude.includes(key))
          .slice(0, limit)
          .map(([, vec]) => cosine(newSheet, vec));
      },
    });
    expect(outcome.verdict).toBe(true);
    if (!outcome.verdict) return;
    // The cut is the background's 0.95 nearest-rank quantile: index floor(0.95 × 40) = 38.
    expect(outcome.calibration.basis).toBe('corpus-relative');
    expect(outcome.calibration.backgroundSamples).toBe(40);
    expect(outcome.calibration.cut).toBeCloseTo(0.48, 9);
    expect(outcome.kept).toEqual([
      { id: 'sheet-1142-rescan', similarity: 0.97 },
      { id: 'sheet-0877', similarity: 0.62 },
      // No stored vector: no evidence to drop it.
      { id: 'sheet-0999-unembedded' },
    ]);
    expect(outcome.dropped).toEqual([{ id: 'sheet-0310', similarity: 0.31 }]);
    // The background excludes the suspected duplicates themselves.
    expect(excluded).toEqual([['sheet-1142-rescan', 'sheet-0877', 'sheet-0310', 'sheet-0999-unembedded']]);
  });

  it('works on similarities the catalogue embedder computes from real text', () => {
    const a = embedSheet('Quercus alba, north ridge, lobed leaves, acorns present');
    const b = embedSheet('Quercus alba, north ridge, lobed leaves, acorns present, rescanned');
    const c = embedSheet('Salix nigra, river bend, narrow leaves, catkins');
    expect(a).toHaveLength(WIDTH);
    expect(cosine(a, b)).toBeGreaterThan(cosine(a, c));
  });
});

// ─── backfill engine ───────────────────────────────────────────────────────

interface CatalogueRow {
  key: string;
  body: string;
  stale: boolean;
  vec?: string;
  labels?: unknown[];
}
interface CatalogueTable {
  rows: CatalogueRow[];
  /** Live width of the vector column. */
  dims: number;
}

/**
 * The catalogue's database, in memory. Like PostgreSQL it refuses a statement
 * whose placeholders and bound parameters disagree. Tables not listed here do
 * not exist.
 */
function catalogueDb(tables: Record<string, CatalogueTable>) {
  const fake = {
    unsafe<T>(query: string, params: unknown[] = []): Promise<T> {
      const refs = [...query.matchAll(/\$(\d+)/g)].map((m) => Number(m[1]));
      const highest = refs.length ? Math.max(...refs) : 0;
      if (highest !== params.length) {
        return Promise.reject(new Error(`bind mismatch: statement references $${highest}, bound ${params.length}`));
      }
      if (query.includes('FROM pg_extension')) return Promise.resolve([{ extname: 'vector' }] as T);
      if (query.includes('information_schema.columns')) {
        const t = tables[String(params[0])];
        if (!t) return Promise.resolve([] as T);
        const col = String(params[1]);
        return Promise.resolve([
          { column_name: col, dims: t.dims },
          { column_name: `${col}_mode`, dims: null },
          { column_name: `${col}_profile`, dims: null },
        ] as T);
      }
      const table = /(?:FROM|UPDATE) (\S+)/.exec(query)?.[1] ?? '';
      const t = tables[table];
      if (!t) return Promise.reject(new Error(`no such table ${table}`));
      if (query.trimStart().startsWith('SELECT')) {
        const limit = Number(params[0]);
        const offset = Number(params[params.length - 1]);
        const pending = t.rows.filter((r) => r.stale && r.body.length > 0);
        return Promise.resolve(pending.slice(offset, offset + limit).map((r) => ({ k0: r.key, body: r.body })) as T);
      }
      if (query.trimStart().startsWith('UPDATE')) {
        const row = t.rows.find((r) => r.key === params[params.length - 1]);
        if (row && row.stale) {
          row.stale = false;
          row.vec = String(params[0]);
          row.labels = params.slice(1, -1);
        }
        return Promise.resolve([] as T);
      }
      return Promise.reject(new Error(`unexpected statement: ${query.slice(0, 40)}`));
    },
  };
  return fake as unknown as BackfillSql;
}

describe('fixture host: backfill engine', () => {
  it('fills the catalogue column with its current model, and refuses the column built at another width', async () => {
    const tables: Record<string, CatalogueTable> = {
      'herbarium.sheets': {
        dims: WIDTH,
        rows: [
          { key: 'sheet-0001', body: 'Quercus alba, north ridge', stale: true },
          { key: 'sheet-0002', body: 'Salix nigra, river bend', stale: true },
          { key: 'sheet-0003', body: '', stale: true },
        ],
      },
      'herbarium.labels': { dims: 640, rows: [{ key: 'label-0001', body: 'Coll. A. Gray 1851', stale: true }] },
    };
    const lines: string[] = [];
    const logger: BackfillLogger = { log: (m) => lines.push(m), warn: (m) => lines.push(m), error: (m) => lines.push(m) };
    const embedded: string[] = [];
    const sweeper = createBackfillSweeper({
      getTargets: () => [
        { table: 'herbarium.sheets', embedCol: 'note_vec', bodySql: 'left(note, 2000)', keyCols: ['sheet_id'] },
        { table: 'herbarium.labels', embedCol: 'label_vec', bodySql: 'label_text', keyCols: ['label_id'] },
        { table: 'herbarium.loans', embedCol: 'terms_vec', bodySql: 'terms', keyCols: ['loan_id'] },
      ],
      getSql: () => catalogueDb(tables),
      resolveEmbedder: async () => ({
        mode: 'sheets',
        dims: WIDTH,
        profile: MODELS.sheetsV2,
        embed: async (text: string) => {
          embedded.push(text);
          return embedSheet(text);
        },
      }),
      resolveProfileSelection: (mode, profile) => space.resolveSelection(mode, profile),
      acceptsWidth: (dims) => space.fitsStorage(dims),
      widthSkew: (measured) => space.computeColumnWidthSkew(measured),
      widthSkewHint: 'Recreate the column at the catalogue width.',
      batchSize: 10,
      maxInputChars: 2000,
      logger,
      logLabel: 'herbarium-backfill',
    });

    const result = await sweeper.run();
    expect(Array.isArray(result)).toBe(true);
    const stats = (result as Array<Record<string, unknown>>).map(({ durationMs: _d, ...s }) => s);
    expect(stats).toEqual([
      { table: 'herbarium.sheets', scanned: 2, embedded: 2, errors: 0 },
      { table: 'herbarium.labels', scanned: 0, embedded: 0, errors: 0 },
      { table: 'herbarium.loans', scanned: 0, embedded: 0, errors: 0 },
    ]);
    expect(embedded).toEqual(['Quercus alba, north ridge', 'Salix nigra, river bend']);

    const [quercus, salix, blank] = tables['herbarium.sheets']!.rows;
    for (const row of [quercus!, salix!]) {
      expect(row.stale).toBe(false);
      expect(JSON.parse(row.vec!)).toHaveLength(WIDTH);
      // The write records the catalogue's mode and its exact current model.
      expect(row.labels).toEqual(['sheets', 'herbarium-sheets@2']);
    }
    expect(blank!.stale).toBe(true);
    expect(tables['herbarium.labels']!.rows[0]!.stale).toBe(true);
    expect(
      lines.some(
        (l) =>
          l.startsWith('[herbarium-backfill] SCHEMA WIDTH SKEW — refusing herbarium.labels.label_vec: it is vector(640) but this code emits 96') &&
          l.endsWith('Recreate the column at the catalogue width.'),
      ),
    ).toBe(true);
  });

  it('refuses to write with a catalogue model its storage does not accept', async () => {
    const tables: Record<string, CatalogueTable> = {
      'herbarium.sheets': { dims: WIDTH, rows: [{ key: 'sheet-0001', body: 'Quercus alba', stale: true }] },
    };
    const lines: string[] = [];
    const sweeper = createBackfillSweeper({
      getTargets: () => [{ table: 'herbarium.sheets', embedCol: 'note_vec', bodySql: 'note', keyCols: ['sheet_id'] }],
      getSql: () => catalogueDb(tables),
      // Same width and metric as storage, but a model the catalogue never accepted.
      resolveEmbedder: async () => ({
        mode: 'sheets',
        dims: WIDTH,
        profile: { profileId: 'herbarium-sheets@3', dimensions: WIDTH, distanceMetric: 'cosine' },
        embed: async (text: string) => embedSheet(text),
      }),
      resolveProfileSelection: (mode, profile) => space.resolveSelection(mode, profile),
      acceptsWidth: (dims) => space.fitsStorage(dims),
      batchSize: 10,
      maxInputChars: 2000,
      logger: { log: (m) => lines.push(m), warn: (m) => lines.push(m), error: (m) => lines.push(m) },
    });
    const result = await sweeper.run();
    expect(tables['herbarium.sheets']!.rows[0]!.stale).toBe(true);
    expect((result as Array<{ embedded: number }>)[0]!.embedded).toBe(0);
    expect(lines.join('\n')).toContain('profile herbarium-sheets@3 is incompatible');
  });
});
