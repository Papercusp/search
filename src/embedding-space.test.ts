/**
 * createEmbeddingSpace — parity with papercusp's pre-move embedding-space code
 * (shared-vector-search-libraries-2026-09-29 P-001: AUTO-BAR-R-1-P-001,
 * AUTO-BAR-R-3-P-001, SPEC-P-001-PAPERCUSP-WIDTH-REFUSAL).
 *
 * __fixtures__/embedding-space-parity-cases.json holds the outputs of the
 * PRE-MOVE functions (see its `provenance`) for every filter call site's
 * columns and every selection kind, plus the host config papercusp passes.
 * Papercusp's embedding-space-parity.test.ts rebuilds the fixture from the
 * pre-move code and the live tree and requires it to equal this copy, so the
 * pins cannot drift and a new call site cannot go unpinned.
 *
 * SQL is compared through a recording tag: text with $n placeholders and the
 * bind values, whitespace collapsed.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { createEmbeddingSpace, type EmbeddingSpaceConfig } from './embedding-space';
import type { PgHandle } from './types';

interface Rendered { text: string; binds: unknown[] }
interface Fixture {
  hostConfig: Required<Pick<EmbeddingSpaceConfig, 'storage' | 'storageLabel'>> & { legacyModes: NonNullable<EmbeddingSpaceConfig['legacyModes']> };
  census: Array<{ fn: string; file: string; args: string; literal: { profileColumn: string; modeColumn: string } | null }>;
  columnPairs: Array<{ profileColumn: string; modeColumn: string; from: string[] }>;
  predicate: Array<{ selection: { profileId: string; legacyMode: string | null } | null; profileColumn: string; modeColumn: string; expected: Rendered }>;
  sourceFilter: Array<{ embeddingProfile: { profileId: string; legacyMode: string | null } | null; profileColumn: string; modeColumn: string; expected: Rendered }>;
  effectiveStored: Array<{ profileColumn: string; modeColumn: string; expected: Rendered }>;
  compatibility: Array<{ profile: { profileId: string; dimensions: number; distanceMetric: string }; expected: string[] }>;
  resolveSelection: Array<{ mode: string; profile: { profileId: string; dimensions: number; distanceMetric: string }; expected: unknown }>;
  resolveCurrent: Array<{ mode: string; expected: unknown }>;
  resolveProfileId: Array<{ profileId: string; legacyMode: string | null; expected: unknown }>;
  resolveAccepted: Array<{ profileId: string; expected: unknown }>;
  identity: Array<{ stored: { profileId?: string | null; mode?: string | null }; selection: { profileId: string; legacyMode: string | null }; expected: boolean }>;
  fits: Array<{ dims: number; expected: boolean }>;
  widthSkew: { measured: Array<{ table: string; column: string; dims: number }>; expected: unknown };
}

const fixture = JSON.parse(
  readFileSync(new URL('./__fixtures__/embedding-space-parity-cases.json', import.meta.url), 'utf8'),
) as Fixture;

const REC = Symbol('recorded-sql');
type Recorded = { [REC]: true; strings?: readonly string[]; values?: unknown[]; unsafe?: string };
function recordingSql(): PgHandle {
  const tag = (strings: TemplateStringsArray, ...values: unknown[]): Recorded => ({ [REC]: true, strings: [...strings], values });
  (tag as unknown as { unsafe: (t: string) => Recorded }).unsafe = (text: string) => ({ [REC]: true, unsafe: text });
  return tag as unknown as PgHandle;
}
function render(fragment: unknown): Rendered {
  const binds: unknown[] = [];
  const walk = (node: unknown): string => {
    const r = node as Recorded;
    if (r && typeof r === 'object' && r[REC]) {
      if (r.unsafe !== undefined) return r.unsafe;
      return r.strings!.map((s, i) => s + (i < r.values!.length ? walk(r.values![i]) : '')).join('');
    }
    binds.push(node);
    return `$${binds.length}`;
  };
  return { text: walk(fragment).replace(/\s+/g, ' ').trim(), binds };
}

const sql = recordingSql();
const space = createEmbeddingSpace(fixture.hostConfig);
const norm = (v: unknown) => JSON.parse(JSON.stringify(v)) as unknown;

/** Every case where `s` disagrees with the fixture, as readable labels. */
function mismatches(s: ReturnType<typeof createEmbeddingSpace>): string[] {
  const out: string[] = [];
  const cmp = (label: string, actual: unknown, expected: unknown) => {
    if (JSON.stringify(norm(actual)) !== JSON.stringify(expected)) out.push(label);
  };
  for (const c of fixture.predicate) cmp(`predicate ${c.profileColumn} ${JSON.stringify(c.selection)}`, render(s.predicateSql(sql, c.selection, c.profileColumn, c.modeColumn)), c.expected);
  for (const c of fixture.sourceFilter) {
    const params = c.embeddingProfile ? { sql, embeddingProfile: c.embeddingProfile } : { sql };
    cmp(`source ${c.profileColumn} ${JSON.stringify(c.embeddingProfile)}`, render(s.sourceFilterSql(params, c.profileColumn, c.modeColumn)), c.expected);
  }
  for (const c of fixture.effectiveStored) cmp(`stored ${c.profileColumn}`, render(s.effectiveStoredProfileIdSql(sql, c.profileColumn, c.modeColumn)), c.expected);
  for (const c of fixture.compatibility) cmp(`compat ${JSON.stringify(c.profile)}`, s.validateCompatibility(c.profile), c.expected);
  for (const c of fixture.resolveSelection) cmp(`select ${c.mode} ${JSON.stringify(c.profile)}`, s.resolveSelection(c.mode, c.profile), c.expected);
  for (const c of fixture.resolveCurrent) cmp(`current ${c.mode}`, s.resolveCurrentSelection(c.mode), c.expected);
  for (const c of fixture.resolveProfileId) cmp(`byId ${c.profileId} ${c.legacyMode}`, s.resolveProfileIdSelection(c.profileId, c.legacyMode), c.expected);
  for (const c of fixture.resolveAccepted) cmp(`accepted ${c.profileId}`, s.resolveAcceptedSelection(c.profileId), c.expected);
  for (const c of fixture.identity) cmp(`identity ${JSON.stringify(c.stored)} ${c.selection.profileId}`, s.storedIdentityMatches(c.stored, c.selection), c.expected);
  for (const c of fixture.fits) cmp(`fits ${c.dims}`, s.fitsStorage(c.dims), c.expected);
  cmp('widthSkew', s.computeColumnWidthSkew(fixture.widthSkew.measured), fixture.widthSkew.expected);
  return out;
}

describe('createEmbeddingSpace — parity with the pre-move papercusp functions', () => {
  it('the fixture pins every census call site and every selection kind', () => {
    expect(fixture.census.length).toBeGreaterThan(0);
    const pairKeys = new Set(fixture.columnPairs.map((p) => `${p.profileColumn}\t${p.modeColumn}`));
    for (const site of fixture.census) {
      if (site.literal) expect(pairKeys.has(`${site.literal.profileColumn}\t${site.literal.modeColumn}`), `${site.fn}@${site.file}`).toBe(true);
    }
    // Non-literal sites receive columns only through a chunk surface's spaceFilter.
    expect(fixture.columnPairs.some((p) => p.from.some((f) => f.startsWith('spaceFilter@chunk-surface:')))).toBe(true);
    const selectionsPerPair = fixture.predicate.length / fixture.columnPairs.length;
    expect(selectionsPerPair).toBeGreaterThanOrEqual(5);
    expect(fixture.predicate.some((c) => c.selection === null)).toBe(true);
    expect(fixture.predicate.some((c) => c.selection?.legacyMode === null)).toBe(true);
    expect(fixture.predicate.some((c) => c.selection?.legacyMode)).toBe(true);
    expect(fixture.effectiveStored).toHaveLength(fixture.columnPairs.length);
    expect(fixture.compatibility.some((c) => c.expected.length === 0)).toBe(true);
    expect(fixture.compatibility.some((c) => c.expected.length > 1)).toBe(true);
  });

  it('every pinned case: same SQL text and binds, same verdicts, same selections (R-1, R-3)', () => {
    expect(mismatches(space)).toEqual([]);
  });

  it('controls: a wrong storage label or a reordered legacy-mode map is caught', () => {
    const relabelled = createEmbeddingSpace({ ...fixture.hostConfig, storageLabel: 'vector storage' });
    expect(mismatches(relabelled).some((l) => l.startsWith('compat '))).toBe(true);
    const reordered = createEmbeddingSpace({
      ...fixture.hostConfig,
      legacyModes: Object.fromEntries(Object.entries(fixture.hostConfig.legacyModes).reverse()),
    });
    expect(mismatches(reordered).some((l) => l.startsWith('stored '))).toBe(true);
  });
});

describe('papercusp width refusal (SPEC-P-001-PAPERCUSP-WIDTH-REFUSAL)', () => {
  it('the host config declares 768', () => {
    expect(space.storage.dimensions).toBe(768);
  });
  it('a 1024-wide stored column is reported as skewed; a 768 one is not', () => {
    const skew = space.computeColumnWidthSkew([
      { table: 'harness_shared.x', column: 'embedding', dims: 1024 },
      { table: 'harness_shared.y', column: 'embedding', dims: 768 },
    ]);
    expect(skew).toEqual([{ table: 'harness_shared.x', column: 'embedding', liveDims: 1024, declaredDims: 768 }]);
  });
  it('a 1024-wide profile with an accepted id is refused, and does not fit', () => {
    const accepted = fixture.hostConfig.storage.acceptedProfileIds[0]!;
    expect(space.validateCompatibility({ profileId: accepted, dimensions: 1024, distanceMetric: 'cosine' })).toEqual([
      `shared prose storage has 768 dimensions; profile ${accepted} emits 1024`,
    ]);
    expect(space.resolveSelection('gemma', { profileId: accepted, dimensions: 1024, distanceMetric: 'cosine' })).toBeNull();
    expect(space.fitsStorage(1024)).toBe(false);
  });
});

describe('createEmbeddingSpace — contract', () => {
  const storage = { acceptedProfileIds: ['a@v1'], dimensions: 3, distanceMetric: 'cosine', indexOperatorClass: 'vector_cosine_ops' };
  it('without legacy modes, a NULL profile maps to NULL (never a guess)', () => {
    const s = createEmbeddingSpace({ storage });
    expect(render(s.effectiveStoredProfileIdSql(sql, 'embedding_profile', 'embedding_mode'))).toEqual({
      text: 'CASE WHEN embedding_profile = ANY($1::text[]) THEN embedding_profile WHEN embedding_profile IS NULL THEN NULL ELSE NULL END',
      binds: [['a@v1']],
    });
    expect(s.resolveAcceptedSelection('a@v1')).toEqual({ profileId: 'a@v1', legacyMode: null });
  });
  it('a missing selection compiles to FALSE', () => {
    expect(render(createEmbeddingSpace({ storage }).predicateSql(sql, null, 'p', 'm')).text).toBe('FALSE');
  });
  it('refuses a column that is not an identifier and a mode label unsafe to inline', () => {
    const s = createEmbeddingSpace({ storage });
    expect(() => s.predicateSql(sql, { profileId: 'a@v1', legacyMode: null }, 'p; DROP TABLE x', 'm')).toThrow(/not a column identifier/);
    expect(() => createEmbeddingSpace({ storage, legacyModes: { "x' OR '1": { profileId: 'a@v1', dimensions: 3, distanceMetric: 'cosine' } } })).toThrow(/not safe to inline/);
  });
  it('refuses a non-positive width', () => {
    expect(() => createEmbeddingSpace({ storage: { ...storage, dimensions: 0 } })).toThrow(/positive integer/);
  });
  it('an unsupported storage metric is a refusal reason, not a throw', () => {
    const s = createEmbeddingSpace({ storage: { ...storage, distanceMetric: 'hamming' } });
    expect(s.validateCompatibility({ profileId: 'a@v1', dimensions: 3, distanceMetric: 'hamming' })).toEqual([
      'vector storage has unsupported metric hamming',
    ]);
  });
});
