/**
 * coverage-gate.ts — does a source's vector leg hold enough of its corpus to be
 * trusted? (shared-vector-search-libraries-2026-09-29 P-002, moved out of
 * papercusp's search/coverage-gate.ts.)
 *
 * A vector query over a near-empty index is indistinguishable from a healthy
 * one from the outside: it returns plausible nearest neighbours, fast. This
 * module turns per-surface coverage samples into a per-source verdict a search
 * tool can report, so a caller learns the semantic leg is running on a partial
 * index instead of receiving confident results drawn from it.
 *
 * Vocabulary. A SURFACE is one stored vector column, named by the host (for
 * example `schema.table.embedding`). A SOURCE is what a search is scoped to; it
 * is backed by zero or more surfaces. The host supplies both through
 * {@link createCoverageGate}, and supplies the samples; this module owns only
 * the verdict logic. {@link buildCoverageCountQuery} is an optional measure a
 * host can use to produce those samples from a table, its vector column and
 * its embedding-label column.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * THE LOAD-BEARING RULE: absence of evidence is NOT health.
 *
 * A missing sample, a stale sample, or an unmapped source all resolve to
 * `unknown`, never to `healthy`. Defaulting the unknown case to healthy would
 * report a green verdict for an index nobody has looked at, which is the
 * silent-confidence failure this module exists to remove.
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Deliberately NOT done here: this does not re-tune scores, apply a cosine
 * threshold, or drop results. It is an honesty signal. Dropping a degraded leg
 * outright would trade one dishonesty for another, because a 72%-covered
 * corpus is still substantially useful.
 *
 * Domain-free by construction (libs/generic contract): no host tables, no host
 * constants, no I/O except through the injected SQL handle.
 */

/** A vector column's latest coverage observation, as the host stored it. */
export interface CoverageSample {
  surface: string;
  observedAt: Date | string;
  /** Rows that should carry a vector. Numeric strings are accepted (bigint columns). */
  eligibleRows: number | string;
  /** Eligible rows that carry a vector in the active embedding space. */
  embeddedRows: number | string;
  /** Eligible rows written inside the host's recent window, when it tracks one. */
  recentEligible?: number | string | null;
  recentEmbedded?: number | string | null;
}

export interface CoverageThresholds {
  /** Embedded fraction below which a leg is `degraded`. */
  coverageFloor: number;
  /** Recent-window fraction at or above which a degraded leg is called historical backlog. */
  recentFloor: number;
  /** Below this many recent eligible rows the recent fraction is noise and is not reported. */
  minRecentSample: number;
  /** Samples older than this stop counting as evidence and read as `unknown`. */
  maxSampleAgeMs: number;
}

/**
 * Defaults: 95% total coverage, 99% recent coverage, 20 recent rows, and a
 * 90-minute sample age (a 30-minute measuring cadence tolerating two missed
 * ticks). A host overrides any of them in {@link createCoverageGate}.
 */
export const DEFAULT_COVERAGE_THRESHOLDS: Readonly<CoverageThresholds> = Object.freeze({
  coverageFloor: 0.95,
  recentFloor: 0.99,
  minRecentSample: 20,
  maxSampleAgeMs: 90 * 60 * 1000,
});

export type CoverageVerdict =
  /** Every known leg is at or above the floor. */
  | 'healthy'
  /** The best known leg is below the floor — hits may be missing. */
  | 'degraded'
  /** No fresh sample. NOT a synonym for healthy. */
  | 'unknown'
  /** This source has no embedding leg; the semantic verdict does not apply. */
  | 'not-semantic';

export interface SurfaceReading {
  surface: string;
  observedAt: Date;
  eligibleRows: number;
  embeddedRows: number;
  /** embedded / eligible, 0..1. `null` when eligible is 0 (nothing to embed). */
  pct: number | null;
  /** Recent-window coverage, when the sample carries enough recent rows. */
  recentPct: number | null;
  stale: boolean;
  ageMs: number;
}

export interface SourceCoverageAssessment {
  source: string;
  verdict: CoverageVerdict;
  /** Best known leg, 0..1 — the number the verdict is derived from. */
  coverage: number | null;
  /** Recent-window coverage of the best known leg, when available. */
  recentCoverage: number | null;
  /** Per-surface detail, including legs that were missing or stale. */
  surfaces: SurfaceReading[];
  /** Surfaces with no fresh sample. Non-empty ⇒ `coverage` is a partial view. */
  unknownSurfaces: string[];
  /** One line an agent or a human can act on. */
  note: string;
}

export interface SearchCoverageReport {
  /** True when ANY scoped source is degraded or unknown. */
  degraded: boolean;
  /** Sources whose semantic results are drawn from a partial index. */
  degradedSources: string[];
  /** Sources with no fresh evidence either way. */
  unknownSources: string[];
  perSource: SourceCoverageAssessment[];
  /** One-line summary, or null when everything semantic is healthy. */
  warning: string | null;
}

export type CoverageSnapshot = Map<string, SurfaceReading>;

const toCount = (v: number | string): number => Number(v);

/** One sample → one reading. Pure: `now` is the only clock. */
export function toSurfaceReading(
  sample: CoverageSample,
  now: Date,
  thresholds: Pick<CoverageThresholds, 'minRecentSample' | 'maxSampleAgeMs'> = DEFAULT_COVERAGE_THRESHOLDS,
): SurfaceReading {
  const observedAt = sample.observedAt instanceof Date ? sample.observedAt : new Date(sample.observedAt);
  const eligibleRows = toCount(sample.eligibleRows);
  const embeddedRows = toCount(sample.embeddedRows);
  const recentEligible =
    sample.recentEligible === null || sample.recentEligible === undefined ? null : toCount(sample.recentEligible);
  const recentEmbedded =
    sample.recentEmbedded === null || sample.recentEmbedded === undefined ? null : toCount(sample.recentEmbedded);
  const ageMs = now.getTime() - observedAt.getTime();
  return {
    surface: sample.surface,
    observedAt,
    eligibleRows,
    embeddedRows,
    pct: eligibleRows > 0 ? embeddedRows / eligibleRows : null,
    recentPct:
      recentEligible !== null && recentEmbedded !== null && recentEligible >= thresholds.minRecentSample
        ? recentEmbedded / recentEligible
        : null,
    stale: ageMs > thresholds.maxSampleAgeMs,
    ageMs,
  };
}

/**
 * Samples → snapshot keyed by surface. When a surface appears more than once
 * the NEWEST observation wins, so a host may pass an unsorted history.
 */
export function buildCoverageSnapshot(
  samples: Iterable<CoverageSample>,
  now: Date = new Date(),
  thresholds: Pick<CoverageThresholds, 'minRecentSample' | 'maxSampleAgeMs'> = DEFAULT_COVERAGE_THRESHOLDS,
): CoverageSnapshot {
  const snapshot: CoverageSnapshot = new Map();
  for (const sample of samples) {
    const reading = toSurfaceReading(sample, now, thresholds);
    const prior = snapshot.get(reading.surface);
    if (!prior || reading.observedAt.getTime() > prior.observedAt.getTime()) {
      snapshot.set(reading.surface, reading);
    }
  }
  return snapshot;
}

const pctStr = (v: number | null): string => (v === null ? 'n/a' : `${(v * 100).toFixed(1)}%`);

/**
 * Assess one set of surfaces under a label. Pure — no I/O.
 *
 * `label` is what the assessment is reported under (a source name, or any
 * consumer that runs a cosine query without being a search source).
 *
 * Multi-leg sources take the BEST known leg: a row is findable if EITHER leg
 * carries its vector, so the union is at least the max. That makes `coverage`
 * a conservative LOWER bound on findability — it can flag `degraded` slightly
 * early, never late, which is the right direction for an honesty signal.
 */
export function assessSurfaceCoverage(
  label: string,
  surfaces: readonly string[],
  snapshot: CoverageSnapshot,
  thresholds: Pick<CoverageThresholds, 'coverageFloor' | 'recentFloor'> = DEFAULT_COVERAGE_THRESHOLDS,
): SourceCoverageAssessment {
  if (surfaces.length === 0) {
    return {
      source: label,
      verdict: 'not-semantic',
      coverage: null,
      recentCoverage: null,
      surfaces: [],
      unknownSurfaces: [],
      note: `source '${label}' has no embedding leg (BM25-only); embedding coverage does not apply.`,
    };
  }

  const readings: SurfaceReading[] = [];
  const unknownSurfaces: string[] = [];
  for (const s of surfaces) {
    const r = snapshot.get(s);
    if (!r || r.stale) unknownSurfaces.push(s);
    if (r) readings.push(r);
  }

  const known = readings.filter((r) => !r.stale && r.pct !== null);
  if (known.length === 0) {
    return {
      source: label,
      verdict: 'unknown',
      coverage: null,
      recentCoverage: null,
      surfaces: readings,
      unknownSurfaces,
      note:
        `no fresh embedding-coverage sample for '${label}' ` +
        `(${unknownSurfaces.join(', ') || 'no surfaces sampled'}) — ` +
        `coverage is UNKNOWN, which is not the same as healthy. ` +
        `Results may be drawn from an under-populated index.`,
    };
  }

  const best = known.reduce((a, b) => ((b.pct ?? 0) > (a.pct ?? 0) ? b : a));
  const coverage = best.pct;
  const degraded = coverage !== null && coverage < thresholds.coverageFloor;
  const partial = unknownSurfaces.length > 0;

  const parts: string[] = [];
  if (degraded) {
    parts.push(
      `'${label}' embedding coverage is ${pctStr(coverage)} ` +
        `(${best.embeddedRows.toLocaleString()}/${best.eligibleRows.toLocaleString()} rows), ` +
        `below the ${pctStr(thresholds.coverageFloor)} floor — semantic hits for this source are ` +
        `drawn from a PARTIAL index and a better match may simply not be embedded yet.`,
    );
    if (best.recentPct !== null && best.recentPct >= thresholds.recentFloor) {
      parts.push(
        `Recent rows are ${pctStr(best.recentPct)} covered, so this is historical backlog ` +
          `rather than a live ingestion failure.`,
      );
    }
  } else {
    parts.push(`'${label}' embedding coverage is ${pctStr(coverage)} (at or above floor).`);
  }
  if (partial) {
    parts.push(`Partial view — no fresh sample for: ${unknownSurfaces.join(', ')}.`);
  }

  return {
    source: label,
    verdict: degraded ? 'degraded' : 'healthy',
    coverage,
    recentCoverage: best.recentPct,
    surfaces: readings,
    unknownSurfaces,
    note: parts.join(' '),
  };
}

/** Roll assessments up into one report for a tool response. Pure. */
export function summarizeCoverage(perSource: SourceCoverageAssessment[]): SearchCoverageReport {
  const degradedSources = perSource.filter((a) => a.verdict === 'degraded').map((a) => a.source);
  const unknownSources = perSource.filter((a) => a.verdict === 'unknown').map((a) => a.source);

  const bits: string[] = [];
  if (degradedSources.length > 0) {
    bits.push(
      `semantic results are DEGRADED for: ${degradedSources.join(', ')} — ` +
        `these surfaces are only partially embedded, so a better match may exist but be unindexed`,
    );
  }
  if (unknownSources.length > 0) {
    bits.push(
      `embedding coverage is UNKNOWN (no fresh sample) for: ${unknownSources.join(', ')} — ` +
        `treat these results as unverified rather than healthy`,
    );
  }

  return {
    degraded: degradedSources.length > 0 || unknownSources.length > 0,
    degradedSources,
    unknownSources,
    perSource,
    warning: bits.length > 0 ? bits.join('; ') : null,
  };
}

export interface CoverageGateConfig {
  /**
   * Which surfaces back each source. An EMPTY array means the source has no
   * embedding leg (lexical-only) — a different statement from "coverage
   * unknown", and the two must not collapse into one verdict.
   */
  sources: Readonly<Record<string, readonly string[]>>;
  thresholds?: Partial<CoverageThresholds>;
  /** The note for a source with no entry in `sources`. The verdict is `unknown` either way. */
  describeUnmappedSource?: (source: string) => string;
}

export interface CoverageGate {
  readonly sources: Readonly<Record<string, readonly string[]>>;
  readonly thresholds: Readonly<CoverageThresholds>;
  /** Samples → snapshot under this gate's age and recent-sample thresholds. */
  snapshot(samples: Iterable<CoverageSample>, now?: Date): CoverageSnapshot;
  /** Assess one configured source. An unmapped source is `unknown`, never healthy. */
  assessSource(source: string, snapshot: CoverageSnapshot): SourceCoverageAssessment;
  /** Assess surfaces a consumer names itself (a cosine query that is not a configured source). */
  assessSurfaces(label: string, surfaces: readonly string[], snapshot: CoverageSnapshot): SourceCoverageAssessment;
  /** Assess every scoped source and roll the result up. */
  assessScope(scope: readonly string[], snapshot: CoverageSnapshot): SearchCoverageReport;
}

const defaultUnmappedNote = (source: string): string =>
  `source '${source}' has no coverage mapping in the coverage gate's sources — ` +
  `treating as unknown. Map it so this reports a real verdict.`;

/** Configure a coverage gate for one host. */
export function createCoverageGate(config: CoverageGateConfig): CoverageGate {
  const thresholds: CoverageThresholds = Object.freeze({ ...DEFAULT_COVERAGE_THRESHOLDS, ...config.thresholds });
  for (const [key, value] of Object.entries(thresholds)) {
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError(`coverage gate threshold ${key} must be a finite non-negative number, got ${String(value)}`);
    }
  }
  if (thresholds.coverageFloor > 1 || thresholds.recentFloor > 1) {
    throw new RangeError('coverage gate floors are fractions and must not exceed 1');
  }
  const sources = config.sources;
  const describeUnmapped = config.describeUnmappedSource ?? defaultUnmappedNote;

  const assessSource = (source: string, snapshot: CoverageSnapshot): SourceCoverageAssessment => {
    const surfaces = Object.prototype.hasOwnProperty.call(sources, source) ? sources[source] : undefined;
    if (surfaces === undefined) {
      return {
        source,
        verdict: 'unknown',
        coverage: null,
        recentCoverage: null,
        surfaces: [],
        unknownSurfaces: [],
        note: describeUnmapped(source),
      };
    }
    return assessSurfaceCoverage(source, surfaces, snapshot, thresholds);
  };

  return {
    sources,
    thresholds,
    snapshot: (samples, now = new Date()) => buildCoverageSnapshot(samples, now, thresholds),
    assessSource,
    assessSurfaces: (label, surfaces, snapshot) => assessSurfaceCoverage(label, surfaces, snapshot, thresholds),
    assessScope: (scope, snapshot) => summarizeCoverage(scope.map((s) => assessSource(s, snapshot))),
  };
}

// ---------------------------------------------------------------------------
// Optional measure: produce a sample from a table, its vector column and its
// embedding-label column.
// ---------------------------------------------------------------------------

/** A stored vector column the host wants measured. */
export interface CoverageSurfaceSpec {
  /** Schema-qualified or bare table name. */
  table: string;
  vectorColumn: string;
  /**
   * Column naming the embedding space each vector was made in. When present, a
   * vector labelled with any other space counts as NOT embedded: coverage means
   * "embedded in the space queries use", and a vector from a retired model is
   * noise to a query in the active one.
   */
  labelColumn?: string;
  /**
   * SQL predicate selecting rows that should carry a vector (for example rows
   * whose text is non-empty). Omitted ⇒ every row is eligible. Host-authored
   * SQL: never build it from user input.
   */
  eligibleSql?: string;
  /** Timestamp column for the recent-window counts. Omitted ⇒ no recent counts. */
  recencyColumn?: string;
  /** Surface key reported in the sample. Defaults to `<table>.<vectorColumn>`. */
  surface?: string;
}

export interface CoverageCountQuery {
  sql: string;
  params: Array<string | number>;
  surface: string;
  hasRecent: boolean;
}

const IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/;

function quoteIdent(name: string, what: string): string {
  if (!IDENT.test(name)) throw new TypeError(`coverage measure: ${what} '${name}' is not a plain SQL identifier`);
  return `"${name}"`;
}

function quoteTable(name: string): string {
  const parts = name.split('.');
  if (parts.length > 2) throw new TypeError(`coverage measure: table '${name}' has more than one schema qualifier`);
  return parts.map((p) => quoteIdent(p, 'table')).join('.');
}

/**
 * Build the one counting query for a surface. Placeholders are allocated only
 * when used: Postgres rejects a Bind carrying a parameter the statement never
 * references, so a surface without a label column must not bind a label.
 */
export function buildCoverageCountQuery(
  spec: CoverageSurfaceSpec,
  activeLabel: string | null,
  recentWindowHours = 24,
): CoverageCountQuery {
  const params: Array<string | number> = [];
  const bind = (v: string | number): string => {
    params.push(v);
    return `$${params.length}`;
  };
  const vector = quoteIdent(spec.vectorColumn, 'vectorColumn');
  const eligible = spec.eligibleSql ? `(${spec.eligibleSql})` : 'TRUE';
  let embedded = `${vector} IS NOT NULL`;
  if (spec.labelColumn) {
    if (activeLabel === null) {
      throw new TypeError('coverage measure: a surface with a labelColumn needs the active label');
    }
    embedded += ` AND ${quoteIdent(spec.labelColumn, 'labelColumn')} = ${bind(activeLabel)}`;
  }
  const cols = [
    `count(*) FILTER (WHERE ${eligible})::bigint AS eligible_rows`,
    `count(*) FILTER (WHERE ${eligible} AND ${embedded})::bigint AS embedded_rows`,
  ];
  if (spec.recencyColumn) {
    if (!(recentWindowHours > 0)) throw new RangeError('coverage measure: recentWindowHours must be positive');
    const recent = `${quoteIdent(spec.recencyColumn, 'recencyColumn')} > now() - make_interval(hours => ${bind(recentWindowHours)})`;
    cols.push(
      `count(*) FILTER (WHERE ${eligible} AND ${recent})::bigint AS recent_eligible`,
      `count(*) FILTER (WHERE ${eligible} AND ${recent} AND ${embedded})::bigint AS recent_embedded`,
    );
  }
  return {
    sql: `SELECT ${cols.join(', ')} FROM ${quoteTable(spec.table)}`,
    params,
    surface: spec.surface ?? `${spec.table}.${spec.vectorColumn}`,
    hasRecent: Boolean(spec.recencyColumn),
  };
}

/** Structural SQL handle — the `sql.unsafe` shape of postgres.js. */
export interface CoverageSqlHandle {
  unsafe(query: string, params?: unknown[]): PromiseLike<unknown>;
}

interface CountRow {
  eligible_rows: string | number;
  embedded_rows: string | number;
  recent_eligible?: string | number | null;
  recent_embedded?: string | number | null;
}

/** Run {@link buildCoverageCountQuery} and return a sample stamped `now`. */
export async function measureSurfaceCoverage(
  sql: CoverageSqlHandle,
  spec: CoverageSurfaceSpec,
  activeLabel: string | null,
  options: { now?: Date; recentWindowHours?: number } = {},
): Promise<CoverageSample> {
  const q = buildCoverageCountQuery(spec, activeLabel, options.recentWindowHours);
  const rows = (await sql.unsafe(q.sql, q.params)) as CountRow[] | null;
  const row = rows?.[0];
  if (!row) throw new Error(`coverage measure: no count row for ${q.surface}`);
  return {
    surface: q.surface,
    observedAt: options.now ?? new Date(),
    eligibleRows: Number(row.eligible_rows),
    embeddedRows: Number(row.embedded_rows),
    recentEligible: q.hasRecent && row.recent_eligible != null ? Number(row.recent_eligible) : null,
    recentEmbedded: q.hasRecent && row.recent_embedded != null ? Number(row.recent_embedded) : null,
  };
}
