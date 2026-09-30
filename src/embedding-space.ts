/**
 * Embedding-space safety: which stored vectors a query may compare against.
 *
 * A vector is only comparable to vectors produced by the same model, recipe and
 * width. Rows record that identity in two label columns:
 *
 *   - `<col>_profile` (e.g. `embedding_profile`) — the exact profile id that
 *     produced the stored vector. Authoritative whenever it is present.
 *   - `<col>_mode` (e.g. `embedding_mode`) — an older, coarser label naming the
 *     embedder MODE. A row whose profile column is NULL can still be placed,
 *     but only through the mode's declared CURRENT profile: equal width or an
 *     equal mode never makes an alternate profile compatible.
 *
 * The host supplies what it knows and the library does not: which profile ids
 * its physical columns accept, their width and distance metric, and the legacy
 * mode labels with the profile each one currently means. Everything else
 * (selection resolution, the SQL filters, the storage-compatibility verdict and
 * the live width check) is here, so every host fails closed the same way.
 *
 * Every SQL filter FAILS CLOSED: a missing or unknown selection compiles to SQL
 * FALSE, never to a mode-only match. Column names are schema identifiers from
 * the host's own call sites; they are checked against an identifier pattern and
 * inlined. Profile ids and modes a query supplies are always bound parameters.
 * Legacy mode LABELS are inlined as quoted SQL literals, because they are host
 * configuration (validated when the space is created), not query input.
 *
 * Extracted from papercusp's prose embedding columns
 * (shared-vector-search-libraries-2026-09-29, P-001).
 */
import type { Fragment } from 'postgres';
import type { PgHandle } from './types';

/** pgvector's distance metrics, as the storage contract names them. */
export type EmbeddingDistanceMetric = 'cosine' | 'l2' | 'inner-product' | 'l1';

/** The index operator class each pgvector metric requires. Closed on purpose:
 * no operator class is ever taken from configuration. */
export const PGVECTOR_INDEX_OPERATOR_CLASS: Readonly<Record<EmbeddingDistanceMetric, string>> = Object.freeze({
  cosine: 'vector_cosine_ops',
  l2: 'vector_l2_ops',
  'inner-product': 'vector_ip_ops',
  l1: 'vector_l1_ops',
});

/** What an embedder emits. `dimensions` is the width of one output vector. */
export interface EmbeddingProfileSpec {
  readonly profileId: string;
  readonly dimensions: number;
  readonly distanceMetric: string;
}

/** The physical contract of the host's vector columns. These are STORAGE
 * facts: they describe what the columns were created as, independently of
 * what any embedder happens to emit. */
export interface EmbeddingStorageContract {
  readonly acceptedProfileIds: readonly string[];
  readonly dimensions: number;
  readonly distanceMetric: string;
  readonly indexOperatorClass: string;
}

/** The exact row identity a query may select. `legacyMode` is non-null only
 * when a mode-only row can be read without guessing: the selected profile is
 * that mode's declared current profile. Same shape as
 * `SearchSourceParams.embeddingProfile`. */
export interface EmbeddingSpaceSelection {
  readonly profileId: string;
  readonly legacyMode: string | null;
}

export interface EmbeddingSpaceConfig {
  readonly storage: EmbeddingStorageContract;
  /**
   * Legacy mode label → the profile that mode CURRENTLY means. A row with a
   * NULL profile column is interpreted through this map and nothing else.
   * Insertion order is the order the stored-identity SQL tests the labels.
   * Omit it when every row records its profile.
   */
  readonly legacyModes?: Readonly<Record<string, EmbeddingProfileSpec>>;
  /** How messages name the storage, e.g. 'shared prose storage'. */
  readonly storageLabel?: string;
  /** Index operator class a metric requires, or undefined when unsupported.
   * Defaults to {@link PGVECTOR_INDEX_OPERATOR_CLASS}. */
  readonly indexOperatorClassFor?: (metric: string) => string | undefined;
}

/** A vector column whose live width disagrees with the storage contract. */
export interface ColumnWidthSkew {
  table: string;
  column: string;
  /** The width the column actually has, measured from the live catalog. */
  liveDims: number;
  /** The width the storage contract declares. */
  declaredDims: number;
}

/** The embedding-space filter a {@link createEmbeddingSpace} call returns. */
export interface EmbeddingSpace {
  readonly storage: EmbeddingStorageContract;
  /** Legacy mode labels, in configured order. */
  readonly legacyModes: readonly string[];
  /** Every reason `profile`'s output cannot be stored here; empty = compatible. */
  validateCompatibility(profile: EmbeddingProfileSpec): string[];
  /** An enabled embedder's exact selection, or null when storage refuses it. */
  resolveSelection(mode: string, profile: EmbeddingProfileSpec): EmbeddingSpaceSelection | null;
  /** The selection for a legacy mode's CURRENT profile; unknown modes are null. */
  resolveCurrentSelection(mode: string): EmbeddingSpaceSelection | null;
  /** A selection from provenance carried by a query embedder (profile id plus
   * the mode it ran under). Unaccepted ids are null; the legacy fallback is
   * granted only to the mode's declared current id. */
  resolveProfileIdSelection(profileId: string, legacyMode: string | null | undefined): EmbeddingSpaceSelection | null;
  /** A selection for an accepted id, granting the legacy fallback when the id
   * is the declared current profile of a legacy mode (first match wins). */
  resolveAcceptedSelection(profileId: string): EmbeddingSpaceSelection | null;
  /** SQL predicate: this row's stored vector is in `selection`'s space. */
  predicateSql(sql: PgHandle, selection: EmbeddingSpaceSelection | null, profileColumn: string, modeColumn: string): Fragment;
  /** The query-side filter a search source applies: the predicate for the
   * query's `embeddingProfile` provenance, FALSE when the query has none. */
  sourceFilterSql(
    params: { sql: PgHandle; embeddingProfile?: { profileId: string; legacyMode: string | null } },
    profileColumn: string,
    modeColumn: string,
  ): Fragment;
  /** SQL expression: a row's effective exact profile id, for row↔row
   * comparisons. Unknown identities are NULL and so never compare equal. */
  effectiveStoredProfileIdSql(sql: PgHandle, profileColumn: string, modeColumn: string): Fragment;
  /** The row-level identity judgement, for non-SQL consumers. */
  storedIdentityMatches(stored: { profileId?: string | null; mode?: string | null }, selection: EmbeddingSpaceSelection): boolean;
  /** Does an embedder's output width fit the columns? Pure and synchronous,
   * for hot paths. It compares against the DECLARED width only; use
   * {@link EmbeddingSpace.computeColumnWidthSkew} for the live schema. */
  fitsStorage(dims: number): boolean;
  /** Classify live column widths (pgvector `pg_attribute.atttypmod`, which
   * holds the dimension directly) against the declared width. Columns with a
   * non-positive or non-finite measurement are unjudgeable and skipped. */
  computeColumnWidthSkew(measured: ReadonlyArray<{ table: string; column: string; dims: number }>): ColumnWidthSkew[];
}

/** Schema identifier, optionally qualified (`alias.col`, `schema.table.col`),
 * each part plain or double-quoted. */
const IDENTIFIER = /^(?:[A-Za-z_][A-Za-z0-9_$]*|"(?:[^"]|"")+")(?:\.(?:[A-Za-z_][A-Za-z0-9_$]*|"(?:[^"]|"")+")){0,2}$/;
/** Legacy mode labels are inlined as SQL literals, so keep them boring. */
const MODE_LABEL = /^[A-Za-z0-9][A-Za-z0-9_.:-]*$/;

function column(name: string, role: string): string {
  if (!IDENTIFIER.test(name)) throw new Error(`embedding-space: ${role} ${JSON.stringify(name)} is not a column identifier`);
  return name;
}

function defaultIndexOperatorClassFor(metric: string): string | undefined {
  return Object.hasOwn(PGVECTOR_INDEX_OPERATOR_CLASS, metric)
    ? PGVECTOR_INDEX_OPERATOR_CLASS[metric as EmbeddingDistanceMetric]
    : undefined;
}

/**
 * Build the embedding-space filter for one storage contract. Throws on a
 * configuration that could not be enforced (a non-positive width, a legacy
 * mode label unsafe to inline); everything after construction fails closed
 * instead of throwing, except for a column name that is not an identifier.
 */
export function createEmbeddingSpace(config: EmbeddingSpaceConfig): EmbeddingSpace {
  const storage: EmbeddingStorageContract = Object.freeze({
    acceptedProfileIds: Object.freeze([...config.storage.acceptedProfileIds]),
    dimensions: config.storage.dimensions,
    distanceMetric: config.storage.distanceMetric,
    indexOperatorClass: config.storage.indexOperatorClass,
  });
  if (!Number.isInteger(storage.dimensions) || storage.dimensions <= 0) {
    throw new Error(`embedding-space: storage dimensions must be a positive integer, got ${storage.dimensions}`);
  }
  const legacy = new Map<string, EmbeddingProfileSpec>();
  for (const [mode, profile] of Object.entries(config.legacyModes ?? {})) {
    if (!MODE_LABEL.test(mode)) throw new Error(`embedding-space: legacy mode label ${JSON.stringify(mode)} is not safe to inline`);
    legacy.set(mode, Object.freeze({ ...profile }));
  }
  const label = config.storageLabel ?? 'vector storage';
  const indexOperatorClassFor = config.indexOperatorClassFor ?? defaultIndexOperatorClassFor;
  const accepted = (id: string) => storage.acceptedProfileIds.includes(id);
  const legacyModeFor = (mode: string | null | undefined, profileId: string): string | null =>
    mode != null && legacy.get(mode)?.profileId === profileId ? mode : null;

  function validateCompatibility(profile: EmbeddingProfileSpec): string[] {
    const problems: string[] = [];
    if (!accepted(profile.profileId)) {
      problems.push(
        `${label} does not accept profile ${profile.profileId}; ` +
          `accepted=${storage.acceptedProfileIds.join(',') || '(none)'}`,
      );
    }
    if (profile.dimensions !== storage.dimensions) {
      problems.push(`${label} has ${storage.dimensions} dimensions; profile ${profile.profileId} emits ${profile.dimensions}`);
    }
    if (profile.distanceMetric !== storage.distanceMetric) {
      problems.push(`${label} uses ${storage.distanceMetric}; profile ${profile.profileId} requires ${profile.distanceMetric}`);
    }
    const required = indexOperatorClassFor(storage.distanceMetric);
    if (!required) {
      problems.push(`${label} has unsupported metric ${storage.distanceMetric}`);
    } else if (required !== storage.indexOperatorClass) {
      problems.push(`${label} index uses ${storage.indexOperatorClass}; ${storage.distanceMetric} requires ${required}`);
    }
    return problems;
  }

  function resolveSelection(mode: string, profile: EmbeddingProfileSpec): EmbeddingSpaceSelection | null {
    if (validateCompatibility(profile).length > 0) return null;
    return { profileId: profile.profileId, legacyMode: legacyModeFor(mode, profile.profileId) };
  }

  function resolveProfileIdSelection(profileId: string, legacyMode: string | null | undefined): EmbeddingSpaceSelection | null {
    if (!accepted(profileId)) return null;
    return { profileId, legacyMode: legacyModeFor(legacyMode, profileId) };
  }

  function predicateSql(sql: PgHandle, selection: EmbeddingSpaceSelection | null, profileColumn: string, modeColumn: string): Fragment {
    if (!selection) return sql`FALSE`;
    const profile = sql.unsafe(column(profileColumn, 'profile column'));
    const mode = sql.unsafe(column(modeColumn, 'mode column'));
    return selection.legacyMode !== null
      ? sql`(${profile} = ${selection.profileId} OR (${profile} IS NULL AND ${mode} = ${selection.legacyMode}))`
      : sql`${profile} = ${selection.profileId}`;
  }

  return {
    storage,
    legacyModes: Object.freeze([...legacy.keys()]),
    validateCompatibility,
    resolveSelection,
    resolveCurrentSelection(mode) {
      const profile = legacy.get(mode);
      return profile ? resolveSelection(mode, profile) : null;
    },
    resolveProfileIdSelection,
    resolveAcceptedSelection(profileId) {
      const mode = [...legacy].find(([, p]) => p.profileId === profileId)?.[0] ?? null;
      return resolveProfileIdSelection(profileId, mode);
    },
    predicateSql,
    sourceFilterSql(params, profileColumn, modeColumn) {
      const selection = params.embeddingProfile
        ? resolveProfileIdSelection(params.embeddingProfile.profileId, params.embeddingProfile.legacyMode)
        : null;
      return predicateSql(params.sql, selection, profileColumn, modeColumn);
    },
    effectiveStoredProfileIdSql(sql, profileColumn, modeColumn) {
      const profile = sql.unsafe(column(profileColumn, 'profile column'));
      const mode = sql.unsafe(column(modeColumn, 'mode column'));
      const ids = [...storage.acceptedProfileIds];
      const branches = [...legacy].map(
        ([label, p]) => sql`WHEN ${mode} = ${sql.unsafe(`'${label}'`)} THEN ${p.profileId}`,
      );
      const legacyCase = branches.length === 0
        ? sql`NULL`
        : sql`CASE ${branches.reduce((acc, b) => sql`${acc} ${b}`)} ELSE NULL END`;
      return sql`CASE WHEN ${profile} = ANY(${ids}::text[]) THEN ${profile} WHEN ${profile} IS NULL THEN ${legacyCase} ELSE NULL END`;
    },
    storedIdentityMatches(stored, selection) {
      if (stored.profileId !== null && stored.profileId !== undefined) return stored.profileId === selection.profileId;
      return selection.legacyMode !== null && stored.mode === selection.legacyMode;
    },
    fitsStorage(dims) {
      return dims === storage.dimensions;
    },
    computeColumnWidthSkew(measured) {
      const skewed: ColumnWidthSkew[] = [];
      for (const m of measured) {
        if (!Number.isFinite(m.dims) || m.dims <= 0) continue;
        if (m.dims !== storage.dimensions) {
          skewed.push({ table: m.table, column: m.column, liveDims: m.dims, declaredDims: storage.dimensions });
        }
      }
      return skewed.sort((a, b) => a.table.localeCompare(b.table) || a.column.localeCompare(b.column));
    },
  };
}
