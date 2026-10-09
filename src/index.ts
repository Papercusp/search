/**
 * @papercusp/search — a host-agnostic BM25 + pgvector hybrid search
 * engine over Postgres.
 *
 * The host registers a set of `SearchSource`s (each owning its own
 * tsvector / pgvector SQL + row→hit mapping) and the engine orchestrates
 * them: `runFullTextSearch` for BM25-only, `runHybridSearch` for
 * BM25 + embeddings fused via Reciprocal Rank Fusion (`@papercusp/rrf`).
 *
 * The Postgres handle and the query embedder are injected, so the package
 * carries no schema coupling and no embedding-provider dependency. Each
 * source call degrades independently (try/catch + log) so a missing
 * index/extension/column never fails the whole search.
 */

export type {
  PgHandle,
  Embedder,
  SearchHit,
  Listing,
  SearchSource,
  SearchSourceParams,
  SearchFilters,
} from './types';
export {
  runFullTextSearch,
  runHybridSearch,
  type SearchContext,
  type SearchResult,
  type HybridResult,
} from './hybrid';
export { applyRecencyRerank, toMillis, type RecencyRank } from './recency';
export { pickTopGroups, countGroups, type GroupKeyOf } from './group';
export {
  hitProvenance,
  isVectorOnly,
  isLexicalRanker,
  isSemanticRanker,
  type MatchProvenance,
  type HitProvenance,
} from './provenance';
export {
  applyMinScore,
  resolveMinScore,
  type MinScoreFloors,
  type MinScoreOutcome,
} from './min-score';
export {
  summariseLegs,
  finaliseLeg,
  newLegAccumulator,
  legOfRanker,
  emptyLegs,
  type LegStatus,
  type LegFailure,
  type LegReport,
  type SearchLegs,
  type LegAccumulator,
} from './legs';
export {
  observeLegs,
  readLegHealth,
  legHealthObservedCount,
  resetLegHealth,
  summariseLegSamples,
  sampleOfLegs,
  LEG_HEALTH_CAPACITY,
  type LegHealthWindow,
  type LegSample,
} from './leg-health';
export {
  observeEmbedLatency,
  beginEmbedLatencyTrace,
  beginEmbedLatencyStage,
  finishEmbedLatencyStage,
  finishEmbedLatencyTrace,
  readEmbedLatency,
  embedLatencyObservedCount,
  resetEmbedLatency,
  summariseEmbedSamples,
  EMBED_LATENCY_CAPACITY,
  EMBED_LATENCY_RECENT_SAMPLE_LIMIT,
  UNATTRIBUTED_CALLER,
  type EmbedLatencyWindow,
  type EmbedCallerLatency,
  type EmbedLatencySample,
  type EmbedLatencyOutcome,
  type EmbedLatencyStage,
  type EmbedLatencyStageDurations,
} from './embed-latency';
export {
  configureSearchDefaults,
  resetSearchDefaults,
  searchDefaultsHost,
  resolveDefault,
  resolveSearchDefaults,
  type SearchDefaultsHost,
  type SearchDefaultsContext,
  type AppliedDefaults,
} from './defaults';
export {
  splitWindows,
  splitMarkdown,
  splitOnLineBoundaries,
  chunkAnchor,
  sectionAnchorBase,
  isContinuationAnchor,
  CHUNK_ANCHOR_SEP,
  type WindowSplitOptions,
  type MarkdownSplitOptions,
  type MarkdownSection,
} from './chunk';
export * from './chunks/index';
export { withIterativeScan, resetIterativeScanProbe, type IterativeScanOptions } from './hnsw-iterative-scan';
export {
  createEmbeddingSpace,
  PGVECTOR_INDEX_OPERATOR_CLASS,
  type ColumnWidthSkew,
  type EmbeddingDistanceMetric,
  type EmbeddingProfileSpec,
  type EmbeddingSpace,
  type EmbeddingSpaceConfig,
  type EmbeddingSpaceSelection,
  type EmbeddingStorageContract,
} from './embedding-space';
export {
  DEFAULT_DESYNC_DISTANCE_THRESHOLD,
  DEFAULT_SELF_CHECK_MAX_AGE_MS,
  cosineDistance,
  createSelfCheckMemo,
  isEmbeddingDesync,
  parseVectorText,
  runStoredRowSelfCheck,
  type SelfCheckCanary,
  type SelfCheckMemo,
  type SelfCheckReading,
  type StoredRowSelfCheckDeps,
  type StoredRowSelfCheckResult,
} from './embedding-space-self-check';
export {
  DEFAULT_COVERAGE_THRESHOLDS,
  assessSurfaceCoverage,
  buildCoverageCountQuery,
  buildCoverageSnapshot,
  createCoverageGate,
  measureSurfaceCoverage,
  summarizeCoverage,
  toSurfaceReading,
  type CoverageCountQuery,
  type CoverageGate,
  type CoverageGateConfig,
  type CoverageSample,
  type CoverageSnapshot,
  type CoverageSqlHandle,
  type CoverageSurfaceSpec,
  type CoverageThresholds,
  type CoverageVerdict,
  type SearchCoverageReport,
  type SourceCoverageAssessment,
  type SurfaceReading,
} from './coverage-gate';
export {
  DEFAULT_BACKGROUND_SAMPLE_LIMIT,
  DEFAULT_MAX_CALIBRATED_CUT,
  DEFAULT_MIN_BACKGROUND_SAMPLES,
  DEFAULT_NEAR_DUPLICATE_QUANTILE,
  DEFAULT_SIMILARITY_DECIMALS,
  calibrateNearDuplicateCut,
  checkNearDuplicates,
  nearestRankQuantile,
  type CalibrateCutOptions,
  type DroppedNearDuplicateCandidate,
  type NearDuplicateCalibration,
  type NearDuplicateCandidate,
  type NearDuplicateCheck,
  type NearDuplicateOutcome,
} from './near-duplicate';
export * from './backfill/index';
export { rrfCombine, RRF_K_DEFAULT, type RankedItem, type FusedItem } from '@papercusp/rrf';
