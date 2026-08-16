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
  configureSearchDefaults,
  resetSearchDefaults,
  searchDefaultsHost,
  resolveDefault,
  resolveSearchDefaults,
  type SearchDefaultsHost,
  type SearchDefaultsContext,
  type AppliedDefaults,
} from './defaults';
export { rrfCombine, RRF_K_DEFAULT, type RankedItem, type FusedItem } from '@papercusp/rrf';
