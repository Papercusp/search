import type { Sql } from 'postgres';
import type { RankedItem } from '@papercusp/rrf';

/** The Postgres handle a search runs against — a postgres-js tagged template. */
export type PgHandle = Sql;

/** Turns a query string into an embedding vector (for the pgvector ranker). */
export type Embedder = (text: string) => Promise<number[]>;

/** One returned search hit, neutral across BM25 / embeddings / fused. */
export interface SearchHit {
  /** Source/ranker label, e.g. `escalations`. */
  source: string;
  source_id: string;
  /** Optional intra-workspace partition key the hit belongs to (host-defined). */
  scope?: string;
  /** First ~200 chars of the matched body. */
  excerpt: string;
  /** ts_headline snippet with <mark> around matched terms. */
  highlight: string;
  /** Ranker-native score (ts_rank_cd or cosine sim) or the fused RRF score. */
  score: number;
  /** Which rankers contributed this hit. */
  rankers: string[];
}

/** A single source's ranked list, keyed for RRF fusion. */
export type Listing = RankedItem<SearchHit>[];

/**
 * Optional structured filters a host may pass alongside the query
 * (session-search-scope-2026-07-05 P-002). Sources OPT IN: a source
 * applies the filters it has columns for and ignores the rest, so
 * passing a filter a source can't honor never errors — it just doesn't
 * narrow that source. Hosts resolve domain concepts (e.g. a fleet or
 * an audience selector) into the flat `owners` set BEFORE calling the
 * engine — this lib stays domain-free.
 */
export interface SearchFilters {
  /** Restrict to rows attributed to one of these owner ids (host-resolved set). */
  owners?: string[];
  /** Restrict to one turn speaker/role (e.g. 'user' | 'assistant' | 'tool'). */
  speaker?: string;
  /** Restrict to one session/conversation id. */
  sessionId?: string;
  /** Restrict to one source kind (e.g. a client/transport label). */
  sourceKind?: string;
  /** ISO timestamp inclusive lower bound. */
  since?: string;
  /** ISO timestamp exclusive upper bound. */
  until?: string;
}

/** Inputs handed to a source's ranker functions. */
export interface SearchSourceParams {
  sql: PgHandle;
  query: string;
  workspaceId: string;
  /** When non-null, restrict to this intra-workspace scope key (sources without a scope column ignore it). */
  scopeFilter: string | null;
  /** Per-source row cap. fulltext passes `limit`; hybrid over-fetches (`limit*3`). */
  limit: number;
  /** Optional structured filters (see SearchFilters). Absent = no narrowing. */
  filters?: SearchFilters;
}

/**
 * A pluggable search surface. Each source owns its own SQL — the table,
 * tsvector/embedding columns, headline expression, and row→hit mapping —
 * so the engine stays schema-agnostic. The host registers its sources.
 */
export interface SearchSource {
  /** Source/ranker label, e.g. `escalations`. */
  name: string;
  /** BM25 (tsvector `ts_rank_cd`) ranked list for this source. */
  bm25(p: SearchSourceParams): Promise<Listing>;
  /** Optional pgvector cosine-similarity ranked list (`<=>`) for this source. */
  embedding?(p: SearchSourceParams & { qVec: string }): Promise<Listing>;
}
