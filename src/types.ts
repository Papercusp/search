import type { Sql } from 'postgres';
import type { RankedItem } from '@papercusp/rrf';

/** The Postgres handle a search runs against — a postgres-js tagged template. */
export type PgHandle = Sql;

/** Turns a query string into an embedding vector (for the pgvector ranker).
 *
 * The optional signal lets a transport-backed embedder stop work when an
 * interactive search gives up. Pure/in-process embedders may ignore it: that
 * preserves the useful legacy behavior where a timed-out cold model keeps
 * warming in the background. */
export type Embedder = (text: string, signal?: AbortSignal) => Promise<number[]>;

/** One returned search hit, neutral across BM25 / embeddings / fused. */
export interface SearchHit {
  /** Source/ranker label, e.g. `escalations`. */
  source: string;
  source_id: string;
  /** Optional intra-workspace partition key the hit belongs to (host-defined). */
  scope?: string;
  /** Optional timestamp for this hit (host-defined, engine-opaque — like `scope`).
   *  ISO string / epoch-ms / Date. Populated by a source that wants recency
   *  ranking; read by `RecencyRank.getTime` (which defaults to this field). */
  ts?: string | number | Date | null;
  /** First ~200 chars of the matched body. */
  excerpt: string;
  /** ts_headline snippet with <mark> around matched terms. */
  highlight: string;
  /** Ranker-native score (ts_rank_cd or cosine sim) or the fused RRF score. */
  score: number;
  /** Which rankers contributed this hit. */
  rankers: string[];
  /**
   * PRE-FUSION per-ranker native scores, keyed by ranker name — e.g.
   * `{ lexical: 0.0731, embeddings: 0.6142 }`. Populated by the engine so a
   * caller can render match PROVENANCE ("semantic match, cosine 0.61")
   * instead of guessing from `score`, which after fusion is an RRF value
   * (≈0.016 per rank-1 contribution) in nobody's units and comparable only
   * against other hits from the same search.
   *
   * Absent for a hit the engine did not fuse (a host constructing hits
   * directly), and for rankers that did not return this hit. Read it, never
   * assume a key is present.
   */
  rankerScores?: Record<string, number>;
  /**
   * POST-RETRIEVAL cross-encoder relevance score, attached by a Stage B rerank
   * (papercusp's `rerankProseHits`). Set by that stage, never by the engine.
   *
   * This is the only per-hit number that is a true query↔document relevance
   * MAGNITUDE, and therefore the only one a relevance THRESHOLD can be
   * expressed in: `score` is rank-derived after fusion, so flooring on it is a
   * rank cutoff that `limit` already performs, and `rankerScores` are
   * per-ranker and absent for rankers that did not return the hit.
   *
   * ABSENT — never 0 — for a hit the cross-encoder did not score, including
   * every hit of a call that degraded to retrieval order. Absence means "not
   * judged", so treat it as unknown rather than as a low score; the rerank
   * stage reports 0 for unscored rows internally, and collapsing that into this
   * field would make a degraded run look uniformly irrelevant.
   */
  rerankScore?: number;
  /**
   * HOST-DEFINED, ENGINE-OPAQUE per-hit metadata — the same contract as `scope`
   * and `ts`: a source sets it, the engine carries it through fusion untouched
   * (fusion spreads the row), and no ranker ever reads it.
   *
   * It exists because a caller downstream of fusion often has to distinguish
   * sub-populations WITHIN one source, and the alternatives are all worse: a
   * second query to re-fetch a column the source already had, parsing it back
   * out of `excerpt`, or promoting a domain concept into this generic type.
   * papercusp's first use is the `work_item` source stamping `lane`, so corpus
   * recall can tell a curated work-item from an observation-lane sample — two
   * populations that share a relation, an id space, and until then a label.
   *
   * Values are strings (or null) on purpose: this is a passthrough label bag,
   * not a place to smuggle structure the engine would then be tempted to rank on.
   */
  meta?: Readonly<Record<string, string | null>>;
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
  /** Restrict transcript rows to a persisted turn-origin verdict. */
  turnOrigin?: string;
  /** Restrict transcript rows to turns affirmatively typed as owner speech. */
  ownerOnly?: boolean;
  /**
   * Restrict transcript rows to turns that COULD be owner speech — proven owner
   * turns plus ones whose origin is unenrolled (a user turn with no
   * machine-injection envelope). Wider than `ownerOnly`, which a host may
   * define so strictly that it cannot match on some source kinds; narrower than
   * no filter, which admits machine-injected rows recorded under a user
   * speaker. Hosts that do not distinguish enrolment may treat it as `ownerOnly`.
   */
  ownerCandidates?: boolean;
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
  /** Optional abort signal threaded from the engine (SearchContext.signal).
   *  Sources OPT IN — e.g. statement-level query cancellation. The engine
   *  already stops STARTING new source calls once it aborts, so ignoring
   *  this is safe (just less prompt). */
  signal?: AbortSignal;
  /** WI-4734 highlight-deferral: when the engine passes `false`, the source MAY
   *  skip its (expensive) highlight expression and return `highlight: ''` for
   *  every row — the engine hydrates highlights AFTER fusion, for the final
   *  top-N only, via {@link SearchSource.hydrateHighlights}. The engine only
   *  passes `false` to a source that implements `hydrateHighlights`; absent /
   *  `true` = inline highlights, byte-identical to the pre-seam behavior.
   *  Rationale: with `limit*3` over-fetch across two rankers, inline
   *  highlighting computes ~6× more `ts_headline`s than the caller displays. */
  wantHighlight?: boolean;
  /**
   * How the LEXICAL leg combines the query's terms. Optional; absent or
   * `'and'` is the historical behaviour and must stay byte-identical.
   *
   * `'and'` — every lexeme must match (`plainto_tsquery`). Precise, and the
   * reason query length is inversely proportional to recall: each extra term
   * ELIMINATES documents.
   *
   * `'coverage-graded'` — anchor on the query's RAREST lexeme, admit any
   * document also matching at least one other, and order by how many DISTINCT
   * query lexemes each document matches before falling back to the native
   * lexical score. Extra terms then ADD EVIDENCE instead of eliminating
   * documents, and the result is a strict SUPERSET of `'and'` with the
   * `'and'` rows ranked first.
   *
   * ⚠ The obvious implementation — plain disjunction scored by the native
   * lexical rank — does NOT work, and fails in the expensive direction: a
   * frequency/density ranker cannot express query-term coverage, so the
   * documents matching every term sink far below the caller's cut while the
   * result set looks larger. Sources implementing this mode must order by
   * coverage FIRST. Measured on a real 421k-row corpus, that mistake put all
   * seven all-terms documents at positions 36-427 for a caller admitting 6.
   */
  lexicalMode?: 'and' | 'coverage-graded';
  /**
   * `coverage-graded` only: a cost budget for the candidate-narrowing anchor,
   * expressed in summed per-lexeme document frequency.
   *
   * Grading every disjunctive match is correct but unaffordable, so a source
   * implementing this mode narrows to an anchor set first. Anchoring on the
   * SINGLE rarest lexeme is the cheapest such narrowing and also a rarity
   * SELECTOR — it silently requires every result to contain the query's rarest
   * term, which drops high-coverage documents that merely lack it. Budgeting the
   * anchor over summed df bounds the same cost without that side effect.
   *
   * Absent ⇒ the source's own default. `0` ⇒ anchor on the single rarest lexeme
   * (the degenerate case), which is what a paired control arm passes.
   */
  lexicalAnchorDfBudget?: number;
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
  lexical(p: SearchSourceParams): Promise<Listing>;
  /** Optional pgvector cosine-similarity ranked list (`<=>`) for this source. */
  embedding?(p: SearchSourceParams & { qVec: string }): Promise<Listing>;
  /** WI-4734 highlight-deferral (opt-in): given the FINAL fused top-N hit ids
   *  for this source, return `source_id → highlight` in ONE batched query.
   *  Implementing this lets the engine pass `wantHighlight: false` to
   *  `lexical`/`embedding` under `SearchContext.deferHighlight`, so the expensive
   *  highlight expression runs for the displayed rows only. Best-effort: a
   *  throw degrades to empty highlights for this source, never fails the
   *  search. */
  hydrateHighlights?(p: SearchSourceParams, sourceIds: string[]): Promise<Map<string, string>>;
}
