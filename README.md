# @papercusp/search

A host-agnostic **BM25 + pgvector hybrid search** engine over Postgres.

The host registers a set of `SearchSource`s — each owns its own SQL (the
table, the `tsvector` column for BM25, the embedding column for pgvector,
the `ts_headline` snippet, and the row→hit mapping) — and the engine
orchestrates them:

- `runFullTextSearch(sources, ctx)` — BM25 only (the `search:fulltext`
  tool). Merges all sources, re-ranks by raw `ts_rank_cd` score, top-N.
- `runHybridSearch(sources, ctx)` — BM25 + embeddings fused via
  Reciprocal Rank Fusion (`@papercusp/rrf`) (the `search:semantic` tool).
  `mode='hybrid'` runs both rankers; `mode='embeddings'` runs only the
  vector ranker.

## Injected seams

- **PG handle** (`PgHandle` = a postgres-js tagged template) — passed in
  per call, so the engine has no DB-connection coupling.
- **Embedder** (`Embedder` = `(text) => Promise<number[]>` | null) — the
  host supplies the query-embedding provider (or `null` to force
  BM25-only). The engine never reaches for a specific embedding API.

```ts
import { runHybridSearch, type SearchSource } from '@papercusp/search';

const result = await runHybridSearch(sources, {
  sql: ctx.tx,              // the PG handle
  query, workspaceId, harnessFilter, limit,
  mode: 'hybrid',
  embedder: await buildQueryEmbedder(), // host-provided, may be null
  log: ctx.log,
});
```

## Graceful degradation (P-013)

Every source call is independently `try/catch`-ed: a source whose index,
pgvector extension, or embedding column is missing degrades to "skipped"
(logged via `ctx.log`) rather than failing the whole search. This is the
explicit form of the old optional-`ctx.tx` contract — the DB handle is a
declared parameter, and absence/error per source yields empty, not a
crash.

## Extraction status

Extracted per `papercusp-systems-abstraction-2026-05-29`, items P-013
(explicit DB contract) + P-020. The Papercusp operator registers its four
prose sources (escalations, brainstorm, operator turns, decisions) and
its embedder cascade in `apps/operator/lib/agent-tools/search/`.
