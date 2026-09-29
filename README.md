# @papercusp/search

A host-agnostic **lexical + pgvector hybrid search** engine over Postgres.

> **Not BM25.** The lexical ranker is Postgres `ts_rank_cd`, which scores
> COVER DENSITY — how tightly the query terms cluster in the document — not
> Okapi BM25's TF/IDF with length normalisation. Earlier versions of this
> README and the package description said "BM25"; that was wrong, and the
> ranker was renamed `bm25` → `lexical` to stop the claim propagating. It
> matters in practice: cover density gives very coarse score granularity on
> single-term queries and does not penalise long documents, so results differ
> from BM25 in exactly the cases people reach for BM25 to fix.

The host registers a set of `SearchSource`s — each owns its own SQL (the
table, the `tsvector` column for the lexical leg, the embedding column for
pgvector, the `ts_headline` snippet, and the row→hit mapping) — and the
engine orchestrates them:

- `runFullTextSearch(sources, ctx)` — lexical only (the `search:fulltext`
  tool). Merges all sources, re-ranks by raw `ts_rank_cd` score, top-N.
- `runHybridSearch(sources, ctx)` — lexical + embeddings fused via
  Reciprocal Rank Fusion (`@papercusp/rrf`) (the `search:semantic` tool).
  `mode='hybrid'` runs both rankers; `mode='embeddings'` runs only the
  vector ranker.

## Injected seams

- **PG handle** (`PgHandle` = a postgres-js tagged template) — passed in
  per call, so the engine has no DB-connection coupling.
- **Embedder** (`Embedder` = `(text) => Promise<number[]>` | null) — the
  host supplies the query-embedding provider (or `null` to force
  lexical-only). The engine never reaches for a specific embedding API.

```ts
import { runHybridSearch, type SearchSource } from '@papercusp/search';

const result = await runHybridSearch(sources, {
  sql: ctx.tx,              // the PG handle
  query, workspaceId, scopeFilter, limit, // scopeFilter: optional intra-workspace scope key, or null
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

## Chunking

An embedder reads a bounded window, so text past it is invisible to the
vector leg. `chunk.ts` splits long text into pieces that are embedded one by
one; a chunked parent is scored by its best chunk, never the mean, since
averaging brings back the dilution chunking removes.

- `splitWindows(text, { size, overlap, maxChunks })` — fixed-width windows
  with overlap, content-blind. For unstructured prose (chat turns).
- `splitMarkdown(body, { maxChars, minChars?, headingDepth?, maxSections?, maxRows? })`
  — one `{ anchor, headingPath, content }` per heading, fence-aware. A section
  longer than `maxChars` is split on line boundaries into continuation parts
  whose anchor is `<anchor>~N` (`chunkAnchor`; `sectionAnchorBase` and
  `isContinuationAnchor` read it back). For documents.

```ts
import { splitMarkdown, splitWindows } from '@papercusp/search';

const windows = splitWindows(turnText, { size: 1500, overlap: 250, maxChunks: 16 });
const sections = splitMarkdown(pageBody, { maxChars: 2000 });
```

Both were moved here verbatim from the Papercusp operator's turn and doc
chunkers, and `chunk.golden.test.ts` pins their output against the pre-move
code, so a change to either shows up as a failing test instead of as stored
chunks that no longer match what the splitter produces.

## Extraction status

Extracted per `papercusp-systems-abstraction-2026-05-29`, items P-013
(explicit DB contract) + P-020. The Papercusp operator registers its four
prose sources (escalations, brainstorm, operator turns, decisions) and
its embedder cascade in `apps/operator/lib/agent-tools/search/`.
