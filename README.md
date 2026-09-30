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

## Chunked collections: adding one is a registry entry

The splitters above are the pure part. `chunks/` keeps stored chunks in step
with the rows they were cut from, embeds them, and searches them, so a
collection whose rows run past the embedder's window becomes searchable past
the cut by adding ONE registry entry: no per-collection code, no migration.

**Once per host** (not per collection):

1. Apply the reference migration
   [`sql/text-chunks.reference.sql`](sql/text-chunks.reference.sql): the shared
   chunk table every collection writes into (pgvector >= 0.8; pick your
   embedder's width).
2. Build one store for it: `const store = sharedChunkStore({ table: 'app.text_chunks' })`.
3. On a tick, run the three generic loops over your registry:
   - `syncChunkSurfaces(sql, REGISTRY, store, { hash })` splits new and edited
     parents, copies the vector of every chunk whose text did not change, and
     prunes the chunks of deleted parents;
   - for each target in `chunkEmbedTargets(REGISTRY)`, call
     `embedPendingChunks(sql, target, { embed })` (or point your own embed sweep
     at the target: table, key columns, vector column, and the SQL of the text to
     embed);
   - search with `chunkAwareVectorLeg(sql, { surface, qVec, limit, mode })`:
     `'retrieve'` pools the parent vector with its chunks (a match anywhere in
     the text finds the parent, with the matched section's anchor), `'gist'`
     ranks by the parent vector alone (use it for duplicate and novelty checks).

**Per collection**, the entry:

```ts
import type { ChunkSurface } from '@papercusp/search';

export const NOTES: ChunkSurface = {
  surface: 'notes',                                   // stored in every chunk row
  parent: { table: 'app.notes', key: [{ column: 'id', type: 'int' }] },
  textSql: 'p.body',                                  // SQL over the parent row `p`
  headerSql: 'p.title',                               // embedded before every chunk
  versionSql: 'p.updated_at',                         // cheap change detection
  splitter: { kind: 'markdown', maxChars: 1500 },     // or { kind: 'window', size, overlap }
  maxChunks: 16,                                      // text past the last chunk is logged and counted
  parentVector: { column: 'embedding' },              // the row's existing vector, for search
  store,
};
```

The SQL fragments are host code, never user input; identifiers are validated
(`resolveChunkSurface`). A collection with its own chunk table instead of the
shared one implements `ChunkStore` and declares `queryTable` and `embedTarget`
so search and embedding still derive from the entry.

`chunks/simple-addition.integration.test.ts` is the proof: a fixture host with
no other code registers `notes` exactly like this and checks the whole path end
to end. Run it with `npm run test:integration` (it starts a throwaway Postgres
from the local binaries, or uses `SEARCH_TEST_PG_URL`).

## Extraction status

Extracted per `papercusp-systems-abstraction-2026-05-29`, items P-013
(explicit DB contract) + P-020. The Papercusp operator registers its four
prose sources (escalations, brainstorm, operator turns, decisions) and
its embedder cascade in `apps/operator/lib/agent-tools/search/`.
