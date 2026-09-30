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

## Embedding-space safety

A vector can only be compared with vectors made by the same model, text
recipe and width. Nothing about a wrong-space vector looks wrong: the query
still returns neighbours, and they are noise. `createEmbeddingSpace` takes the
host's storage contract and returns the filters every query and every write
goes through.

```ts
import { createEmbeddingSpace } from '@papercusp/search';

const space = createEmbeddingSpace({
  storage: {
    acceptedProfileIds: ['notes-embedder@384'],
    dimensions: 384,
    distanceMetric: 'cosine',
    indexOperatorClass: 'vector_cosine_ops',
  },
  // Rows written before profile ids were recorded: mode label -> the profile it means now.
  legacyModes: { local: { profileId: 'notes-embedder@384', dimensions: 384, distanceMetric: 'cosine' } },
});

space.validateCompatibility(profile);            // [] means this embedder's output fits the columns
const selection = space.resolveSelection('local', profile); // null when storage refuses it
await sql`SELECT id FROM app.notes
          WHERE ${space.predicateSql(sql, selection, 'embedding_profile', 'embedding_mode')}`;
```

Rows carry two label columns beside the vector. `<col>_profile` holds the exact
profile id and is authoritative when present. `<col>_mode` is an older, coarser
label: a row with only a mode is read through that mode's declared current
profile and nothing else, so an equal width or an equal mode never makes a
different profile compatible.

Every filter fails closed. A missing or unknown selection compiles to SQL
`FALSE`, never to a mode-only match. `sourceFilterSql` is the version a
`SearchSource` applies to its vector leg. `computeColumnWidthSkew` compares the
live width of each vector column (pgvector's `pg_attribute.atttypmod`) with the
declared one.

`runStoredRowSelfCheck({ embed, pickCanary, alert, clear? })` catches drift
after the fact. It re-embeds one stored row's own text with the active embedder
and measures the cosine distance to the stored vector. Unchanged text under an
unchanged model lands near 0; past `threshold` (default 0.05) it calls `alert`.
A fixed canary string would only prove the embedder agrees with itself; a
stored row proves that what is on disk is what the active embedder produces
today. It never throws: a failing dependency returns a `skipped` result.

## Coverage gate: is the vector leg holding enough of the corpus?

A vector query over a half-built index returns plausible neighbours quickly and
looks healthy from the outside. The coverage gate turns per-column coverage
samples into a verdict a search tool can return with its results, so a caller
learns that hits may be missing.

```ts
import { createCoverageGate, measureSurfaceCoverage } from '@papercusp/search';

const gate = createCoverageGate({
  // A source maps to the vector columns behind it. [] means lexical-only.
  sources: { notes: ['app.notes.embedding'], tags: [] },
});

const sample = await measureSurfaceCoverage(sql, {
  table: 'app.notes',
  vectorColumn: 'embedding',
  labelColumn: 'embedding_profile', // a vector in any other space counts as missing
  eligibleSql: "body <> ''",
  recencyColumn: 'created_at',
}, activeProfileId);

const report = gate.assessScope(['notes'], gate.snapshot([sample]));
// report.degraded, report.warning, report.perSource[0].verdict
```

The verdict is `healthy`, `degraded` (the best column is below
`coverageFloor`, default 95%), `unknown` or `not-semantic`. Absence of evidence
is not health: a missing sample, a sample older than `maxSampleAgeMs` (default
90 minutes) and a source with no entry in `sources` are all `unknown`. The gate
does not drop results or change scores. A 72%-covered index is still useful;
the caller just has to know.

`measureSurfaceCoverage` is optional. A host that already counts coverage
passes its own `CoverageSample`s to `gate.snapshot`.

## Near-duplicate check with a corpus-calibrated cut

For "is this new document a copy of one we already have": a cheap matcher
(tokens, titles) proposes candidates, and the check keeps the ones whose vectors
are close enough to count as duplicates.

```ts
import { checkNearDuplicates } from '@papercusp/search';

const outcome = await checkNearDuplicates({
  candidates,                                      // what the cheap matcher suspects
  keyOf: (c) => c.id,
  similarities: (keys) => cosinesTo(newDoc, keys), // Map<key, cosine>
  sampleBackground: (excludeKeys, limit) => cosinesToRandomDocs(newDoc, excludeKeys, limit),
});
if (outcome.verdict) {
  outcome.kept;              // likely duplicates, plus candidates with no stored vector
  outcome.dropped;           // topically close, not duplicates
  outcome.calibration.cut;
}
```

The cut is relative to the corpus: by default, the 95th percentile of the new
document's similarity to 256 background documents. A fixed cosine cut only means
something for one embedding procedure. When pooling, normalisation or a prompt
prefix changes, the whole similarity scale shifts: in the corpus this came from,
a 0.6 cut chosen against controls at 0.46-0.49 later sat below every pair, so
nothing could ever be dropped.

Declining to decide is a normal outcome. With fewer than 32 usable background
values, or a cut above 0.98 (a background of near-copies), the check returns
`verdict: false` and keeps every candidate, so the caller falls back to its
cheap matcher. `absoluteOverride` forces a fixed cut. This compares documents
with documents; it is not a search floor, and neither threshold licenses the
other.

## Embedding backfill

A table's vector column falls behind whenever rows are written, edited, or
embedded under a profile or text recipe that is no longer current. The backfill
engine finds those rows and embeds them.

```ts
import { createBackfillSweeper } from '@papercusp/search';

const sweeper = createBackfillSweeper({
  getTargets: () => [{
    table: 'app.notes',
    embedCol: 'embedding',          // labels: embedding_mode, _profile, _recipe
    keyCols: ['id'],
    bodySql: "coalesce(title, '') || E'\\n' || left(body, 8000)",
    orderBySql: 'created_at DESC',  // drain order; name an indexed column
    recipeVersion: 2,               // bump with bodySql; older rows become stale
  }],
  getSql: () => sql,
  resolveEmbedder: async () => ({ mode: 'local', dims: 384, profile, embed, embedMany }),
  resolveProfileSelection: (mode, p) => space.resolveSelection(mode, p),
  acceptsWidth: (dims) => space.fitsStorage(dims),
  widthSkew: (measured) => space.computeColumnWidthSkew(measured),
  maxInputChars: 8000,
  batchSize: 64,
});

const result = await sweeper.run(); // BackfillStats per target, or { skipped: 'already_running' }
```

A row is stale when its vector is missing, or its labels name another embedding
space or an older text recipe. Texts are embedded in batches, and a failed batch
falls back to one row at a time so a single bad row cannot block the rest. Input
is truncated to `maxInputChars`, so an oversized row cannot fail and be retried
forever. A vector whose width `acceptsWidth` rejects is counted as an error and
not written.

Each sweep reads every target's columns from the catalog once: which label
columns exist (a table with no `_mode` column only ever fills missing vectors)
and the vector column's live width. If the host passes `widthSkew` and it
reports a mismatch, that target is skipped with an error naming the column;
without the hook, width is never refused. The sweep then drains the targets
round-robin, one batch per target per round, until they are done or the time
budget (default 200 s) runs out.

A sweeper runs one sweep at a time. Its latch lives in a `BackfillSweepState`;
pass a pinned one if the host module can load twice.

## Extraction status

Extracted per `papercusp-systems-abstraction-2026-05-29`, items P-013
(explicit DB contract) + P-020. The Papercusp operator registers its four
prose sources (escalations, brainstorm, operator turns, decisions) and
its embedder cascade in `apps/operator/lib/agent-tools/search/`.

Embedding-space safety, the coverage gate, the near-duplicate check and the
backfill engine were moved here by `shared-vector-search-libraries-2026-09-29`
(P-001, P-002, P-004, P-006). **As of 2026-09-30 they are in `src/` with their
tests, but they are not exported from the package root yet, and the Papercusp
operator still runs its own copies.** The root exports and the operator's
switch to them land together, after `generic-rag-chunking-2026-09-29` ships:
that plan's acceptance evidence pins the operator files the switch edits.

What stays in Papercusp is configuration: the 768-dimension prose storage and
its accepted profiles, the backfill target list, embed admission and the
sidecar embedder. Papercusp's `plans:new` adopts the near-duplicate check. The
work-item duplicate guard does not. On 493 real filings the calibrated cut alone
missed fewer duplicates (25 of 133 against the guard's 64), but it merged 251 of
581 distinct pairs against the guard's 32, and at filing time a false merge
refuses a legitimate item. The comparison is in
`docs/evidence/shared-vector-search-libraries-2026-09-29/p004-dupe-guard-comparison.json`
(decision D-005 of that plan).

Tests that run each module with no Papercusp code:
`embedding-space.integration.test.ts` and `backfill/backfill.integration.test.ts`
(real PostgreSQL with pgvector), `embedding-space-self-check.test.ts`,
`coverage-gate.test.ts` and `near-duplicate.test.ts`. `import-boundary.test.ts`
fails if anything in `src/` imports outside the package.
