-- Reference migration: the shared chunk store read and written by
-- sharedChunkStore() in @papercusp/search.
--
-- Copy this into your own migrations. The table is created UNQUALIFIED, so it
-- lands in the first schema on your search_path; pass the same schema-qualified
-- name to sharedChunkStore({ table: 'your_schema.text_chunks' }).
--
-- Requires pgvector >= 0.8 (CREATE EXTENSION vector) for the HNSW index and
-- iterative scans. vector(768) is an example width: use your embedder's.
--
-- One row is one chunk of one parent row. Every collection you register writes
-- into this one table, told apart by `surface`, so registering a collection is
-- a registry entry in code and never DDL.
--
--   surface           the registry entry's `surface` name
--   parent_key        the parent's primary key as text, in the entry's key order
--   chunk_idx         0-based position within the parent
--   anchor, header    the markdown heading anchor (NULL for window chunks) and
--                     the context line embedded before the content
--   parent_sha        hash of the text the chunks were cut from
--   chunk_sha         hash of header + content: a re-split copies the vector of
--                     an old chunk with the same chunk_sha, so an edit
--                     re-embeds only the chunks whose text changed
--   splitter_version  how the parent was cut; a registry edit to the splitter
--                     re-cuts instead of trusting chunks cut the old way
--   embedding         NULL until your embed sweep fills it (embedPendingChunks)
--   updated_at        the parent's versionSql value as read at sync time, so a
--                     parent edited mid-sync is picked up again
--
-- There is no foreign key: parents live in tables with different key shapes.
-- The sync engine's prune removes the chunks of deleted parents, and every
-- query (chunkAwareVectorLeg) joins back to the parent, so a chunk of a
-- deleted row is never returned even before the prune reaches it.

CREATE TABLE IF NOT EXISTS text_chunks (
  surface           text        NOT NULL,
  parent_key        text[]      NOT NULL,
  chunk_idx         integer     NOT NULL,
  anchor            text,
  header            text,
  content           text        NOT NULL,
  parent_sha        text        NOT NULL,
  chunk_sha         text        NOT NULL,
  splitter_version  text        NOT NULL,
  embedding         vector(768),
  embedding_mode    text,
  embedding_profile text,
  updated_at        timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (surface, parent_key, chunk_idx),
  CONSTRAINT text_chunks_surface_nonempty CHECK (surface <> ''),
  CONSTRAINT text_chunks_parent_key_nonempty CHECK (cardinality(parent_key) > 0),
  CONSTRAINT text_chunks_chunk_idx_nonnegative CHECK (chunk_idx >= 0)
);

-- Freshest-first walks (an embed sweep, a recency window).
CREATE INDEX IF NOT EXISTS text_chunks_updated_idx ON text_chunks (updated_at);

-- The chunk leg of chunkAwareVectorLeg orders by cosine distance.
CREATE INDEX IF NOT EXISTS text_chunks_embedding_hnsw_idx
  ON text_chunks USING hnsw (embedding vector_cosine_ops);
