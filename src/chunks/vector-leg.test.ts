/**
 * chunkAwareVectorLegSql SQL shape (generic-rag-chunking-2026-09-29 D-046).
 *
 * The mixed form `scan: 'exact', chunkScan: 'ann'` exists so a per-surface
 * partial HNSW index can serve the chunk leg. That only works if three things
 * about the emitted SQL hold, and none of them is visible in the rows a query
 * returns, so they are pinned here as text:
 *   1. the surface is a LITERAL (a bind parameter never matches a partial
 *      index's predicate under a generic plan);
 *   2. the chunk leg reads the chunk table alone and tests slice membership as
 *      a jsonb filter, not a join (given a join, the planner probes per parent);
 *   3. the chunk leg orders by the vector operator itself.
 * The integration test (vector-leg-ann-chunk.integration.test.ts) checks that
 * real Postgres then uses the index and returns the right parents.
 */
import { describe, expect, it } from 'vitest';
import type { PgHandle } from '../types';
import { chunkAwareVectorLegSpaceColumns, chunkAwareVectorLegSql, type ChunkAwareVectorLegOptions } from './vector-leg';

type Frag = { kind: 'frag'; strings: readonly string[]; values: readonly unknown[] } | { kind: 'raw'; text: string };

/** A postgres.js stand-in that records fragments instead of executing them. */
function fakeSql(): PgHandle {
  const tag = ((strings: TemplateStringsArray, ...values: unknown[]): Frag => ({ kind: 'frag', strings: [...strings], values })) as unknown as PgHandle;
  (tag as unknown as { unsafe: (text: string) => Frag }).unsafe = (text: string) => ({ kind: 'raw', text });
  return tag;
}

function isFrag(v: unknown): v is Frag {
  return typeof v === 'object' && v !== null && ((v as Frag).kind === 'frag' || (v as Frag).kind === 'raw');
}

/** Render a fragment tree to text, with each bound value as $n. */
function render(root: unknown): { text: string; params: unknown[] } {
  const params: unknown[] = [];
  const walk = (f: Frag): string => {
    if (f.kind === 'raw') return f.text;
    let out = f.strings[0] ?? '';
    f.values.forEach((v, i) => {
      if (isFrag(v)) out += walk(v);
      else {
        params.push(v);
        out += `$${params.length}`;
      }
      out += f.strings[i + 1] ?? '';
    });
    return out;
  };
  return { text: walk(root as Frag).replace(/\s+/g, ' '), params };
}

const SURFACE = {
  surface: 'consult_questions',
  parent: { table: 'harness_shared.consult_state', key: ['workspace_id', 'conversation_id'] },
  chunkMargin: 0.04,
  parentVector: { column: 'query_embedding', profileColumn: 'query_embedding_profile', modeColumn: 'query_embedding_mode' },
} as ChunkAwareVectorLegOptions['surface'];

function build(extra: Partial<ChunkAwareVectorLegOptions> = {}): { text: string; params: unknown[] } {
  const sql = fakeSql();
  return render(
    chunkAwareVectorLegSql(sql, {
      surface: SURFACE,
      parentAlias: 'cs',
      qVec: '[0.1,0.2]',
      limit: 1,
      mode: 'retrieve',
      scan: 'exact',
      parentFilter: sql`cs.state = 'closed_answered'` as never,
      ...extra,
    }),
  );
}

/** The chunk leg: the UNION ALL branch after the parent leg. */
const chunkLegOf = (text: string) => text.slice(text.indexOf('UNION ALL'));

describe("chunkAwareVectorLegSql chunkScan 'ann' over an exact slice (D-046)", () => {
  it('names the surface as a SQL literal, never a bind parameter', () => {
    const { text, params } = build({ chunkScan: 'ann' });
    expect(chunkLegOf(text)).toContain("c.surface = 'consult_questions'");
    expect(params).not.toContain('consult_questions');
  });

  it('reads the chunk table alone, with slice membership as a jsonb filter', () => {
    const leg = chunkLegOf(build({ chunkScan: 'ann' }).text);
    const inner = leg.slice(leg.indexOf('FROM (SELECT'), leg.indexOf(') hit'));
    expect(inner).toContain('FROM harness_shared.text_chunks c WHERE');
    expect(inner).toContain(
      'to_jsonb(c.parent_key) = ANY (ARRAY( SELECT to_jsonb(ARRAY[(cs.workspace_id)::text, (cs.conversation_id)::text]) FROM chunk_leg_slice cs))',
    );
    // No join inside the ranked query: the only join is back to the slice, after the LIMIT.
    expect(inner).not.toMatch(/\bJOIN\b/);
    expect(leg).toMatch(/LIMIT \$\d+\) hit JOIN chunk_leg_slice cs ON hit\.parent_key = ARRAY\[\(cs\.workspace_id\)::text, \(cs\.conversation_id\)::text\]/);
  });

  it('orders the chunk leg by the vector operator, so an HNSW index can serve it', () => {
    const leg = chunkLegOf(build({ chunkScan: 'ann' }).text);
    expect(leg).toMatch(/ORDER BY c\.embedding <=> \$\d+::vector LIMIT \$\d+\) hit/);
    // The margin is still added to the reported distance.
    expect(leg).toMatch(/\(c\.embedding <=> \$\d+::vector\) \+ \$\d+::float8 AS distance/);
  });

  it('leaves the parent leg exact over the materialised slice', () => {
    const { text } = build({ chunkScan: 'ann' });
    expect(text).toContain('WITH chunk_leg_slice AS MATERIALIZED');
    const parentLeg = text.slice(0, text.indexOf('UNION ALL'));
    expect(parentLeg).toContain('FROM chunk_leg_slice cs');
  });

  it('control: without chunkScan the exact chunk leg joins the slice and binds the surface', () => {
    const { text, params } = build();
    const leg = chunkLegOf(text);
    expect(leg).toContain('JOIN chunk_leg_slice cs ON c.parent_key = ARRAY[(cs.workspace_id)::text, (cs.conversation_id)::text]');
    expect(leg).toMatch(/c\.surface = \$\d+ AND/);
    expect(params).toContain('consult_questions');
    expect(leg).not.toContain('to_jsonb');
  });

  it('hands the space filter the chunk columns in the ANN chunk leg too', () => {
    const cols = chunkAwareVectorLegSpaceColumns(fakeSql(), {
      surface: SURFACE,
      parentAlias: 'cs',
      mode: 'retrieve',
      scan: 'exact',
      chunkScan: 'ann',
    });
    expect(cols).toEqual([
      { profileColumn: 'cs.query_embedding_profile', modeColumn: 'cs.query_embedding_mode' },
      { profileColumn: 'c.embedding_profile', modeColumn: 'c.embedding_mode' },
    ]);
  });

  it("'gist' with chunkScan 'ann' has no chunk leg to change", () => {
    const { text } = build({ mode: 'gist', chunkScan: 'ann' });
    expect(text).not.toContain('UNION ALL');
    expect(text).not.toContain('text_chunks');
  });

  it("refuses chunkScan 'exact' without scan 'exact'", () => {
    expect(() => build({ scan: 'ann', chunkScan: 'exact' })).toThrow(/chunkScan 'exact' needs scan 'exact'/);
  });

  it('refuses an unknown chunkScan value', () => {
    expect(() => build({ chunkScan: 'approximate' as never })).toThrow(/unknown chunkScan 'approximate'/);
  });

  it('refuses the mixed form on a typed-keying chunk table', () => {
    expect(() => build({ chunkScan: 'ann', chunks: { table: 'harness_shared.session_turn_chunks', keying: 'typed' } })).toThrow(
      /needs a shared-keying chunk table/,
    );
  });

  it('refuses a surface name that cannot be inlined as a literal', () => {
    expect(() => build({ chunkScan: 'ann', surface: { ...SURFACE, surface: "x' OR '1'='1" } })).toThrow(
      /cannot be inlined as a SQL literal/,
    );
  });
});
