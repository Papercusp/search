/**
 * chunkAwareVectorLegSql SQL shape (generic-rag-chunking-2026-09-29 D-046).
 *
 * The mixed form `scan: 'exact', chunkScan: 'ann'` exists so a per-surface
 * partial HNSW index can serve the chunk leg. That only works if three things
 * about the emitted SQL hold, and none of them is visible in the rows a query
 * returns, so they are pinned here as text:
 *   1. the surface is a LITERAL (a bind parameter never matches a partial
 *      index's predicate under a generic plan);
 *   2. the chunk leg reads the chunk table alone, never the slice: membership
 *      is the join back to the slice after the LIMIT (D-048; D-046's in-scan
 *      jsonb membership filter forced an iterative scan and cost ~8 ms);
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

  it('reads the chunk table alone; slice membership is the join after the LIMIT (D-048)', () => {
    const leg = chunkLegOf(build({ chunkScan: 'ann' }).text);
    const inner = leg.slice(leg.indexOf('FROM (SELECT'), leg.indexOf(') hit'));
    expect(inner).toContain('FROM harness_shared.text_chunks c WHERE');
    // The ranked query never reads the slice, so membership cannot thin the index scan.
    expect(inner).not.toContain('chunk_leg_slice');
    expect(inner).not.toContain('to_jsonb');
    expect(inner).not.toMatch(/\bJOIN\b/);
    expect(leg).toMatch(/LIMIT \$\d+\) hit JOIN chunk_leg_slice cs ON hit\.parent_key = ARRAY\[\(cs\.workspace_id\)::text, \(cs\.conversation_id\)::text\]/);
  });

  it('applies chunkFilter inside the ranked query and chunkCandidates as its LIMIT', () => {
    const sql = fakeSql();
    const { text, params } = render(
      chunkAwareVectorLegSql(sql, {
        surface: SURFACE,
        parentAlias: 'cs',
        qVec: '[0.1,0.2]',
        limit: 1,
        mode: 'retrieve',
        scan: 'exact',
        chunkScan: 'ann',
        parentFilter: sql`cs.workspace_id = ${'ws-1'}` as never,
        chunkFilter: sql`c.parent_key[1] = ${'ws-1'}` as never,
        chunkCandidates: 40,
      }),
    );
    const leg = chunkLegOf(text);
    const inner = leg.slice(leg.indexOf('FROM (SELECT'), leg.indexOf(') hit'));
    const filter = inner.match(/AND \(c\.parent_key\[1\] = \$(\d+)\) ORDER BY c\.embedding <=> \$\d+::vector LIMIT \$(\d+)$/);
    expect(filter, inner).not.toBeNull();
    expect(params[Number(filter![1]) - 1]).toBe('ws-1');
    expect(params[Number(filter![2]) - 1]).toBe(40);
  });

  it('defaults chunkCandidates to limit and chunkFilter to TRUE', () => {
    const { text, params } = build({ chunkScan: 'ann', limit: 7 });
    const leg = chunkLegOf(text);
    const inner = leg.slice(leg.indexOf('FROM (SELECT'), leg.indexOf(') hit'));
    const m = inner.match(/AND \(TRUE\) ORDER BY c\.embedding <=> \$\d+::vector LIMIT \$(\d+)$/);
    expect(m, inner).not.toBeNull();
    expect(params[Number(m![1]) - 1]).toBe(7);
  });

  it('refuses chunkFilter or chunkCandidates outside the ANN-over-slice form', () => {
    const sql = fakeSql();
    expect(() => build({ chunkFilter: sql`TRUE` as never })).toThrow(/apply only to scan 'exact' with chunkScan 'ann'/);
    expect(() => build({ scan: 'ann', chunkCandidates: 40 })).toThrow(/apply only to scan 'exact' with chunkScan 'ann'/);
    expect(() => build({ chunkScan: 'ann', chunkCandidates: 0 })).toThrow(/chunkCandidates must be a positive integer/);
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

  // generic-rag-chunking D-047: a materialised parent vector is a detoasted copy per
  // row that the parent leg then re-reads; the slice carries the distance instead.
  it('materialises the parent distance, not the parent vector, in the exact slice', () => {
    for (const chunkScan of ['ann', 'exact'] as const) {
      const { text } = build({ chunkScan });
      const sliceSql = text.slice(text.indexOf('WITH chunk_leg_slice AS MATERIALIZED'), text.indexOf('SELECT workspace_id, conversation_id, distance'));
      expect(sliceSql).toMatch(/SELECT cs\.workspace_id, cs\.conversation_id, CASE WHEN cs\.query_embedding IS NOT NULL AND \(/);
      expect(sliceSql).toMatch(/THEN cs\.query_embedding <=> \$\d+::vector END AS chunk_leg_parent_distance/);
      // The vector and space columns are read only inside the CASE, never copied out.
      expect(sliceSql).not.toMatch(/, cs\.query_embedding,/);
      const parentLeg = text.slice(0, text.indexOf('UNION ALL'));
      expect(parentLeg).toMatch(/SELECT cs\.workspace_id, cs\.conversation_id, cs\.chunk_leg_parent_distance AS distance/);
      expect(parentLeg).toMatch(/WHERE cs\.chunk_leg_parent_distance IS NOT NULL ORDER BY cs\.chunk_leg_parent_distance LIMIT \$\d+\)/);
    }
  });

  it("control: scan 'ann' keeps the parent leg on the table, ordered by the vector", () => {
    const { text } = build({ scan: 'ann', chunkScan: undefined });
    expect(text).not.toContain('chunk_leg_slice');
    expect(text).toMatch(/FROM harness_shared\.consult_state cs WHERE/);
    expect(text).toMatch(/ORDER BY cs\.query_embedding <=> \$\d+::vector LIMIT/);
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
