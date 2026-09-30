/**
 * Read every schema fact the sweep needs about one target in ONE catalog query:
 * does the vector column exist, which label columns exist, and how wide is the
 * vector column really.
 */
import { modeColOf, profileColOf, recipeColOf } from './predicates';
import type { BackfillSql, BackfillTarget, BackfillWidthSkew } from './types';

export interface BackfillTargetInspection {
  readonly exists: boolean;
  /** `<embedCol>_mode` exists, so a vector in another space can be detected. */
  readonly spaceAware: boolean;
  readonly profileColPresent: boolean;
  readonly recipeColPresent: boolean;
  /** Live width of the vector column (pgvector's typmod IS the dimension), or
   * null when it could not be measured. */
  readonly liveDims: number | null;
  /** The host's width verdict on the live width; empty = the column fits. */
  readonly widthSkew: readonly BackfillWidthSkew[];
}

export type BackfillWidthSkewFn = (
  measured: ReadonlyArray<{ table: string; column: string; dims: number }>,
) => readonly BackfillWidthSkew[];

/**
 * `target.table` must be schema-qualified. An unmeasured width is unjudgeable,
 * never skew: absence of evidence must not refuse a target.
 */
export async function inspectBackfillTarget(
  sql: BackfillSql,
  target: BackfillTarget,
  widthSkew: BackfillWidthSkewFn = () => [],
): Promise<BackfillTargetInspection> {
  const rows = await sql.unsafe<Array<{ column_name: string; dims: number | null }>>(
    `SELECT col.column_name, a.atttypmod AS dims
       FROM information_schema.columns col
       LEFT JOIN pg_namespace n ON n.nspname = col.table_schema
       LEFT JOIN pg_class pc ON pc.relnamespace = n.oid AND pc.relname = col.table_name
       LEFT JOIN pg_attribute a
         ON a.attrelid = pc.oid AND a.attname = col.column_name AND NOT a.attisdropped
      WHERE col.table_schema = split_part($1, '.', 1)
        AND col.table_name   = split_part($1, '.', 2)
        AND col.column_name  IN ($2, $3, $4, $5)`,
    [target.table, target.embedCol, modeColOf(target), profileColOf(target), recipeColOf(target)],
  );
  const cols = new Set(rows.map((r) => r.column_name));
  const measured = rows.find((r) => r.column_name === target.embedCol)?.dims;
  const liveDims = measured === null || measured === undefined ? null : Number(measured);
  return {
    exists: cols.has(target.embedCol),
    spaceAware: cols.has(modeColOf(target)),
    profileColPresent: cols.has(profileColOf(target)),
    recipeColPresent: cols.has(recipeColOf(target)),
    liveDims,
    widthSkew: liveDims === null ? [] : widthSkew([{ table: target.table, column: target.embedCol, dims: liveDims }]),
  };
}
