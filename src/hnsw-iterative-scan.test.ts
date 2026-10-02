import { afterEach, describe, expect, it, vi } from 'vitest';
import type { PgHandle } from './types';
import { resetIterativeScanProbe, withIterativeScan } from './hnsw-iterative-scan';

type FakeSql = PgHandle;
type ReadOnlyTransactionRunner = <T>(body: (tx: PgHandle) => Promise<T>) => Promise<T>;

function makeSql(probeError?: Error & { code?: string }): {
  sql: FakeSql;
  probeCount: () => number;
  bodyTransactions: () => number;
} {
  let probeCount = 0;
  let bodyTransactions = 0;
  const root = vi.fn(async () => []) as unknown as FakeSql;
  root.begin = vi.fn(async (callback: (tx: PgHandle) => Promise<unknown>) => {
    let readOnly = false;
    const tx = vi.fn(async (strings: TemplateStringsArray) => {
      const statement = strings.join('');
      if (/SET TRANSACTION READ ONLY/i.test(statement)) {
        readOnly = true;
        bodyTransactions += 1;
      }
      if (/SET LOCAL hnsw\.iterative_scan/i.test(statement) && !readOnly) {
        probeCount += 1;
        if (probeError && probeCount === 1) throw probeError;
      }
      return [];
    }) as unknown as PgHandle;
    return callback(tx);
  }) as unknown as PgHandle['begin'];
  return { sql: root, probeCount: () => probeCount, bodyTransactions: () => bodyTransactions };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('withIterativeScan capability probing', () => {
  it('runs a transaction-scoped handle as-is, without probing or warning', async () => {
    // The measured shape (WI-10002536): sessions:search passes its transaction
    // handle, which has no `begin`, so every probe threw and warned "transiently".
    const tx = vi.fn(async () => []) as unknown as FakeSql;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const bodies: PgHandle[] = [];

    const result = await withIterativeScan(tx, async (handle) => {
      bodies.push(handle);
      return 'rows';
    });

    expect(result).toBe('rows');
    expect(bodies).toEqual([tx]);
    expect(tx).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
  });

  it('uses an injected bounded read-only transaction for both probe and query', async () => {
    const { sql, probeCount, bodyTransactions } = makeSql();
    const runReadOnlyTransaction = vi.fn(async (body: (tx: PgHandle) => Promise<unknown>) =>
      sql.begin(async (tx) => {
        await tx`SET TRANSACTION READ ONLY`;
        return body(tx as unknown as PgHandle);
      }),
    );
    const bodies: PgHandle[] = [];

    const result = await withIterativeScan(
      sql,
      async (handle) => {
        bodies.push(handle);
        return 'rows';
      },
      { runReadOnlyTransaction: runReadOnlyTransaction as unknown as ReadOnlyTransactionRunner },
    );

    expect(result).toBe('rows');
    expect(runReadOnlyTransaction).toHaveBeenCalledTimes(2);
    expect(bodyTransactions()).toBe(2);
    expect(probeCount()).toBe(0);
    expect(bodies[0]).not.toBe(sql);
  });

  it('keeps the unsupported-server fallback inside the injected bounded transaction', async () => {
    const { sql, bodyTransactions } = makeSql();
    const missingSetting = Object.assign(new Error('unrecognized configuration parameter "hnsw.iterative_scan"'), {
      code: '42704',
    });
    const runReadOnlyTransaction = vi.fn(async (body: (tx: PgHandle) => Promise<unknown>) => {
      if (runReadOnlyTransaction.mock.calls.length === 1) throw missingSetting;
      return sql.begin(async (tx) => {
        await tx`SET TRANSACTION READ ONLY`;
        return body(tx as unknown as PgHandle);
      });
    });
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const bodies: PgHandle[] = [];

    await withIterativeScan(sql, async (handle) => {
      bodies.push(handle);
      return 'rows';
    }, { runReadOnlyTransaction: runReadOnlyTransaction as unknown as ReadOnlyTransactionRunner });

    expect(runReadOnlyTransaction).toHaveBeenCalledTimes(2);
    expect(bodyTransactions()).toBe(1);
    expect(bodies[0]).not.toBe(sql);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('does not support'));
  });

  it('retries after a transient probe failure instead of poisoning the handle', async () => {
    const { sql, probeCount, bodyTransactions } = makeSql(
      Object.assign(new Error('connection reset by peer'), { code: '08006' }),
    );
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const bodies: PgHandle[] = [];

    await withIterativeScan(sql, async (handle) => {
      bodies.push(handle);
      return 'first';
    });
    await withIterativeScan(sql, async (handle) => {
      bodies.push(handle);
      return 'second';
    });

    expect(probeCount()).toBe(2);
    expect(bodyTransactions()).toBe(1);
    expect(bodies[0]).toBe(sql);
    expect(bodies[1]).not.toBe(sql);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('transiently'));
  });

  it('caches an undefined-object capability result and logs it once', async () => {
    const { sql, probeCount, bodyTransactions } = makeSql(
      Object.assign(new Error('unrecognized configuration parameter "hnsw.iterative_scan"'), {
        code: '42704',
      }),
    );
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const bodies: PgHandle[] = [];

    await withIterativeScan(sql, async (handle) => {
      bodies.push(handle);
      return 'first';
    });
    await withIterativeScan(sql, async (handle) => {
      bodies.push(handle);
      return 'second';
    });

    expect(probeCount()).toBe(1);
    expect(bodyTransactions()).toBe(0);
    expect(bodies[0]).toBe(sql);
    expect(bodies[1]).toBe(sql);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('does not support'));

    resetIterativeScanProbe(sql);
  });
});

/** Records every statement each transaction runs, in order, with its bound values. */
function recordingSql(): { sql: FakeSql; statements: Array<{ text: string; values: unknown[] }> } {
  const statements: Array<{ text: string; values: unknown[] }> = [];
  const root = vi.fn(async () => []) as unknown as FakeSql;
  root.begin = vi.fn(async (callback: (tx: PgHandle) => Promise<unknown>) => {
    const tx = vi.fn(async (strings: TemplateStringsArray, ...values: unknown[]) => {
      statements.push({ text: strings.join('$'), values });
      return [];
    }) as unknown as PgHandle;
    return callback(tx);
  }) as unknown as PgHandle['begin'];
  return { sql: root, statements };
}

describe('withIterativeScan efSearch (generic-rag-chunking D-046)', () => {
  const EF = /set_config\('hnsw\.ef_search'/;

  it('sets hnsw.ef_search transaction-locally, after iterative scan and before the body', async () => {
    const { sql, statements } = recordingSql();
    await withIterativeScan(sql, async (tx) => tx`SELECT body`, { efSearch: 100 });

    const body = statements.findIndex((s) => s.text === 'SELECT body');
    const ef = statements.findIndex((s) => EF.test(s.text));
    const iterative = statements.map((s) => s.text).lastIndexOf('SET LOCAL hnsw.iterative_scan = relaxed_order');
    expect(ef).toBeGreaterThan(iterative);
    expect(body).toBeGreaterThan(ef);
    // is_local => true is what makes it SET LOCAL; the value travels as a bind parameter.
    expect(statements[ef]!.text).toBe("SELECT set_config('hnsw.ef_search', $, true)");
    expect(statements[ef]!.values).toEqual(['100']);
  });

  it('sets it inside an injected read-only transaction runner too', async () => {
    const { sql, statements } = recordingSql();
    const runReadOnlyTransaction = vi.fn(async (body: (tx: PgHandle) => Promise<unknown>) =>
      sql.begin(async (tx) => body(tx as unknown as PgHandle)),
    );
    await withIterativeScan(sql, async (tx) => tx`SELECT body`, {
      efSearch: 200,
      runReadOnlyTransaction: runReadOnlyTransaction as unknown as ReadOnlyTransactionRunner,
    });
    const ef = statements.filter((s) => EF.test(s.text));
    expect(ef).toHaveLength(1);
    expect(ef[0]!.values).toEqual(['200']);
    expect(statements.findIndex((s) => s.text === 'SELECT body')).toBeGreaterThan(statements.findIndex((s) => EF.test(s.text)));
  });

  it('leaves ef_search alone when no efSearch is given', async () => {
    const { sql, statements } = recordingSql();
    await withIterativeScan(sql, async (tx) => tx`SELECT body`);
    expect(statements.some((s) => EF.test(s.text))).toBe(false);
    expect(statements.some((s) => s.text === 'SELECT body')).toBe(true);
  });

  it.each([0, 1001, 2.5, Number.NaN])('refuses efSearch %s before running anything', async (efSearch) => {
    const { sql, statements } = recordingSql();
    const body = vi.fn(async () => 'rows');
    await expect(withIterativeScan(sql, body, { efSearch })).rejects.toThrow(/efSearch must be an integer in 1\.\.1000/);
    expect(body).not.toHaveBeenCalled();
    expect(statements).toEqual([]);
  });
});
