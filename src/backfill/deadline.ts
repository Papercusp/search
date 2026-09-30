/**
 * Bound a promise that must not hang forever. The engine's embeds end in a
 * network call or a native model with no deadline of their own, and one hung
 * call would otherwise hold the sweep's single-flight latch indefinitely.
 *
 * `label` names the phase in the thrown error (`<label>_timeout_after_<ms>ms`),
 * so a hung resolver is not reported as a hung row.
 */
export function withDeadline<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return Promise.race([
    p,
    new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label}_timeout_after_${ms}ms`)), ms);
      // Never hold the process open on this timer alone.
      (timer as unknown as { unref?: () => void }).unref?.();
    }),
  ]).finally(() => {
    if (timer) clearTimeout(timer);
  });
}

/**
 * The first `max` characters of `text`, counting Unicode code points the way
 * PostgreSQL's `left()` does, so a surrogate pair is never split.
 */
export function truncateToChars(text: string, max: number): string {
  // UTF-16 length is at least the code-point count, so this is exact when it holds.
  if (text.length <= max) return text;
  let out = '';
  let n = 0;
  for (const ch of text) {
    if (n >= max) break;
    out += ch;
    n += 1;
  }
  return out;
}
