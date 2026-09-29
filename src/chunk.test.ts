/**
 * Behaviour tests for the splitters (the byte-for-byte pins against the
 * pre-move code live in chunk.golden.test.ts).
 *
 * The window tests defend a MEASURED algorithm: the retrieval gain it was
 * shipped on is a measurement of this exact loop, and does not transfer to a
 * splitter that is merely similar (snapping to word boundaries, dropping a runt
 * final window, re-anchoring the last window). So the loop is pinned to a
 * frozen reference oracle, and the coverage/overlap properties are checked
 * against deliberately-wrong controls kept here permanently.
 */
import { describe, expect, it } from 'vitest';
import {
  chunkAnchor,
  isContinuationAnchor,
  sectionAnchorBase,
  splitMarkdown,
  splitOnLineBoundaries,
  splitWindows,
} from './chunk';

const W = { size: 1500, overlap: 250, maxChunks: 16 };

/** The windowing loop as it stood when it was measured, frozen. Not a copy to keep in sync. */
function referenceSplit(text: string, size: number, overlap: number): string[] {
  const out: string[] = [];
  const step = Math.max(1, size - overlap);
  for (let s = 0; s < text.length; s += step) {
    out.push(text.slice(s, s + size));
    if (s + size >= text.length) break;
  }
  return out;
}

/** Deterministic, non-repeating filler: repeated text would hide an off-by-one. */
const lorem = (n: number): string => {
  let s = '';
  let i = 0;
  while (s.length < n) s += `${i++} the quick brown fox jumps over the lazy dog. `;
  return s.slice(0, n);
};

describe('splitWindows', () => {
  it('matches the reference oracle across the boundary lengths', () => {
    for (const n of [1, 2, 999, 1499, 1500, 1501, 1749, 1750, 1751, 2000, 2001, 3000, 4730, 5000, 7999, 8000]) {
      const text = lorem(n);
      expect(splitWindows(text, W), `length ${n}`).toEqual(referenceSplit(text, 1500, 250));
    }
  });

  it('matches the oracle at other sizes', () => {
    const text = lorem(6000);
    for (const [size, overlap] of [[1000, 100], [2000, 500], [1500, 0], [800, 799]]) {
      expect(splitWindows(text, { size, overlap, maxChunks: 10_000 }), `${size}/${overlap}`).toEqual(
        referenceSplit(text, size, overlap),
      );
    }
  });

  it('overlaps by exactly `overlap` and reaches the end; the wrong controls do not', () => {
    const text = lorem(5000);
    const real = splitWindows(text, W);
    const noOverlap: string[] = [];
    for (let s = 0; s < text.length; s += 1500) noOverlap.push(text.slice(s, s + 1500));
    const droppedTail = referenceSplit(text, 1500, 250).slice(0, -1);

    const overlapsBy = (cs: string[], n: number): boolean =>
      cs.slice(1).every((c, i) => cs[i].slice(-n) === c.slice(0, n));
    expect(overlapsBy(real, 250)).toBe(true);
    expect(overlapsBy(noOverlap, 250)).toBe(false);
    expect(real[real.length - 1].endsWith(text.slice(-50))).toBe(true);
    expect(droppedTail[droppedTail.length - 1].endsWith(text.slice(-50))).toBe(false);
  });

  it('covers every character: the windows rebuild the text exactly', () => {
    const text = lorem(5000);
    const chunks = splitWindows(text, W);
    const rebuilt = chunks.slice(0, -1).map((c) => c.slice(0, 1250)).join('') + chunks[chunks.length - 1];
    expect(rebuilt).toBe(text);
  });

  it('yields one window for text at or under the width, none for empty text', () => {
    expect(splitWindows(lorem(1500), W)).toHaveLength(1);
    expect(splitWindows(lorem(900), W)).toEqual([lorem(900)]);
    expect(splitWindows('', W)).toEqual([]);
  });

  it('enforces maxChunks', () => {
    expect(splitWindows(lorem(100_000), { ...W, maxChunks: 4 })).toHaveLength(4);
  });

  it('never emits a zero-length window when overlap >= size', () => {
    const chunks = splitWindows(lorem(3000), { size: 100, overlap: 500, maxChunks: 50 });
    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks.every((c) => c.length > 0)).toBe(true);
  });
});

describe('splitOnLineBoundaries', () => {
  it('returns the text unchanged when it fits, nothing for empty text', () => {
    expect(splitOnLineBoundaries('short enough', 100)).toEqual(['short enough']);
    expect(splitOnLineBoundaries('', 100)).toEqual([]);
  });

  it('sizes parts evenly, so no orphan tail falls under a minimum-length floor', () => {
    const parts = splitOnLineBoundaries('z'.repeat(2050), 2000);
    expect(parts).toHaveLength(2);
    for (const p of parts) expect(p.length).toBeGreaterThanOrEqual(20);
    expect(parts.join('').length).toBe(2050);
  });

  it('breaks on line boundaries when it can', () => {
    const text = ['aaaa', 'bbbb', 'cccc', 'dddd'].join('\n');
    const parts = splitOnLineBoundaries(text, 10);
    for (const p of parts) expect(p.length).toBeLessThanOrEqual(10);
    expect(parts.join('\n')).toBe(text);
  });

  it('hard-splits a single line longer than the cap', () => {
    const parts = splitOnLineBoundaries('q'.repeat(45), 10);
    for (const p of parts) expect(p.length).toBeLessThanOrEqual(10);
    expect(parts.join('').length).toBe(45);
  });
});

describe('splitMarkdown', () => {
  const M = { maxChars: 2000 };

  it('splits on headings with a preamble section', () => {
    const body = [
      'Intro paragraph long enough to keep around.',
      '',
      '## Alpha Part',
      'Alpha body text that is long enough.',
      '',
      '### Beta Sub',
      'Beta body text that is long enough.',
    ].join('\n');
    const s = splitMarkdown(body, M);
    expect(s.map((x) => x.anchor)).toEqual(['', 'alpha-part', 'beta-sub']);
    expect(s[1].content).toContain('Alpha body');
  });

  it('reports the heading ancestry, outermost first', () => {
    const body = [
      'preamble text long enough to keep',
      '# Top',
      'top body long enough to keep here',
      '## Mid',
      'mid body long enough to keep here',
      '### Low',
      'low body long enough to keep here',
      '## Mid two',
      'second mid body long enough to keep',
      '# Other top',
      'other top body long enough to keep',
    ].join('\n');
    expect(splitMarkdown(body, M).map((s) => s.headingPath)).toEqual([
      [],
      ['Top'],
      ['Top', 'Mid'],
      ['Top', 'Mid', 'Low'],
      ['Top', 'Mid two'],
      ['Other top'],
    ]);
  });

  it('never splits on a heading-looking line inside a code fence', () => {
    const body = [
      '## Real Section',
      'Some intro long enough to count here.',
      '```bash',
      '# not a heading, just a comment',
      '## also not a heading',
      '```',
      'trailing text after the fence block.',
    ].join('\n');
    const s = splitMarkdown(body, M);
    expect(s).toHaveLength(1);
    expect(s[0].anchor).toBe('real-section');
    expect(s[0].content).toContain('also not a heading');
    expect(s[0].content).toContain('trailing text');
  });

  it('dedupes repeated heading anchors with -2/-3 suffixes', () => {
    const body = ['## Setup', 'first setup section body, long enough.', '## Setup', 'second setup section body, long enough.'].join('\n');
    expect(splitMarkdown(body, M).map((x) => x.anchor)).toEqual(['setup', 'setup-2']);
  });

  it('drops sections under minChars', () => {
    const body = ['## Big', 'a real section body, long enough to keep.', '## Tiny', 'ok'].join('\n');
    const s = splitMarkdown(body, M);
    expect(s.map((x) => x.anchor)).toEqual(['big']);
  });

  it('splits an over-long section into parts instead of truncating it', () => {
    const body = ['## Big', 'x'.repeat(5000), '## After', 'a following section, long enough.'].join('\n');
    const s = splitMarkdown(body, M);
    const big = s.filter((x) => sectionAnchorBase(x.anchor) === 'big');
    expect(big.map((x) => x.anchor)).toEqual(['big', 'big~2', 'big~3']);
    expect(big.map((x) => x.content).join('').length).toBe(5000);
    for (const part of big) {
      expect(part.content.length).toBeLessThanOrEqual(2000);
      expect(part.headingPath).toEqual(['Big']);
    }
    expect(s.some((x) => x.anchor === 'after')).toBe(true);
  });

  it('does not let a continuation part consume a duplicate-heading slot', () => {
    const body = ['## Setup', 'y'.repeat(3000), '## Setup', 'second setup body, long enough.'].join('\n');
    expect(splitMarkdown(body, M).map((x) => x.anchor)).toEqual(['setup', 'setup~2', 'setup-2']);
  });

  it('keeps headings deeper than headingDepth inside their parent section', () => {
    const body = ['## Parent', 'parent body long enough to keep.', '#### Deep detail', 'deep body text.'].join('\n');
    const s = splitMarkdown(body, M);
    expect(s).toHaveLength(1);
    expect(s[0].content).toContain('#### Deep detail');
    const deeper = splitMarkdown(body, { ...M, headingDepth: 4 });
    expect(deeper.map((x) => x.headingPath)).toEqual([['Parent']]);
  });

  it('every emitted anchor round-trips to a real heading anchor', () => {
    const body = ['## Long Heading', 'w'.repeat(4500), '## Short', 'short body, long enough to keep.'].join('\n');
    for (const s of splitMarkdown(body, M)) expect(['long-heading', 'short']).toContain(sectionAnchorBase(s.anchor));
  });
});

describe('anchor convention', () => {
  it('round-trips a part anchor back to its heading', () => {
    expect(chunkAnchor('setup', 0)).toBe('setup');
    expect(chunkAnchor('setup', 1)).toBe('setup~2');
    expect(sectionAnchorBase('setup~2')).toBe('setup');
    expect(sectionAnchorBase('setup')).toBe('setup');
    expect(sectionAnchorBase(chunkAnchor('', 1))).toBe('');
  });

  it('does not mistake a real heading ending in a digit for a part', () => {
    expect(isContinuationAnchor('step-2')).toBe(false);
    expect(sectionAnchorBase('step-2')).toBe('step-2');
    expect(isContinuationAnchor('step~2')).toBe(true);
  });
});
