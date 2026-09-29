/**
 * chunk — split long text into pieces a vector index can embed one by one, so
 * text past an embedder's input window stays findable by meaning.
 *
 * Two strategies, both lifted VERBATIM from the chunkers that were measured in
 * production (golden-pinned in chunk.golden.test.ts, so a change to either one
 * shows up as a failing test rather than as silently different chunks):
 *
 *   - splitWindows   fixed-width windows with overlap. Content-blind: a window
 *                    boundary can land mid-word. Right for unstructured prose
 *                    (chat turns), where it was measured to lift past-the-cut
 *                    MRR from ~0.25 to ~0.73-0.84 against 59 distractors.
 *   - splitMarkdown  one section per heading (fence-aware), with an oversized
 *                    section split on line boundaries into `<anchor>~N` parts.
 *                    Right for documents, where a heading is a natural unit and
 *                    an edit to one section leaves the others' text unchanged.
 *
 * Score a chunked parent by the BEST match over its chunks, never the mean: a
 * parent is relevant if any part of it is, and averaging brings back exactly
 * the dilution chunking exists to remove.
 */

export interface WindowSplitOptions {
  /** Window width, in UTF-16 code units. */
  size: number;
  /** Code units shared by consecutive windows. The step is max(1, size - overlap). */
  overlap: number;
  /** Hard cap on windows per text. Text past the last window is not returned. */
  maxChunks: number;
}

/**
 * Split text into overlapping fixed-width windows.
 *
 * The loop stops as soon as a window reaches the end of the text, so the last
 * window is whatever remains — never padded, never re-anchored. Empty text
 * yields no windows (never one empty window).
 */
export function splitWindows(text: string, opts: WindowSplitOptions): string[] {
  const { size, overlap, maxChunks } = opts;
  if (text.length === 0) return [];
  const step = Math.max(1, size - overlap);
  const out: string[] = [];
  for (let s = 0; s < text.length; s += step) {
    out.push(text.slice(s, s + size));
    if (out.length >= maxChunks) break;
    if (s + size >= text.length) break;
  }
  return out;
}

export interface MarkdownSplitOptions {
  /** Per-part cap. A longer section is split on line boundaries into several parts. */
  maxChars: number;
  /** A section whose trimmed content is shorter than this is dropped. Default 20. */
  minChars?: number;
  /** Deepest heading level that starts a section; deeper headings stay inside it. Default 3. */
  headingDepth?: number;
  /** Cap on heading sections per document. Counts headings, not parts. Default 80. */
  maxSections?: number;
  /** Backstop on parts per document (sections plus continuations). Default 240. */
  maxRows?: number;
}

export interface MarkdownSection {
  /**
   * Heading anchor ('' for the preamble), deduped within the document with
   * -2/-3 suffixes. Continuation parts of an oversized section carry
   * `<anchor>~N` (see chunkAnchor); sectionAnchorBase() recovers the heading.
   */
  anchor: string;
  /**
   * The section's heading ancestry, outermost first, over headings no deeper
   * than `headingDepth`. The last entry is the section's own heading; the
   * preamble has none.
   */
  headingPath: string[];
  content: string;
}

/** Separator between a heading anchor and its continuation index. */
export const CHUNK_ANCHOR_SEP = '~';

/**
 * The anchor for part `index` (0-based) of the section anchored at `anchor`.
 * Part 0 keeps the bare heading anchor, so a section that fits under the cap
 * gets exactly the anchor it would have had without chunking.
 *
 * `~` is safe because the slug rule collapses every non-`[a-z0-9]` run to `-`:
 * 'step-2' is a real heading, 'step~2' is part 2 of 'step'.
 */
export function chunkAnchor(anchor: string, index: number): string {
  return index === 0 ? anchor : `${anchor}${CHUNK_ANCHOR_SEP}${index + 1}`;
}

/** The navigable heading anchor behind a stored anchor; a no-op on a heading anchor. */
export function sectionAnchorBase(anchor: string): string {
  const i = anchor.lastIndexOf(CHUNK_ANCHOR_SEP);
  return i === -1 ? anchor : anchor.slice(0, i);
}

/** True when this anchor names a continuation part rather than a heading's first part. */
export function isContinuationAnchor(anchor: string): boolean {
  return anchor.includes(CHUNK_ANCHOR_SEP);
}

/** Heading text to anchor id: lower-case, every non-`[a-z0-9]` run collapsed to `-`. */
function slugifyAnchor(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

/**
 * Split one section's text into parts of at most `max` chars, breaking on line
 * boundaries. Parts are sized EVENLY (ceil(len/parts)) rather than greedily
 * filled, which keeps a 2,050-char section from becoming a 2,000-char part plus
 * a 50-char orphan that a minimum-length floor would then drop.
 *
 * A single line longer than `max` has no break to honour and is hard-split;
 * that is the one case where a boundary can land mid-word.
 */
export function splitOnLineBoundaries(text: string, max: number): string[] {
  if (text.length === 0) return [];
  if (text.length <= max) return [text];

  const parts = Math.ceil(text.length / max);
  const target = Math.ceil(text.length / parts);

  const units: string[] = [];
  for (const line of text.split('\n')) {
    if (line.length <= max) units.push(line);
    else for (let i = 0; i < line.length; i += max) units.push(line.slice(i, i + max));
  }

  const out: string[] = [];
  let cur = '';
  for (const u of units) {
    const candidate = cur === '' ? u : `${cur}\n${u}`;
    // Break when the part would overflow the hard cap, or once it has reached
    // its even-split target — never when `cur` is still empty, so a single
    // over-long unit is emitted rather than dropped.
    if (cur !== '' && (candidate.length > max || cur.length >= target)) {
      out.push(cur);
      cur = u;
    } else {
      cur = candidate;
    }
  }
  if (cur !== '') out.push(cur);
  return out;
}

/**
 * Split markdown into sections on headings of depth 1..headingDepth.
 *
 * Fence-aware: a `#` line inside a ``` or ~~~ block never starts a section.
 * Sections shorter than `minChars` are dropped (heading-only sections are
 * ranking noise). A section longer than `maxChars` becomes several parts, the
 * first keeping the heading's anchor and the rest `<anchor>~N`.
 */
export function splitMarkdown(body: string, opts: MarkdownSplitOptions): MarkdownSection[] {
  const maxChars = opts.maxChars;
  const minChars = opts.minChars ?? 20;
  const headingDepth = opts.headingDepth ?? 3;
  const maxSections = opts.maxSections ?? 80;
  const maxRows = opts.maxRows ?? 240;
  const headingRe = new RegExp(`^(#{1,${headingDepth}})\\s+(.+?)\\s*$`);

  const raw: Array<{ heading: string | null; path: string[]; lines: string[] }> = [
    { heading: null, path: [], lines: [] },
  ];
  const open: Array<{ level: number; text: string }> = [];
  let fenceMark: string | null = null;
  for (const line of body.split('\n')) {
    const fm = /^\s*(`{3,}|~{3,})/.exec(line);
    if (fm) {
      if (fenceMark === null) fenceMark = fm[1][0];
      else if (fm[1][0] === fenceMark) fenceMark = null;
      raw[raw.length - 1].lines.push(line);
      continue;
    }
    const hm = fenceMark === null ? headingRe.exec(line) : null;
    if (hm) {
      const level = hm[1].length;
      const text = hm[2].trim();
      while (open.length > 0 && open[open.length - 1].level >= level) open.pop();
      open.push({ level, text });
      raw.push({ heading: text, path: open.map((h) => h.text), lines: [] });
    } else raw[raw.length - 1].lines.push(line);
  }

  const seen = new Map<string, number>();
  const out: MarkdownSection[] = [];
  let headings = 0;
  for (const s of raw) {
    if (headings >= maxSections || out.length >= maxRows) break;
    const content = s.lines.join('\n').trim();
    if (content.length < minChars) continue;
    let anchor = s.heading === null ? '' : slugifyAnchor(s.heading) || 'section';
    // Dedupe on the BASE anchor only — continuation suffixes are storage
    // identity, not headings, and must not consume a -2/-3 slot.
    const n = seen.get(anchor) ?? 0;
    seen.set(anchor, n + 1);
    if (n > 0) anchor = `${anchor}-${n + 1}`;
    headings += 1;
    const parts = splitOnLineBoundaries(content, maxChars);
    for (let i = 0; i < parts.length && out.length < maxRows; i += 1) {
      out.push({ anchor: chunkAnchor(anchor, i), headingPath: s.path, content: parts[i] });
    }
  }
  return out;
}
