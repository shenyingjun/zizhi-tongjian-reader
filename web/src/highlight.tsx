import type { Paragraph } from './corpus';

export interface Segment {
  kind: 'text' | 'note';
  text: string;
  noteIdx?: number;
  /** For text segments, the [mainStart, mainEnd) range within p.main this slice covers. */
  mainStart?: number;
  mainEnd?: number;
}

/** Split a paragraph into interleaved main-text and 胡注 segments in reading order. */
export function splitParagraph(p: Paragraph): Segment[] {
  if (!p.notes.length) {
    return [{ kind: 'text', text: p.main, mainStart: 0, mainEnd: p.main.length }];
  }
  const segs: Segment[] = [];
  let cursor = 0;
  p.notes.forEach((n, i) => {
    const at = Math.min(n.after, p.main.length);
    if (at > cursor) segs.push({ kind: 'text', text: p.main.slice(cursor, at), mainStart: cursor, mainEnd: at });
    segs.push({ kind: 'note', text: n.text, noteIdx: i });
    cursor = at;
  });
  if (cursor < p.main.length) segs.push({ kind: 'text', text: p.main.slice(cursor), mainStart: cursor, mainEnd: p.main.length });
  return segs;
}

/** Return all [start, end) ranges where `q` matches in `text`. */
export function findMatches(text: string, q: string): Array<[number, number]> {
  if (!q) return [];
  const ranges: Array<[number, number]> = [];
  let from = 0;
  while (true) {
    const idx = text.indexOf(q, from);
    if (idx < 0) break;
    ranges.push([idx, idx + q.length]);
    from = idx + q.length;
  }
  return ranges;
}

/**
 * Render `text` (which spans p.main[mainStart..mainEnd)) with all overlapping
 * portions of `mainMatches` wrapped in <mark>. Used so a match like "曹有道"
 * stays highlighted even when an inline note (校勘 / 胡注) splits it across
 * two text segments.
 */
export function highlightWithRanges(
  text: string,
  mainStart: number,
  mainMatches: Array<[number, number]>,
): React.ReactNode {
  if (!mainMatches.length) return text;
  const mainEnd = mainStart + text.length;
  const parts: React.ReactNode[] = [];
  let cursor = mainStart;
  let k = 0;
  for (const [ms, me] of mainMatches) {
    if (me <= mainStart) continue;
    if (ms >= mainEnd) break;
    const segStart = Math.max(ms, mainStart);
    const segEnd = Math.min(me, mainEnd);
    if (segStart > cursor) parts.push(text.slice(cursor - mainStart, segStart - mainStart));
    parts.push(
      <mark key={k++} className="search-hit">
        {text.slice(segStart - mainStart, segEnd - mainStart)}
      </mark>,
    );
    cursor = segEnd;
  }
  if (cursor < mainEnd) parts.push(text.slice(cursor - mainStart));
  return parts.length ? parts : text;
}

/** Standalone substring highlighter — for self-contained strings (note text, full paragraphs). */
export function highlight(text: string, q: string): React.ReactNode {
  if (!q) return text;
  const parts: React.ReactNode[] = [];
  let from = 0;
  let k = 0;
  while (true) {
    const idx = text.indexOf(q, from);
    if (idx < 0) break;
    if (idx > from) parts.push(text.slice(from, idx));
    parts.push(<mark key={k++} className="search-hit">{text.slice(idx, idx + q.length)}</mark>);
    from = idx + q.length;
  }
  if (!parts.length) return text;
  if (from < text.length) parts.push(text.slice(from));
  return parts;
}
