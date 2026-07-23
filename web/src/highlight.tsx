import { Fragment } from 'react';
import type { Paragraph } from './corpus';

export interface Segment {
  kind: 'text' | 'note';
  text: string;
  noteIdx?: number;
  /** For text segments, the [mainStart, mainEnd) range within p.main this slice covers. */
  mainStart?: number;
  mainEnd?: number;
}

// All offsets emitted by the Python data pipeline (mention start/end, note.after)
// are Unicode CODE POINT indices. JS strings are UTF-16, so a non-BMP character
// (e.g. CJK Ext-B 𨁂 U+28042) is a 2-unit surrogate pair and would desync every
// offset after it by +1 if we sliced with String.prototype.slice. We therefore
// index every paragraph by code point. `cpSlice`/`cpLen` make that explicit and
// are a no-op for the (vast majority of) paragraphs with no astral characters.
const cpLen = (s: string): number => Array.from(s).length;

/** Split a paragraph into interleaved main-text and 胡注 segments in reading order. */
export function splitParagraph(p: Paragraph): Segment[] {
  const mainLen = cpLen(p.main);
  if (!p.notes.length) {
    return [{ kind: 'text', text: p.main, mainStart: 0, mainEnd: mainLen }];
  }
  const cps = Array.from(p.main);
  const segs: Segment[] = [];
  let cursor = 0;
  p.notes.forEach((n, i) => {
    const at = Math.min(n.after, mainLen);
    if (at > cursor) segs.push({ kind: 'text', text: cps.slice(cursor, at).join(''), mainStart: cursor, mainEnd: at });
    segs.push({ kind: 'note', text: n.text, noteIdx: i });
    cursor = at;
  });
  if (cursor < mainLen) segs.push({ kind: 'text', text: cps.slice(cursor).join(''), mainStart: cursor, mainEnd: mainLen });
  return segs;
}

/** Return all [start, end) ranges (in CODE POINTS) where `q` matches in `text`. */
export function findMatches(text: string, q: string | string[]): Array<[number, number]> {
  const needles = Array.from(
    new Set((Array.isArray(q) ? q : [q]).map(s => s.trim()).filter(Boolean)),
  );
  if (needles.length === 0) return [];
  const cps = Array.from(text);
  // Code-point arrays, longest needle first (so a longer needle is preferred
  // over a shorter overlapping one — e.g. 王翦 over 翦).
  const ndArr = needles.map(n => Array.from(n)).sort((a, b) => b.length - a.length);
  const raw: Array<[number, number]> = [];
  for (const nd of ndArr) {
    const L = nd.length;
    if (!L) continue;
    for (let i = 0; i + L <= cps.length; i++) {
      let ok = true;
      for (let j = 0; j < L; j++) { if (cps[i + j] !== nd[j]) { ok = false; break; } }
      if (ok) { raw.push([i, i + L]); i += L - 1; }
    }
  }
  // Drop overlaps (longer needle already preferred via sort) so multi-needle
  // highlighting never double-marks the same span.
  raw.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
  const ranges: Array<[number, number]> = [];
  let lastEnd = -1;
  for (const r of raw) {
    if (r[0] >= lastEnd) { ranges.push(r); lastEnd = r[1]; }
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
  const cps = Array.from(text);
  const mainEnd = mainStart + cps.length;
  const parts: React.ReactNode[] = [];
  let cursor = mainStart;
  let k = 0;
  for (const [ms, me] of mainMatches) {
    if (me <= mainStart) continue;
    if (ms >= mainEnd) break;
    const segStart = Math.max(ms, mainStart);
    const segEnd = Math.min(me, mainEnd);
    if (segStart > cursor) parts.push(cps.slice(cursor - mainStart, segStart - mainStart).join(''));
    parts.push(
      <mark key={k++} className="search-hit">
        {cps.slice(segStart - mainStart, segEnd - mainStart).join('')}
      </mark>,
    );
    cursor = segEnd;
  }
  if (cursor < mainEnd) parts.push(cps.slice(cursor - mainStart).join(''));
  return parts.length ? parts : text;
}

/** A person mention projected onto p.main coordinates for one paragraph. */
export interface PersonSpan {
  start: number;
  end: number;
  personId?: string;
  confidence: string;
}

/**
 * Render a main-text segment spanning p.main[mainStart..mainStart+text.length)
 * with two overlaid editorial layers:
 *   1. search-hit <mark>s from `mainMatches` (in p.main coords), and
 *   2. person affordances from `personSpans` (in p.main coords) rendered as
 *      inline <button class="person-name"> that open the spoiler-safe card.
 * Person spans are non-overlapping (the pipeline enforces consumed ranges).
 * Search highlighting is preserved inside person buttons. Drag-selection still
 * wins because the buttons are inline and do not preventDefault on pointerdown.
 */
export function renderTextSegment(
  text: string,
  mainStart: number,
  mainMatches: Array<[number, number]>,
  personSpans: PersonSpan[],
  onPersonClick: (personId: string, pid: number, source?: 'main' | 'guide', clickedLabel?: string) => void,
  pid: number,
  activePersonId?: string | null,
): React.ReactNode {
  const mainEnd = mainStart + cpLen(text);
  const cps = Array.from(text);
  const spans = personSpans
    .filter(s => s.end > mainStart && s.start < mainEnd)
    .sort((a, b) => a.start - b.start);
  if (!spans.length) return highlightWithRanges(text, mainStart, mainMatches);

  const out: React.ReactNode[] = [];
  let cursor = mainStart;
  let k = 0;
  for (const s of spans) {
    const segStart = Math.max(s.start, mainStart);
    const segEnd = Math.min(s.end, mainEnd);
    if (segStart > cursor) {
      out.push(
        <Fragment key={'t' + k++}>
          {highlightWithRanges(cps.slice(cursor - mainStart, segStart - mainStart).join(''), cursor, mainMatches)}
        </Fragment>,
      );
    }
    const label = cps.slice(segStart - mainStart, segEnd - mainStart).join('');
    if (s.personId) {
      const isActive = !!activePersonId && s.personId === activePersonId;
      out.push(
        <button
          key={'p' + k++}
          type="button"
          className={'person-name' + (isActive ? ' is-active' : '')}
          data-confidence={s.confidence}
          title="编者人物信息 · 非原文"
          aria-label={`${label}，编者人物信息，非原文，查看此前出现`}
          onClick={(e) => {
            // Let copy/select gestures win — but only when the live selection
            // actually covers THIS name (the user is dragging across it). A
            // stray selection elsewhere on the page (e.g. left over from a
            // select-to-search) must NOT swallow a deliberate tap on a person.
            const sel = typeof window !== 'undefined' ? window.getSelection() : null;
            if (sel && !sel.isCollapsed && sel.rangeCount > 0) {
              const btn = e.currentTarget as Node;
              for (let i = 0; i < sel.rangeCount; i++) {
                if (sel.getRangeAt(i).intersectsNode(btn)) return;
              }
            }
            onPersonClick(s.personId!, pid, 'main', label);
          }}
        >
          {highlightWithRanges(label, segStart, mainMatches)}
        </button>,
      );
    } else {
      out.push(
        <span
          key={'p' + k++}
          className="person-name is-tag-only"
          data-confidence={s.confidence}
          title="自动人名标注"
        >
          {highlightWithRanges(label, segStart, mainMatches)}
        </span>,
      );
    }
    cursor = segEnd;
  }
  if (cursor < mainEnd) {
    out.push(
      <Fragment key={'t' + k++}>
        {highlightWithRanges(cps.slice(cursor - mainStart).join(''), cursor, mainMatches)}
      </Fragment>,
    );
  }
  return out;
}

/** Standalone substring highlighter — for self-contained strings (note text, full paragraphs). */
export function highlight(text: string, q: string | string[]): React.ReactNode {
  const ranges = findMatches(text, q);
  if (!ranges.length) return text;
  const cps = Array.from(text);
  const parts: React.ReactNode[] = [];
  let from = 0;
  let k = 0;
  for (const [s, e] of ranges) {
    if (s > from) parts.push(cps.slice(from, s).join(''));
    parts.push(<mark key={k++} className="search-hit">{cps.slice(s, e).join('')}</mark>);
    from = e;
  }
  if (from < cps.length) parts.push(cps.slice(from).join(''));
  return parts;
}
