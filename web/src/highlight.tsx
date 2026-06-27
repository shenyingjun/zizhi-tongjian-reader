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
export function findMatches(text: string, q: string | string[]): Array<[number, number]> {
  const needles = Array.from(
    new Set((Array.isArray(q) ? q : [q]).map(s => s.trim()).filter(Boolean)),
  ).sort((a, b) => b.length - a.length);
  if (needles.length === 0) return [];
  const raw: Array<[number, number]> = [];
  for (const nd of needles) {
    let from = 0;
    while (true) {
      const idx = text.indexOf(nd, from);
      if (idx < 0) break;
      raw.push([idx, idx + nd.length]);
      from = idx + nd.length;
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

/** A person mention projected onto p.main coordinates for one paragraph. */
export interface PersonSpan {
  start: number;
  end: number;
  personId: string;
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
  const mainEnd = mainStart + text.length;
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
          {highlightWithRanges(text.slice(cursor - mainStart, segStart - mainStart), cursor, mainMatches)}
        </Fragment>,
      );
    }
    const label = text.slice(segStart - mainStart, segEnd - mainStart);
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
          onPersonClick(s.personId, pid, 'main', label);
        }}
      >
        {highlightWithRanges(label, segStart, mainMatches)}
      </button>,
    );
    cursor = segEnd;
  }
  if (cursor < mainEnd) {
    out.push(
      <Fragment key={'t' + k++}>
        {highlightWithRanges(text.slice(cursor - mainStart), cursor, mainMatches)}
      </Fragment>,
    );
  }
  return out;
}

/** Standalone substring highlighter — for self-contained strings (note text, full paragraphs). */
export function highlight(text: string, q: string | string[]): React.ReactNode {
  const ranges = findMatches(text, q);
  if (!ranges.length) return text;
  const parts: React.ReactNode[] = [];
  let from = 0;
  let k = 0;
  for (const [s, e] of ranges) {
    if (s > from) parts.push(text.slice(from, s));
    parts.push(<mark key={k++} className="search-hit">{text.slice(s, e)}</mark>);
    from = e;
  }
  if (from < text.length) parts.push(text.slice(from));
  return parts;
}
