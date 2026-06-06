import { useMemo, useState } from 'react';
import type { Juan, Paragraph, HuNote } from './corpus';

interface Props {
  juan: Juan;
  showHu: boolean;
  highlightQuery: string;
  highlightPid: number | null;
}

interface Segment {
  kind: 'text' | 'note';
  text: string;
  noteIdx?: number;
}

function splitParagraph(p: Paragraph): Segment[] {
  if (!p.notes.length) return [{ kind: 'text', text: p.main }];
  const segs: Segment[] = [];
  let cursor = 0;
  p.notes.forEach((n, i) => {
    const at = Math.min(n.after, p.main.length);
    if (at > cursor) segs.push({ kind: 'text', text: p.main.slice(cursor, at) });
    segs.push({ kind: 'note', text: n.text, noteIdx: i });
    cursor = at;
  });
  if (cursor < p.main.length) segs.push({ kind: 'text', text: p.main.slice(cursor) });
  return segs;
}

function highlight(text: string, q: string) {
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

function ParagraphView({
  p, showHu, highlightQuery, isTarget,
}: { p: Paragraph; showHu: boolean; highlightQuery: string; isTarget: boolean }) {
  const segments = useMemo(() => splitParagraph(p), [p]);
  const [overridden, setOverridden] = useState<Set<number>>(new Set());

  const toggleNote = (i: number) =>
    setOverridden(prev => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });

  const cls =
    'paragraph paragraph-' + (p.type || 'event') + (isTarget ? ' is-target' : '');

  return (
    <p className={cls} data-pid={p.id} data-type={p.type}>
      {segments.map((seg, i) => {
        if (seg.kind === 'text') return <span key={i}>{highlight(seg.text, highlightQuery)}</span>;
        const idx = seg.noteIdx!;
        const open = overridden.has(idx) ? !showHu : showHu;
        return (
          <span key={i} className="hu-note-wrap">
            <button
              className={'hu-marker' + (open ? ' open' : '')}
              onClick={() => toggleNote(idx)}
              title={open ? '收起此条音注' : '展开此条胡三省音注'}
            >
              {open ? '［收］' : '［注］'}
            </button>
            {open && <span className="hu-note">{highlight(seg.text, highlightQuery)}</span>}
          </span>
        );
      })}
    </p>
  );
}

export default function Reader({ juan, showHu, highlightQuery, highlightPid }: Props) {
  return (
    <article className="reader-body">
      {juan.paragraphs.map(p => (
        <ParagraphView
          key={p.id}
          p={p}
          showHu={showHu}
          highlightQuery={highlightQuery}
          isTarget={highlightPid === p.id}
        />
      ))}
    </article>
  );
}

// Suppress unused-import warnings
export type { HuNote };
