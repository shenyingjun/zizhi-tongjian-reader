import { useMemo, useState } from 'react';
import type { Juan, Paragraph, HuNote } from './corpus';

interface Props {
  juan: Juan;
  showHu: boolean;
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

function ParagraphView({ p, showHu }: { p: Paragraph; showHu: boolean }) {
  const segments = useMemo(() => splitParagraph(p), [p]);
  // Per-marker override of the global default. If a marker isn't in the set,
  // it follows `showHu`; if it is, its state is inverted.
  const [overridden, setOverridden] = useState<Set<number>>(new Set());

  const toggleNote = (i: number) =>
    setOverridden(prev => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });

  const cls =
    'paragraph paragraph-' + (p.type || 'event');

  return (
    <p className={cls} data-pid={p.id} data-type={p.type}>
      {segments.map((seg, i) => {
        if (seg.kind === 'text') return <span key={i}>{seg.text}</span>;
        const idx = seg.noteIdx!;
        // Default = showHu; per-marker toggle inverts that default.
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
            {open && <span className="hu-note">{seg.text}</span>}
          </span>
        );
      })}
    </p>
  );
}

export default function Reader({ juan, showHu }: Props) {
  return (
    <article className="reader-body">
      {juan.paragraphs.map(p => (
        <ParagraphView key={p.id} p={p} showHu={showHu} />
      ))}
    </article>
  );
}

// Suppress unused-import warnings
export type { HuNote };
