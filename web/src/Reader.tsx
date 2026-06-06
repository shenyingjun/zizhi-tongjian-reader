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
  const [openNotes, setOpenNotes] = useState<Set<number>>(new Set());

  const toggleNote = (i: number) =>
    setOpenNotes(prev => {
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
        if (!showHu) return null;
        const idx = seg.noteIdx!;
        const open = openNotes.has(idx);
        return (
          <span key={i} className="hu-note-wrap">
            <button
              className={'hu-marker' + (open ? ' open' : '')}
              onClick={() => toggleNote(idx)}
              title={open ? '收起音注' : '展开胡三省音注'}
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
