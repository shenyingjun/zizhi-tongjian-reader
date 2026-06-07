import { useMemo, useState } from 'react';
import type { Juan, Paragraph, HuNote } from './corpus';
import { splitParagraph, findMatches, highlightWithRanges, highlight } from './highlight';

interface Props {
  juan: Juan;
  showHu: boolean;
  highlightQuery: string;
  highlightPid: number | null;
}

function ParagraphView({
  p, showHu, highlightQuery, isTarget, isDisclaimer,
}: { p: Paragraph; showHu: boolean; highlightQuery: string; isTarget: boolean; isDisclaimer: boolean }) {
  const segments = useMemo(() => splitParagraph(p), [p]);
  const mainMatches = useMemo(() => findMatches(p.main, highlightQuery), [p.main, highlightQuery]);
  const [overridden, setOverridden] = useState<Set<number>>(new Set());

  const toggleNote = (i: number) =>
    setOverridden(prev => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });

  const cls =
    'paragraph paragraph-' + (p.type || 'event') +
    (isTarget ? ' is-target' : '') +
    (isDisclaimer ? ' paragraph-disclaimer' : '');

  return (
    <p className={cls} data-pid={p.id} data-type={p.type}>
      {segments.map((seg, i) => {
        if (seg.kind === 'text') {
          return (
            <span key={i}>
              {highlightWithRanges(seg.text, seg.mainStart!, mainMatches)}
            </span>
          );
        }
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
      {juan.paragraphs.map((p, i) => (
        <ParagraphView
          key={p.id}
          p={p}
          showHu={showHu}
          highlightQuery={highlightQuery}
          isTarget={highlightPid === p.id}
          isDisclaimer={i === juan.paragraphs.length - 1}
        />
      ))}
    </article>
  );
}

// Suppress unused-import warnings
export type { HuNote };
