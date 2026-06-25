import { useMemo, useState, type ReactNode } from 'react';
import type { Juan, Paragraph, HuNote, GuideSummary } from './corpus';
import { splitParagraph, findMatches, highlightWithRanges, highlight } from './highlight';
import GuideBlock from './GuideBlock';

interface Props {
  juan: Juan;
  showHu: boolean;
  highlightQuery: string;
  highlightPid: number | null;
  // 'off' hides all guide blocks; 'brief'/'full' render them (full = expanded).
  guideMode: 'off' | 'brief' | 'full';
  // anchor_pid → reviewed editorial summary for this 卷.
  guideByAnchorPid: Map<number, GuideSummary>;
  onGuideJump: (pid: number) => void;
  onPersonSearch: (query: string) => void;
}

function ParagraphView({
  p, showHu, highlightQuery, isTarget, isDisclaimer, after,
}: { p: Paragraph; showHu: boolean; highlightQuery: string; isTarget: boolean; isDisclaimer: boolean; after?: ReactNode }) {
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
    <>
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
    {after}
    </>
  );
}

export default function Reader({
  juan, showHu, highlightQuery, highlightPid,
  guideMode, guideByAnchorPid, onGuideJump, onPersonSearch,
}: Props) {
  return (
    <article className="reader-body">
      {juan.paragraphs.map((p, i) => {
        const guide =
          guideMode !== 'off' ? guideByAnchorPid.get(p.id) : undefined;
        return (
          <ParagraphView
            key={p.id}
            p={p}
            showHu={showHu}
            highlightQuery={highlightQuery}
            isTarget={highlightPid === p.id}
            isDisclaimer={i === juan.paragraphs.length - 1}
            after={
              guide ? (
                <GuideBlock
                  // Key by mode so switching global mode resets local expands.
                  key={`guide-${p.id}-${guideMode}`}
                  summary={guide}
                  mode={guideMode === 'full' ? 'full' : 'brief'}
                  onJump={onGuideJump}
                  onPersonSearch={onPersonSearch}
                />
              ) : null
            }
          />
        );
      })}
    </article>
  );
}

// Suppress unused-import warnings
export type { HuNote };
