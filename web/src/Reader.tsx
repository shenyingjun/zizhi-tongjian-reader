import { useMemo, useState, type ReactNode } from 'react';
import type { Juan, Paragraph, HuNote, GuideSummary, GuidePersonRef } from './corpus';
import { splitParagraph, findMatches, highlightWithRanges, highlight, renderTextSegment, type PersonSpan } from './highlight';
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
  onPersonSearch: (query: string) => void;
  // pid → main-text person mention spans (empty/absent when no person data).
  personSpansByPid: Map<number, PersonSpan[]>;
  onPersonClick: (personId: string, pid: number, source?: 'main' | 'guide', clickedLabel?: string) => void;
  // Resolve a guide 关键人物 ref to a canonical person id, or null for literal fallback.
  resolveGuidePerson: (ref: GuidePersonRef) => string | null;
  // The person whose card is currently open (for inline active highlighting).
  activePersonId: string | null;
}

function ParagraphView({
  p, showHu, highlightQuery, isTarget, isDisclaimer, after, personSpans, onPersonClick, activePersonId,
}: {
  p: Paragraph; showHu: boolean; highlightQuery: string; isTarget: boolean;
  isDisclaimer: boolean; after?: ReactNode;
  personSpans: PersonSpan[]; onPersonClick: (personId: string, pid: number, source?: 'main' | 'guide', clickedLabel?: string) => void;
  activePersonId: string | null;
}) {
  const segments = useMemo(() => splitParagraph(p), [p]);
  const mainMatches = useMemo(() => findMatches(p.main, highlightQuery), [p.main, highlightQuery]);
  const [overridden, setOverridden] = useState<Set<number>>(new Set());

  // 纲目 event markers are circled numbers ①..㊿ (Unicode only provides 1..50).
  // For the 51st item onward the digitiser falls back to plain ASCII digits,
  // which otherwise render as a bare "51". Detect that leading digit run so we
  // can wrap it in a CSS circled badge and keep it visually consistent with ㊿.
  const markerLen = useMemo(() => {
    if (p.type !== 'event') return 0;
    const m = /^\d+/.exec(p.main);
    return m ? m[0].length : 0;
  }, [p.type, p.main]);

  const renderText = (text: string, mainStart: number) =>
    personSpans.length
      ? renderTextSegment(text, mainStart, mainMatches, personSpans, onPersonClick, p.id, activePersonId)
      : highlightWithRanges(text, mainStart, mainMatches);

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
          if (markerLen && seg.mainStart === 0) {
            const cps = Array.from(seg.text);
            const num = cps.slice(0, markerLen).join('');
            const rest = cps.slice(markerLen).join('');
            return (
              <span key={i}>
                <span className="event-marker">{num}</span>
                {renderText(rest, markerLen)}
              </span>
            );
          }
          return (
            <span key={i}>
              {renderText(seg.text, seg.mainStart!)}
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
  guideMode, guideByAnchorPid, onPersonSearch,
  personSpansByPid, onPersonClick, resolveGuidePerson, activePersonId,
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
            personSpans={personSpansByPid.get(p.id) ?? []}
            onPersonClick={onPersonClick}
            activePersonId={activePersonId}
            after={
              guide ? (
                <GuideBlock
                  // Key by mode so switching global mode resets local expands.
                  key={`guide-${p.id}-${guideMode}`}
                  summary={guide}
                  mode={guideMode === 'full' ? 'full' : 'brief'}
                  onPersonSearch={onPersonSearch}
                  resolveGuidePerson={resolveGuidePerson}
                  onPersonClick={onPersonClick}
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
