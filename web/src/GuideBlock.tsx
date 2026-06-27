import { useId, useState } from 'react';
import type { GuideSummary, GuidePersonRef } from './corpus';

interface Props {
  summary: GuideSummary;
  // Global default reading mode. 'off' is handled by the caller (block not
  // rendered at all); here we only ever see 'brief' or 'full'.
  mode: 'brief' | 'full';
  // Look a person up via the existing 出处检索.
  onPersonSearch: (query: string) => void;
  // Resolve a 关键人物 ref to a canonical person id, or null for literal fallback.
  resolveGuidePerson: (ref: GuidePersonRef) => string | null;
  // Open the spoiler-safe person card. The guide's anchor_pid is its "current
  // position" for spoiler filtering.
  onPersonClick: (personId: string, pid: number, source?: 'main' | 'guide', clickedLabel?: string) => void;
}

export default function GuideBlock({ summary, mode, onPersonSearch, resolveGuidePerson, onPersonClick }: Props) {
  // Per-block expand is LOCAL — independent of the global mode. In 'full'
  // mode every block is already expanded and the toggle is hidden; switching
  // the global mode remounts blocks (keyed by mode upstream) so local state
  // resets, matching the "global mode clears local overrides" rule.
  const [localOpen, setLocalOpen] = useState(false);
  const detailId = useId();

  const expanded = mode === 'full' || localOpen;
  const hasDetail =
    !!summary.what_happened ||
    !!summary.why_it_matters ||
    !!summary.background ||
    (summary.key_people?.length ?? 0) > 0;

  return (
    <aside className="guide-block" aria-label="白话导读">
      <div className="guide-head">
        <span className="guide-tag">白话导读</span>
        {summary.title && <span className="guide-title">{summary.title}</span>}
        <span className="guide-prov" title="本段为编者整理的现代汉语导读，并非《资治通鉴》原文">
          非原文
        </span>
      </div>

      <p className="guide-oneliner">{summary.one_liner}</p>

      {hasDetail && mode !== 'full' && (
        <button
          type="button"
          className="guide-expand"
          aria-expanded={expanded}
          aria-controls={detailId}
          onClick={() => setLocalOpen(o => !o)}
        >
          {expanded ? '收起导读' : '展开导读'}
        </button>
      )}

      {hasDetail && expanded && (
        <div className="guide-detail" id={detailId}>
          {summary.what_happened && (
            <section className="guide-field">
              <h4 className="guide-field-label">发生了什么</h4>
              <p>{summary.what_happened}</p>
            </section>
          )}

          {summary.why_it_matters && (
            <section className="guide-field">
              <h4 className="guide-field-label">为什么重要</h4>
              <p>{summary.why_it_matters}</p>
            </section>
          )}

          {summary.background && (
            <section className="guide-field">
              <h4 className="guide-field-label">背景</h4>
              <p>{summary.background}</p>
            </section>
          )}

          {summary.key_people && summary.key_people.length > 0 && (
            <section className="guide-field">
              <h4 className="guide-field-label">关键人物</h4>
              <div className="guide-people">
                {summary.key_people.map((person, i) => {
                  const boundId = resolveGuidePerson(person);
                  if (boundId) {
                    return (
                      <button
                        key={i}
                        type="button"
                        className="pchip bound"
                        onClick={() => onPersonClick(boundId, summary.anchor_pid, 'guide', person.name)}
                        title={`查看「${person.name}」的人物信息（编者整理·非原文）`}
                        aria-label={`${person.name}，查看人物信息`}
                      >
                        <span className="pchip-dot" aria-hidden="true" />
                        <span className="pchip-name">{person.name}</span>
                        {person.role && <span className="pchip-role">{person.role}</span>}
                      </button>
                    );
                  }
                  return (
                    <button
                      key={i}
                      type="button"
                      className="pchip unbound"
                      onClick={() => onPersonSearch(person.query || person.name)}
                      title={`检索「${person.name}」的出处`}
                      aria-label={`${person.name}，检索出处`}
                    >
                      <svg className="pchip-search" viewBox="0 0 24 24" fill="none"
                           stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"
                           strokeLinejoin="round" aria-hidden="true">
                        <circle cx="11" cy="11" r="7" />
                        <line x1="21" y1="21" x2="16.5" y2="16.5" />
                      </svg>
                      <span className="pchip-name">{person.name}</span>
                      {person.role && <span className="pchip-role">{person.role}</span>}
                      <span className="pchip-tag" aria-hidden="true">检索</span>
                    </button>
                  );
                })}
              </div>
            </section>
          )}
        </div>
      )}
    </aside>
  );
}
