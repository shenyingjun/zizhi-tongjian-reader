import { useId, useState } from 'react';
import type { GuideSummary } from './corpus';

interface Props {
  summary: GuideSummary;
  // Global default reading mode. 'off' is handled by the caller (block not
  // rendered at all); here we only ever see 'brief' or 'full'.
  mode: 'brief' | 'full';
  // Look a person up via the existing 出处检索.
  onPersonSearch: (query: string) => void;
}

export default function GuideBlock({ summary, mode, onPersonSearch }: Props) {
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
              <ul className="guide-people">
                {summary.key_people.map((person, i) => (
                  <li key={i} className="guide-person">
                    <button
                      type="button"
                      className="guide-person-name"
                      onClick={() => onPersonSearch(person.query || person.name)}
                      title={`检索「${person.name}」的出处`}
                    >
                      {person.name}
                    </button>
                    {person.role && <span className="guide-person-role">{person.role}</span>}
                  </li>
                ))}
              </ul>
            </section>
          )}

          <div className="guide-foot">
            <span className="guide-note">
              {summary.editorial_note || '编者整理，非原文'}
            </span>
          </div>
        </div>
      )}
    </aside>
  );
}
