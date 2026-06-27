import { useEffect, useRef } from 'react';
import type { Person, PersonConfidence } from './corpus';

interface Props {
  person: Person;
  // When true, show the full editorial identity (may contain later events).
  // When false, show the spoiler-safe establishing brief.
  spoiler: boolean;
  // Toggle between brief (spoiler-safe) and full identity. Only offered when
  // the two differ.
  onToggleSpoiler: () => void;
  onClose: () => void;
  // Move focus to the heading on open. Only for click-opened banners — never
  // when the banner appears because the reader typed a matching name (that
  // would yank focus out of the search box mid-keystroke).
  autoFocus?: boolean;
  // How the user reached this person — drives a small provenance tag so they
  // can retrace 来自正文 / 来自导读 / 来自检索.
  origin?: 'inline' | 'guide' | 'lookup-promoted';
  // The exact label the user clicked (underline surface / pill name). When it
  // differs from canonical_name, the card shows a reconciliation line.
  clickedLabel?: string;
}

const CONF_LABEL: Record<PersonConfidence, string> = {
  reviewed: '身份已复核',
  high: '自动识别',
  medium: '较高',
  low: '可能',
  unresolved: '不确定',
};

const CONF_HINT: Record<PersonConfidence, string> = {
  reviewed: '编者已确认此处指向该人物；非《资治通鉴》原文。',
  high: '程序自动识别，置信较高；非原文。',
  medium: '程序自动识别，置信中等；非原文。',
  low: '可能指向该人物，尚不确定；非原文。',
  unresolved: '未能确定指向，仅供参考；非原文。',
};

// A slim identity banner that sits atop the 出处检索 results. The result list
// below IS this person's occurrences (a literal name search), so the banner
// only carries the editorial "who is this" — name, station, and a spoiler
// switch between the establishing brief and the full life arc.
export default function PersonCard({ person, spoiler, onToggleSpoiler, onClose, autoFocus, origin, clickedLabel }: Props) {
  const aliases = person.names.filter(n => n.text !== person.canonical_name);
  const headingRef = useRef<HTMLHeadingElement>(null);

  const brief = person.brief ?? person.identity;
  const hasSpoiler = !!person.identity && person.identity !== brief;
  const summary = spoiler && hasSpoiler ? person.identity : brief;

  // Move focus to the banner heading when a person is opened by an explicit
  // click, so keyboard / screen-reader users land on the new editorial panel.
  useEffect(() => {
    if (autoFocus) headingRef.current?.focus();
  }, [person.id, autoFocus]);

  return (
    <section
      className="person-banner"
      role="region"
      aria-label={`编者人物信息，非原文：${person.canonical_name}`}
    >
      <div className="person-card-head">
        <div className="person-card-id">
          <h3 className="person-card-name" tabIndex={-1} ref={headingRef}>
            {person.canonical_name}
          </h3>
          {person.dynasty && <span className="person-meta-chip">{person.dynasty}</span>}
          <span
            className="person-meta-conf"
            data-confidence={person.confidence}
            title={CONF_HINT[person.confidence]}
          >
            {CONF_LABEL[person.confidence]}
          </span>
          <span className="person-prov" title="本卡片为编者整理的人物信息，并非《资治通鉴》原文">
            非原文
          </span>
          {origin && (
            <span
              className="person-prov-origin"
              data-origin={origin}
              title="此卡片的来源"
              aria-label={
                origin === 'inline' ? '来自正文'
                : origin === 'guide' ? '来自导读'
                : '来自检索'
              }
            />
          )}
        </div>
        <button
          type="button"
          className="person-card-close"
          onClick={onClose}
          title="关闭人物信息"
          aria-label="关闭人物信息"
        >×</button>
      </div>

      {clickedLabel && clickedLabel !== person.canonical_name && (
        <p className="person-card-reconcile">
          {clickedLabel}，即{person.canonical_name}（同一人）
        </p>
      )}

      {aliases.length > 0 && (
        <p className="person-card-aliases">
          <span className="person-aliases-label">别称</span>
          {aliases.map((n, i) => (
            <span key={i} className="person-alias">{n.text}</span>
          ))}
        </p>
      )}

      {hasSpoiler && (
        <button
          type="button"
          className={'person-spoiler-toggle' + (spoiler ? ' is-on' : '')}
          onClick={onToggleSpoiler}
          aria-pressed={spoiler}
        >
          {spoiler ? '只看登场身份' : '展开完整生平（含后文剧透）'}
        </button>
      )}

      <p className={'person-banner-summary' + (spoiler && hasSpoiler ? ' is-spoiler' : '')}>
        {summary}
      </p>
    </section>
  );
}
