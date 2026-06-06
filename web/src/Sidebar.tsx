import { useState } from 'react';
import type { Manifest, JuanMeta } from './corpus';

interface Props {
  manifest: Manifest;
  currentJuan: number;
  onSelect: (n: number) => void;
}

function fmtYear(y: number): string {
  return y < 0 ? `前${-y}` : String(y);
}

function formatMeta(j: JuanMeta): { years: string; span: string; emperors: string } {
  let years = '';
  let span = '';
  if (j.ce_start != null && j.ce_end != null) {
    years = j.ce_start === j.ce_end
      ? `${fmtYear(j.ce_start)}年`
      : `${fmtYear(j.ce_start)}-${fmtYear(j.ce_end)}年`;
    const n = j.ce_end - j.ce_start + 1;
    span = n > 1 ? `共${n}年` : '';
  }
  const chunks = j.title.split(/[\s\u3000]+/);
  const list: string[] = [];
  for (const c of chunks) {
    const m = /^([^年载\s\u3000]{1,8}?(?:帝|王|后|公))/.exec(c);
    if (m && !list.includes(m[1])) list.push(m[1]);
  }
  let emperors = '';
  if (list.length === 1) emperors = list[0];
  else if (list.length >= 2) emperors = `${list[0]}-${list[list.length - 1]}`;
  return { years, span, emperors };
}

export default function Sidebar({ manifest, currentJuan, onSelect }: Props) {
  const currentDynasty =
    manifest.juans.find(j => j.juan_no === currentJuan)?.dynasty || '';
  const [openGroups, setOpenGroups] = useState<Set<string>>(
    new Set(currentDynasty ? [currentDynasty] : [])
  );

  const toggle = (d: string) => {
    setOpenGroups(prev => {
      const next = new Set(prev);
      if (next.has(d)) next.delete(d);
      else next.add(d);
      return next;
    });
  };

  return (
    <nav className="sidebar">
      <div className="brand">资治通鉴 · 胡三省音注</div>
      <div className="brand-sub">简体中文阅读器</div>
      <ul className="dynasty-list">
        {manifest.grouped.map(group => {
          const open = openGroups.has(group.dynasty);
          return (
            <li key={group.dynasty} className="dynasty">
              <button
                className="dynasty-toggle"
                onClick={() => toggle(group.dynasty)}
              >
                <span className="caret">{open ? '▾' : '▸'}</span>
                <span>{group.dynasty}</span>
                <span className="count">{group.juans.length}</span>
              </button>
              {open && (
                <ul className="juan-list">
                  {group.juans.map(j => {
                    const { years, span, emperors } = formatMeta(j);
                    return (
                      <li key={j.juan_no}>
                        <button
                          className={'juan-link' + (j.juan_no === currentJuan ? ' active' : '')}
                          onClick={() => onSelect(j.juan_no)}
                          title={j.title}
                        >
                          <span className="juan-line juan-line-top">
                            <span className="juan-no">卷{String(j.juan_no).padStart(3, '0')}</span>
                            <span className="juan-years">{years}</span>
                          </span>
                          <span className="juan-line juan-line-bot">
                            <span className="juan-emperors">{emperors}</span>
                            {span && <span className="juan-span">{span}</span>}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
