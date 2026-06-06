import { useState } from 'react';
import type { Manifest, JuanMeta } from './corpus';

interface Props {
  manifest: Manifest;
  currentJuan: number;
  readJuans: Set<number>;
  onSelect: (n: number) => void;
}

function fmtYear(y: number): string {
  return y < 0 ? `前${-y}` : String(y);
}

function formatMeta(j: JuanMeta): { years: string; emperors: string } {
  let years = '';
  if (j.ce_start != null && j.ce_end != null) {
    const range = j.ce_start === j.ce_end
      ? `${fmtYear(j.ce_start)}`
      : `${fmtYear(j.ce_start)}-${fmtYear(j.ce_end)}`;
    const n = j.ce_end - j.ce_start + 1;
    years = n > 1 ? `${range}年 · ${n}年` : `${range}年`;
  }
  const chunks = j.title.split(/[\s\u3000]+/);
  const list: string[] = [];
  for (const c of chunks) {
    const m = /^([^年载\s\u3000]{1,8}?(?:帝|王|后|公|祖|宗|主|侯|莽))/.exec(c);
    if (m && !list.includes(m[1])) list.push(m[1]);
  }
  let emperors = '';
  if (list.length === 1) emperors = list[0];
  else if (list.length >= 2) emperors = `${list[0]}-${list[list.length - 1]}`;
  return { years, emperors };
}

export default function Sidebar({ manifest, currentJuan, readJuans, onSelect }: Props) {
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
          const readCount = group.juans.reduce(
            (n, j) => n + (readJuans.has(j.juan_no) ? 1 : 0),
            0,
          );
          return (
            <li key={group.dynasty} className="dynasty">
              <button
                className="dynasty-toggle"
                onClick={() => toggle(group.dynasty)}
              >
                <span className="caret">{open ? '▾' : '▸'}</span>
                <span>{group.dynasty}</span>
                <span className="count">
                  {readCount > 0 ? `${readCount}/${group.juans.length}` : group.juans.length}
                </span>
              </button>
              {open && (
                <ul className="juan-list">
                  {group.juans.map(j => {
                    const { years, emperors } = formatMeta(j);
                    const isRead = readJuans.has(j.juan_no);
                    const isActive = j.juan_no === currentJuan;
                    const cls = 'juan-link'
                      + (isActive ? ' active' : '')
                      + (isRead ? ' read' : '');
                    return (
                      <li key={j.juan_no}>
                        <button
                          className={cls}
                          onClick={() => onSelect(j.juan_no)}
                          title={isRead ? `${j.title}（已读）` : j.title}
                        >
                          <span className="juan-no">卷{String(j.juan_no).padStart(3, '0')}</span>
                          <span className="juan-body">
                            <span className="juan-emperors">{emperors}</span>
                            <span className="juan-years">{years}</span>
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
