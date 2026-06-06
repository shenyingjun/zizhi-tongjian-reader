import { useState } from 'react';
import type { Manifest } from './corpus';

interface Props {
  manifest: Manifest;
  currentJuan: number;
  onSelect: (n: number) => void;
}

export default function Sidebar({ manifest, currentJuan, onSelect }: Props) {
  // Auto-expand the group containing the current juan.
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
                  {group.juans.map(j => (
                    <li key={j.juan_no}>
                      <button
                        className={'juan-link' + (j.juan_no === currentJuan ? ' active' : '')}
                        onClick={() => onSelect(j.juan_no)}
                        title={j.title}
                      >
                        <span className="juan-no">卷{String(j.juan_no).padStart(3, '0')}</span>
                        <span className="juan-range">{j.year_range}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
