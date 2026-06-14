import { useMemo } from 'react';
import type { YearEntry } from './corpus';

interface Props {
  years: YearEntry[];
  activeParagraphId: number | null;
  /** When set, takes precedence over scroll-derived active year. Lets a
   *  year clicked in this TOC stay highlighted even if scroll-tracking would
   *  otherwise pick a neighboring year (last-year edge case, year with no
   *  body paragraphs, trailing scroll events from the jump animation). */
  selectedYearPid?: number | null;
  onJump: (paragraphId: number) => void;
}

function formatCE(ce: number | null): string {
  if (ce === null) return '';
  if (ce < 0) return `前${-ce}`;
  return String(ce);
}

export default function YearToc({ years, activeParagraphId, selectedYearPid, onJump }: Props) {
  // Determine which year is currently active (largest paragraph_id <= active).
  const scrollActiveYear = useMemo(() => {
    if (activeParagraphId === null) return null;
    let last: YearEntry | null = null;
    for (const y of years) {
      if (y.paragraph_id <= activeParagraphId) last = y;
      else break;
    }
    return last;
  }, [years, activeParagraphId]);
  // An explicit click takes precedence over the scroll-derived guess.
  const activeYear = useMemo(() => {
    if (selectedYearPid != null) {
      const match = years.find(y => y.paragraph_id === selectedYearPid);
      if (match) return match;
    }
    return scrollActiveYear;
  }, [years, selectedYearPid, scrollActiveYear]);

  if (!years.length) return null;

  return (
    <nav className="year-toc">
      <div className="year-toc-title">本卷纪年</div>
      <ul>
        {years.map(y => {
          const active = activeYear && y.paragraph_id === activeYear.paragraph_id;
          return (
            <li key={y.paragraph_id}>
              <button
                className={'year-link' + (active ? ' active' : '')}
                onClick={() => onJump(y.paragraph_id)}
                title={y.label}
              >
                <span className="year-ce">{formatCE(y.ce_year)}</span>
                <span className="year-label">{y.label}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
