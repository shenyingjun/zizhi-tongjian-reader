import { useMemo } from 'react';
import type { YearEntry } from './corpus';

interface Props {
  years: YearEntry[];
  activeParagraphId: number | null;
  onJump: (paragraphId: number) => void;
}

function formatCE(ce: number | null): string {
  if (ce === null) return '';
  if (ce < 0) return `前${-ce}`;
  return String(ce);
}

export default function YearToc({ years, activeParagraphId, onJump }: Props) {
  // Determine which year is currently active (largest paragraph_id <= active).
  const activeYear = useMemo(() => {
    if (activeParagraphId === null) return null;
    let last: YearEntry | null = null;
    for (const y of years) {
      if (y.paragraph_id <= activeParagraphId) last = y;
      else break;
    }
    return last;
  }, [years, activeParagraphId]);

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
