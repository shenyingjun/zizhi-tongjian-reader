import { useState, useEffect, useRef } from 'react';
import type { Manifest, Juan } from './corpus';
import { loadManifest, loadJuan } from './corpus';
import Sidebar from './Sidebar';
import Reader from './Reader';
import YearToc from './YearToc';
import LookupPanel from './LookupPanel';
import './styles.css';

const LAST_JUAN_KEY = 'zztj.lastJuan';

export default function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [juan, setJuan] = useState<Juan | null>(null);
  const [juanNo, setJuanNo] = useState<number>(() => {
    const saved = localStorage.getItem(LAST_JUAN_KEY);
    return saved ? Number(saved) : 1;
  });
  const [showHu, setShowHu] = useState<boolean>(() => {
    return localStorage.getItem('zztj.showHu') !== '0';
  });
  const [activeParagraphId, setActiveParagraphId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const readerPaneRef = useRef<HTMLElement | null>(null);

  // Lookup state: query is what the user has selected/clicked.
  const [lookupQuery, setLookupQuery] = useState<string>('');
  const [filterByYear, setFilterByYear] = useState<boolean>(true);
  // Paragraph to scroll to once the target juan finishes loading (cross-juan jump).
  const pendingScrollRef = useRef<number | null>(null);

  useEffect(() => {
    loadManifest().then(setManifest).catch(e => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!manifest) return;
    setJuan(null);
    setActiveParagraphId(null);
    loadJuan(juanNo)
      .then(j => {
        setJuan(j);
        localStorage.setItem(LAST_JUAN_KEY, String(juanNo));
        if (pendingScrollRef.current !== null) {
          const pid = pendingScrollRef.current;
          pendingScrollRef.current = null;
          // Wait for next paint so the paragraph DOM is in place.
          requestAnimationFrame(() => {
            const pane = readerPaneRef.current;
            const el = pane?.querySelector<HTMLElement>(`[data-pid="${pid}"]`);
            if (pane && el) pane.scrollTo({ top: el.offsetTop - 70, behavior: 'auto' });
          });
        } else if (readerPaneRef.current) {
          readerPaneRef.current.scrollTop = 0;
        }
      })
      .catch(e => setError(String(e)));
  }, [juanNo, manifest]);

  useEffect(() => {
    localStorage.setItem('zztj.showHu', showHu ? '1' : '0');
  }, [showHu]);

  // Track which paragraph is most prominent in viewport.
  useEffect(() => {
    if (!juan) return;
    const pane = readerPaneRef.current;
    if (!pane) return;
    const onScroll = () => {
      const paraEls = pane.querySelectorAll<HTMLElement>('[data-pid]');
      const top = pane.scrollTop + 80;
      let activePid: number | null = null;
      for (const el of paraEls) {
        const offset = el.offsetTop;
        if (offset <= top) activePid = Number(el.dataset.pid);
        else break;
      }
      setActiveParagraphId(activePid);
    };
    onScroll();
    pane.addEventListener('scroll', onScroll, { passive: true });
    return () => pane.removeEventListener('scroll', onScroll);
  }, [juan]);

  // Capture text selection within the reader pane to feed the lookup panel.
  useEffect(() => {
    const pane = readerPaneRef.current;
    if (!pane) return;
    const onMouseUp = () => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed) return;
      const txt = sel.toString().trim();
      if (!txt || txt.length > 20) return;
      // Only react if the selection is inside the reader pane.
      const anchor = sel.anchorNode;
      if (!anchor || !pane.contains(anchor instanceof Element ? anchor : anchor.parentElement)) return;
      setLookupQuery(txt);
    };
    pane.addEventListener('mouseup', onMouseUp);
    return () => pane.removeEventListener('mouseup', onMouseUp);
  }, [juan]);

  const jumpToParagraph = (pid: number) => {
    const pane = readerPaneRef.current;
    if (!pane) return;
    const el = pane.querySelector<HTMLElement>(`[data-pid="${pid}"]`);
    if (el) {
      pane.scrollTo({ top: el.offsetTop - 70, behavior: 'smooth' });
    }
  };

  // Jump from a lookup hit: may be same juan or another juan.
  const jumpToHit = (targetJuan: number, paragraphId: number) => {
    if (targetJuan === juanNo) {
      jumpToParagraph(paragraphId);
    } else {
      pendingScrollRef.current = paragraphId;
      setJuanNo(targetJuan);
    }
  };

  if (error) return (
    <div className="error">
      加载失败：{error}<br />
      <span className="muted">请先运行 <code>python -m emit</code> 生成 <code>web/public/text/</code>。</span>
    </div>
  );
  if (!manifest) return <div className="loading">载入目录中……</div>;

  // The active year is what gates the person history (no spoilers).
  const activeYear = (() => {
    if (!juan || activeParagraphId === null) return null;
    let last: number | null = null;
    for (const y of juan.years) {
      if (y.paragraph_id <= activeParagraphId && y.ce_year !== null) last = y.ce_year;
    }
    return last;
  })();

  // If no active year yet (haven't scrolled), use the 卷's start year as a sensible floor.
  const effectiveYear = activeYear ?? juan?.years.find(y => y.ce_year !== null)?.ce_year ?? null;

  return (
    <div className="layout">
      <Sidebar
        manifest={manifest}
        currentJuan={juanNo}
        onSelect={setJuanNo}
      />
      <main className="reader-pane" ref={readerPaneRef as React.RefObject<HTMLElement>}>
        <header className="reader-header">
          <div className="reader-title">
            {juan?.title || `卷${String(juanNo).padStart(3, '0')}`}
          </div>
          <label className="toggle">
            <input
              type="checkbox"
              checked={showHu}
              onChange={e => setShowHu(e.target.checked)}
            />
            <span>显示胡三省音注</span>
          </label>
        </header>
        {juan
          ? <Reader juan={juan} showHu={showHu} />
          : <div className="loading">载入卷 {juanNo} 中……</div>}
      </main>
      <aside className="person-pane">
        {juan && (
          <YearToc
            years={juan.years}
            activeParagraphId={activeParagraphId}
            onJump={jumpToParagraph}
          />
        )}
        <div className="lookup-section">
          <div className="lookup-header">
            <h3>出处检索</h3>
            <div className="lookup-controls">
              <input
                type="text"
                className="lookup-input"
                value={lookupQuery}
                placeholder="选中正文或在此输入"
                onChange={e => setLookupQuery(e.target.value)}
              />
              {lookupQuery && (
                <button
                  type="button"
                  className="lookup-clear"
                  onClick={() => setLookupQuery('')}
                  title="清除"
                >×</button>
              )}
            </div>
            <label className="toggle small">
              <input
                type="checkbox"
                checked={filterByYear}
                onChange={e => setFilterByYear(e.target.checked)}
              />
              <span>仅显示当前年份之前
                {effectiveYear !== null && (
                  <span className="muted">（≤ {effectiveYear < 0 ? `前${-effectiveYear}` : effectiveYear}）</span>
                )}
              </span>
            </label>
          </div>
          <LookupPanel
            query={lookupQuery}
            maxYear={filterByYear ? effectiveYear : null}
            onJump={jumpToHit}
          />
        </div>
      </aside>
    </div>
  );
}


