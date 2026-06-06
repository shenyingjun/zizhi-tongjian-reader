import { useState, useEffect, useLayoutEffect, useRef } from 'react';
import type { Manifest, Juan } from './corpus';
import { loadManifest, loadJuan } from './corpus';
import Sidebar from './Sidebar';
import Reader from './Reader';
import YearToc from './YearToc';
import LookupPanel from './LookupPanel';
import './styles.css';

const LAST_JUAN_KEY = 'zztj.lastJuan';
const READ_JUANS_KEY = 'zztj.readJuans';
const SCROLL_BY_JUAN_KEY = 'zztj.scrollByJuan';
// Mark a juan as "read" once the reader has been scrolled to within this
// fraction of the bottom.
const READ_THRESHOLD = 0.9;

function loadReadJuans(): Set<number> {
  try {
    const raw = localStorage.getItem(READ_JUANS_KEY);
    if (!raw) return new Set();
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return new Set();
    return new Set(arr.filter(n => typeof n === 'number'));
  } catch {
    return new Set();
  }
}

function loadScrollMap(): Record<number, number> {
  try {
    const raw = localStorage.getItem(SCROLL_BY_JUAN_KEY);
    if (!raw) return {};
    const obj = JSON.parse(raw);
    return obj && typeof obj === 'object' ? obj : {};
  } catch {
    return {};
  }
}

interface RouteState {
  juanNo: number;
  q: string;
  p: number | null;
}

function parseHash(): RouteState | null {
  const m = /^#\/juan\/(\d+)(?:\?(.*))?/.exec(window.location.hash);
  if (!m) return null;
  const params = new URLSearchParams(m[2] || '');
  const p = params.get('p');
  return {
    juanNo: Number(m[1]),
    q: params.get('q') || '',
    p: p ? Number(p) : null,
  };
}

function buildHash(juanNo: number, q: string, p: number | null): string {
  const params = new URLSearchParams();
  if (q) params.set('q', q);
  if (p != null) params.set('p', String(p));
  const qs = params.toString();
  return `#/juan/${juanNo}${qs ? '?' + qs : ''}`;
}

export default function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [juan, setJuan] = useState<Juan | null>(null);

  const initialRoute: RouteState = (() => {
    const fromHash = parseHash();
    if (fromHash) return fromHash;
    const saved = localStorage.getItem(LAST_JUAN_KEY);
    return { juanNo: saved ? Number(saved) : 1, q: '', p: null };
  })();

  const [juanNo, setJuanNo] = useState<number>(initialRoute.juanNo);
  const [showHu, setShowHu] = useState<boolean>(() => {
    return localStorage.getItem('zztj.showHu') !== '0';
  });
  const [showSidebar, setShowSidebar] = useState<boolean>(() => {
    return localStorage.getItem('zztj.showSidebar') !== '0';
  });
  const [activeParagraphId, setActiveParagraphId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const readerPaneRef = useRef<HTMLElement | null>(null);
  const [readJuans, setReadJuans] = useState<Set<number>>(() => loadReadJuans());
  const scrollMapRef = useRef<Record<number, number>>(loadScrollMap());
  // Whether the juan currently being loaded should restore its saved scroll
  // position (true on initial load and after sidebar navigation; false when
  // jumping to a specific paragraph from a lookup hit).
  const restoreScrollRef = useRef<boolean>(initialRoute.p === null);

  const [lookupQuery, setLookupQuery] = useState<string>(initialRoute.q);
  const [filterByYear, setFilterByYear] = useState<boolean>(true);
  // Paragraph that should be visually highlighted as the lookup target.
  const [highlightPid, setHighlightPid] = useState<number | null>(initialRoute.p);
  // Paragraph to scroll to once the target juan finishes loading.
  const pendingScrollRef = useRef<number | null>(initialRoute.p);
  // Suppress URL writes when state was just synced FROM the URL (popstate).
  const skipUrlSyncRef = useRef<boolean>(true);

  useEffect(() => {
    loadManifest().then(setManifest).catch(e => setError(String(e)));
  }, []);

  // Push URL on juanNo / query / highlight change (unless coming from popstate).
  useEffect(() => {
    if (skipUrlSyncRef.current) {
      skipUrlSyncRef.current = false;
      // Still seed the initial hash so the first state is bookmarkable.
      const hash = buildHash(juanNo, lookupQuery, highlightPid);
      if (window.location.hash !== hash) {
        window.history.replaceState(null, '', hash);
      }
      return;
    }
    const hash = buildHash(juanNo, lookupQuery, highlightPid);
    if (window.location.hash === hash) return;
    window.history.pushState(null, '', hash);
  }, [juanNo, lookupQuery, highlightPid]);

  // React to browser back/forward.
  useEffect(() => {
    const onPop = () => {
      const r = parseHash();
      if (!r) return;
      skipUrlSyncRef.current = true;
      if (r.p !== null) pendingScrollRef.current = r.p;
      setHighlightPid(r.p);
      setLookupQuery(r.q);
      setJuanNo(r.juanNo);
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  useEffect(() => {
    if (!manifest) return;
    setJuan(null);
    setActiveParagraphId(null);
    loadJuan(juanNo)
      .then(j => {
        setJuan(j);
        localStorage.setItem(LAST_JUAN_KEY, String(juanNo));
        if (pendingScrollRef.current === null && readerPaneRef.current) {
          const pane = readerPaneRef.current;
          const saved = restoreScrollRef.current ? scrollMapRef.current[juanNo] : undefined;
          restoreScrollRef.current = true;
          if (saved && saved > 0) {
            requestAnimationFrame(() => {
              pane.scrollTop = saved;
            });
          } else {
            pane.scrollTop = 0;
          }
        }
        // Note: when pendingScrollRef is set, the scroll is handled by the
        // useLayoutEffect below, which runs after the new juan's DOM is
        // committed.
      })
      .catch(e => setError(String(e)));
  }, [juanNo, manifest]);

  // After the new juan's DOM is committed, scroll to the pending target
  // paragraph (set by jumpToHit when navigating across juans from search).
  useLayoutEffect(() => {
    if (!juan || pendingScrollRef.current === null) return;
    const pid = pendingScrollRef.current;
    pendingScrollRef.current = null;
    scrollParagraphIntoView(pid);
  }, [juan]);

  useEffect(() => {
    localStorage.setItem('zztj.showHu', showHu ? '1' : '0');
  }, [showHu]);

  useEffect(() => {
    localStorage.setItem('zztj.showSidebar', showSidebar ? '1' : '0');
  }, [showSidebar]);

  // Track which paragraph is most prominent in viewport. Also persist scroll
  // position per juan and mark a juan as "read" once the reader has scrolled
  // to the bottom area.
  useEffect(() => {
    if (!juan) return;
    const pane = readerPaneRef.current;
    if (!pane) return;
    let saveTimer: number | null = null;
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

      // Persist scroll position (debounced).
      if (saveTimer !== null) window.clearTimeout(saveTimer);
      saveTimer = window.setTimeout(() => {
        scrollMapRef.current[juanNo] = pane.scrollTop;
        try {
          localStorage.setItem(SCROLL_BY_JUAN_KEY, JSON.stringify(scrollMapRef.current));
        } catch { /* quota */ }
      }, 250);

      // Mark as read once scrolled near the bottom.
      const maxScroll = pane.scrollHeight - pane.clientHeight;
      if (maxScroll > 0 && pane.scrollTop / maxScroll >= READ_THRESHOLD) {
        setReadJuans(prev => {
          if (prev.has(juanNo)) return prev;
          const next = new Set(prev);
          next.add(juanNo);
          try {
            localStorage.setItem(READ_JUANS_KEY, JSON.stringify([...next]));
          } catch { /* quota */ }
          return next;
        });
      }
    };
    onScroll();
    pane.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      if (saveTimer !== null) window.clearTimeout(saveTimer);
      pane.removeEventListener('scroll', onScroll);
    };
  }, [juan, juanNo]);

  // Capture text selection within the reader pane to feed the lookup panel.
  useEffect(() => {
    const pane = readerPaneRef.current;
    if (!pane) return;
    const onMouseDown = () => setHighlightPid(null);
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
    pane.addEventListener('mousedown', onMouseDown);
    pane.addEventListener('mouseup', onMouseUp);
    return () => {
      pane.removeEventListener('mousedown', onMouseDown);
      pane.removeEventListener('mouseup', onMouseUp);
    };
  }, [juan]);

  // Scroll the given paragraph into view. Reliable even with
  // .paragraph { content-visibility: auto }, which makes offsetTop and
  // single-shot scrollIntoView wrong for off-screen paragraphs (above
  // content uses placeholder heights). We iterate a few times with
  // instant scrolls — each pass paints paragraphs near the target so
  // `contain-intrinsic-size: auto` records their real sizes, and the
  // estimate converges within ~3 frames. We finish with one smooth scroll
  // to the final position so the motion feels intentional.
  const scrollParagraphIntoView = (pid: number) => {
    const pane = readerPaneRef.current;
    if (!pane) return;
    const HEADER_OFFSET = 70;
    const DURATION = 350;

    // Re-measure the target's desired scrollTop based on current layout.
    // .paragraph uses content-visibility: auto, so this estimate may shift
    // across frames as nearby paragraphs paint — re-measuring per frame
    // lets us keep the animation pointed at the right place.
    const measureTarget = (): number | null => {
      const el = pane.querySelector<HTMLElement>(`[data-pid="${pid}"]`);
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      const paneRect = pane.getBoundingClientRect();
      return Math.max(0, pane.scrollTop + (rect.top - paneRect.top) - HEADER_OFFSET);
    };

    const start = pane.scrollTop;
    let startTime: number | null = null;
    const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);

    const animate = (now: number) => {
      if (startTime === null) startTime = now;
      const progress = Math.min(1, (now - startTime) / DURATION);
      const target = measureTarget();
      if (target === null) {
        // Element not in DOM yet; try again next frame.
        if (progress < 1) requestAnimationFrame(animate);
        return;
      }
      pane.scrollTop = start + (target - start) * easeOutCubic(progress);
      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        // Final instant settle in case layout shifted right at the end.
        let settleAttempts = 0;
        const settle = () => {
          const t = measureTarget();
          if (t === null) return;
          if (Math.abs(t - pane.scrollTop) > 2) pane.scrollTop = t;
          if (settleAttempts++ < 3) requestAnimationFrame(settle);
        };
        settle();
      }
    };
    requestAnimationFrame(animate);
  };

  const jumpToParagraph = (pid: number) => scrollParagraphIntoView(pid);

  // Jump from a lookup hit: may be same juan or another juan.
  const jumpToHit = (targetJuan: number, paragraphId: number) => {
    setHighlightPid(paragraphId);
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

  // Spoiler floor: use the LAST year of the current 卷 so scrolling within
  // a 卷 doesn't constantly re-run the filter. Within-卷 hits are always
  // shown anyway (see searchCorpus currentJuan bypass).
  const effectiveYear = (() => {
    if (!juan) return null;
    let last: number | null = null;
    for (const y of juan.years) {
      if (y.ce_year !== null) last = y.ce_year;
    }
    return last;
  })();

  return (
    <div className={`layout${showSidebar ? '' : ' sidebar-collapsed'}`}>
      <Sidebar
        manifest={manifest}
        currentJuan={juanNo}
        readJuans={readJuans}
        onSelect={n => { setHighlightPid(null); setJuanNo(n); }}
      />
      <main className="reader-pane" ref={readerPaneRef as React.RefObject<HTMLElement>}>
        <header className="reader-header">
          <button
            type="button"
            className="sidebar-toggle"
            onClick={() => setShowSidebar(s => !s)}
            title={showSidebar ? '隐藏目录' : '显示目录'}
            aria-label={showSidebar ? '隐藏目录' : '显示目录'}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
                 aria-hidden="true">
              <rect x="3" y="4" width="18" height="16" rx="2" />
              <line x1="9" y1="4" x2="9" y2="20" />
            </svg>
          </button>
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
          ? <Reader
              juan={juan}
              showHu={showHu}
              highlightQuery={lookupQuery}
              highlightPid={highlightPid}
            />
          : <div className="loading">载入卷 {juanNo} 中……</div>}
      </main>
      <aside className="person-pane">
        {juan && (
          <YearToc
            years={juan.years}
            activeParagraphId={activeParagraphId}
            onJump={pid => { setHighlightPid(null); jumpToParagraph(pid); }}
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
                onChange={e => { setHighlightPid(null); setLookupQuery(e.target.value); }}
              />
              {lookupQuery && (
                <button
                  type="button"
                  className="lookup-clear"
                  onClick={() => { setHighlightPid(null); setLookupQuery(''); }}
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
            currentJuan={juanNo}
            onJump={jumpToHit}
          />
        </div>
      </aside>
    </div>
  );
}


