import { useState, useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import { createPortal } from 'react-dom';
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
    const saved = localStorage.getItem('zztj.showSidebar');
    if (saved !== null) return saved !== '0';
    // First visit: collapsed on mobile by default, open on desktop.
    if (typeof window !== 'undefined'
        && window.matchMedia('(max-width: 900px)').matches) {
      return false;
    }
    return true;
  });
  const [showLookup, setShowLookup] = useState<boolean>(false);
  // Font size scale for the reader body — persisted so iOS / mobile users
  // who can't comfortably pinch-zoom (and would lose layout if they did)
  // have a stable way to bump up text size. Desktop benefits too.
  const [fontScale, setFontScale] = useState<number>(() => {
    const raw = localStorage.getItem('zztj.fontScale');
    const n = raw ? Number(raw) : NaN;
    return Number.isFinite(n) && n >= 0.8 && n <= 1.8 ? n : 1;
  });
  useEffect(() => {
    localStorage.setItem('zztj.fontScale', String(fontScale));
  }, [fontScale]);
  const bumpFont = (delta: number) => {
    setFontScale(s => {
      const next = Math.round((s + delta) * 20) / 20; // 0.05 step
      return Math.min(1.8, Math.max(0.8, next));
    });
  };
  const [activeParagraphId, setActiveParagraphId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const readerPaneRef = useRef<HTMLDivElement | null>(null);
  const [readJuans, setReadJuans] = useState<Set<number>>(() => loadReadJuans());
  const scrollMapRef = useRef<Record<number, number>>(loadScrollMap());
  // Whether the juan currently being loaded should restore its saved scroll
  // position (true on initial load and after sidebar navigation; false when
  // jumping to a specific paragraph from a lookup hit).
  const restoreScrollRef = useRef<boolean>(initialRoute.p === null);

  // `lookupQuery` is what's in the search input and drives the lookup panel.
  // It updates eagerly — including when the user selects text in the reader.
  //
  // `committedQuery` is what the reader body highlights. It only changes on
  // explicit user actions (typing in the input, clicking "搜出处", URL nav).
  //
  // The split exists so that text-selection auto-fill does NOT rebuild the
  // reader's paragraph text nodes. Doing so would (a) collapse the user's
  // in-flight selection — making it appear to "jump" while dragging —
  // and (b) destroy the just-finished selection before Ctrl+C can read it.
  const [lookupQuery, setLookupQuery] = useState<string>(initialRoute.q);
  const [committedQuery, setCommittedQuery] = useState<string>(initialRoute.q);
  const [filterByJuan, setFilterByJuan] = useState<boolean>(true);
  // Paragraph that should be visually highlighted as the lookup target.
  const [highlightPid, setHighlightPid] = useState<number | null>(initialRoute.p);
  // Paragraph to scroll to once the target juan finishes loading.
  const pendingScrollRef = useRef<number | null>(initialRoute.p);
  // Suppress URL writes when state was just synced FROM the URL (popstate).
  const skipUrlSyncRef = useRef<boolean>(true);

  useEffect(() => {
    loadManifest().then(setManifest).catch(e => setError(String(e)));
  }, []);

  // Push URL on juanNo / committed query / highlight change (unless coming from popstate).
  // We intentionally key off `committedQuery` rather than `lookupQuery` so that
  // passive selection auto-fills don't spam the browser history.
  useEffect(() => {
    if (skipUrlSyncRef.current) {
      skipUrlSyncRef.current = false;
      // Still seed the initial hash so the first state is bookmarkable.
      const hash = buildHash(juanNo, committedQuery, highlightPid);
      if (window.location.hash !== hash) {
        window.history.replaceState(null, '', hash);
      }
      return;
    }
    const hash = buildHash(juanNo, committedQuery, highlightPid);
    if (window.location.hash === hash) return;
    window.history.pushState(null, '', hash);
  }, [juanNo, committedQuery, highlightPid]);

  // React to browser back/forward.
  useEffect(() => {
    const onPop = () => {
      const r = parseHash();
      if (!r) return;
      skipUrlSyncRef.current = true;
      if (r.p !== null) pendingScrollRef.current = r.p;
      setHighlightPid(r.p);
      setLookupQuery(r.q);
      setCommittedQuery(r.q);
      setJuanNo(r.juanNo);
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  useEffect(() => {
    if (!manifest) return;
    // Don't blank the current juan while the next one loads — that causes
    // the right-pane YearToc to unmount and the LookupPanel below it to
    // visibly jump. Keep the old juan on screen and swap atomically once
    // the new one resolves.
    let cancelled = false;
    loadJuan(juanNo)
      .then(j => {
        if (cancelled) return;
        setActiveParagraphId(null);
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
      .catch(e => !cancelled && setError(String(e)));
    return () => { cancelled = true; };
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

  // Capture text selection within the reader pane to drive the lookup.
  //
  // Two important rules keep the user's selection stable:
  //
  //   1. We never call `setLookupQuery` while the user is actively dragging
  //      (between pointerdown and pointerup). A React state update here would
  //      re-render the LookupPanel and, more critically, could trigger work
  //      that runs on the same frame as the browser updating the selection —
  //      contributing to the "jumpy" feel.
  //
  //   2. We only ever write to `lookupQuery` (the input value + lookup panel
  //      query), never to `committedQuery`. That means the reader's paragraph
  //      DOM — and therefore the user's live Selection range — is never torn
  //      down by a selection-driven update. This is what makes drag-extend
  //      and Ctrl+C reliable.
  //
  // selectionchange (vs. mouseup) reliably fires for both pointer and touch
  // gestures, so this works on mobile where mouseup is unreliable after
  // the OS text-selection long-press. On hover-capable devices the selection
  // auto-fills the always-visible search input. On touch the lookup drawer
  // is hidden by default, so we defer until the user explicitly taps the
  // floating "搜出处" pill — making "I'm just copying" still work.
  const hoverCapable = useMemo(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return false;
    try { return window.matchMedia('(hover: hover)').matches; } catch { return false; }
  }, []);
  const [pendingSelection, setPendingSelection] = useState<string | null>(null);

  useEffect(() => {
    const pane = readerPaneRef.current;
    if (!pane) return;

    let dragging = false;
    let timer: number | null = null;

    const flushSelection = () => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed) {
        setPendingSelection(null);
        return;
      }
      const txt = sel.toString().trim();
      if (!txt || txt.length > 20) {
        setPendingSelection(null);
        return;
      }
      const anchor = sel.anchorNode;
      const anchorEl = anchor instanceof Element ? anchor : anchor?.parentElement ?? null;
      if (!anchorEl || !pane.contains(anchorEl)) {
        setPendingSelection(null);
        return;
      }
      if (hoverCapable) {
        // Fill the input + lookup panel only. Do NOT touch committedQuery —
        // re-rendering the reader body would collapse this very selection.
        setLookupQuery(prev => (prev === txt ? prev : txt));
        setPendingSelection(null);
      } else {
        setPendingSelection(prev => (prev === txt ? prev : txt));
      }
    };

    const scheduleFlush = (delay: number) => {
      if (timer !== null) window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        timer = null;
        // If the user started a new drag while we were waiting, bail —
        // pointerup will reschedule.
        if (dragging) return;
        flushSelection();
      }, delay);
    };

    const onPointerDown = () => {
      dragging = true;
      setHighlightPid(null);
      // Cancel any pending flush from a previous selection — the user is
      // starting fresh.
      if (timer !== null) { window.clearTimeout(timer); timer = null; }
    };
    const onPointerUp = () => {
      if (!dragging) return;
      dragging = false;
      // Selection's final state may settle one tick after pointerup
      // (Chrome fires a trailing selectionchange). A short delay catches it
      // without making the lookup panel feel laggy.
      scheduleFlush(60);
    };
    const onPointerCancel = () => {
      dragging = false;
    };

    const onSelectionChange = () => {
      // While the user is mid-drag, do nothing. Any state update fanning
      // out to the lookup panel here can cause jank that visibly disturbs
      // the selection. pointerup will pick up the final range.
      if (dragging) return;
      // Touch / keyboard / programmatic selection: debounce briefly to
      // coalesce iOS's flurry of events while dragging the selection handles.
      scheduleFlush(200);
    };

    pane.addEventListener('pointerdown', onPointerDown);
    // Listen on window so we still hear the release if the user drags
    // out of the reader pane before letting go.
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerCancel);
    document.addEventListener('selectionchange', onSelectionChange);
    return () => {
      pane.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerCancel);
      document.removeEventListener('selectionchange', onSelectionChange);
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [juan, hoverCapable]);

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
    const HEADER_OFFSET = 12;
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

  const isMobileWidth = () =>
    typeof window !== 'undefined' && window.matchMedia('(max-width: 900px)').matches;

  // Jump from a lookup hit: may be same juan or another juan.
  const jumpToHit = (targetJuan: number, paragraphId: number) => {
    setHighlightPid(paragraphId);
    if (targetJuan === juanNo) {
      jumpToParagraph(paragraphId);
    } else {
      pendingScrollRef.current = paragraphId;
      setJuanNo(targetJuan);
    }
    if (isMobileWidth()) setShowLookup(false);
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
  // (Spoiler filter is now juan-based, so no need to compute a year cutoff.)

  return (
    <div className={`layout${showSidebar ? '' : ' sidebar-collapsed'}${showLookup ? ' lookup-open' : ''}`}>
      <Sidebar
        manifest={manifest}
        currentJuan={juanNo}
        readJuans={readJuans}
        onSelect={n => {
          setHighlightPid(null);
          setJuanNo(n);
          if (isMobileWidth()) setShowSidebar(false);
        }}
      />
      <main className="reader-pane" style={{ ['--reader-font-scale' as any]: fontScale }}>
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
            {(() => {
              if (!juan) return `卷${String(juanNo).padStart(3, '0')}`;
              // Glue the 卷号 + 纪名 prefix together with a non-breaking
              // space so it never wraps mid-unit, while leaving normal
              // spaces and 　 (U+3000) between later sections as natural
              // wrap points. This keeps the long title from breaking at
              // arbitrary spots without forcing a rigid multi-line layout
              // (which would overlap the header controls at narrow widths).
              const m = /^(\S+)\s+(\S+)(\s[\s\S]*)?$/.exec(juan.title);
              if (!m) return juan.title;
              return `${m[1]}\u00A0${m[2]}${m[3] ?? ''}`;
            })()}
          </div>
          <label className="toggle">
            <input
              type="checkbox"
              checked={showHu}
              onChange={e => setShowHu(e.target.checked)}
            />
            <span>胡三省音注</span>
          </label>
          <span className="font-size-controls" role="group" aria-label="正文字号">
            <button
              type="button"
              className="font-size-btn"
              onClick={() => bumpFont(-0.1)}
              disabled={fontScale <= 0.8 + 1e-6}
              title="缩小字号"
              aria-label="缩小字号"
            >A−</button>
            <button
              type="button"
              className="font-size-btn font-size-reset"
              onClick={() => setFontScale(1)}
              disabled={Math.abs(fontScale - 1) < 1e-6}
              title={`重置字号（当前 ${Math.round(fontScale * 100)}%）`}
              aria-label="重置字号"
            >{Math.round(fontScale * 100)}%</button>
            <button
              type="button"
              className="font-size-btn"
              onClick={() => bumpFont(0.1)}
              disabled={fontScale >= 1.8 - 1e-6}
              title="放大字号"
              aria-label="放大字号"
            >A+</button>
          </span>
          <button
            type="button"
            className="lookup-toggle"
            onClick={() => setShowLookup(s => !s)}
            title={showLookup ? '隐藏检索' : '出处检索'}
            aria-label={showLookup ? '隐藏检索' : '出处检索'}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
                 aria-hidden="true">
              <circle cx="11" cy="11" r="7" />
              <line x1="21" y1="21" x2="16.5" y2="16.5" />
            </svg>
          </button>
        </header>
        <div className="reader-scroller" ref={readerPaneRef}>
          {juan
            ? <Reader
                juan={juan}
                showHu={showHu}
                highlightQuery={committedQuery}
                highlightPid={highlightPid}
              />
            : <div className="loading">载入卷 {juanNo} 中……</div>}
        </div>
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
                onChange={e => {
                  setHighlightPid(null);
                  setLookupQuery(e.target.value);
                  // Typing in the input is an explicit search action — commit
                  // so the reader body highlights too.
                  setCommittedQuery(e.target.value);
                }}
              />
              {lookupQuery && (
                <button
                  type="button"
                  className="lookup-clear"
                  onClick={() => { setHighlightPid(null); setLookupQuery(''); setCommittedQuery(''); }}
                  title="清除"
                >×</button>
              )}
            </div>
            <label className="toggle small">
              <input
                type="checkbox"
                checked={filterByJuan}
                onChange={e => setFilterByJuan(e.target.checked)}
              />
              <span>仅显示当前卷之前</span>
            </label>
          </div>
          <div className="lookup-body">
            <LookupPanel
              query={lookupQuery}
              maxJuan={filterByJuan ? juanNo : null}
              currentJuan={juanNo}
              highlightPid={highlightPid}
              onJump={jumpToHit}
            />
          </div>
        </div>
      </aside>
      <div
        className="drawer-backdrop"
        onClick={() => { setShowSidebar(false); setShowLookup(false); }}
        aria-hidden="true"
      />
      {pendingSelection && createPortal(
        <div className="selection-action-bar" role="toolbar" aria-label="选词检索">
          <span className="selection-action-text" title={pendingSelection}>
            「{pendingSelection.length > 8 ? pendingSelection.slice(0, 7) + '…' : pendingSelection}」
          </span>
          <button
            type="button"
            className="selection-action-btn"
            onClick={() => {
              const q = pendingSelection;
              setLookupQuery(q);
              // Explicit user action — commit so the reader body highlights.
              setCommittedQuery(q);
              setShowLookup(true);
              setPendingSelection(null);
              // Drop the OS selection so its menu and our pill both go away;
              // otherwise selectionchange will re-fire when the user taps
              // elsewhere and the pill might briefly flash back.
              try { window.getSelection()?.removeAllRanges(); } catch { /* noop */ }
            }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                 aria-hidden="true">
              <circle cx="11" cy="11" r="7" />
              <line x1="21" y1="21" x2="16.5" y2="16.5" />
            </svg>
            搜出处
          </button>
        </div>,
        document.body,
      )}
    </div>
  );
}


