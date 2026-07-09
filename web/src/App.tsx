import { useState, useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import { createPortal } from 'react-dom';
import type { Manifest, Juan, GuideSummary, Person, PersonMention, GuidePersonRef, AppearanceRow, PersonVariant } from './corpus';
import { loadManifest, loadJuan, loadJuanGuide, loadPeople, loadPersonMentions, loadAppearances, setPersonVariant } from './corpus';
import Sidebar from './Sidebar';
import Reader from './Reader';
import YearToc from './YearToc';
import LookupPanel from './LookupPanel';
import PersonCard from './PersonCard';
import type { PersonSpan } from './highlight';
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
  // 白话导读 (plain-language comprehension layer) reading mode.
  //   'off'   — never show guide blocks
  //   'brief' — one-liner inline, expandable per block (default)
  //   'full'  — every block expanded
  const [guideMode, setGuideMode] = useState<'off' | 'brief' | 'full'>(() => {
    const saved = localStorage.getItem('zztj.guideMode');
    return saved === 'off' || saved === 'full' ? saved : 'brief';
  });
  useEffect(() => {
    localStorage.setItem('zztj.guideMode', guideMode);
  }, [guideMode]);
  // Person-data pipeline variant: 'v1' = current production underlines/cards,
  // 'v2' = the two-stage local-first pipeline (experimental). Applied to the
  // corpus module synchronously at init so the very first people/mentions fetch
  // already targets the right directory.
  const [personVariant, setPersonVariantState] = useState<PersonVariant>(() => {
    const v: PersonVariant = localStorage.getItem('zztj.personVariant') === 'v2' ? 'v2' : 'v1';
    setPersonVariant(v);
    return v;
  });
  const changePersonVariant = (v: PersonVariant) => {
    if (v === personVariant) return;
    setPersonVariant(v);                 // clears person caches synchronously
    localStorage.setItem('zztj.personVariant', v);
    setPersonVariantState(v);            // re-renders → people/mentions effects refetch
  };
  // anchor_pid → reviewed summary for the currently loaded 卷. Empty when the
  // 卷 ships no guide file (graceful absence).
  const [guideByAnchorPid, setGuideByAnchorPid] =
    useState<Map<number, GuideSummary>>(new Map());
  // ── 人物识别 (person identity) state ──
  // Canonical person KB (loaded once; empty when no person assets ship).
  const [people, setPeople] = useState<Map<string, Person>>(new Map());
  // Cross-卷 verified appearance index (persons/appearances.json). Drives the
  // NER-accurate occurrence list for a bound person; empty when assets absent.
  const [appearances, setAppearances] = useState<Map<string, AppearanceRow[]>>(new Map());
  // Person mentions for the currently loaded 卷 (empty when the 卷 has no
  // person sidecar — a graceful, non-blocking absence).
  const [mentions, setMentions] = useState<PersonMention[]>([]);
  // Opt-in reveal of an open person's future (spoiler) appearances.
  // Whether the identity banner shows the full life arc (剧透) vs the
  // spoiler-safe establishing brief.
  const [spoilerSummary, setSpoilerSummary] = useState<boolean>(false);
  // When the reader uses 跳至此段 to revisit an earlier mention, we stash the
  // paragraph they jumped *from* so a "返回刚才阅读处" pill can bring them back.
  const [jumpReturnPid, setJumpReturnPid] = useState<number | null>(null);
  // Pid of a year the user explicitly clicked in YearToc. When set, the
  // YearToc highlight uses this directly rather than the scroll-derived
  // activeParagraphId — so the highlight stays locked on what the user
  // clicked, immune to any mid-animation overshoot or end-of-pane edge
  // cases (last year unreachable, year-with-no-body, etc.). Cleared when
  // the user does a real scroll.
  const [selectedYearPid, setSelectedYearPid] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const readerPaneRef = useRef<HTMLDivElement | null>(null);
  const settingsBtnRef = useRef<HTMLButtonElement | null>(null);
  const settingsMenuRef = useRef<HTMLDivElement | null>(null);
  // True while a programmatic scroll (year click, lookup jump) is in flight.
  // During this window the scroll handler must not update activeParagraphId —
  // measureTarget re-estimates mid-animation as paragraphs paint and can
  // briefly overshoot, which otherwise causes the YearToc highlight to flash
  // to the wrong year before settling.
  const programmaticScrollRef = useRef<boolean>(false);
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
  // ── The dock's single source of truth: one persistent "subject" ──
  // A discriminated union — years (resting) | lookup (出处检索) | person
  // (人物身份). The dock shows a person card IFF subject.kind === 'person';
  // a typed/selected name NEVER auto-opens a card. An empty lookup query is
  // unrepresentable (it renders 'years'). The transient selection popover is
  // owned by window.getSelection() and is never written into the subject.
  type DockSubject =
    | { kind: 'years' }
    | { kind: 'lookup'; query: string;
        origin: 'selection-promoted' | 'typed' | 'unbound-pill' }
    | { kind: 'person'; personId: string; atPid: number;
        origin: 'inline' | 'guide' | 'lookup-promoted';
        from: DockSubject | null; clickedLabel?: string };
  const [subject, setSubject] = useState<DockSubject>(
    initialRoute.q ? { kind: 'lookup', query: initialRoute.q, origin: 'typed' } : { kind: 'years' },
  );
  // Live mirror of the subject readable from effect/handler closures.
  const subjectRef = useRef<DockSubject>(subject);
  subjectRef.current = subject;
  // Most recent lookup query / person — power the header seg control's
  // "检索"/"人物" buttons, which stay re-selectable once a subject of that
  // kind has existed.
  const [lastLookup, setLastLookup] = useState<string>(initialRoute.q || '');
  const [lastPerson, setLastPerson] = useState<
    { personId: string; atPid: number; origin: 'inline' | 'guide' | 'lookup-promoted'; clickedLabel?: string } | null
  >(null);
  // Mobile bottom-sheet detent for the SUBJECT (lookup/person). 'peek' == the
  // sheet is collapsed (== years/dismissed); 'half'/'full' show the subject.
  // Pure function of the subject by default (set on subject change); the user
  // can swipe between half and full.
  type SheetDetent = 'peek' | 'half' | 'full';
  const [sheetDetent, setSheetDetent] = useState<SheetDetent>(
    initialRoute.q ? 'half' : 'peek',
  );
  // Transient "compose a fresh typed search" intent — opens the 检索 input
  // even before a (non-empty) lookup subject exists, without violating the
  // "empty lookup is unrepresentable" rule (the subject stays 'years' until a
  // non-empty query is typed). Entry point for typed search (the header ⌕).
  const [searchCompose, setSearchCompose] = useState<boolean>(false);
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
        setSelectedYearPid(null);
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

  // Load the per-卷 白话导读 file alongside the 卷 text. Missing files resolve
  // to null (loadJuanGuide swallows 404/parse errors) → empty map → no blocks.
  useEffect(() => {
    let cancelled = false;
    setGuideByAnchorPid(new Map());
    loadJuanGuide(juanNo).then(guide => {
      if (cancelled || !guide) return;
      const map = new Map<number, GuideSummary>();
      for (const s of guide.summaries) {
        if (s.confidence === 'omit') continue;
        map.set(s.anchor_pid, s);
      }
      setGuideByAnchorPid(map);
    });
    return () => { cancelled = true; };
  }, [juanNo]);

  useEffect(() => {
    localStorage.setItem('zztj.showSidebar', showSidebar ? '1' : '0');
  }, [showSidebar]);

  // Load the canonical person KB once. Absent assets resolve to an empty map
  // (loadPeople swallows errors) → no identity layer, literal search only.
  useEffect(() => {
    loadPeople().then(setPeople);
  }, [personVariant]);

  // Load the per-卷 person mention sidecar alongside the 卷 text. Missing files
  // resolve to null → empty mentions → no person affordances for that 卷.
  // Opening a new 卷 also dismisses any open person card (pop the subject back
  // to 纪年 if a person was open).
  useEffect(() => {
    let cancelled = false;
    setMentions([]);
    setSpoilerSummary(false);
    if (subjectRef.current.kind === 'person') {
      setSubject({ kind: 'years' });
      setLookupQuery('');
      setCommittedQuery('');
    }
    loadPersonMentions(juanNo).then(file => {
      if (cancelled || !file) return;
      setMentions(file.mentions);
    });
    return () => { cancelled = true; };
  }, [juanNo, personVariant]);

  // Track which paragraph is most prominent in viewport. Also persist scroll
  // position per juan and mark a juan as "read" once the reader has scrolled
  // to the bottom area.
  useEffect(() => {
    if (!juan) return;
    const pane = readerPaneRef.current;
    if (!pane) return;
    let saveTimer: number | null = null;
    const onScroll = () => {
      const paneRect = pane.getBoundingClientRect();
      // ".paragraph" has no positioned ancestor, so el.offsetTop is body-
      // relative, not pane-relative, and mixing it with pane.scrollTop gave
      // increasingly wrong answers further down a juan (which made YearToc
      // highlight year N-1 after jumping to year N). Use rects in the pane's
      // coordinate space instead.
      const anchorY = paneRect.top + 80;
      const paraEls = pane.querySelectorAll<HTMLElement>('[data-pid]');
      let activePid: number | null = null;
      for (const el of paraEls) {
        const top = el.getBoundingClientRect().top;
        if (top <= anchorY) activePid = Number(el.dataset.pid);
        else break;
      }
      // When the pane is scrolled to its very end the last year's heading may
      // sit below the anchor line (the pane can't scroll any further), which
      // would otherwise leave it un-highlightable. Snap to the last paragraph
      // so the bottom-most year always wins at the bottom of the juan.
      const atBottom = pane.scrollHeight - (pane.scrollTop + pane.clientHeight) <= 2;
      if (atBottom && paraEls.length > 0) {
        activePid = Number(paraEls[paraEls.length - 1].dataset.pid);
      }
      // Skip overriding the active paragraph while a programmatic scroll is
      // animating — the click handler has already set the desired pid and
      // mid-animation overshoot would only cause a flicker.
      if (!programmaticScrollRef.current) {
        setActiveParagraphId(activePid);
      }

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

    // Clear the YearToc "selected year" lock ONLY on real user-initiated
    // scroll gestures. We can't use the generic `scroll` event for this —
    // programmatic scrollTop writes (and their trailing async scroll events
    // that may sneak through right after our lock clears) would otherwise
    // drop the lock and let the YearToc highlight flicker to a neighboring
    // year. Wheel / touchmove / keyboard on the pane are unambiguous user
    // gestures, so they're safe signals.
    const dropSelection = () => setSelectedYearPid(null);
    pane.addEventListener('wheel', dropSelection, { passive: true });
    pane.addEventListener('touchmove', dropSelection, { passive: true });
    pane.addEventListener('keydown', dropSelection);

    return () => {
      if (saveTimer !== null) window.clearTimeout(saveTimer);
      pane.removeEventListener('scroll', onScroll);
      pane.removeEventListener('wheel', dropSelection);
      pane.removeEventListener('touchmove', dropSelection);
      pane.removeEventListener('keydown', dropSelection);
    };
  }, [juan, juanNo]);

  // Edge-swipe gestures on mobile open the side drawers:
  //   - left-edge → 卷 navigation sidebar
  //   - right-edge → the 3-in-1 dock, opened on its 本卷纪年 view
  // The dock can also be opened by its 检索 button in the header (two triggers,
  // one drawer).
  //
  // When a drawer is already open, a swipe in the closing direction
  // dismisses it (swipe left for the sidebar, swipe right for the dock).
  // Close gestures may start anywhere — they don't require the edge strip —
  // so the user can grab the drawer itself and shove it.
  //
  // Notes:
  //   - In Safari (not standalone PWA) iOS reserves edge swipes for browser
  //     back/forward; the system gesture usually wins. The sidebar still has
  //     its toolbar button as a fallback; the dock has its 检索 button.
  //   - The trigger zone is a thin strip (24px) so reader text selection is
  //     not affected by touches that start inside the body.
  const drawerStateRef = useRef({ sidebar: false, dock: false });
  drawerStateRef.current = {
    sidebar: showSidebar,
    dock: sheetDetent !== 'peek',
  };
  // Stable handle to the latest openDockYears so the (deps: []) gesture effect
  // always opens with current state.
  const openDockYearsRef = useRef<() => void>(() => {});
  useEffect(() => {
    const EDGE = 24;
    const THRESHOLD_X = 50;
    const MAX_OFF_AXIS = 35;
    let startX: number | null = null;
    let startY = 0;
    let fromLeft = false;
    let fromRight = false;

    const isMobile = () => window.matchMedia('(max-width: 900px)').matches;

    const onTouchStart = (e: TouchEvent) => {
      // Multi-touch (e.g. pinch-zoom) is not a drawer gesture.
      if (e.touches.length !== 1 || !isMobile()) { startX = null; return; }
      const t = e.touches[0];
      const w = window.innerWidth;
      fromLeft = t.clientX <= EDGE;
      fromRight = t.clientX >= w - EDGE;
      const anyOpen = drawerStateRef.current.sidebar
        || drawerStateRef.current.dock;
      // Only track the gesture if it could possibly do something:
      // either it starts in an edge strip (potential open) or a drawer
      // is already open (potential close).
      if (!fromLeft && !fromRight && !anyOpen) { startX = null; return; }
      startX = t.clientX;
      startY = t.clientY;
    };
    const onTouchEnd = (e: TouchEvent) => {
      if (startX === null) return;
      const t = e.changedTouches[0];
      const dx = t.clientX - startX;
      const dy = t.clientY - startY;
      startX = null;
      if (Math.abs(dy) > MAX_OFF_AXIS) return;
      if (Math.abs(dx) < THRESHOLD_X) return;

      const { sidebar, dock } = drawerStateRef.current;
      // Close gestures win over open gestures so the user can dismiss a
      // drawer without an accidental re-open on the opposite edge.
      if (sidebar && dx < 0) { setShowSidebar(false); return; }
      if (dock && dx > 0) { setSheetDetent('peek'); return; }

      if (fromLeft && dx > 0) {
        setShowSidebar(true);
        setSheetDetent('peek');
      } else if (fromRight && dx < 0) {
        // Right-edge swipe opens the 3-in-1 dock on its 纪年 view.
        openDockYearsRef.current();
      }
    };
    const onTouchCancel = () => { startX = null; };
    document.addEventListener('touchstart', onTouchStart, { passive: true });
    document.addEventListener('touchend', onTouchEnd, { passive: true });
    document.addEventListener('touchcancel', onTouchCancel, { passive: true });
    return () => {
      document.removeEventListener('touchstart', onTouchStart);
      document.removeEventListener('touchend', onTouchEnd);
      document.removeEventListener('touchcancel', onTouchCancel);
    };
  }, []);

  // Capture text selection within the reader pane to drive a transient,
  // selection-owned popover (one explicit "search" affordance on ALL devices —
  // no more desktop auto-fill). The popover is NEVER written into the dock
  // subject; it auto-dies on deselect by construction.
  //
  // Two important rules keep the user's selection stable:
  //
  //   1. We never fan a React state update out to the reader body while the
  //      user is actively dragging (between pointerdown and pointerup).
  //
  //   2. We only ever paint other-occurrence matches via the CSS Custom
  //      Highlight API (selectionMatch) — never via committedQuery / <mark>
  //      wrapping. Rebuilding the reader text nodes would collapse the user's
  //      live Selection and break Ctrl+C.
  //
  // selectionchange (vs. mouseup) reliably fires for both pointer and touch
  // gestures, so this works on mobile where mouseup is unreliable after the OS
  // text-selection long-press.
  // Text of the current reader selection, used to paint yellow highlights on
  // every other occurrence via the CSS Custom Highlight API. We intentionally
  // do NOT route this through committedQuery / <mark> wrapping: rebuilding the
  // reader text nodes would collapse the user's live selection and break
  // Ctrl+C. Custom highlights are painted over the existing DOM, leaving the
  // selection (and clipboard) untouched.
  const [selectionMatch, setSelectionMatch] = useState<string>('');
  // The transient selection popover, owned by window.getSelection(). `rect` is
  // the viewport anchor on desktop (null → mobile bottom pill). `personId` is
  // set only when the selection exactly equals a unique in-卷 person name.
  const [selectionPopover, setSelectionPopover] = useState<
    { text: string; rect: { cx: number; top: number } | null; personId: string | null } | null
  >(null);
  const selectionPopoverRef = useRef<typeof selectionPopover>(null);
  selectionPopoverRef.current = selectionPopover;
  // Resolve a selected string to a unique in-卷 person id (else null). Assigned
  // after nameIndex/juanSurfaceIndex are defined; read via ref so the selection
  // effect always sees the current resolver.
  const exactPersonRef = useRef<(t: string) => string | null>(() => null);

  useEffect(() => {
    const pane = readerPaneRef.current;
    if (!pane) return;

    let dragging = false;
    let timer: number | null = null;

    const flushSelection = () => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed) {
        setSelectionPopover(null);
        setSelectionMatch('');
        return;
      }
      const txt = sel.toString().trim();
      if (!txt || txt.length > 20) {
        setSelectionPopover(null);
        setSelectionMatch('');
        return;
      }
      const anchor = sel.anchorNode;
      const anchorEl = anchor instanceof Element ? anchor : anchor?.parentElement ?? null;
      if (!anchorEl || !pane.contains(anchorEl)) {
        setSelectionPopover(null);
        setSelectionMatch('');
        return;
      }
      setSelectionMatch(prev => (prev === txt ? prev : txt));
      // Desktop anchors the popover BELOW the selection rect (so it doesn't
      // collide with the Edge "mini menu" / native selection toolbar, which
      // sits above the selection). Mobile keeps the bottom pill (rect == null).
      let rect: { cx: number; top: number } | null = null;
      if (!isMobileWidth()) {
        try {
          const r = sel.getRangeAt(sel.rangeCount - 1).getBoundingClientRect();
          if (r && (r.width || r.height)) rect = { cx: r.left + r.width / 2, top: r.bottom };
        } catch { /* noop */ }
      }
      const personId = exactPersonRef.current(txt);
      setSelectionPopover({ text: txt, rect, personId });
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

    const onPointerDown = (e: PointerEvent) => {
      dragging = true;
      setHighlightPid(null);
      // Cancel any pending flush from a previous selection — the user is
      // starting fresh.
      if (timer !== null) { window.clearTimeout(timer); timer = null; }

      // If the user mousedowns inside an existing selection, browsers enter
      // text-drag mode (mouse moves become a native drag-and-drop of the
      // selected text) instead of starting a fresh selection. In a reader,
      // dragging text out is near-useless and the inability to re-select on
      // top of an existing selection is a constant annoyance. Clear the
      // selection here, BEFORE the browser's mousedown handler decides
      // drag-vs-select, so the next drag starts cleanly.
      //
      // Skip when modifiers are held so shift+click extension and OS gestures
      // (Ctrl/Cmd, Alt) keep working.
      if (e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || sel.rangeCount === 0) return;
      const x = e.clientX, y = e.clientY;
      let insideSelection = false;
      outer: for (let i = 0; i < sel.rangeCount; i++) {
        const rects = sel.getRangeAt(i).getClientRects();
        for (let j = 0; j < rects.length; j++) {
          const r = rects[j];
          if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) {
            insideSelection = true;
            break outer;
          }
        }
      }
      if (insideSelection) sel.removeAllRanges();
    };

    // Belt-and-suspenders: even if the browser still tries to start a text
    // drag (e.g. the pointer was just outside our rect tolerance), cancel
    // it so the user never sees the drag cursor flicker on the reader.
    const onDragStart = (e: DragEvent) => { e.preventDefault(); };
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

    // Suppress the browser's native context / selection menu (right-click on
    // desktop, long-press callout on touch) when the user has selected reader
    // text. The native menu otherwise pops over our own selection popover at
    // the same anchor. We only swallow it when there's an active in-pane
    // selection, so right-clicking elsewhere (and long-press to *start* a
    // selection) still behaves normally.
    const onContextMenu = (e: MouseEvent) => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || !sel.toString().trim()) return;
      const anchor = sel.anchorNode;
      const anchorEl = anchor instanceof Element ? anchor : anchor?.parentElement ?? null;
      if (anchorEl && pane.contains(anchorEl)) e.preventDefault();
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
    pane.addEventListener('dragstart', onDragStart);
    pane.addEventListener('contextmenu', onContextMenu);
    // Listen on window so we still hear the release if the user drags
    // out of the reader pane before letting go.
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerCancel);
    document.addEventListener('selectionchange', onSelectionChange);
    return () => {
      pane.removeEventListener('pointerdown', onPointerDown);
      pane.removeEventListener('dragstart', onDragStart);
      pane.removeEventListener('contextmenu', onContextMenu);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerCancel);
      document.removeEventListener('selectionchange', onSelectionChange);
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [juan]);

  // Paint a yellow highlight over every other occurrence of the selected text.
  // Uses the CSS Custom Highlight API so the existing reader DOM (and thus the
  // user's live selection + clipboard) is never rebuilt. Falls back to a no-op
  // on browsers without the API.
  useEffect(() => {
    const cssHighlights = (CSS as unknown as { highlights?: Map<string, unknown> }).highlights;
    const HighlightCtor = (window as unknown as { Highlight?: new (...ranges: Range[]) => unknown }).Highlight;
    if (!cssHighlights || !HighlightCtor) return;
    const HL_NAME = 'selection-match';
    const pane = readerPaneRef.current;
    const q = selectionMatch;
    if (!pane || !q) {
      cssHighlights.delete(HL_NAME);
      return;
    }
    const ranges: Range[] = [];
    const walker = document.createTreeWalker(pane, NodeFilter.SHOW_TEXT);
    let node: Node | null;
    while ((node = walker.nextNode())) {
      const text = node.nodeValue ?? '';
      let from = 0;
      while (true) {
        const idx = text.indexOf(q, from);
        if (idx < 0) break;
        const r = document.createRange();
        r.setStart(node, idx);
        r.setEnd(node, idx + q.length);
        ranges.push(r);
        from = idx + q.length;
      }
    }
    if (ranges.length) {
      cssHighlights.set(HL_NAME, new HighlightCtor(...ranges));
    } else {
      cssHighlights.delete(HL_NAME);
    }
    return () => { cssHighlights.delete(HL_NAME); };
  }, [selectionMatch, juan, showHu, committedQuery]);

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

    programmaticScrollRef.current = true;
    // Defer unlock past the trailing scroll event and explicitly set the
    // final activeParagraphId. Setting pane.scrollTop dispatches `scroll`
    // asynchronously; if we cleared the flag synchronously at the end of
    // settle, that final event would run the handler and set
    // activeParagraphId to whatever the anchor heuristic picks (which can
    // legitimately differ from `pid` — the paragraph just below a short
    // year heading, or the next year's heading when this year has no body).
    // Two rAFs reliably outlive the trailing event; the setActiveParagraphId
    // makes the post-animation state authoritative regardless of which
    // entry point (year click, lookup hit) initiated the jump.
    const finish = () => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          programmaticScrollRef.current = false;
          setActiveParagraphId(pid);
        });
      });
    };

    const animate = (now: number) => {
      if (startTime === null) startTime = now;
      const progress = Math.min(1, (now - startTime) / DURATION);
      const target = measureTarget();
      if (target === null) {
        // Element not in DOM yet; try again next frame.
        if (progress < 1) requestAnimationFrame(animate);
        else finish();
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
          if (t === null) { finish(); return; }
          if (Math.abs(t - pane.scrollTop) > 2) pane.scrollTop = t;
          if (settleAttempts++ < 3) requestAnimationFrame(settle);
          else finish();
        };
        settle();
      }
    };
    requestAnimationFrame(animate);
  };

  // Open/position the 设置 popover (native top-layer popover so it escapes the
  // scroller's overflow). CSS anchor positioning is unreliable in target
  // browsers, so we position by JS under the button on desktop; on mobile we
  // clear the inline coords and let CSS render it as a bottom sheet.
  const toggleSettings = () => {
    const menu = settingsMenuRef.current;
    const btn = settingsBtnRef.current;
    if (!menu || !btn) return;
    if (menu.matches(':popover-open')) { menu.hidePopover(); return; }
    menu.showPopover();
    if (isMobileWidth()) { menu.style.left = ''; menu.style.top = ''; return; }
    const r = btn.getBoundingClientRect();
    const mw = menu.offsetWidth || 240;
    const left = Math.max(8, Math.min(r.right - mw, window.innerWidth - mw - 8));
    menu.style.left = `${left}px`;
    menu.style.top = `${r.bottom + 6}px`;
  };

  const jumpToParagraph = (pid: number) => scrollParagraphIntoView(pid);

  const isMobileWidth = () =>
    typeof window !== 'undefined' && window.matchMedia('(max-width: 900px)').matches;

  // Apply a new dock subject and keep the lookupQuery/committedQuery plumbing
  // (input box, reader <mark>, URL, LookupPanel) in sync with it. Remembers the
  // last lookup / person so the header seg control can re-select them, and
  // reconciles the mobile sheet detent (pure function of subject by default).
  const applySubject = (next: DockSubject) => {
    setSubject(next);
    if (next.kind === 'lookup') {
      setLookupQuery(next.query);
      setCommittedQuery(next.query);
      setLastLookup(next.query);
      setSheetDetent('half');
    } else if (next.kind === 'person') {
      const q = people.get(next.personId)?.canonical_name ?? '';
      setLookupQuery(q);
      setCommittedQuery(q);
      setLastPerson({ personId: next.personId, atPid: next.atPid, origin: next.origin, clickedLabel: next.clickedLabel });
      setSheetDetent('half');
    } else {
      setLookupQuery('');
      setCommittedQuery('');
      setSheetDetent('peek');
    }
  };

  // Return to 纪年 (breadcrumb-home / full dismiss). Drops the live OS selection
  // so a debounced selectionchange can't re-open a popover and snap back.
  const goToYears = () => {
    setSpoilerSummary(false);
    setHighlightPid(null);
    setSearchCompose(false);
    setSelectionPopover(null);
    setSelectionMatch('');
    applySubject({ kind: 'years' });
    try { window.getSelection()?.removeAllRanges(); } catch { /* noop */ }
  };

  // Open the dock on its 纪年 view (mobile right-edge swipe entry point). On
  // mobile applySubject({kind:'years'}) collapses the sheet to peek, so we
  // re-raise it to half here to actually slide the drawer in.
  const openDockYears = () => {
    goToYears();
    setSheetDetent('half');
    setShowSidebar(false);
  };
  openDockYearsRef.current = openDockYears;

  // The single back/dismiss ladder: person → (from ?? years), lookup → years,
  // years → no-op (desktop) / collapse to peek (mobile). Always drops the live
  // selection. Routed by card ×, breadcrumb ‹, Esc, and the sheet swipe-down.
  const dockBack = () => {
    try { window.getSelection()?.removeAllRanges(); } catch { /* noop */ }
    setSelectionPopover(null);
    setSelectionMatch('');
    setSearchCompose(false);
    setSpoilerSummary(false);
    setHighlightPid(null);
    const cur = subjectRef.current;
    if (cur.kind === 'person') {
      applySubject(cur.from ?? { kind: 'years' });
    } else if (cur.kind === 'lookup') {
      applySubject({ kind: 'years' });
    } else {
      setSheetDetent('peek');
    }
  };
  const dockBackRef = useRef(dockBack);
  dockBackRef.current = dockBack;

  // Look a term up as a fresh 出处检索 — used by 白话导读 unbound pills.
  const searchFor = (query: string) => {
    const q = query.trim();
    if (!q) return;
    setSpoilerSummary(false);
    setHighlightPid(null);
    setSearchCompose(true);
    setSelectionPopover(null);
    setSelectionMatch('');
    applySubject({ kind: 'lookup', query: q, origin: 'unbound-pill' });
    if (isMobileWidth()) { setShowSidebar(false); }
  };

  // Jump from a lookup hit: may be same juan or another juan. On mobile,
  // collapse the sheet to peek so the jumped-to passage is visible.
  const jumpToHit = (targetJuan: number, paragraphId: number) => {
    setHighlightPid(paragraphId);
    if (targetJuan === juanNo) {
      jumpToParagraph(paragraphId);
    } else {
      pendingScrollRef.current = paragraphId;
      setJuanNo(targetJuan);
    }
    if (isMobileWidth()) setSheetDetent('peek');
  };

  // ── 人物识别 derived data ──
  // Main-text mention spans grouped by paragraph id, for inline affordances.
  const personSpansByPid = useMemo(() => {
    const m = new Map<number, PersonSpan[]>();
    for (const mt of mentions) {
      if (mt.source !== 'main' || !mt.person_id) continue;
      const arr = m.get(mt.pid) ?? [];
      arr.push({ start: mt.start, end: mt.end, personId: mt.person_id, confidence: mt.confidence });
      m.set(mt.pid, arr);
    }
    return m;
  }, [mentions]);

  // Surface → person ids index for binding 白话导读 关键人物 to the KB.
  // Only canonical/alias surfaces of reviewed/high people are bindable; an
  // ambiguous surface (mapping to >1 person) stays unbound (literal search).
  const nameIndex = useMemo(() => {
    const idx = new Map<string, Set<string>>();
    for (const p of people.values()) {
      if (p.confidence !== 'reviewed' && p.confidence !== 'high') continue;
      for (const n of p.names) {
        const set = idx.get(n.text) ?? new Set<string>();
        set.add(p.id);
        idx.set(n.text, set);
      }
    }
    return idx;
  }, [people]);

  // People that actually appear in the current 卷 — guide binding is gated to
  // these so a name only opens a card when there's at least one safe in-卷 hit.
  const peopleInJuan = useMemo(() => {
    const s = new Set<string>();
    for (const mt of mentions) if (mt.person_id) s.add(mt.person_id);
    return s;
  }, [mentions]);

  // Per-卷 surface → person ids, harvested from THIS 卷's mentions — i.e. the
  // exact collision-free table the body text underlines from. A recurring name
  // like 李德裕 is globally ambiguous (split into several person instances across
  // 卷 windows) so `nameIndex` can't bind it, yet within a single 卷 it resolves
  // to one instance. This lets a 白话导读 name bind to the same person the
  // paragraph already underlines, instead of falling back to white-dot search.
  const juanSurfaceIndex = useMemo(() => {
    const idx = new Map<string, Set<string>>();
    for (const mt of mentions) {
      if (!mt.person_id || !mt.surface) continue;
      const set = idx.get(mt.surface) ?? new Set<string>();
      set.add(mt.person_id);
      idx.set(mt.surface, set);
    }
    return idx;
  }, [mentions]);


  const resolveGuidePerson = (ref: GuidePersonRef): string | null => {
    if (ref.person_id && people.has(ref.person_id) && peopleInJuan.has(ref.person_id)) {
      return ref.person_id;
    }
    const ids = nameIndex.get(ref.name);
    if (ids && ids.size === 1) {
      const id = [...ids][0];
      if (peopleInJuan.has(id)) return id;
    }
    // Fall back to the per-卷 mention table (the same source the paragraph
    // underlines from). A name that is globally ambiguous but appears for
    // exactly one person in THIS 卷 binds to that in-卷 instance.
    const local = juanSurfaceIndex.get(ref.name);
    if (local && local.size === 1) {
      const id = [...local][0];
      if (peopleInJuan.has(id)) return id;
    }
    return null;
  };

  // Resolve a selected/typed exact string to a unique in-卷 person id, mirroring
  // resolveGuidePerson's binding rules. Powers the selection popover's optional
  // 看人物身份 affordance and the header 人物 availability. Read via ref from the
  // selection effect (which is created before nameIndex et al. exist).
  exactPersonRef.current = (txt: string): string | null => {
    const q = txt.trim();
    if (!q) return null;
    const ids = nameIndex.get(q);
    if (ids && ids.size === 1) {
      const id = [...ids][0];
      if (peopleInJuan.has(id)) return id;
    }
    const local = juanSurfaceIndex.get(q);
    if (local && local.size === 1) {
      const id = [...local][0];
      if (peopleInJuan.has(id)) return id;
    }
    return null;
  };

  // Open a person card (inline underline or guide bound pill). The dock shows a
  // person card IFF subject.kind === 'person'.
  const openPerson = (
    personId: string,
    atPid: number,
    source: 'main' | 'guide' = 'main',
    clickedLabel?: string,
  ) => {
    const person = people.get(personId);
    if (!person) return;
    // Pre-step (critical, finding P1 #1): clear any stray OS selection BEFORE
    // opening, so a leftover selection can never swallow the tap.
    try { window.getSelection()?.removeAllRanges(); } catch { /* noop */ }
    setSelectionPopover(null);
    setSelectionMatch('');
    setSpoilerSummary(false);
    setHighlightPid(null);
    setSearchCompose(false);
    applySubject({
      kind: 'person',
      personId,
      atPid,
      origin: source === 'guide' ? 'guide' : 'inline',
      from: subjectRef.current,
      clickedLabel,
    });
    if (isMobileWidth()) { setShowSidebar(false); }
  };

  // Promote the selection popover's 看人物身份 affordance: open the person with
  // origin 'lookup-promoted' and from=null (per spec §2.1).
  const promoteSelectionToPerson = (personId: string, atPid: number, clickedLabel?: string) => {
    const person = people.get(personId);
    if (!person) return;
    try { window.getSelection()?.removeAllRanges(); } catch { /* noop */ }
    setSelectionPopover(null);
    setSelectionMatch('');
    setSpoilerSummary(false);
    setHighlightPid(null);
    setSearchCompose(false);
    applySubject({ kind: 'person', personId, atPid, origin: 'lookup-promoted', from: null, clickedLabel });
    if (isMobileWidth()) { setShowSidebar(false); }
  };

  // The currently open person object (card shown IFF subject.kind==='person').
  const activePersonObj = subject.kind === 'person' ? people.get(subject.personId) ?? null : null;

  // When a person is open, build the NER-accurate occurrence inputs: the set of
  // their verified appearance paragraphs ("j:p") plus every name surface to
  // highlight. null when no person is open → LookupPanel falls back to literal
  // substring search on the typed query.
  const occurrenceNames = useMemo(
    () => (activePersonObj ? activePersonObj.names.map(n => n.text) : null),
    [activePersonObj],
  );
  const occurrencePids = useMemo(() => {
    if (!activePersonObj) return null;
    const rows = appearances.get(activePersonObj.id);
    if (!rows || rows.length === 0) return null;
    return new Set(rows.map(r => r.juan + ':' + r.pid));
  }, [activePersonObj, appearances]);

  // Lazy-load the cross-卷 appearance index (≈3MB at full 294-卷 scale) only
  // once a person is actually open — keeps first paint light. loadAppearances
  // is process-cached, so this fetch happens at most once.
  useEffect(() => {
    if (activePersonObj && appearances.size === 0) {
      loadAppearances().then(setAppearances);
    }
  }, [activePersonObj, appearances.size]);

  // On variant switch, drop the loaded appearance index so the lazy effect above
  // refetches from the new directory (the corpus cache was already cleared).
  const firstVariantRun = useRef(true);
  useEffect(() => {
    if (firstVariantRun.current) { firstVariantRun.current = false; return; }
    setAppearances(new Map());
  }, [personVariant]);

  // Compact "current position" label for the dock's persistent year chip:
  // the latest year-anchor at or before the active paragraph.
  const currentYearLabel = useMemo(() => {
    if (!juan) return '';
    if (activeParagraphId == null) return juan.label;
    let y: number | null = null;
    for (const yr of juan.years) {
      if (yr.paragraph_id <= activeParagraphId) y = yr.ce_year;
      else break;
    }
    const yl = y === null ? '' : (y < 0 ? `前${-y}年` : `${y}年`);
    return yl ? `${juan.label} · ${yl}` : juan.label;
  }, [juan, activeParagraphId]);

  // Paragraph id of the current year anchor (latest year at/before the active
  // paragraph). Tapping the breadcrumb year label in 纪年 mode jumps here.
  const currentYearPid = useMemo(() => {
    if (!juan || juan.years.length === 0) return null;
    let pid: number | null = null;
    for (const yr of juan.years) {
      if (activeParagraphId == null) { pid = yr.paragraph_id; break; }
      if (yr.paragraph_id <= activeParagraphId) pid = yr.paragraph_id;
      else break;
    }
    return pid ?? juan.years[0].paragraph_id;
  }, [juan, activeParagraphId]);

  // The breadcrumb: in 纪年 mode it jumps to the current year's paragraph; in
  // lookup/person mode the ‹ chevron pops the subject one level (dockBack).
  const onCrumbClick = () => {
    if (currentYearPid != null) {
      setSelectedYearPid(currentYearPid);
      jumpToParagraph(currentYearPid);
    }
  };

  // Global Esc handler: first Esc dismisses a live selection popover; otherwise
  // it pops the subject via the single dockBack ladder.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      if (selectionPopoverRef.current) {
        try { window.getSelection()?.removeAllRanges(); } catch { /* noop */ }
        setSelectionPopover(null);
        setSelectionMatch('');
        return;
      }
      dockBackRef.current();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // The header subject segmented control / legend availability.
  const hasPersonCtx = subject.kind === 'person' || !!lastPerson;
  const legendText =
    subject.kind === 'lookup' ? `出处检索：「${subject.query}」`
    : subject.kind === 'person' ? `人物身份：${activePersonObj?.canonical_name ?? ''}`
    : '本卷纪年';
  // Mobile sheet detent class (inert on desktop, where the dock is column 3).
  const detentClass = ` dock-${sheetDetent}`;

  // Which body view the dock renders (person > lookup > years).
  const showPersonView = subject.kind === 'person';
  const showLookupView = !showPersonView && (subject.kind === 'lookup' || searchCompose);
  const showYearsView = !showPersonView && !showLookupView;
  // When in a lookup whose query is a unique in-卷 person name, offer an
  // explicit promote ("看人物身份 ›") — never an automatic morph (AC3).
  const lookupPersonId = subject.kind === 'lookup' ? exactPersonRef.current(subject.query) : null;
  const lookupPerson = lookupPersonId ? people.get(lookupPersonId) ?? null : null;

  // Promote the selection popover's 搜全部出处: write a lookup subject.
  const promoteSelectionToLookup = () => {
    const txt = selectionPopover?.text;
    if (!txt) return;
    setSearchCompose(true);
    applySubject({ kind: 'lookup', query: txt, origin: 'selection-promoted' });
    setSelectionPopover(null);
    setSelectionMatch('');
    try { window.getSelection()?.removeAllRanges(); } catch { /* noop */ }
    if (isMobileWidth()) { setShowSidebar(false); }
  };

  // Promote a lookup → person (from = the lookup, so Back returns to it).
  const promoteLookupToPerson = () => {
    if (subject.kind !== 'lookup' || !lookupPersonId) return;
    setSpoilerSummary(false);
    setHighlightPid(null);
    setSearchCompose(false);
    applySubject({
      kind: 'person',
      personId: lookupPersonId,
      atPid: activeParagraphId ?? currentYearPid ?? 0,
      origin: 'lookup-promoted',
      from: subject,
      clickedLabel: subject.query,
    });
  };

  // Header seg: open / re-select the lookup context.
  const selectLookupSeg = () => {
    const q = subject.kind === 'lookup' ? subject.query : lastLookup;
    setSearchCompose(true);
    if (q.trim()) applySubject({ kind: 'lookup', query: q, origin: 'typed' });
    else setSheetDetent('half');
    if (isMobileWidth()) { setShowSidebar(false); }
  };

  // Header seg: re-select the last person context.
  const selectPersonSeg = () => {
    if (subject.kind === 'person' || !lastPerson) return;
    setSpoilerSummary(false);
    setHighlightPid(null);
    setSearchCompose(false);
    applySubject({
      kind: 'person',
      personId: lastPerson.personId,
      atPid: lastPerson.atPid,
      origin: lastPerson.origin,
      from: subject.kind === 'years' ? null : subject,
      clickedLabel: lastPerson.clickedLabel,
    });
    if (isMobileWidth()) { setShowSidebar(false); }
  };

  // Mobile drawer drag handling (grabber). The dock is a right slide-in drawer,
  // so a swipe RIGHT dismisses it (dockBack ladder). A small horizontal handle
  // on the drawer's left edge is the affordance; tap is a no-op.
  const sheetDragRef = useRef<number | null>(null);
  const onSheetPointerDown = (e: React.PointerEvent) => { sheetDragRef.current = e.clientX; };
  const onSheetPointerUp = (e: React.PointerEvent) => {
    const start = sheetDragRef.current;
    sheetDragRef.current = null;
    if (start == null) return;
    const dx = e.clientX - start;
    const TH = 40;
    if (dx > TH) dockBack();
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
    <div className={`layout${showSidebar ? '' : ' sidebar-collapsed'}${detentClass}${sheetDetent !== 'peek' ? ' dock-open' : ''}`}>
      <Sidebar
        manifest={manifest}
        currentJuan={juanNo}
        readJuans={readJuans}
        onSelect={n => {
          setHighlightPid(null);
          setJuanNo(n);
          if (isMobileWidth()) {
            setShowSidebar(false);
          }
        }}
      />
      <main className="reader-pane" style={{ ['--reader-font-scale' as any]: fontScale }}>
        <header className="reader-header">
          <button
            type="button"
            className="sidebar-toggle"
            onClick={() => setShowSidebar(s => {
              const next = !s;
              if (next && isMobileWidth()) { setSheetDetent('peek'); }
              return next;
            })}
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
          <button
            type="button"
            ref={settingsBtnRef}
            className="settings-btn"
            onClick={toggleSettings}
            aria-haspopup="dialog"
            title="阅读设置"
            aria-label="阅读设置"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
                 aria-hidden="true">
              <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z" />
            </svg>
            <span className="settings-btn-label">设置</span>
          </button>
          <div
            ref={settingsMenuRef}
            className="settings-menu"
            {...({ popover: 'auto' } as any)}
            role="dialog"
            aria-label="阅读设置"
          >
            <div className="settings-row" role="group" aria-label="白话导读">
              <span className="settings-label">白话导读</span>
              <div className="seg">
                <button
                  type="button"
                  className={'seg-btn' + (guideMode === 'off' ? ' is-on' : '')}
                  aria-pressed={guideMode === 'off'}
                  onClick={() => setGuideMode('off')}
                >关</button>
                <button
                  type="button"
                  className={'seg-btn' + (guideMode === 'brief' ? ' is-on' : '')}
                  aria-pressed={guideMode === 'brief'}
                  onClick={() => setGuideMode('brief')}
                >简</button>
                <button
                  type="button"
                  className={'seg-btn' + (guideMode === 'full' ? ' is-on' : '')}
                  aria-pressed={guideMode === 'full'}
                  onClick={() => setGuideMode('full')}
                >全</button>
              </div>
            </div>
            <div className="settings-row" role="group" aria-label="胡三省音注">
              <span className="settings-label">胡三省音注</span>
              <div className="seg">
                <button
                  type="button"
                  className={'seg-btn' + (!showHu ? ' is-on' : '')}
                  aria-pressed={!showHu}
                  onClick={() => setShowHu(false)}
                >隐藏</button>
                <button
                  type="button"
                  className={'seg-btn' + (showHu ? ' is-on' : '')}
                  aria-pressed={showHu}
                  onClick={() => setShowHu(true)}
                >显示</button>
              </div>
            </div>
            <div className="settings-row" role="group" aria-label="正文字号">
              <span className="settings-label">正文字号</span>
              <div className="seg">
                <button
                  type="button"
                  className="seg-btn"
                  onClick={() => bumpFont(-0.1)}
                  disabled={fontScale <= 0.8 + 1e-6}
                  aria-label="缩小字号"
                >A−</button>
                <button
                  type="button"
                  className="seg-btn"
                  onClick={() => setFontScale(1)}
                  disabled={Math.abs(fontScale - 1) < 1e-6}
                  aria-label={`重置字号（当前 ${Math.round(fontScale * 100)}%）`}
                >{Math.round(fontScale * 100)}%</button>
                <button
                  type="button"
                  className="seg-btn"
                  onClick={() => bumpFont(0.1)}
                  disabled={fontScale >= 1.8 - 1e-6}
                  aria-label="放大字号"
                >A+</button>
              </div>
            </div>
            <div className="settings-row" role="group" aria-label="人名标注">
              <span className="settings-label">人名标注</span>
              <div className="seg">
                <button
                  type="button"
                  className={'seg-btn' + (personVariant === 'v1' ? ' is-on' : '')}
                  aria-pressed={personVariant === 'v1'}
                  onClick={() => changePersonVariant('v1')}
                >旧</button>
                <button
                  type="button"
                  className={'seg-btn' + (personVariant === 'v2' ? ' is-on' : '')}
                  aria-pressed={personVariant === 'v2'}
                  onClick={() => changePersonVariant('v2')}
                >新</button>
              </div>
            </div>
          </div>
          <button
            type="button"
            className={'lookup-toggle' + (searchCompose || subject.kind === 'lookup' ? ' is-on' : '')}
            onClick={() => {
              const open = searchCompose || subject.kind === 'lookup';
              if (open) {
                goToYears();
              } else {
                if (subject.kind === 'person') goToYears();
                setSearchCompose(true);
                setSheetDetent('half');
                if (isMobileWidth()) { setShowSidebar(false); }
              }
            }}
            title={searchCompose || subject.kind === 'lookup' ? '隐藏检索' : '出处检索'}
            aria-label={searchCompose || subject.kind === 'lookup' ? '隐藏检索' : '出处检索'}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
                 aria-hidden="true">
              <circle cx="11" cy="11" r="7" />
              <line x1="21" y1="21" x2="16.5" y2="16.5" />
            </svg>
          </button>
        </header>
        <div className={'reader-scroller' + (selectionMatch ? ' sel-live' : '')} ref={readerPaneRef}>
          {juan
            ? <Reader
                juan={juan}
                showHu={showHu}
                highlightQuery={committedQuery}
                highlightPid={highlightPid}
                guideMode={guideMode}
                guideByAnchorPid={guideByAnchorPid}
                onPersonSearch={searchFor}
                personSpansByPid={personSpansByPid}
                onPersonClick={openPerson}
                resolveGuidePerson={resolveGuidePerson}
                activePersonId={subject.kind === 'person' ? subject.personId : null}
              />
            : <div className="loading">载入卷 {juanNo} 中……</div>}
        </div>
        {jumpReturnPid !== null && (
          <button
            type="button"
            className="jump-return-pill"
            onClick={() => { const pid = jumpReturnPid; setJumpReturnPid(null); setHighlightPid(pid); jumpToParagraph(pid); }}
          >
            ↩ 返回刚才阅读处
          </button>
        )}
      </main>
      <aside className="person-pane context-dock">
        <button
          type="button"
          className="dock-grabber"
          onPointerDown={onSheetPointerDown}
          onPointerUp={onSheetPointerUp}
          title="向右滑动关闭"
          aria-label="向右滑动关闭"
        >
          <span className="dock-grabber-bar" aria-hidden="true" />
        </button>
        <div className="dock-head">
          <button
            type="button"
            className="dock-crumb"
            onClick={onCrumbClick}
            title="跳至本卷当前纪年"
            aria-label="跳至本卷当前纪年"
          >
            <span className="dock-crumb-year">{currentYearLabel || '本卷纪年'}</span>
          </button>
          <div className="dock-subject">
            <div className="seg" role="group" aria-label="上下文">
              <button
                type="button"
                className={'seg-btn' + (subject.kind === 'years' ? ' is-on' : '')}
                aria-pressed={subject.kind === 'years'}
                onClick={goToYears}
              >纪年</button>
              <button
                type="button"
                className={'seg-btn' + (subject.kind === 'lookup' ? ' is-on' : '')}
                aria-pressed={subject.kind === 'lookup'}
                onClick={selectLookupSeg}
              >检索</button>
              <button
                type="button"
                className={'seg-btn' + (subject.kind === 'person' ? ' is-on' : '')}
                aria-pressed={subject.kind === 'person'}
                disabled={!hasPersonCtx}
                onClick={selectPersonSeg}
              >人物</button>
            </div>
            <span className="dock-legend" title={legendText}>{legendText}</span>
          </div>
        </div>

        <div className="dock-body">
          {showPersonView && activePersonObj && (
            <div className="dock-view" key={'person-' + activePersonObj.id}>
              <PersonCard
                key={activePersonObj.id}
                person={activePersonObj}
                spoiler={spoilerSummary}
                origin={subject.kind === 'person' ? subject.origin : undefined}
                clickedLabel={subject.kind === 'person' ? subject.clickedLabel : undefined}
                autoFocus
                onToggleSpoiler={() => setSpoilerSummary(s => !s)}
                onClose={dockBack}
              />
              <div className="dock-search-toolbar">
                <label className="toggle small">
                  <input
                    type="checkbox"
                    checked={filterByJuan}
                    onChange={e => setFilterByJuan(e.target.checked)}
                  />
                  <span>仅本卷之前</span>
                </label>
              </div>
              <div className="lookup-body">
                <LookupPanel
                  query={lookupQuery}
                  maxJuan={filterByJuan ? juanNo : null}
                  currentJuan={juanNo}
                  highlightPid={highlightPid}
                  onJump={jumpToHit}
                  occurrenceNames={occurrencePids ? occurrenceNames : null}
                  occurrencePids={occurrencePids}
                  occurrenceKey={occurrencePids ? activePersonObj.id : null}
                />
              </div>
            </div>
          )}

          {showLookupView && (
            <div className="dock-view" key="lookup">
              <div className="dock-search">
                <input
                  type="text"
                  className="lookup-input"
                  value={lookupQuery}
                  placeholder="检索出处…"
                  aria-label="检索出处"
                  autoFocus={searchCompose && subject.kind !== 'lookup'}
                  onChange={e => {
                    const v = e.target.value;
                    setHighlightPid(null);
                    if (v.trim() === '') {
                      // Empty query is unrepresentable as a lookup subject —
                      // fall back to 纪年, but keep the compose input open.
                      setSubject({ kind: 'years' });
                      setLookupQuery('');
                      setCommittedQuery('');
                    } else {
                      applySubject({ kind: 'lookup', query: v, origin: 'typed' });
                    }
                  }}
                />
                {lookupQuery && (
                  <button
                    type="button"
                    className="lookup-clear"
                    onClick={goToYears}
                    title="清除"
                  >×</button>
                )}
              </div>
              <div className="dock-search-toolbar">
                <label className="toggle small">
                  <input
                    type="checkbox"
                    checked={filterByJuan}
                    onChange={e => setFilterByJuan(e.target.checked)}
                  />
                  <span>仅本卷之前</span>
                </label>
              </div>
              <div className="lookup-body">
                <LookupPanel
                  query={lookupQuery}
                  maxJuan={filterByJuan ? juanNo : null}
                  currentJuan={juanNo}
                  highlightPid={highlightPid}
                  onJump={jumpToHit}
                  occurrenceNames={null}
                  occurrencePids={null}
                  occurrenceKey={null}
                />
                {lookupPerson && subject.kind === 'lookup' && (
                  <div className="lookup-promote">
                    <span className="lookup-promote-text">
                      「{subject.query}」也是 {lookupPerson.canonical_name} 之名？
                    </span>
                    <button
                      type="button"
                      className="lookup-promote-btn"
                      onClick={promoteLookupToPerson}
                    >看人物身份 ›</button>
                  </div>
                )}
              </div>
            </div>
          )}

          {showYearsView && juan && (
            <YearToc
              years={juan.years}
              activeParagraphId={activeParagraphId}
              selectedYearPid={selectedYearPid}
              onJump={pid => {
                setHighlightPid(null);
                // Lock the YearToc highlight on the clicked year. Decoupled
                // from activeParagraphId so end-of-pane / no-body edge cases
                // and any trailing scroll events from the animation can't
                // shift it. scrollParagraphIntoView's finish() updates
                // activeParagraphId itself once the scroll settles.
                setSelectedYearPid(pid);
                jumpToParagraph(pid);
              }}
            />
          )}
        </div>
      </aside>
      <div
        className="drawer-backdrop"
        onClick={() => {
          if (showSidebar) { setShowSidebar(false); return; }
          // Mobile sheet backdrop: one tap returns to reading (纪年).
          goToYears();
        }}
        aria-hidden="true"
      />
      {selectionPopover && createPortal(
        <div
          className={'selection-action-bar' + (selectionPopover.rect ? ' is-anchored' : '')}
          style={selectionPopover.rect
            ? { left: `${selectionPopover.rect.cx}px`, top: `${selectionPopover.rect.top + 10}px`, bottom: 'auto' }
            : undefined}
          role="toolbar"
          aria-label="选词检索"
        >
          <span className="selection-action-text" title={selectionPopover.text}>
            「{selectionPopover.text.length > 8 ? selectionPopover.text.slice(0, 7) + '…' : selectionPopover.text}」
          </span>
          <button
            type="button"
            className="selection-action-btn"
            onClick={promoteSelectionToLookup}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                 aria-hidden="true">
              <circle cx="11" cy="11" r="7" />
              <line x1="21" y1="21" x2="16.5" y2="16.5" />
            </svg>
            搜全部出处
          </button>
          {selectionPopover.personId && (
            <button
              type="button"
              className="selection-action-btn selection-action-person"
              onClick={() => promoteSelectionToPerson(
                selectionPopover.personId!,
                activeParagraphId ?? currentYearPid ?? 0,
                selectionPopover.text,
              )}
            >
              看人物身份 ›
            </button>
          )}
        </div>,
        document.body,
      )}
    </div>
  );
}


