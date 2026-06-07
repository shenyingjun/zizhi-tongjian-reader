import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import type { LookupHit, Paragraph } from './corpus';
import { loadJuan, loadLookup, searchCorpus } from './corpus';
import { splitParagraph, findMatches, highlight, highlightWithRanges } from './highlight';

interface Props {
  query: string;
  maxJuan: number | null;
  currentJuan: number;
  highlightPid: number | null;
  onJump: (juanNo: number, paragraphId: number) => void;
}

function formatCE(y: number | null): string {
  if (y === null) return '?';
  return y < 0 ? `前${-y}年` : `${y}年`;
}

// Each LookupHit may bundle multiple nearby matches into one snippet,
// so "处" (occurrences) is the sum of match ranges, not hits.length.
function countMatches(hits: LookupHit[]): number {
  let n = 0;
  for (const h of hits) n += h.matches.length;
  return n;
}

// Render a snippet that may contain multiple highlight ranges. The ranges
// are sorted and non-overlapping by construction in searchCorpus.
function renderSnippet(h: LookupHit): ReactNode[] {
  const out: ReactNode[] = [];
  let cur = 0;
  h.matches.forEach((m, i) => {
    if (m.start > cur) out.push(h.snippet.slice(cur, m.start));
    out.push(<mark key={i}>{h.snippet.slice(m.start, m.start + m.len)}</mark>);
    cur = m.start + m.len;
  });
  if (cur < h.snippet.length) out.push(h.snippet.slice(cur));
  return out;
}

interface YearGroup {
  y: number | null;
  hits: LookupHit[];
}
interface JuanGroup {
  j: number;
  hits: LookupHit[];
  years: YearGroup[];
  pids: number[]; // distinct paragraph ids in reading order
}

function groupHits(hits: LookupHit[]): JuanGroup[] {
  const byJuan = new Map<number, JuanGroup>();
  for (const h of hits) {
    let jg = byJuan.get(h.j);
    if (!jg) {
      jg = { j: h.j, hits: [], years: [], pids: [] };
      byJuan.set(h.j, jg);
    }
    jg.hits.push(h);
    const last = jg.years[jg.years.length - 1];
    if (last && last.y === h.y) last.hits.push(h);
    else jg.years.push({ y: h.y, hits: [h] });
    if (jg.pids[jg.pids.length - 1] !== h.p) jg.pids.push(h.p);
  }
  return Array.from(byJuan.values());
}

/**
 * Render a paragraph in its natural reading order — main text with 胡注
 * inline at their original positions. Used by the hover popover so users
 * see the same flow they'd encounter in the reader, including multiple
 * interleaved notes. Matches of `q` are highlighted in both main and notes;
 * matches that span a note insertion point stay highlighted across the split.
 */
function FullParagraphInterleaved({ p, q }: { p: Paragraph; q: string }) {
  const segments = useMemo(() => splitParagraph(p), [p]);
  const mainMatches = useMemo(() => findMatches(p.main, q), [p.main, q]);
  return (
    <div className="lookup-full">
      {segments.map((seg, i) => {
        if (seg.kind === 'text') {
          return (
            <span key={i}>
              {highlightWithRanges(seg.text, seg.mainStart!, mainMatches)}
            </span>
          );
        }
        return (
          <span key={i} className="lookup-full-hu-inline">
            （{highlight(seg.text, q)}）
          </span>
        );
      })}
    </div>
  );
}

interface PopoverState {
  key: string;
  rect: DOMRect;
  hit: LookupHit;
  para: Paragraph | null;     // null while the juan is loading
  error: string | null;
}

/** Position the popover beside the anchor card. Prefers the left side (the
 *  lookup panel sits on the right of the layout) and clamps to the viewport. */
function popoverStyle(rect: DOMRect): React.CSSProperties {
  const W = 400;
  const GAP = 6;
  const vw = typeof window !== 'undefined' ? window.innerWidth : 1024;
  const vh = typeof window !== 'undefined' ? window.innerHeight : 768;
  let left = rect.left - GAP - W;
  if (left < 8) left = rect.right + GAP;
  if (left + W > vw - 8) left = Math.max(8, vw - 8 - W);
  const maxH = Math.floor(vh * 0.6);
  let top = rect.top;
  if (top + maxH > vh - 8) top = Math.max(8, vh - 8 - maxH);
  return { position: 'fixed', left, top, width: W, maxHeight: maxH };
}

export default function LookupPanel({ query, maxJuan, currentJuan, highlightPid, onJump }: Props) {
  const [hits, setHits] = useState<LookupHit[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [futureCount, setFutureCount] = useState(0);
  const activeHitRef = useRef<HTMLLIElement | null>(null);

  // Hover-only full-paragraph peek. Popover is interactive so users can move
  // the cursor into it to scroll long paragraphs without it disappearing.
  const [popover, setPopover] = useState<PopoverState | null>(null);
  const enterTimerRef = useRef<number | null>(null);
  const leaveTimerRef = useRef<number | null>(null);

  const hoverCapable = useMemo(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return false;
    try { return window.matchMedia('(hover: hover)').matches; } catch { return false; }
  }, []);

  const clearEnterTimer = () => {
    if (enterTimerRef.current !== null) {
      window.clearTimeout(enterTimerRef.current);
      enterTimerRef.current = null;
    }
  };
  const clearLeaveTimer = () => {
    if (leaveTimerRef.current !== null) {
      window.clearTimeout(leaveTimerRef.current);
      leaveTimerRef.current = null;
    }
  };
  useEffect(() => () => { clearEnterTimer(); clearLeaveTimer(); }, []);

  // Drop any open peek when the query changes — its content would be stale.
  useEffect(() => { setPopover(null); setSheet(null); }, [query]);

  // The popover is anchored to a snapshot rect; any layout shift moves the
  // anchor out from under it. Resize is always dismiss-worthy. For scroll,
  // ignore scrolls that originate inside the popover itself (the user is
  // scrolling its content) by checking the event target.
  useEffect(() => {
    if (!popover) return;
    const onResize = () => setPopover(null);
    const onScroll = (e: Event) => {
      const t = e.target as Node | null;
      const pop = document.querySelector('.lookup-popover');
      if (pop && t && pop.contains(t)) return;
      setPopover(null);
    };
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onResize);
    };
  }, [popover]);

  // Lazy-load the juan for the hovered hit, then attach its Paragraph to the
  // popover. corpus.loadJuan caches per juan, so re-hovers within the same
  // juan are instant.
  useEffect(() => {
    if (!popover || popover.para || popover.error) return;
    const myKey = popover.key;
    let cancelled = false;
    loadJuan(popover.hit.j)
      .then(j => {
        if (cancelled) return;
        const para = j.paragraphs.find(p => p.id === popover.hit.p) || null;
        setPopover(prev => (prev && prev.key === myKey ? { ...prev, para } : prev));
      })
      .catch(err => {
        if (cancelled) return;
        setPopover(prev => (prev && prev.key === myKey ? { ...prev, error: String(err) } : prev));
      });
    return () => { cancelled = true; };
  }, [popover]);

  const openPeek = (key: string, target: HTMLElement, hit: LookupHit) => {
    clearLeaveTimer();
    clearEnterTimer();
    enterTimerRef.current = window.setTimeout(() => {
      // If the popover is already showing this exact card (e.g. cursor moved
      // popover → card → popover), keep the existing state — re-creating it
      // would discard the already-loaded paragraph and flash "loading".
      setPopover(prev => (
        prev && prev.key === key
          ? prev
          : { key, rect: target.getBoundingClientRect(), hit, para: null, error: null }
      ));
    }, 350);
  };
  const schedulePeekClose = () => {
    clearEnterTimer();
    clearLeaveTimer();
    leaveTimerRef.current = window.setTimeout(() => setPopover(null), 180);
  };
  const keepPeekAlive = () => {
    // Cursor moved into the popover — cancel any pending close so the user
    // can scroll and read.
    clearLeaveTimer();
  };

  // Mobile / touch peek: a bottom sheet variant of the popover. Same content,
  // different shell (modal sheet instead of hover popover) because there's
  // no hover gesture and the floating popover doesn't suit narrow screens.
  // Tap a card → sheet opens; sheet has an explicit 跳转 button.
  interface SheetState { key: string; hit: LookupHit; para: Paragraph | null; error: string | null }
  const [sheet, setSheet] = useState<SheetState | null>(null);
  const openSheet = (key: string, hit: LookupHit) => {
    setSheet({ key, hit, para: null, error: null });
  };
  const closeSheet = () => setSheet(null);

  // Reuse the same lazy-load pattern as the popover.
  useEffect(() => {
    if (!sheet || sheet.para || sheet.error) return;
    const myKey = sheet.key;
    let cancelled = false;
    loadJuan(sheet.hit.j)
      .then(j => {
        if (cancelled) return;
        const para = j.paragraphs.find(p => p.id === sheet.hit.p) || null;
        setSheet(prev => (prev && prev.key === myKey ? { ...prev, para } : prev));
      })
      .catch(err => {
        if (cancelled) return;
        setSheet(prev => (prev && prev.key === myKey ? { ...prev, error: String(err) } : prev));
      });
    return () => { cancelled = true; };
  }, [sheet]);

  // While the sheet is open: Esc closes it, and lock background scroll so
  // the page underneath doesn't drift as the user scrolls the paragraph.
  useEffect(() => {
    if (!sheet) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeSheet(); };
    window.addEventListener('keydown', onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [sheet]);

  // Bring the highlighted card into view (instant scroll) whenever the
  // active paragraph changes. Compensates for the sticky group header.
  useEffect(() => {
    const el = activeHitRef.current;
    if (!el) return;
    let scroller: HTMLElement | null = el.parentElement;
    while (scroller && !scroller.classList.contains('lookup-body')) {
      scroller = scroller.parentElement;
    }
    if (!scroller) return;
    const stickyHeader = scroller.querySelector<HTMLElement>(
      '.lookup-juan-group.is-current .lookup-juan-header',
    );
    const headerH = stickyHeader ? stickyHeader.getBoundingClientRect().height : 0;
    const elRect = el.getBoundingClientRect();
    const scRect = scroller.getBoundingClientRect();
    if (elRect.top >= scRect.top + headerH && elRect.bottom <= scRect.bottom) return;
    const targetTop = scroller.scrollTop + (elRect.top - scRect.top) - headerH - 8;
    scroller.scrollTop = Math.max(0, targetTop);
  }, [highlightPid]);

  useEffect(() => {
    if (!query) {
      setHits(null);
      setFutureCount(0);
      return;
    }
    let cancelled = false;
    // Only show the loading skeleton on the very first fetch. Once we have
    // hits, re-filters (e.g. when currentJuan changes after a jump) should
    // be silent — the corpus is cached and the recompute is synchronous, so
    // flashing the loading state would just make the panel flicker.
    if (hits === null) setLoading(true);
    setError(null);
    loadLookup()
      .then(corpus => {
        if (cancelled) return;
        const filtered = searchCorpus(query, corpus, { maxJuan, limit: 500 });
        const all = maxJuan === null ? filtered : searchCorpus(query, corpus, { limit: 5000 });
        setHits(filtered);
        setFutureCount(all.length - filtered.length);
      })
      .catch(e => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
    // hits intentionally omitted from deps: it's only read to decide whether
    // to show the loading skeleton on first fetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, maxJuan, currentJuan]);

  if (!query) {
    return (
      <div className="lookup-empty muted small">
        <p>在正文中选中任意文字（人名、地名、官职等），此处将显示其在《通鉴》其他出处。</p>
        <p>结果默认按当前阅读年份过滤——只展示「此前」的出处，避免剧透。</p>
      </div>
    );
  }
  if (loading) return <div className="loading small">搜索 “{query}” 中……</div>;
  if (error) return <div className="error small">搜索失败：{error}</div>;
  if (!hits || hits.length === 0) {
    return (
      <div className="lookup-empty small">
        <p>未找到 “<b>{query}</b>” 此前的其他出处。</p>
        {futureCount > 0 && (
          <p className="muted">（此后共有<b>{futureCount}</b>处出现，已隐藏以避免剧透）</p>
        )}
      </div>
    );
  }
  const groups = groupHits(hits);
  const paraCount = groups.reduce((s, g) => s + g.pids.length, 0);
  const matchCount = countMatches(hits);
  return (
    <div className="lookup-results">
      <p className="lookup-summary small muted">
        “<b>{query}</b>” 共<b>{paraCount}</b>段
        {paraCount !== matchCount && <>（{matchCount}处匹配）</>}
        {futureCount > 0 && <>（此后另有<b>{futureCount}</b>处已隐藏）</>}
      </p>
      <div className="lookup-groups">
        {groups.map(jg => {
          const isCurrent = jg.j === currentJuan;
          const navPids = jg.pids;
          const navIndex = isCurrent && highlightPid !== null
            ? navPids.indexOf(highlightPid) : -1;
          const navJump = (delta: number) => {
            const total = navPids.length;
            if (total === 0) return;
            const next = navIndex < 0
              ? (delta > 0 ? 0 : total - 1)
              : (navIndex + delta + total) % total;
            onJump(jg.j, navPids[next]);
          };
          return (
          <section
            key={jg.j}
            className={`lookup-juan-group${isCurrent ? ' is-current' : ''}`}
          >
            <header className="lookup-juan-header">
              <span className="lookup-juan-label">卷{jg.j}</span>
              {isCurrent && navPids.length > 0 && (
                <span className="hit-nav" role="group" aria-label="本卷检索结果导航">
                  <button
                    type="button"
                    className="hit-nav-btn"
                    onClick={() => navJump(-1)}
                    title="上一处（本卷）"
                    aria-label="上一处"
                  >
                    <svg viewBox="0 0 12 12" aria-hidden="true">
                      <path d="M3 7.5 L6 4.5 L9 7.5" fill="none"
                        stroke="currentColor" strokeWidth="1.6"
                        strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                  <span className="hit-nav-count">
                    {navIndex < 0 ? '–' : navIndex + 1}<span className="hit-nav-sep">/</span>{navPids.length}
                  </span>
                  <button
                    type="button"
                    className="hit-nav-btn"
                    onClick={() => navJump(1)}
                    title="下一处（本卷）"
                    aria-label="下一处"
                  >
                    <svg viewBox="0 0 12 12" aria-hidden="true">
                      <path d="M3 4.5 L6 7.5 L9 4.5" fill="none"
                        stroke="currentColor" strokeWidth="1.6"
                        strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                </span>
              )}
              <span
                className="lookup-juan-count"
                title={countMatches(jg.hits) === jg.pids.length
                  ? undefined
                  : `共 ${countMatches(jg.hits)} 处匹配，分布在 ${jg.pids.length} 段`}
              >{jg.pids.length}段</span>
            </header>
            {jg.years.map((yg, yi) => (
              <div key={yi} className="lookup-year-group">
                <div className="lookup-year-header">{formatCE(yg.y)}</div>
                <ol className="lookup-list">
                  {(() => {
                    // Group consecutive same-paragraph hits so one paragraph
                    // = one clickable entry, with stacked snippets inside.
                    const paras: { pid: number; hits: LookupHit[] }[] = [];
                    for (const h of yg.hits) {
                      const last = paras[paras.length - 1];
                      if (last && last.pid === h.p) last.hits.push(h);
                      else paras.push({ pid: h.p, hits: [h] });
                    }
                    return paras.map((para, i) => {
                      const first = para.hits[0];
                      const matchCountPara = countMatches(para.hits);
                      const multi = matchCountPara > 1;
                      const isActive = isCurrent && highlightPid === para.pid;
                      const key = `${jg.j}:${para.pid}`;
                      const handleEnter = (e: React.MouseEvent<HTMLLIElement> | React.FocusEvent<HTMLLIElement>) => {
                        if (!hoverCapable) return;
                        openPeek(key, e.currentTarget, first);
                      };
                      const handleLeave = () => {
                        if (!hoverCapable) return;
                        schedulePeekClose();
                      };
                      // One <li> per paragraph. Each LookupHit inside may
                      // already bundle several nearby matches sharing one
                      // snippet (collapsed in searchCorpus to avoid showing
                      // overlapping context twice); render each as its own
                      // line and mark every match range inside it.
                      return (
                        <li
                          key={i}
                          ref={isActive ? activeHitRef : undefined}
                          className={`lookup-hit kind-${first.k}${isActive ? ' is-active-hit' : ''}`}
                          onMouseEnter={handleEnter}
                          onMouseLeave={handleLeave}
                          onFocus={handleEnter}
                          onBlur={handleLeave}
                        >
                          <button
                            type="button"
                            className="lookup-jump"
                            onClick={() => {
                              // On hover-capable devices the popover is the
                              // peek surface; tap goes straight to navigation.
                              // On touch, the same tap opens a bottom sheet
                              // so users can verify context before committing.
                              if (hoverCapable) onJump(jg.j, para.pid);
                              else openSheet(key, first);
                            }}
                            title={multi
                              ? `跳转：卷${jg.j} 段${para.pid}（${matchCountPara} 处匹配）`
                              : `跳转：卷${jg.j} 段${para.pid}`}
                          >
                            {para.hits.length > 1 ? (
                              <div className="lookup-snippets-multi">
                                {para.hits.map((h, k) => (
                                  <span key={k} className={`lookup-snippet${h.inHu ? ' in-hu' : ''}`}>
                                    {h.atStart ? '' : '…'}{renderSnippet(h)}{h.atEnd ? '' : '…'}
                                  </span>
                                ))}
                              </div>
                            ) : (
                              <span className={`lookup-snippet${first.inHu ? ' in-hu' : ''}`}>
                                {first.atStart ? '' : '…'}{renderSnippet(first)}{first.atEnd ? '' : '…'}
                              </span>
                            )}
                            {multi && (
                              <span className="lookup-multi-badge">{matchCountPara}处</span>
                            )}
                          </button>
                        </li>
                      );
                    });
                  })()}
                </ol>
              </div>
            ))}
          </section>
          );
        })}
      </div>
      {popover && createPortal(
        <div
          className="lookup-popover"
          role="tooltip"
          style={popoverStyle(popover.rect)}
          onMouseEnter={keepPeekAlive}
          onMouseLeave={schedulePeekClose}
        >
          <div className="lookup-popover-head">
            卷{popover.hit.j} · 段{popover.hit.p}
            {popover.hit.y !== null && <> · {formatCE(popover.hit.y)}</>}
          </div>
          <div className="lookup-popover-body">
            {popover.error ? (
              <div className="error small">加载失败：{popover.error}</div>
            ) : popover.para ? (
              <FullParagraphInterleaved p={popover.para} q={query} />
            ) : (
              <div className="muted small">加载中……</div>
            )}
          </div>
        </div>,
        document.body,
      )}
      {sheet && createPortal(
        <>
          <div className="lookup-sheet-backdrop" onClick={closeSheet} />
          <div className="lookup-sheet" role="dialog" aria-modal="true" aria-label="搜索结果全文预览">
            <header className="lookup-sheet-head">
              <span className="lookup-sheet-title">
                卷{sheet.hit.j} · 段{sheet.hit.p}
                {sheet.hit.y !== null && <> · {formatCE(sheet.hit.y)}</>}
              </span>
              <button
                type="button"
                className="lookup-sheet-close"
                onClick={closeSheet}
                aria-label="关闭"
              >×</button>
            </header>
            <div className="lookup-sheet-body">
              {sheet.error ? (
                <div className="error small">加载失败：{sheet.error}</div>
              ) : sheet.para ? (
                <FullParagraphInterleaved p={sheet.para} q={query} />
              ) : (
                <div className="muted small">加载中……</div>
              )}
            </div>
            <footer className="lookup-sheet-foot">
              <button
                type="button"
                className="lookup-sheet-jump"
                onClick={() => {
                  const { j, p } = sheet.hit;
                  closeSheet();
                  onJump(j, p);
                }}
              >跳转到此段 →</button>
            </footer>
          </div>
        </>,
        document.body,
      )}
    </div>
  );
}
