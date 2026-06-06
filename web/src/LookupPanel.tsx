import { useEffect, useState } from 'react';
import type { LookupHit } from './corpus';
import { loadLookup, searchCorpus } from './corpus';

interface Props {
  query: string;
  maxJuan: number | null;
  currentJuan: number;
  highlightPid: number | null;
  onJump: (juanNo: number, paragraphId: number) => void;
}

function formatCE(y: number | null): string {
  if (y === null) return '?';
  return y < 0 ? `前${-y}` : String(y);
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

export default function LookupPanel({ query, maxJuan, currentJuan, highlightPid, onJump }: Props) {
  const [hits, setHits] = useState<LookupHit[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [futureCount, setFutureCount] = useState(0);

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
  return (
    <div className="lookup-results">
      <p className="lookup-summary small muted">
        “<b>{query}</b>” 共<b>{paraCount}</b>段
        {paraCount !== hits.length && <>（{hits.length}处匹配）</>}
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
                    onClick={() => navJump(-1)}
                    title="上一处（本卷）"
                    aria-label="上一处"
                  >↑</button>
                  <span className="hit-nav-count">
                    {navIndex < 0 ? '–' : navIndex + 1}/{navPids.length}
                  </span>
                  <button
                    type="button"
                    onClick={() => navJump(1)}
                    title="下一处（本卷）"
                    aria-label="下一处"
                  >↓</button>
                </span>
              )}
              <span
                className="lookup-juan-count"
                title={jg.hits.length === jg.pids.length
                  ? undefined
                  : `共 ${jg.hits.length} 处匹配，分布在 ${jg.pids.length} 段`}
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
                      const multi = para.hits.length > 1;
                      return (
                        <li key={i} className={`lookup-hit kind-${first.k}`}>
                          <button
                            type="button"
                            className="lookup-jump"
                            onClick={() => onJump(jg.j, para.pid)}
                            title={multi
                              ? `跳转：卷${jg.j} 段${para.pid}（${para.hits.length} 处匹配）`
                              : `跳转：卷${jg.j} 段${para.pid}`}
                          >
                            {multi ? (
                              <div className="lookup-snippets-multi">
                                {para.hits.map((h, k) => (
                                  <span key={k} className={`lookup-snippet${h.inHu ? ' in-hu' : ''}`}>
                                    …{h.snippet.slice(0, h.matchStart)}
                                    <mark>{h.snippet.slice(h.matchStart, h.matchStart + h.matchLen)}</mark>
                                    {h.snippet.slice(h.matchStart + h.matchLen)}…
                                  </span>
                                ))}
                              </div>
                            ) : (
                              <span className={`lookup-snippet${first.inHu ? ' in-hu' : ''}`}>
                                …{first.snippet.slice(0, first.matchStart)}
                                <mark>{first.snippet.slice(first.matchStart, first.matchStart + first.matchLen)}</mark>
                                {first.snippet.slice(first.matchStart + first.matchLen)}…
                              </span>
                            )}
                            {multi && (
                              <span className="lookup-multi-badge">{para.hits.length}处</span>
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
    </div>
  );
}
