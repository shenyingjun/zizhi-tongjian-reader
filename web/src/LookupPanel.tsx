import { useEffect, useState } from 'react';
import type { LookupHit } from './corpus';
import { loadLookup, searchCorpus } from './corpus';

interface Props {
  query: string;
  maxYear: number | null;
  onJump: (juanNo: number, paragraphId: number) => void;
}

function formatCE(y: number | null): string {
  if (y === null) return '?';
  return y < 0 ? `前${-y}` : String(y);
}

export default function LookupPanel({ query, maxYear, onJump }: Props) {
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
    setLoading(true);
    setError(null);
    loadLookup()
      .then(corpus => {
        if (cancelled) return;
        const filtered = searchCorpus(query, corpus, { maxYear, limit: 500 });
        const all = maxYear === null ? filtered : searchCorpus(query, corpus, { limit: 5000 });
        setHits(filtered);
        setFutureCount(all.length - filtered.length);
      })
      .catch(e => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [query, maxYear]);

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
          <p className="muted">（此后共有 {futureCount} 处出现，已隐藏以避免剧透）</p>
        )}
      </div>
    );
  }
  return (
    <div className="lookup-results">
      <p className="lookup-summary small muted">
        “<b>{query}</b>” 共找到 <b>{hits.length}</b> 处
        {futureCount > 0 && ` （此后另有 ${futureCount} 处已隐藏）`}
      </p>
      <ol className="lookup-list">
        {hits.map((h, i) => (
          <li key={i} className={`lookup-hit kind-${h.k}`}>
            <button
              type="button"
              className="lookup-jump"
              onClick={() => onJump(h.j, h.p)}
              title={`跳转：卷${h.j} 段${h.p}`}
            >
              <span className="lookup-meta">
                卷{h.j} · {formatCE(h.y)}
              </span>
              <span className="lookup-snippet">
                …{h.snippet.slice(0, h.matchStart)}
                <mark>{h.snippet.slice(h.matchStart, h.matchStart + h.matchLen)}</mark>
                {h.snippet.slice(h.matchStart + h.matchLen)}…
              </span>
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}
