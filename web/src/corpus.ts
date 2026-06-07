export interface JuanMeta {
  juan_no: number;
  label: string;
  title: string;
  dynasty: string;
  year_range: string;
  paragraph_count: number;
  ce_start: number | null;
  ce_end: number | null;
}

export interface DynastyGroup {
  dynasty: string;
  juans: JuanMeta[];
}

export interface Manifest {
  juans: JuanMeta[];
  grouped: DynastyGroup[];
}

export interface HuNote {
  after: number;
  text: string;
}

export type ParagraphType = 'emperor' | 'year' | 'season' | 'event' | 'commentary';

export interface Paragraph {
  id: number;
  kind: 'text';
  type?: ParagraphType;
  ce_year?: number | null;
  main: string;
  notes: HuNote[];
}

export interface YearEntry {
  ce_year: number | null;
  ganzhi_idx: number | null;
  label: string;
  paragraph_id: number;
}

export interface Juan {
  juan_no: number;
  label: string;
  title: string;
  dynasty: string;
  year_range: string;
  years: YearEntry[];
  paragraphs: Paragraph[];
}

const BASE = import.meta.env.BASE_URL || './';

export async function loadManifest(): Promise<Manifest> {
  const r = await fetch(`${BASE}text/manifest.json`);
  if (!r.ok) throw new Error(`manifest ${r.status}`);
  return r.json();
}

const _juanCache = new Map<number, Promise<Juan>>();

export function loadJuan(no: number): Promise<Juan> {
  let cached = _juanCache.get(no);
  if (!cached) {
    const padded = String(no).padStart(3, '0');
    cached = fetch(`${BASE}text/juan_${padded}.json`).then(r => {
      if (!r.ok) throw new Error(`juan ${no} ${r.status}`);
      return r.json();
    });
    _juanCache.set(no, cached);
  }
  return cached;
}

// Flat lookup corpus for selection-based search.
// One entry per paragraph; `t` contains main text + concatenated 胡注 text.
export interface LookupEntry {
  j: number;       // juan number
  p: number;       // paragraph id within juan
  y: number | null; // CE year (negative for BCE)
  k: string;       // kind/type (event, commentary, year, emperor, ...)
  t: string;       // searchable text (main + " " + 胡三省音注 if present)
  m?: number;      // index within `t` where 胡三省音注 begins (omitted if no notes)
}

let _lookupCache: Promise<LookupEntry[]> | null = null;

export function loadLookup(): Promise<LookupEntry[]> {
  if (!_lookupCache) {
    _lookupCache = fetch(`${BASE}text/lookup.json`)
      .then(r => { if (!r.ok) throw new Error(`lookup ${r.status}`); return r.json(); });
  }
  return _lookupCache;
}

export interface LookupHit {
  j: number;
  p: number;
  y: number | null;
  k: string;
  snippet: string;   // text with the matched query in context
  matchStart: number; // index of match within snippet
  matchLen: number;
  inHu: boolean;     // true when the match falls inside 胡三省音注
}

const SNIPPET_PAD = 30;

export function searchCorpus(
  query: string,
  corpus: LookupEntry[],
  opts: { maxJuan?: number | null; limit?: number } = {},
): LookupHit[] {
  const q = query.trim();
  if (!q) return [];
  const { maxJuan = null, limit = 200 } = opts;
  const hits: LookupHit[] = [];
  for (const entry of corpus) {
    // Spoiler filter: hide hits from 卷 the user hasn't reached yet.
    if (maxJuan !== null && entry.j > maxJuan) continue;
    let from = 0;
    while (true) {
      const idx = entry.t.indexOf(q, from);
      if (idx < 0) break;
      const start = Math.max(0, idx - SNIPPET_PAD);
      const end = Math.min(entry.t.length, idx + q.length + SNIPPET_PAD);
      hits.push({
        j: entry.j,
        p: entry.p,
        y: entry.y,
        k: entry.k,
        snippet: entry.t.slice(start, end),
        matchStart: idx - start,
        matchLen: q.length,
        inHu: entry.m !== undefined && idx >= entry.m,
      });
      from = idx + q.length;
    }
  }
  // Sort by 卷 descending (latest first), but keep paragraphs within a
  // 卷 in their natural reading order.
  hits.sort((a, b) => (b.j - a.j) || (a.p - b.p));
  return hits.length > limit ? hits.slice(0, limit) : hits;
}
