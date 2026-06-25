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

// ── 白话导读 (plain-language comprehension layer) ──────────────────────────
// Editorial, pre-generated content rendered alongside the source text. It is
// NOT原文 and never generated at runtime — each 卷 may ship a static guide
// file;卷 without one simply render no guide blocks (graceful absence).

export interface GuidePersonRef {
  name: string;
  // Optional search query to look the person up via the existing 出处检索.
  query?: string;
  role?: string;
}

export interface GuideSummary {
  id: string;
  juan_no: number;
  granularity: 'year' | 'paragraph' | 'span';
  // Render the guide block immediately after this paragraph. For the MVP this
  // is the `type:'year'` paragraph id (see Juan.years[].paragraph_id).
  anchor_pid: number;
  ce_year: number | null;
  title?: string;
  one_liner: string;
  what_happened?: string;
  key_people?: GuidePersonRef[];
  why_it_matters?: string;
  background?: string;
  source_range?: {
    start_pid: number;
    end_pid?: number;
    label: string;
  };
  editorial_note?: string;
  confidence?: 'reviewed' | 'draft' | 'omit';
}

export interface JuanGuideFile {
  juan_no: number;
  version: 1;
  summaries: GuideSummary[];
}

// Cache per 卷 like _juanCache. A missing guide file (404) resolves to null —
// it is an expected, non-fatal state for any 卷 we haven't authored yet.
const _guideCache = new Map<number, Promise<JuanGuideFile | null>>();

export function loadJuanGuide(no: number): Promise<JuanGuideFile | null> {
  let cached = _guideCache.get(no);
  if (!cached) {
    const padded = String(no).padStart(3, '0');
    cached = fetch(`${BASE}text/guide/juan_${padded}.json`)
      .then(r => {
        if (r.status === 404) return null;
        if (!r.ok) throw new Error(`guide ${no} ${r.status}`);
        return r.json() as Promise<JuanGuideFile>;
      })
      .catch(() => null); // network/parse errors are non-blocking — source text still renders
    _guideCache.set(no, cached);
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
  t: string;       // searchable text (main + " " + each 胡注 joined by spaces)
  m?: number[];    // start index of each 胡注 within `t`; omitted if none
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
  snippet: string;   // text with one or more matches in shared context
  // Match positions within `snippet`, sorted by start. Multiple entries
  // mean nearby occurrences were merged into a single combined snippet to
  // avoid showing the same context twice.
  matches: { start: number; len: number }[];
  inHu: boolean;     // true when the match falls inside 胡三省音注
  atStart: boolean;  // snippet begins at the region start (omit leading …)
  atEnd: boolean;    // snippet ends at the region end (omit trailing …)
}

const SNIPPET_PAD = 30;
// When a punctuation mark sits a bit past SNIPPET_PAD, extend the snippet
// out to it (up to SNIPPET_MAX) instead of cutting mid-phrase. Snapping
// only ever grows the excerpt — it never trims context below SNIPPET_PAD.
const SNIPPET_MAX = 60;
// Chinese punctuation we'll prefer as snippet cut points so excerpts start
// and end on natural clause boundaries instead of mid-phrase.
const PUNCT = '，。！？；：、「」『』“”‘’（）《》〈〉…—,.!?;:';
// Sentence-ending punctuation — when the snippet boundary lands on one of
// these, the cut already reads like a complete clause so we suppress the
// trailing "…" ellipsis.
const SENT_END = '。！？.!?';

function isPunct(c: string): boolean { return PUNCT.indexOf(c) >= 0; }
function isSentEnd(c: string): boolean { return SENT_END.indexOf(c) >= 0; }

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
    // Collect all raw match indices, then merge ones whose context
    // windows would overlap so the user sees a single combined snippet
    // with multiple highlights instead of near-duplicate excerpts.
    const idxs: number[] = [];
    let from = 0;
    while (true) {
      const idx = entry.t.indexOf(q, from);
      if (idx < 0) break;
      idxs.push(idx);
      from = idx + q.length;
    }
    if (idxs.length === 0) continue;
    const noteStarts = entry.m;
    // Build region boundaries: each region is [start, end) within `t`. A
    // single-space separator sits between consecutive regions. Regions
    // are: main text, then one per 胡注 slot. Matches in different regions
    // are never merged so excerpts don't cross paragraph/note boundaries.
    const regions: [number, number][] = [];
    if (!noteStarts || noteStarts.length === 0) {
      regions.push([0, entry.t.length]);
    } else {
      regions.push([0, noteStarts[0] - 1]);
      for (let r = 0; r < noteStarts.length; r++) {
        const end = r + 1 < noteStarts.length ? noteStarts[r + 1] - 1 : entry.t.length;
        regions.push([noteStarts[r], end]);
      }
    }
    const regionOf = (i: number): number => {
      for (let r = 0; r < regions.length; r++) {
        if (i >= regions[r][0] && i < regions[r][1]) return r;
      }
      return 0;
    };
    const huStart = noteStarts && noteStarts.length > 0 ? noteStarts[0] : -1;
    const inHuFor = (i: number) => huStart >= 0 && i >= huStart;
    let group: number[] = [idxs[0]];
    const flush = () => {
      const first = group[0];
      const last = group[group.length - 1];
      const [lo, hi] = regions[regionOf(first)];
      const matchEnd = last + q.length;
      // Soft window: at least SNIPPET_PAD chars on each side, clamped to
      // the region.
      let start = Math.max(lo, first - SNIPPET_PAD);
      let end = Math.min(hi, matchEnd + SNIPPET_PAD);
      // Extend outward (never inward) up to SNIPPET_MAX looking for a
      // punctuation cut, so excerpts read as complete clauses without
      // sacrificing context.
      const extLo = Math.max(lo, first - SNIPPET_MAX);
      for (let i = start - 1; i >= extLo; i--) {
        if (isPunct(entry.t[i])) { start = i + 1; break; }
      }
      const extHi = Math.min(hi, matchEnd + SNIPPET_MAX);
      let cutAtSentEnd = false;
      for (let i = end; i < extHi; i++) {
        if (isPunct(entry.t[i])) {
          end = i + 1;
          cutAtSentEnd = isSentEnd(entry.t[i]);
          break;
        }
      }
      // If the trimmed `end` already happens to sit right after a sentence
      // ender (because the region itself ends there), treat that as a clean
      // cut too so we can drop the trailing "…".
      if (!cutAtSentEnd && end > 0 && isSentEnd(entry.t[end - 1])) {
        cutAtSentEnd = true;
      }
      // Skip any leading whitespace that crept in (e.g., when start landed
      // right after a region separator) — it would render as a stray gap.
      while (start < first && entry.t[start] === ' ') start++;
      hits.push({
        j: entry.j,
        p: entry.p,
        y: entry.y,
        k: entry.k,
        snippet: entry.t.slice(start, end),
        matches: group.map(i => ({ start: i - start, len: q.length })),
        inHu: inHuFor(first),
        atStart: start === lo,
        atEnd: end === hi || cutAtSentEnd,
      });
    };
    for (let i = 1; i < idxs.length; i++) {
      const prev = group[group.length - 1];
      // Merge when the gap between two matches is small enough that their
      // ±SNIPPET_PAD windows would overlap, AND both live in the same
      // region. Crossing a main/胡注 or note-to-note boundary always breaks
      // the group so distinct excerpts stay distinct.
      const close = idxs[i] - (prev + q.length) <= 2 * SNIPPET_MAX;
      const sameRegion = regionOf(idxs[i]) === regionOf(group[0]);
      if (close && sameRegion) {
        group.push(idxs[i]);
      } else {
        flush();
        group = [idxs[i]];
      }
    }
    flush();
  }
  // Sort by 卷 descending (latest first), but keep paragraphs within a
  // 卷 in their natural reading order.
  hits.sort((a, b) => (b.j - a.j) || (a.p - b.p));
  return hits.length > limit ? hits.slice(0, limit) : hits;
}
