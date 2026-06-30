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
  // ── P0 person-identity binding (backward-compatible) ──
  // When present, this key person resolves to a canonical Person in the KB and
  // opens the spoiler-safe person card instead of a literal search. Unbound
  // refs (no person_id and unresolved at runtime) keep the `query || name`
  // literal-search behavior exactly as before.
  person_id?: string;
  candidate_ids?: string[];
  confidence?: PersonConfidence;
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

// ── 人物识别 (person identity) ─────────────────────────────────────────────
// A static, offline person knowledge base + per-卷 mention sidecars, shipped
// beside the text/ and guide/ assets. Source paragraph JSON is never mutated —
// person data lives entirely in these sidecars (mirroring the guide/ layer).
// Confidence tiers are honest about resolution quality; only 'reviewed'/'high'
// assert identity, everything else falls back to literal 出处检索.

export type PersonConfidence =
  | 'reviewed' | 'high' | 'medium' | 'low' | 'unresolved';

export interface PersonName {
  text: string;
  type: 'name' | 'style' | 'title' | 'posthumous' | 'clan' | 'alias';
}

export interface Person {
  id: string;
  canonical_name: string;
  names: PersonName[];
  dynasty?: string;
  era_hint?: string;
  floruit?: { start: number | null; end: number | null };
  // Spoiler-safe establishing identity (station/origin/kin, no later outcomes).
  // Shown by default in the card. NOT 原文 — rendered as 编者信息.
  brief?: string;
  // Full editorial identity; may narrate later events, so revealed only behind
  // the explicit spoiler toggle. NOT 原文.
  identity: string;
  confidence: PersonConfidence;
}

export interface PeopleFile {
  version: 1;
  people: Person[];
}

// One detected occurrence of a person in a 卷, keyed to an existing Paragraph.id.
export interface PersonMention {
  pid: number;
  ce_year: number | null;
  source: 'main' | 'hu';
  note_index?: number;
  start: number;          // char offset within main text or the 胡注 text
  end: number;
  surface: string;        // matched surface form
  person_id?: string;
  candidate_ids?: string[];
  // How the surface binds to a person:
  //   'alias'    surface -> exactly one person, valid anywhere (1:1, the default)
  //   'anaphora' 省称 short form resolved by position to the nearest full-name anchor
  //   'role'     称谓 (吴主/魏主/帝/上) resolved to the reigning holder at this point
  // Absent is treated as 'alias' for backward compatibility.
  kind?: 'alias' | 'anaphora' | 'role';
  confidence: PersonConfidence;
}

export interface JuanPersonMentions {
  juan_no: number;
  version: 1;
  mentions: PersonMention[];
}

// A resolved prior appearance shown in the person card (spoiler-filtered).
export interface PersonAppearance {
  person_id: string;
  juan: number;
  pid: number;
  ce_year: number | null;
  source: 'main' | 'hu';
  surface: string;
}

// One row of the cross-卷 appearance index (persons/appearances.json). It is a
// per-(person, paragraph) appearance in reading order, used to assemble
// spoiler-filtered prior/future appearances that span 卷 boundaries (the
// per-卷 mention sidecars only cover the current 卷).
export interface AppearanceRow {
  juan: number;
  pid: number;
  ce_year: number | null;
  source: 'main' | 'hu';
}

export interface AppearancesFile {
  version: 1;
  appearances: Record<string, AppearanceRow[]>;
}

// Cross-卷 appearance index, cached process-wide. Absent file → empty map
// (graceful: the app falls back to current-卷 mentions only).
let _appearancesCache: Promise<Map<string, AppearanceRow[]>> | null = null;

export function loadAppearances(): Promise<Map<string, AppearanceRow[]>> {
  if (!_appearancesCache) {
    _appearancesCache = fetch(`${BASE}text/persons/appearances.json`)
      .then(r => {
        if (!r.ok) throw new Error(`appearances ${r.status}`);
        return r.json() as Promise<AppearancesFile>;
      })
      .then(f => {
        const m = new Map<string, AppearanceRow[]>();
        for (const [pid, rows] of Object.entries(f.appearances)) m.set(pid, rows);
        return m;
      })
      .catch(() => new Map<string, AppearanceRow[]>());
  }
  return _appearancesCache;
}

// The people KB is one small file for the slice; cached process-wide.
let _peopleCache: Promise<Map<string, Person>> | null = null;

export function loadPeople(): Promise<Map<string, Person>> {
  if (!_peopleCache) {
    _peopleCache = fetch(`${BASE}text/persons/people.json`)
      .then(r => {
        if (!r.ok) throw new Error(`people ${r.status}`);
        return r.json() as Promise<PeopleFile>;
      })
      .then(f => {
        const m = new Map<string, Person>();
        for (const p of f.people) m.set(p.id, p);
        return m;
      })
      .catch(() => new Map<string, Person>()); // absent KB → no identity, graceful
  }
  return _peopleCache;
}

// Per-卷 mention sidecar. A missing file (404) resolves to null — expected for
// any 卷 we haven't built person data for yet (graceful absence).
const _mentionsCache = new Map<number, Promise<JuanPersonMentions | null>>();

export function loadPersonMentions(no: number): Promise<JuanPersonMentions | null> {
  let cached = _mentionsCache.get(no);
  if (!cached) {
    const padded = String(no).padStart(3, '0');
    cached = fetch(`${BASE}text/persons/mentions/juan_${padded}.json`)
      .then(r => {
        if (r.status === 404) return null;
        if (!r.ok) throw new Error(`mentions ${no} ${r.status}`);
        return r.json() as Promise<JuanPersonMentions>;
      })
      .catch(() => null);
    _mentionsCache.set(no, cached);
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

// Turn a set of (already deduped, sorted, non-overlapping) match positions
// inside one corpus entry's `t` into one or more LookupHit snippets, merging
// matches whose ±context windows touch (but never across a main/胡注 region
// boundary). Shared by the literal substring search and the span-driven
// person-occurrence builder so both produce identical snippet framing.
function buildEntryHits(
  entry: LookupEntry,
  idxs: { pos: number; len: number }[],
  out: LookupHit[],
): void {
  if (idxs.length === 0) return;
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
  let group: { pos: number; len: number }[] = [idxs[0]];
  const flush = () => {
    const first = group[0].pos;
    const lastM = group[group.length - 1];
    const [lo, hi] = regions[regionOf(first)];
    const matchEnd = lastM.pos + lastM.len;
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
    out.push({
      j: entry.j,
      p: entry.p,
      y: entry.y,
      k: entry.k,
      snippet: entry.t.slice(start, end),
      matches: group.map(m => ({ start: m.pos - start, len: m.len })),
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
    const close = idxs[i].pos - (prev.pos + prev.len) <= 2 * SNIPPET_MAX;
    const sameRegion = regionOf(idxs[i].pos) === regionOf(group[0].pos);
    if (close && sameRegion) {
      group.push(idxs[i]);
    } else {
      flush();
      group = [idxs[i]];
    }
  }
  flush();
}

// Index a (cached) lookup corpus by "j:p" so per-paragraph lookups are O(1).
// Keyed on the array identity — the corpus array is itself process-cached, so
// the index is built once.
const _lookupIndexCache = new WeakMap<LookupEntry[], Map<string, LookupEntry>>();
function lookupIndex(corpus: LookupEntry[]): Map<string, LookupEntry> {
  let m = _lookupIndexCache.get(corpus);
  if (!m) {
    m = new Map<string, LookupEntry>();
    for (const e of corpus) m.set(e.j + ':' + e.p, e);
    _lookupIndexCache.set(corpus, m);
  }
  return m;
}

// Build a person's occurrence list directly from the pipeline's NER mention
// spans (alias / anaphora / role) rather than re-substring-matching the card's
// name surfaces. This is what lets an emperor's 上/帝/魏主 mentions and a
// 省称 given-name (收/发) actually appear in the occurrence panel — the spans
// are already resolved to this exact person, so there is no over-highlight and
// no missed occurrence. `mentionsByJuan` carries each 卷's full mention list;
// only mentions whose person_id matches are mapped into the snippet builder.
export function buildPersonHits(
  personId: string,
  corpus: LookupEntry[],
  mentionsByJuan: { juan: number; mentions: PersonMention[] }[],
  opts: { limit?: number } = {},
): LookupHit[] {
  const { limit = 5000 } = opts;
  const idx = lookupIndex(corpus);
  const hits: LookupHit[] = [];
  for (const { juan, mentions } of mentionsByJuan) {
    // Group this person's spans by paragraph, mapping each into `t` coords:
    // main spans are already `t`-relative; a 胡注 span sits at m[note_index]
    // plus its in-note offset (every shipped note is non-empty, so note_index
    // aligns 1:1 with the `m` array).
    const byPara = new Map<number, { pos: number; len: number }[]>();
    for (const mn of mentions) {
      if (mn.person_id !== personId) continue;
      const entry = idx.get(juan + ':' + mn.pid);
      if (!entry) continue;
      let pos: number;
      if (mn.source === 'main') {
        pos = mn.start;
      } else {
        const ns = entry.m;
        const ni = mn.note_index ?? 0;
        if (!ns || ni >= ns.length) continue;
        pos = ns[ni] + mn.start;
      }
      const len = Math.max(1, mn.end - mn.start);
      const arr = byPara.get(mn.pid);
      if (arr) arr.push({ pos, len });
      else byPara.set(mn.pid, [{ pos, len }]);
    }
    for (const [pid, raw] of byPara) {
      const entry = idx.get(juan + ':' + pid);
      if (!entry) continue;
      raw.sort((a, b) => a.pos - b.pos || b.len - a.len);
      const dedup: { pos: number; len: number }[] = [];
      let lastEnd = -1;
      for (const m of raw) {
        if (m.pos >= lastEnd) { dedup.push(m); lastEnd = m.pos + m.len; }
      }
      buildEntryHits(entry, dedup, hits);
    }
  }
  // Latest 卷 first; natural reading order within a 卷.
  hits.sort((a, b) => (b.j - a.j) || (a.p - b.p));
  return hits.length > limit ? hits.slice(0, limit) : hits;
}

export function searchCorpus(
  query: string | string[],
  corpus: LookupEntry[],
  opts: { maxJuan?: number | null; limit?: number; restrictPids?: Set<string> | null } = {},
): LookupHit[] {
  // Accept one needle (literal 出处检索) or several (a bound person's
  // canonical_name + aliases). Longest-first so an alias nested inside the
  // canonical form (翦 ⊂ 王翦) never pre-empts the longer match.
  const needles = Array.from(
    new Set((Array.isArray(query) ? query : [query]).map(s => s.trim()).filter(Boolean)),
  ).sort((a, b) => b.length - a.length);
  if (needles.length === 0) return [];
  const { maxJuan = null, limit = 200, restrictPids = null } = opts;
  const hits: LookupHit[] = [];
  for (const entry of corpus) {
    // Spoiler filter: hide hits from 卷 the user hasn't reached yet.
    if (maxJuan !== null && entry.j > maxJuan) continue;
    // Curated-occurrence filter: when a bound person drives the query, only
    // their verified appearance paragraphs (from appearances.json) qualify —
    // this is what makes the list NER-accurate instead of substring-fuzzy.
    if (restrictPids !== null && !restrictPids.has(entry.j + ':' + entry.p)) continue;
    // Collect every match (position + its own length, since needles vary),
    // then drop overlaps (longer needle wins) and merge ones whose context
    // windows touch so the user sees one combined snippet, not duplicates.
    const raw: { pos: number; len: number }[] = [];
    for (const nd of needles) {
      let from = 0;
      while (true) {
        const idx = entry.t.indexOf(nd, from);
        if (idx < 0) break;
        raw.push({ pos: idx, len: nd.length });
        from = idx + nd.length;
      }
    }
    if (raw.length === 0) continue;
    raw.sort((a, b) => a.pos - b.pos || b.len - a.len);
    const idxs: { pos: number; len: number }[] = [];
    let lastEnd = -1;
    for (const m of raw) {
      if (m.pos >= lastEnd) { idxs.push(m); lastEnd = m.pos + m.len; }
    }
    buildEntryHits(entry, idxs, hits);
  }
  // Sort by 卷 descending (latest first), but keep paragraphs within a
  // 卷 in their natural reading order.
  hits.sort((a, b) => (b.j - a.j) || (a.p - b.p));
  return hits.length > limit ? hits.slice(0, limit) : hits;
}
