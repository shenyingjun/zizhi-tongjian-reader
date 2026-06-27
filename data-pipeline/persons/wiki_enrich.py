"""Wikipedia/Wikidata enrichment of verified persons.

Reads wiki_verdicts.json (surface -> {qid, verdict, p31}) produced by
wiki_verify.py and, for every surface whose verdict is 'person' (Wikidata
P31=Q5, a human) with a QID, fetches:

  * the Wikidata short DESCRIPTION in Chinese  (e.g. 曹操 -> "東漢末年的軍事家…")
  * the zh.Wikipedia intro EXTRACT (lead paragraph, plain text)

Both come from ONE zh.wikipedia action-API call (prop=extracts|pageterms),
20 titles/call, after a Wikidata wbgetentities pass (50 QIDs/call) maps each
QID to its zhwiki title.

Output: wiki_person_info.json  ->  { surface: {qid, title, desc, extract} }
Consumed by seed.py to populate informative brief / identity for auto people.

Rerunnable & resumable: existing wiki_person_info.json entries are kept and
their QIDs skipped, so an interrupted run can resume cheaply.

Rate-limit hygiene (per Wikimedia 2024-2026 policy): a compliant User-Agent
(no UA => 10 req/min => 429s), maxlag=5, Retry-After honouring, ~1 req/s.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
from itertools import islice
import urllib.request, urllib.parse, urllib.error

HERE = Path(__file__).resolve().parent
VERDICTS = HERE / "wiki_verdicts.json"
OUT = HERE / "wiki_person_info.json"

UA = ("ZizhiTongjianReader/1.0 "
      "(https://github.com/shenyingjun/zizhi-tongjian-reader; reader@example.com)")
WD_API = "https://www.wikidata.org/w/api.php"
ZH_API = "https://zh.wikipedia.org/w/api.php"


def chunks(seq, n):
    it = iter(seq)
    while batch := list(islice(it, n)):
        yield batch


def safe_get(base, params):
    """GET JSON with 429 / 503 / maxlag backoff."""
    url = base + "?" + urllib.parse.urlencode(params)
    while True:
        req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                   "Accept-Encoding": "gzip"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    raw = gzip.decompress(raw)
                data = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", "60") or 60)
                print(f"  429 -> sleep {wait}s", flush=True); time.sleep(wait); continue
            if e.code == 503:
                print("  503 -> sleep 30s", flush=True); time.sleep(30); continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  net error {e} -> sleep 10s", flush=True); time.sleep(10); continue
        if isinstance(data, dict) and "error" in data and data["error"].get("code") == "maxlag":
            print(f"  maxlag {data['error'].get('lag','?')}s -> sleep 5s", flush=True)
            time.sleep(5); continue
        return data


def phase1_titles(qids):
    """{qid: zhwiki_title or None} via Wikidata wbgetentities (50/call)."""
    out = {}
    batches = list(chunks(qids, 50))
    for i, batch in enumerate(batches, 1):
        data = safe_get(WD_API, {
            "action": "wbgetentities", "ids": "|".join(batch),
            "props": "sitelinks|descriptions",
            "languages": "zh|zh-hans|zh-hant|zh-cn|zh-tw|zh-hk",
            "sitefilter": "zhwiki", "format": "json", "formatversion": "2",
        })
        for qid, ent in (data.get("entities") or {}).items():
            title = (ent.get("sitelinks") or {}).get("zhwiki", {}).get("title")
            desc_map = ent.get("descriptions") or {}
            desc = None
            for lang in ("zh", "zh-hans", "zh-hant", "zh-cn", "zh-tw", "zh-hk"):
                if lang in desc_map:
                    desc = desc_map[lang]["value"]; break
            out[qid] = {"title": title, "desc": desc}
        print(f"  phase1 {i}/{len(batches)} ({len(out)} resolved)", flush=True)
        time.sleep(1.0)
    return out


def phase2_extracts(titles):
    """{title: {desc, extract}} via zh.wikipedia extracts|pageterms (20/call)."""
    out = {}
    batches = list(chunks(titles, 20))
    for i, batch in enumerate(batches, 1):
        data = safe_get(ZH_API, {
            "action": "query", "prop": "extracts|pageterms",
            "exintro": "1", "explaintext": "1", "exlimit": "20",
            "wbptterms": "description", "titles": "|".join(batch),
            "redirects": "1", "maxlag": "5",
            "format": "json", "formatversion": "2",
        })
        # Map any redirect/normalized source titles back to what we asked for.
        remap = {}
        for r in (data.get("query", {}).get("redirects") or []):
            remap[r["to"]] = r["from"]
        for r in (data.get("query", {}).get("normalized") or []):
            remap[r["to"]] = r.get("from", remap.get(r["to"]))
        for pg in data.get("query", {}).get("pages", []):
            title = pg.get("title")
            asked = remap.get(title, title)
            terms = (pg.get("terms") or {}).get("description") or []
            out[asked] = {
                "extract": (pg.get("extract") or "").strip(),
                "desc": (terms[0].strip() if terms else None),
                "resolved_title": title,
            }
        print(f"  phase2 {i}/{len(batches)} ({len(out)} fetched)", flush=True)
        time.sleep(1.0)
    return out


def main():
    verdicts = json.loads(VERDICTS.read_text(encoding="utf-8"))
    existing = {}
    if OUT.exists():
        try:
            existing = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    # surface -> qid for confirmed humans, minus already-enriched surfaces.
    targets = {s: v["qid"] for s, v in verdicts.items()
               if v.get("verdict") == "person" and v.get("qid")
               and s not in existing}
    print(f"verdicts={len(verdicts)} person={sum(1 for v in verdicts.values() if v.get('verdict')=='person')} "
          f"already={len(existing)} to_fetch={len(targets)}", flush=True)
    if not targets:
        print("nothing to do", flush=True); return

    qids = sorted(set(targets.values()))
    print(f"phase1: {len(qids)} distinct QIDs -> titles", flush=True)
    qinfo = phase1_titles(qids)

    titles = sorted({d["title"] for d in qinfo.values() if d.get("title")})
    print(f"phase2: {len(titles)} titles -> extracts", flush=True)
    tinfo = phase2_extracts(titles) if titles else {}

    result = dict(existing)
    enriched = 0
    for surf, qid in targets.items():
        qi = qinfo.get(qid, {})
        title = qi.get("title")
        rec = {"qid": qid, "title": title,
               "desc": qi.get("desc"), "extract": None}
        if title and title in tinfo:
            ti = tinfo[title]
            rec["extract"] = ti.get("extract") or None
            # Prefer the zh.wikipedia pageterms description (same Wikidata source,
            # better cached) but fall back to the wbgetentities one.
            rec["desc"] = ti.get("desc") or rec["desc"]
        if rec["desc"] or rec["extract"]:
            enriched += 1
        result[surf] = rec

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=0),
                   encoding="utf-8")
    print(f"wrote {OUT.name}: {len(result)} entries, {enriched} with content", flush=True)


if __name__ == "__main__":
    main()
