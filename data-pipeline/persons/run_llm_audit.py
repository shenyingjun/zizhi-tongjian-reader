#!/usr/bin/env python3
"""run_llm_audit.py — v4 AUDIT pass (veto / binding / card) over the 294-卷 corpus.

Unlike ``run_llm_pass.py`` (which *detects* names by reading the raw 卷 text), this
pass **audits the pipeline's own output**. After a deterministic ``build_persons.py``
run it reads the local artifacts —

    web/public/text/persons/people.json                 (the cards)
    web/public/text/persons/mentions/juan_NNN.json      (every tagged span)
    web/public/text/juan_NNN.json                       (原文 for ±windows)

— and builds a COMPACT per-卷 *digest* (candidate cards + distinct tagged surfaces +
封号/官职 spans the build left unbound). The LLM sees the digest, not the 30KB 卷, so
the marginal cost of the audit layer is a fraction of the detection pass. One combined
completion per 卷 returns veto + binding + card records, appended (dedup-guarded) to the
SAME durable ``llm_annotations/juan_NNN.jsonl`` cache that ``build_persons.py`` consumes.

Precision-first by construction: veto is delete-only, binding/card offsets are still
placed + re-verified by the build, single-char 省称 routes through the gated anaphora
channel. The auditor only proposes mappings; the deterministic guards keep precision.

Usage
-----
    set LLM_API_KEY=sk-...
    set LLM_API_BASE=https://api.openai.com/v1     # optional
    set LLM_MODEL=gpt-4o-mini                       # optional

    python run_llm_audit.py                 # ALL 294 卷 (blanket audit), skip done
    python run_llm_audit.py --juans 16,108,195   # a subset
    python run_llm_audit.py --measure       # no API — print digest token/char sizes
    python run_llm_audit.py --dry-run       # no API — print the digest + prompt
    python run_llm_audit.py --mock FILE     # no API — feed a canned JSON response
    python run_llm_audit.py --force         # re-audit 卷 that already carry v4 records

Resumable: a 卷 whose jsonl already contains v4 audit records is skipped unless --force.
Cost note: the digest is ~5–10k input tokens/卷, so a full 294-卷 blanket audit on a
mini-class model is well under US$1 (batched). Supply your own key.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXT_DIR = HERE.parents[1] / "web" / "public" / "text"
PERSONS_DIR = TEXT_DIR / "persons"
MENTIONS_DIR = PERSONS_DIR / "mentions"
CACHE_DIR = HERE / "llm_annotations"

API_BASE = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

WIN = 8                       # ±context window (chars) around a surface / 封号 span
PLACEHOLDER_RE = re.compile(r"见于卷\d")
# 封号-looking span: 地名(1–3 汉字) + 爵. A candidate binding target when it is a person
# reference the build did NOT already tag (fresh recall the deterministic scan missed).
FENG_RE = re.compile(r"[\u4e00-\u9fff]{1,3}(?:王|公|侯|君|太子|皇后|夫人)")

SYSTEM_PROMPT = (
    "你是《资治通鉴》人物卡审校助手。给你**一卷**的结构化摘要（不是全文），包括：\n"
    "A. cards：本卷出现的人物卡（canonical 真名 / dynasty 朝代 / brief 简介 / 出现次数）；\n"
    "B. surfaces：流水线在本卷标注的表面串及一个上下文片段；\n"
    "C. feng_spans：本卷正文中「封号/官职」样式、但流水线尚未绑定到具体人物的片段。\n\n"
    "请只依据《资治通鉴》书内事实，输出三类**修正记录**（精度优先，宁缺毋滥）：\n"
    "1. veto：surfaces 中**非人名**的串（文言词、纯官职、地名、边界截断），"
    "给 {\"surface\":..,\"reason\":..}。\n"
    "2. binding：把 feng_spans（或 surfaces 中的封号/省称）映射到真名，"
    "给 {\"surface\":..,\"canonical\":..,\"dynasty\":..,\"role\":..}；"
    "若该封号在本卷**轮换指代不同人**，加 {\"para_range\":[lo,hi]}；"
    "**单个汉字**的省称（如 卬→刘卬）只在该字**不与常用词/其他人重名**时才给。\n"
    "3. card：对 cards 的**仅元数据**修缮："
    "{\"canonical\":..,\"dynasty\":..} 重标朝代（十六国实体常被误标为晋）、"
    "{\"canonical\":..,\"brief\":..} 用一句书内事实替换「见于卷NNN」占位、"
    "{\"canonical\":..,\"merge_into\":..} 把**异名同人**并入幸存卡"
    "（仅证据确凿，如 魏其侯≡窦婴；跨代同名同形异人不要并）。\n\n"
    "只用一个 JSON 对象回答，形如 "
    "{\"veto\":[..],\"binding\":[..],\"card\":[..]}，不要任何解释。空类给空数组。"
)


def _iter_juans(spec: str | None):
    if not spec:
        yield from range(1, 295)
        return
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            yield from range(int(a), int(b) + 1)
        elif part:
            yield int(part)


def _load_people():
    P = json.loads((PERSONS_DIR / "people.json").read_text(encoding="utf-8"))
    people = P["people"] if isinstance(P, dict) else P
    return {p["id"]: p for p in people}


def _existing_audit(juan_no: int):
    """Return (surfaces, canonicals) already carried as v4 audit records for this 卷,
    so a re-run never duplicates a hand-authored or prior-pass record."""
    jl = CACHE_DIR / f"juan_{juan_no:03d}.jsonl"
    surf, canon = set(), set()
    if not jl.exists():
        return surf, canon, False
    has_v4 = False
    for raw in jl.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = rec.get("type")
        if t in ("veto", "binding", "card"):
            has_v4 = True
            if rec.get("surface"):
                surf.add(rec["surface"])
            if rec.get("canonical"):
                canon.add(rec["canonical"])
    return surf, canon, has_v4


def build_digest(juan_no: int, by_id: dict) -> dict | None:
    """Assemble the compact per-卷 audit digest from local build artifacts."""
    jf = TEXT_DIR / f"juan_{juan_no:03d}.json"
    mf = MENTIONS_DIR / f"juan_{juan_no:03d}.json"
    if not jf.exists() or not mf.exists():
        return None
    juan = json.loads(jf.read_text(encoding="utf-8"))
    paras = juan.get("paragraphs", [])
    main_by_idx = {p.get("id"): p.get("main", "") for p in paras}
    full = "\n".join(p.get("main", "") for p in paras)

    M = json.loads(mf.read_text(encoding="utf-8"))
    rows = M if isinstance(M, list) else M.get("mentions", M)

    # ── A. cards present in this 卷 (via the mentions' person_ids) ──
    card_counts: dict[str, int] = {}
    for r in rows:
        pid = r.get("person_id")
        if pid:
            card_counts[pid] = card_counts.get(pid, 0) + 1
    cards = []
    for pid, cnt in sorted(card_counts.items(), key=lambda kv: -kv[1]):
        p = by_id.get(pid)
        if not p:
            continue
        brief = p.get("brief") or p.get("identity") or ""
        cards.append({
            "canonical": p.get("canonical_name", ""),
            "dynasty": p.get("dynasty", ""),
            "brief": brief,
            "n": cnt,
            "placeholder": bool(PLACEHOLDER_RE.search(brief)),
        })

    # ── B. distinct tagged surfaces + one window (veto candidates) ──
    surf_seen: dict[str, dict] = {}
    covered = []  # (para_id, start, end) spans already tagged — to find unbound 封号
    for r in rows:
        s = r.get("surface", "")
        if not s:
            continue
        covered.append((r.get("pid"), r.get("start"), r.get("end")))
        if s not in surf_seen:
            pid_para = r.get("pid")
            txt = main_by_idx.get(pid_para, "")
            st = r.get("start", 0)
            win = txt[max(0, st - WIN): st + len(s) + WIN].replace("\n", " ")
            surf_seen[s] = {"surface": s, "kind": r.get("kind", ""),
                            "n": 0, "ctx": win}
        surf_seen[s]["n"] += 1
    surfaces = sorted(surf_seen.values(), key=lambda d: -d["n"])

    # ── C. 封号/官职 spans NOT already tagged (unbound recall candidates) ──
    covered_by_para: dict[object, list] = {}
    for pid_para, st, en in covered:
        if st is None:
            continue
        covered_by_para.setdefault(pid_para, []).append((st, en))
    feng_seen: dict[str, dict] = {}
    for p in paras:
        pid_para = p.get("id")
        txt = p.get("main", "")
        spans = covered_by_para.get(pid_para, [])
        for m in FENG_RE.finditer(txt):
            a, b = m.start(), m.end()
            if any(cs <= a < ce or cs < b <= ce for cs, ce in spans):
                continue  # already tagged by the build → not a recall gap
            s = m.group(0)
            if s in feng_seen:
                feng_seen[s]["n"] += 1
                continue
            win = txt[max(0, a - WIN): b + WIN].replace("\n", " ")
            feng_seen[s] = {"span": s, "n": 1, "ctx": win}
    # keep only 封号 that recur or sit next to a given char (signal of a person ref)
    feng_spans = sorted(feng_seen.values(), key=lambda d: -d["n"])[:40]

    return {"juan": juan_no, "cards": cards, "surfaces": surfaces,
            "feng_spans": feng_spans, "_chars": len(full)}


def digest_to_prompt(dg: dict) -> str:
    """Render the digest as the compact user message."""
    lines = [f"卷{dg['juan']:03d} 审校摘要", "", "A. cards（真名｜朝代｜出现次数｜简介）："]
    for c in dg["cards"]:
        flag = " «占位»" if c["placeholder"] else ""
        lines.append(f"  {c['canonical']}｜{c['dynasty']}｜{c['n']}｜{c['brief']}{flag}")
    lines.append("")
    lines.append("B. surfaces（表面串｜kind｜次数｜片段）：")
    for s in dg["surfaces"]:
        lines.append(f"  {s['surface']}｜{s['kind']}｜{s['n']}｜…{s['ctx']}…")
    lines.append("")
    lines.append("C. feng_spans（未绑定的封号/官职｜次数｜片段）：")
    for f in dg["feng_spans"]:
        lines.append(f"  {f['span']}｜{f['n']}｜…{f['ctx']}…")
    return "\n".join(lines)


def _call_llm(user_msg: str, retries: int = 4) -> dict:
    payload = json.dumps({
        "model": MODEL, "temperature": 0,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user_msg}],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"].strip()
            a, b = content.find("{"), content.rfind("}")
            if a < 0 or b < 0:
                return {"veto": [], "binding": [], "card": []}
            return json.loads(content[a:b + 1])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            wait = 2 ** attempt
            print(f"    ! API error ({e}); retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"    ! bad response ({e}); skip", file=sys.stderr)
            return {"veto": [], "binding": [], "card": []}
    return {"veto": [], "binding": [], "card": []}


def _records_from_response(resp: dict, dg: dict, seen_surf: set, seen_canon: set):
    """Normalize the model's JSON into dedup-guarded v4 JSONL records + a count."""
    out = []
    full_cards = {c["canonical"] for c in dg["cards"]}
    for v in resp.get("veto", []) or []:
        s = (v.get("surface") or "").strip()
        if s and s not in seen_surf:
            out.append({"type": "veto", "surface": s,
                        "reason": (v.get("reason") or "").strip()})
            seen_surf.add(s)
    for b in resp.get("binding", []) or []:
        s = (b.get("surface") or "").strip()
        canon = (b.get("canonical") or "").strip()
        if not s or not canon or s in seen_surf:
            continue
        rec = {"type": "binding", "surface": s, "canonical": canon,
               "dynasty": (b.get("dynasty") or "").strip(),
               "role": (b.get("role") or "").strip()}
        pr = b.get("para_range")
        if isinstance(pr, list) and len(pr) == 2:
            rec["para_range"] = [int(pr[0]), int(pr[1])]
        out.append(rec)
        seen_surf.add(s)
    for c in resp.get("card", []) or []:
        canon = (c.get("canonical") or "").strip()
        if not canon:
            continue
        rec = {"type": "card", "canonical": canon}
        if c.get("dynasty"):
            rec["dynasty"] = c["dynasty"].strip()
        if c.get("brief"):
            rec["brief"] = c["brief"].strip()
        mi = (c.get("merge_into") or "").strip()
        # only accept a merge whose survivor is a real card in this 卷 (precision)
        if mi and mi in full_cards and mi != canon:
            rec["merge_into"] = mi
        if len(rec) > 2:  # more than just canonical → a real edit
            out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--juans", help="e.g. 16,108,195 or 1-40 (default: all 294)")
    ap.add_argument("--force", action="store_true", help="re-audit cached 卷")
    ap.add_argument("--measure", action="store_true", help="print digest sizes, no API")
    ap.add_argument("--dry-run", action="store_true", help="print digest+prompt, no API")
    ap.add_argument("--mock", help="path to a canned JSON response (no API)")
    args = ap.parse_args()

    if not (args.measure or args.dry_run or args.mock) and not API_KEY:
        sys.exit("LLM_API_KEY is not set. Export it, or use --measure/--dry-run/--mock.")
    by_id = _load_people()
    CACHE_DIR.mkdir(exist_ok=True)
    mock_resp = json.loads(Path(args.mock).read_text(encoding="utf-8")) if args.mock else None

    tot_chars = tot_written = audited = 0
    for juan_no in _iter_juans(args.juans):
        seen_surf, seen_canon, has_v4 = _existing_audit(juan_no)
        if has_v4 and not args.force and not (args.measure or args.dry_run):
            print(f"卷{juan_no:03d}: audited, skip")
            continue
        dg = build_digest(juan_no, by_id)
        if dg is None:
            continue
        prompt = digest_to_prompt(dg)
        approx_tok = (len(SYSTEM_PROMPT) + len(prompt)) // 2  # ~2 chars/token for 中文
        tot_chars += len(prompt)
        if args.measure:
            print(f"卷{juan_no:03d}: {dg['_chars']:6d} 原文字 → digest "
                  f"{len(prompt):5d} 字 (~{approx_tok} tok)  "
                  f"cards={len(dg['cards'])} surf={len(dg['surfaces'])} "
                  f"feng={len(dg['feng_spans'])}")
            continue
        if args.dry_run:
            print(f"\n===== 卷{juan_no:03d} PROMPT (~{approx_tok} tok) =====")
            print(prompt)
            continue
        resp = mock_resp if mock_resp is not None else _call_llm(prompt)
        recs = _records_from_response(resp, dg, seen_surf, seen_canon)
        if not recs:
            print(f"卷{juan_no:03d}: no new records")
            continue
        jl = CACHE_DIR / f"juan_{juan_no:03d}.jsonl"
        with open(jl, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"# \u2500\u2500 R-audit ({MODEL}) veto/binding/card \u2500\u2500\n")
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tot_written += len(recs)
        audited += 1
        print(f"卷{juan_no:03d}: +{len(recs)} records "
              f"({sum(r['type']=='veto' for r in recs)}v/"
              f"{sum(r['type']=='binding' for r in recs)}b/"
              f"{sum(r['type']=='card' for r in recs)}c)")

    if args.measure:
        n = max(audited, 1)
        print(f"\nTOTAL digest chars: {tot_chars}  "
              f"(~{tot_chars//2} input tokens across the audited 卷)")
    else:
        print(f"\ndone: {audited} 卷 audited, {tot_written} records written")


if __name__ == "__main__":
    main()
