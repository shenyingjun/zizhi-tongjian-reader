"""Extract per-year text bundles from emitted 卷 JSON.

Each 卷 is already segmented by year (paragraphs of type "year" with the year
heading; the narrative follows until the next year heading). This slices a 卷
into one record per year so the downstream authoring/validation steps work on a
clean, anchored unit.

Main text and 胡三省音注 (notes[]) are kept separate: 胡注 is reference gloss fed
as *context*, never summarised.

Usage:
    python extract.py 41 67 179
    python extract.py --all
Output: data-pipeline/guide_gen/out/years.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # data-pipeline/
TEXT_DIR = ROOT.parent / "web" / "public" / "text"     # emitted corpus
OUT_DIR = Path(__file__).resolve().parent / "out"

SALIENCE_PATTERNS: dict[str, str] = {
    "commentary": r"臣光曰",
    "state_ending": r"灭.{0,4}国|韩灭郑|宋.*亡|遂并|废.*为家人|分其地|迁.*公于海上",
    "title_change": r"初称王|称王|称帝|相王|列为诸侯|列侯|王号|不肯.*王|无其实，敢处其名",
    "regicide_deposition_succession": r"弑|废|传位|不立太子|争立|立其子|薨.*子.*立|劫|拘|质",
    "major_battle_territory": r"大破|斩首[二三四五六七八九十百千万0-9]+|拔|取城大小|取.*六十一|入郢|迁都|献.*地|割.*地",
    "diplomacy_alliance": r"合纵|从约|约从|从约皆解|连横|事秦|绝齐|欺齐、魏|会盟",
    "governance_reform": r"变法|求贤|下令国中|为相|刑名|县|度量衡|胡服骑射",
}

SALIENCE_SCORES: dict[str, int] = {
    "commentary": 5,
    "dense_year": 3,
    "state_ending": 3,
    "title_change": 3,
    "regicide_deposition_succession": 3,
    "major_battle_territory": 3,
    "diplomacy_alliance": 2,
    "governance_reform": 2,
}


def load_juan(no: int) -> dict:
    path = TEXT_DIR / f"juan_{no:03d}.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def note_text(p: dict) -> str:
    return "　".join(n.get("text", "") for n in p.get("notes", []) if n.get("text"))


def salience_for(main_text: str, event_paras: int, has_commentary: bool) -> dict:
    reasons: list[str] = []
    keyword_hits: list[dict[str, str]] = []

    if has_commentary:
        reasons.append("commentary")
        keyword_hits.append({"type": "commentary", "text": "has_commentary"})
    for typ, pat in SALIENCE_PATTERNS.items():
        for m in re.finditer(pat, main_text):
            if typ not in reasons:
                reasons.append(typ)
            keyword_hits.append({"type": typ, "text": m.group(0)})

    if event_paras >= 4:
        reasons.append("dense_year")

    score = sum(SALIENCE_SCORES[r] for r in reasons)
    return {"score": score, "reasons": reasons, "keyword_hits": keyword_hits}


def extract_juan(no: int) -> list[dict]:
    juan = load_juan(no)
    paras = juan["paragraphs"]
    by_id = {p["id"]: i for i, p in enumerate(paras)}
    years = juan["years"]
    records: list[dict] = []

    for i, y in enumerate(years):
        start_pid = y["paragraph_id"]
        # Span runs until the next year heading (exclusive), or end of 卷.
        if i + 1 < len(years):
            end_pid = years[i + 1]["paragraph_id"] - 1
        else:
            end_pid = paras[-1]["id"]

        si = by_id[start_pid]
        # Find the index of end_pid (or last paragraph whose id <= end_pid).
        ei = si
        while ei + 1 < len(paras) and paras[ei + 1]["id"] <= end_pid:
            ei += 1

        span = paras[si : ei + 1]
        main_parts: list[str] = []
        hu_parts: list[str] = []
        event_paras = 0
        has_commentary = False
        for p in span:
            t = p.get("type")
            if t == "commentary":
                has_commentary = True
            if t == "event":
                event_paras += 1
            if p.get("main"):
                main_parts.append(p["main"])
            hu = note_text(p)
            if hu:
                hu_parts.append(hu)

        main_text = "\n".join(main_parts)
        records.append(
            {
                "juan_no": no,
                "anchor_pid": start_pid,
                "ce_year": y.get("ce_year"),
                "label": y.get("label"),
                "source_range": {"start_pid": start_pid, "end_pid": end_pid},
                "event_paras": event_paras,
                "has_commentary": has_commentary,
                "salience": salience_for(main_text, event_paras, has_commentary),
                "main_text": main_text,
                "hu_text": "\n".join(hu_parts),
            }
        )
    return records


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    if argv[0] == "--all":
        manifest = json.loads((TEXT_DIR / "manifest.json").read_text(encoding="utf-8"))
        nos = [j["juan_no"] for j in manifest["juans"]]
    else:
        nos = [int(a) for a in argv]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "years.jsonl"
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for no in nos:
            for rec in extract_juan(no):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"extracted {n} year records from {len(nos)} 卷 -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
