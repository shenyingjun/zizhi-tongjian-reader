"""
Emit static assets into web/public/ for the React reader to consume.

Produces:
  web/public/text/juan_NNN.json   (one per 卷, simplified)
  web/public/text/manifest.json   (list of all 卷 with metadata)

Run:
    python -m emit
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "cache" / "simplified"
WEB_PUBLIC = ROOT.parent / "web" / "public" / "text"

# Dynasty grouping for the sidebar (kept in Simplified).
DYNASTY_ORDER = [
    "周纪", "秦纪", "汉纪", "魏纪", "晋纪",
    "宋纪", "齐纪", "梁纪", "陈纪", "隋纪", "唐纪",
    "后梁纪", "后唐纪", "后晋纪", "后汉纪", "后周纪",
]


def main() -> int:
    if WEB_PUBLIC.exists():
        shutil.rmtree(WEB_PUBLIC)
    WEB_PUBLIC.mkdir(parents=True)

    manifest: list[dict] = []
    for src in sorted(SRC.glob("juan_*.json")):
        data = json.loads(src.read_text(encoding="utf-8"))
        # Re-serialize compact (no indent) — the source is pretty-printed at
        # indent=1, which roughly doubles file size for Chinese text. Stripping
        # it cuts each juan from ~110 KB to ~70 KB and noticeably speeds up
        # in-app navigation between 卷.
        (WEB_PUBLIC / src.name).write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        manifest.append({
            "juan_no": data["juan_no"],
            "label": data["label"],
            "title": data["title"],
            "dynasty": data.get("dynasty", ""),
            "year_range": data.get("year_range", ""),
            "paragraph_count": len(data["paragraphs"]),
            "ce_start": data["years"][0]["ce_year"] if data.get("years") else None,
            "ce_end": data["years"][-1]["ce_year"] if data.get("years") else None,
        })

    # Group by dynasty for the sidebar.
    by_dynasty: dict[str, list[dict]] = {}
    for m in manifest:
        by_dynasty.setdefault(m["dynasty"] or "其他", []).append(m)
    grouped = []
    for d in DYNASTY_ORDER:
        if d in by_dynasty:
            grouped.append({"dynasty": d, "juans": by_dynasty.pop(d)})
    for d, juans in by_dynasty.items():
        grouped.append({"dynasty": d, "juans": juans})

    (WEB_PUBLIC / "manifest.json").write_text(
        json.dumps({"juans": manifest, "grouped": grouped},
                   ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    # Build flat lookup corpus for selection-based search.
    # Each entry: {j: juan_no, p: paragraph_id, y: ce_year, k: kind, t: text}
    # `text` includes the main text + concatenated notes text so users can
    # find names mentioned in 胡三省音注 too.
    lookup: list[dict] = []
    for src in sorted(SRC.glob("juan_*.json")):
        data = json.loads(src.read_text(encoding="utf-8"))
        juan_no = data["juan_no"]
        for para in data["paragraphs"]:
            main_text = para.get("main", "")
            notes_text = "".join(n.get("text", "") for n in para.get("notes", []))
            if notes_text:
                text = main_text + " " + notes_text
            else:
                text = main_text
            entry = {
                "j": juan_no,
                "p": para["id"],
                "y": para.get("ce_year"),
                "k": para.get("type", para.get("kind", "")),
                "t": text,
            }
            # `m` marks where 胡三省音注 begins within `t` (after the
            # separator space). Omitted when there are no notes.
            if notes_text:
                entry["m"] = len(main_text) + 1
            lookup.append(entry)
    (WEB_PUBLIC / "lookup.json").write_text(
        json.dumps(lookup, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    size_mb = (WEB_PUBLIC / "lookup.json").stat().st_size / (1024 * 1024)
    print(f"emitted {len(manifest)} 卷 → {WEB_PUBLIC}")
    print(f"emitted lookup index: {len(lookup):,} paragraphs, {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
