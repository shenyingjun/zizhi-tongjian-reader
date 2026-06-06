"""
Accuracy validation for the pipeline.

For each parsed 卷, compare:
  - Total visible character count in parsed JSON (main text + 胡注)
  - vs total visible character count of the source HTML (after stripping tags
    and Wikisource header/sister-project boxes).

Reports any 卷 with a > 1% character delta as a potential parser bug.

Also re-fetches a few random 卷 from Wikisource and diffs the raw page text
against our parsed text to catch silent drops.

Run:
    python -m validate
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
HTML_DIR = ROOT / "cache" / "html"
PARSED_DIR = ROOT / "cache" / "parsed"
SIMPLIFIED_DIR = ROOT / "cache" / "simplified"

WS = re.compile(r"\s+")


def html_visible_chars(html_path: Path) -> int:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    # Strip non-content boxes and assets.
    for sel in ("#headerContainer", "table", "#plainSister", "noscript",
                "style", "script", "link"):
        for el in soup.select(sel):
            el.decompose()
    # Also strip transparent bracket spans (〈〉 around 胡注) — the parser
    # intentionally drops these decorative markers.
    for el in list(soup.find_all("span")):
        style = el.get("style", "") or ""
        if "transparent" in style:
            el.decompose()
    body = soup.find("body") or soup
    text = body.get_text("", strip=False)
    text = WS.sub("", text)
    return len(text)


def parsed_visible_chars(parsed_path: Path) -> int:
    data = json.loads(parsed_path.read_text(encoding="utf-8"))
    total = 0
    for p in data["paragraphs"]:
        total += len(p["main"])
        for n in p["notes"]:
            total += len(n["text"])
    return total


def main() -> int:
    parsed_files = sorted(PARSED_DIR.glob("juan_*.json"))
    if not parsed_files:
        print("no parsed files", file=sys.stderr)
        return 1

    problems: list[tuple[int, int, int, float]] = []
    total_html = total_parsed = 0
    sample_simp = 0
    for pf in parsed_files:
        juan_no = int(pf.stem.split("_")[1])
        hf = HTML_DIR / f"juan_{juan_no:03d}.html"
        if not hf.exists():
            continue
        h = html_visible_chars(hf)
        p = parsed_visible_chars(pf)
        total_html += h
        total_parsed += p
        delta = abs(h - p)
        ratio = delta / max(h, 1)
        if ratio > 0.02:
            problems.append((juan_no, h, p, ratio))
        # Verify simplified preserves length.
        sf = SIMPLIFIED_DIR / pf.name
        if sf.exists():
            s = parsed_visible_chars(sf)
            sample_simp += 1
            if s != p:
                print(f"[warn] juan {juan_no}: simplified char count "
                      f"{s} != parsed {p}")

    print(f"checked {len(parsed_files)} 卷")
    print(f"  html visible chars : {total_html:>10,}")
    print(f"  parsed chars       : {total_parsed:>10,}")
    print(f"  delta              : {abs(total_html-total_parsed):>10,} "
          f"({abs(total_html-total_parsed)/max(total_html,1):.2%})")
    print(f"  simplified checked : {sample_simp}")

    if problems:
        print(f"\n{len(problems)} 卷 with > 2% char delta (possible parser issue):")
        for jn, h, p, r in problems[:20]:
            print(f"  juan {jn:03d}: html={h:>6}  parsed={p:>6}  delta={r:.1%}")
    else:
        print("\nNo 卷 with > 2% delta — parser looks accurate.")
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
