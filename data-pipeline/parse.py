"""
Parse cached Wikisource HTML (Parsoid REST output) into structured JSON.

Per-卷 output schema:
{
  "juan_no": 1,
  "title": "卷001 周紀一 起戊寅(前403)盡壬子(前369) ...",
  "dynasty": "周紀",
  "year_range": "起戊寅(前403)盡壬子(前369)",
  "header": "資治通鑑卷第一 ... 司馬光奉敕編集",
  "paragraphs": [
    { "id": 0, "kind": "text",  "main": "...", "notes": [{"after": 12, "text": "胡注..."}] },
    ...
  ]
}

`main` is the visible body of the paragraph (no inline 胡注 expanded into it).
`notes` records 胡三省音注 with character offsets into `main` so the UI can render
them at the right spot (collapsible markers).

Run:
    python -m parse
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
HTML_DIR = ROOT / "cache" / "html"
INDEX_JSON = ROOT / "cache" / "index.json"
PARSED_DIR = ROOT / "cache" / "parsed"

# 胡三省 notes are rendered by Template:* with this inline style.
# Match by substring on the style attr (the full string contains a CSS var fallback).
HU_NOTE_STYLE_FRAGMENT = "996666"

# Empty span markers like <span style="color:transparent;font-size:0px">〈</span>
# wrap each 胡注 to visually bracket it. We drop them.
TRANSPARENT_RE = re.compile(r"transparent")


def _is_hu_note(node: Tag) -> bool:
    if node.name != "small":
        return False
    style = node.get("style", "") or ""
    return HU_NOTE_STYLE_FRAGMENT in style


def _clean_text(s: str) -> str:
    # Collapse runs of whitespace (Wikisource emits stray newlines).
    return re.sub(r"\s+", "", s)


def _node_text_without_notes(node: Tag) -> str:
    """Concatenate text of a paragraph excluding 胡注 <small> nodes."""
    parts: list[str] = []
    for child in node.descendants:
        if isinstance(child, Tag):
            if _is_hu_note(child):
                continue
            # If we hit a transparent marker span, skip its text contents.
            if child.name == "span" and TRANSPARENT_RE.search(child.get("style", "") or ""):
                # Don't double-walk; mark to skip.
                # We do nothing; we'll handle by checking ancestors below.
                pass
        elif isinstance(child, NavigableString):
            # Skip strings that are inside a 胡注 or transparent marker.
            skip = False
            for anc in child.parents:
                if not isinstance(anc, Tag):
                    continue
                if _is_hu_note(anc):
                    skip = True
                    break
                if anc.name == "span" and TRANSPARENT_RE.search(anc.get("style", "") or ""):
                    skip = True
                    break
            if skip:
                continue
            parts.append(str(child))
    return _clean_text("".join(parts))


def _hu_note_text(node: Tag) -> str:
    """Text of a <small> 胡注, stripping the transparent bracket markers."""
    out: list[str] = []
    for child in node.descendants:
        if isinstance(child, NavigableString):
            in_marker = any(
                isinstance(p, Tag)
                and p.name == "span"
                and TRANSPARENT_RE.search(p.get("style", "") or "")
                for p in child.parents
            )
            if in_marker:
                continue
            out.append(str(child))
    return _clean_text("".join(out))


def _extract_notes(node: Tag) -> list[dict]:
    """Walk the paragraph in document order and record each 胡注 with the
    character offset into the main text (text seen so far, excluding notes)."""
    notes: list[dict] = []
    main_so_far = 0
    # We need an in-order walk that, when we enter a 胡注, records the offset
    # then skips its subtree. Use an explicit stack.
    stack: list = [iter(node.children)]
    while stack:
        try:
            child = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(child, NavigableString):
            # Check we're not inside a transparent marker (parent only).
            text = _clean_text(str(child))
            main_so_far += len(text)
        elif isinstance(child, Tag):
            if _is_hu_note(child):
                note_text = _hu_note_text(child)
                if note_text:
                    notes.append({"after": main_so_far, "text": note_text})
                # don't recurse into the note
                continue
            if child.name == "span" and TRANSPARENT_RE.search(child.get("style", "") or ""):
                # ignore transparent marker entirely
                continue
            stack.append(iter(child.children))
    return notes


def parse_juan(html_path: Path, entry: dict) -> dict:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")

    # Strip header container (book metadata) and any navigation/sister-project boxes.
    for header in soup.select("#headerContainer"):
        header.decompose()
    for cat in soup.select('link[rel="mw:PageProp/Category"]'):
        cat.decompose()

    body = soup.find("body") or soup

    # Pull a header line (title + author) from the original soup before strip — we
    # already removed it, so reconstruct from index entry title.
    title = entry["title"]
    # The title looks like: 卷001 周紀一 起戊寅(前403)盡壬子(前369)凡三十五年 ...
    m = re.match(r"卷\d+\s+(\S+?)([一二三四五六七八九十百千]+)?\s+(.+)$", title)
    dynasty = m.group(1) if m else ""
    year_range = m.group(3).split("　")[0] if m else ""

    paragraphs: list[dict] = []
    para_id = 0
    capture_tags = ("p", "dl", "dd", "blockquote")
    for el in body.find_all(capture_tags, recursive=True):
        # Skip if any ancestor is also a capture-target — avoids double-counting
        # text that lives in nested structures like <dl><dd><p>...</p></dd></dl>.
        skip = False
        for anc in el.parents:
            if isinstance(anc, Tag) and anc.name in capture_tags:
                skip = True
                break
        if skip:
            continue
        main = _node_text_without_notes(el)
        if not main:
            continue
        notes = _extract_notes(el)
        paragraphs.append({
            "id": para_id,
            "kind": "text",
            "main": main,
            "notes": notes,
        })
        para_id += 1

    return {
        "juan_no": entry["juan_no"],
        "label": entry["label"],
        "title": title,
        "dynasty": dynasty,
        "year_range": year_range,
        "paragraphs": paragraphs,
    }


def main() -> int:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    entries = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    juan_entries = [e for e in entries if e["juan_no"] is not None]
    for entry in tqdm(juan_entries, desc="parse", unit="卷"):
        html_path = HTML_DIR / f"juan_{entry['juan_no']:03d}.html"
        if not html_path.exists():
            continue
        out = parse_juan(html_path, entry)
        out_path = PARSED_DIR / f"juan_{entry['juan_no']:03d}.json"
        out_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    print(f"parsed → {PARSED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
