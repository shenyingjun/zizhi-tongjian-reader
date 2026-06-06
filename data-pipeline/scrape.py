"""
Scrape 《资治通鉴》(胡三省音注) from Wikisource.

Strategy:
1. Fetch the index page once.
2. Extract all 卷 URLs (294 卷 + 附录).
3. For each 卷, fetch the page (cached on disk in cache/html/),
   throttled to >=1 req/sec with a real User-Agent.

Run:
    python -m scrape

Re-running is safe: cached pages are not re-fetched.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
HTML_DIR = CACHE / "html"
INDEX_HTML = CACHE / "index.html"
INDEX_JSON = CACHE / "index.json"

INDEX_URL = (
    "https://zh.wikisource.org/zh-hant/"
    "%E8%B3%87%E6%B2%BB%E9%80%9A%E9%91%92_(%E8%83%A1%E4%B8%89%E7%9C%81%E9%9F%B3%E6%B3%A8)"
)
BASE = "https://zh.wikisource.org"
# Parsoid REST endpoint returns clean parsed HTML without skin chrome.
REST_HTML = "https://zh.wikisource.org/api/rest_v1/page/html/"

# Be a good citizen.
HEADERS = {
    "User-Agent": (
        "ZizhiTongjianReader/0.1 (personal study tool; "
        "contact: github.com/local-user) python-requests"
    ),
    "Accept-Language": "zh-Hant,zh;q=0.9",
}
THROTTLE_SECONDS = 1.2
MAX_RETRIES = 5


def _ensure_dirs() -> None:
    HTML_DIR.mkdir(parents=True, exist_ok=True)


def _get(url: str, dest: Path) -> str:
    """Fetch a URL to dest if not already cached. Returns the text.
    Retries with exponential backoff on 429/5xx."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_text(encoding="utf-8")
    delay = THROTTLE_SECONDS
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                # Honor Retry-After if present.
                retry_after = resp.headers.get("Retry-After")
                sleep_for = float(retry_after) if retry_after else delay * (2 ** attempt)
                sleep_for = min(sleep_for, 60.0)
                time.sleep(sleep_for)
                last_exc = requests.HTTPError(f"{resp.status_code} on {url}")
                continue
            resp.raise_for_status()
            resp.encoding = "utf-8"
            dest.write_text(resp.text, encoding="utf-8")
            time.sleep(THROTTLE_SECONDS)
            return resp.text
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(delay * (2 ** attempt))
    raise last_exc or RuntimeError(f"failed: {url}")


def fetch_index() -> str:
    return _get(INDEX_URL, INDEX_HTML)


def parse_index(html: str) -> list[dict]:
    """Extract a list of {juan_no, title, url, label_raw} from the index page."""
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one("div.mw-parser-output") or soup
    entries: list[dict] = []
    seen: set[str] = set()

    # Match links of the form /wiki/.../卷001 (or 附录...).
    pat = re.compile(r"/(?:wiki|zh-han[ts])/[^?#]+/(卷\d+|附[錄录][^/?#]*)$")

    for a in content.find_all("a", href=True):
        href = a["href"]
        decoded = unquote(href)
        m = pat.search(decoded)
        if not m:
            continue
        if href in seen:
            continue
        seen.add(href)

        label = m.group(1)
        juan_no: int | None = None
        if label.startswith("卷"):
            try:
                juan_no = int(label[1:])
            except ValueError:
                juan_no = None

        # Caption text often includes the descriptive title (周纪一 起戊寅...).
        # Walk the containing list item to grab the full descriptive line.
        li = a.find_parent("li")
        caption = li.get_text(" ", strip=True) if li else a.get_text(" ", strip=True)

        entries.append(
            {
                "juan_no": juan_no,
                "label": label,
                "title": caption,
                "url": urljoin(BASE, href),
            }
        )

    # Sort: 卷 by number, 附录 last.
    entries.sort(key=lambda e: (e["juan_no"] is None, e["juan_no"] or 0, e["label"]))
    return entries


def cache_path_for(entry: dict) -> Path:
    if entry["juan_no"] is not None:
        name = f"juan_{entry['juan_no']:03d}.html"
    else:
        safe = re.sub(r"[^\w\-]+", "_", entry["label"])
        name = f"appendix_{safe}.html"
    return HTML_DIR / name


def _rest_url_for(entry: dict) -> str:
    """Build the Parsoid REST URL from a wiki article URL."""
    from urllib.parse import quote
    # Extract the part after /wiki/ or /zh-han[ts]/.
    decoded = unquote(entry["url"])
    m = re.search(r"/(?:wiki|zh-han[ts])/(.+)$", decoded)
    if not m:
        raise ValueError(f"can't extract title from {entry['url']}")
    title = m.group(1)
    # REST API wants the title path-encoded but with `/` preserved.
    return REST_HTML + quote(title, safe="")


def fetch_all(entries: list[dict]) -> None:
    for entry in tqdm(entries, desc="scrape", unit="卷"):
        dest = cache_path_for(entry)
        try:
            _get(_rest_url_for(entry), dest)
        except requests.HTTPError as exc:
            print(f"[warn] {entry['label']}: {exc}", file=sys.stderr)


def main() -> int:
    _ensure_dirs()
    print(f"index → {INDEX_URL}")
    index_html = fetch_index()
    entries = parse_index(index_html)
    print(f"found {len(entries)} entries "
          f"({sum(1 for e in entries if e['juan_no'] is not None)} 卷)")
    INDEX_JSON.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if "--index-only" in sys.argv:
        return 0

    fetch_all(entries)
    print(f"cached HTML → {HTML_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
