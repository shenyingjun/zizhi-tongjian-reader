# Data pipeline (offline)

Builds the static corpus + person index that the web reader ships with.

## Steps

1. `scrape.py` — fetch all 卷 of 《资治通鉴》(胡三省音注) from Wikisource.
2. `parse.py` — convert MediaWiki HTML to structured per-卷 JSON (separating main text and 胡注).
3. `simplify.py` — OpenCC traditional → simplified conversion.
4. `year_segment.py` — detect year boundaries inside each 卷.
5. `persons/` — build the person knowledge base (seed from JY0284 + NER + LLM disambig + CBDB). See [`persons/README.md`](persons/README.md) for the recognition rules, wave log, and backlog.
6. `emit.py` — emit `web/public/text/{juan}.json` and `web/public/index/zztj.sqlite`.

## Setup

```
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Cache

`cache/` holds raw HTML and intermediate JSON; safe to delete (regenerable). It is gitignored.
