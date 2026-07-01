#!/usr/bin/env python3
"""run_llm_pass.py — one-time LLM person-name pass over the 294-卷 corpus.

Produces the durable cache in ``llm_annotations/juan_NNN.tsv`` that
``build_persons.py`` consumes as a precision-guarded recall tier. The goal is to
run the LLM **once** for the whole book and never again: the cache is authored
here, checked in, and reused on every deterministic rebuild.

Design (from the 卷251 pipeline-vs-LLM bake-off): the LLM contributes *detection*
only — it lists full person names it sees in each 卷. The pipeline still owns
offsets, anaphora enumeration and disambiguation, and re-verifies every asserted
name against the 卷 text before minting a card. So the LLM is allowed to be a
little over-eager; the build-time guards keep precision first.

Usage
-----
    # OpenAI-compatible endpoint; key + model from the environment
    set LLM_API_KEY=sk-...
    set LLM_API_BASE=https://api.openai.com/v1     # optional, this is the default
    set LLM_MODEL=gpt-4o-mini                       # optional, this is the default

    python run_llm_pass.py                 # all 294 卷, skip cached ones
    python run_llm_pass.py --juans 250-253 # a range
    python run_llm_pass.py --juans 251     # a single 卷
    python run_llm_pass.py --force         # re-annotate even if cached
    python run_llm_pass.py --dry-run       # print prompts, make no API calls

The pass is resumable: a 卷 whose ``.tsv`` already exists is skipped unless
``--force`` is given, so an interrupted run just picks up where it left off.

Cost note: a full 294-卷 run on a small model (gpt-4o-mini-class) is roughly
$40–70 one-time, batched. No API key is bundled with this repo — supply your own.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXT_DIR = HERE.parents[1] / "web" / "public" / "text"
CACHE_DIR = HERE / "llm_annotations"

API_BASE = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# Keep each request well under the model's context; 通鉴 卷 run ~15–40k chars, so we
# batch paragraphs to stay cheap and avoid truncated completions.
CHARS_PER_BATCH = 6000

SYSTEM_PROMPT = (
    "你是《资治通鉴》人名识别助手。给你一段文言正文，请列出其中出现的**真实人物**的"
    "**完整姓名**（2–4 个汉字，姓+名）。\n"
    "严格要求（精度优先，宁缺毋滥）：\n"
    "1. 只输出完整姓名，且必须以真实姓氏或复姓开头（如 李德裕、长孙无忌）。\n"
    "2. 不要输出：单字省称（收、发）、官职（刺史、节度使）、称号谥号庙号（太宗、"
    "文正公）、封号（秦王、彭城王）、地名、民族/部族名、年号。\n"
    "3. 不确定是不是人名，就不要输出。\n"
    "4. 每个姓名只输出一次。\n"
    "只用 JSON 数组回答，形如 [\"李德裕\",\"王仲甫\"]，不要任何解释。"
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
        else:
            yield int(part)


def _batches(juan: dict):
    """Yield newline-joined paragraph batches (main + 胡注) under CHARS_PER_BATCH."""
    buf, size = [], 0
    for para in juan.get("paragraphs", []):
        pieces = [para.get("main", "")]
        pieces += [nt.get("text", "") for nt in para.get("notes", [])]
        chunk = "\n".join(p for p in pieces if p)
        if not chunk:
            continue
        if size + len(chunk) > CHARS_PER_BATCH and buf:
            yield "\n".join(buf)
            buf, size = [], 0
        buf.append(chunk)
        size += len(chunk)
    if buf:
        yield "\n".join(buf)


def _call_llm(text: str, retries: int = 4) -> list[str]:
    payload = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"].strip()
            start, end = content.find("["), content.rfind("]")
            if start < 0 or end < 0:
                return []
            names = json.loads(content[start:end + 1])
            return [n.strip() for n in names if isinstance(n, str) and n.strip()]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            wait = 2 ** attempt
            print(f"    ! API error ({e}); retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"    ! bad response ({e}); skipping batch", file=sys.stderr)
            return []
    return []


def annotate_juan(juan_no: int, dry_run: bool) -> list[str]:
    jf = TEXT_DIR / f"juan_{juan_no:03d}.json"
    if not jf.exists():
        return []
    juan = json.loads(jf.read_text(encoding="utf-8"))
    seen: dict[str, str] = {}  # name -> first evidence snippet
    for batch in _batches(juan):
        if dry_run:
            print(f"    [dry-run] batch {len(batch)} chars")
            continue
        for name in _call_llm(batch):
            if name in seen:
                continue
            idx = batch.find(name)
            snippet = ""
            if idx >= 0:
                snippet = batch[max(0, idx - 6): idx + len(name) + 6].replace("\n", " ")
            seen[name] = snippet
    return [f"{n}\thigh\t{ev}" for n, ev in seen.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--juans", help="e.g. 251 or 250-253 or 1,5,7 (default: all 294)")
    ap.add_argument("--force", action="store_true", help="re-annotate cached 卷")
    ap.add_argument("--dry-run", action="store_true", help="no API calls")
    args = ap.parse_args()

    if not args.dry_run and not API_KEY:
        sys.exit("LLM_API_KEY is not set. Export it (and optionally LLM_API_BASE / "
                 "LLM_MODEL) before running, or use --dry-run.")
    CACHE_DIR.mkdir(exist_ok=True)

    for juan_no in _iter_juans(args.juans):
        out = CACHE_DIR / f"juan_{juan_no:03d}.tsv"
        if out.exists() and not args.force:
            print(f"卷{juan_no:03d}: cached, skip")
            continue
        print(f"卷{juan_no:03d}: annotating…")
        rows = annotate_juan(juan_no, args.dry_run)
        if args.dry_run:
            continue
        header = (f"# 卷{juan_no:03d} LLM 人名校补（{MODEL}）\n"
                  f"# 格式：姓名<TAB>置信度<TAB>证据片段\n")
        out.write_text(header + "\n".join(rows) + ("\n" if rows else ""),
                       encoding="utf-8")
        print(f"卷{juan_no:03d}: {len(rows)} names -> {out.name}")


if __name__ == "__main__":
    main()
