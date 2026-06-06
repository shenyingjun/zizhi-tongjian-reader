"""
Convert parsed JSON from Traditional → Simplified Chinese using OpenCC.

Input:  cache/parsed/juan_NNN.json   (Traditional)
Output: cache/simplified/juan_NNN.json (Simplified)

Run:
    python -m simplify
"""
from __future__ import annotations

import json
from pathlib import Path

from opencc import OpenCC
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "cache" / "parsed"
DST = ROOT / "cache" / "simplified"

# t2s = Traditional → Simplified (uses HK/TW variants by default; t2s is generic CN target).
cc = OpenCC("t2s")


def convert_obj(obj):
    if isinstance(obj, str):
        return cc.convert(obj)
    if isinstance(obj, list):
        return [convert_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_obj(v) for k, v in obj.items()}
    return obj


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("juan_*.json"))
    for src_path in tqdm(files, desc="t2s", unit="卷"):
        data = json.loads(src_path.read_text(encoding="utf-8"))
        out = convert_obj(data)
        (DST / src_path.name).write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    print(f"simplified → {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
