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

# Characters OpenCC merges that lose meaningful distinction in 通鉴 context.
# We round-trip them through Private Use Area codepoints so OpenCC sees an
# unknown char (no conversion) and we restore the traditional form after.
#
#   乾 (qián, dry/heaven/era-name) — OpenCC maps to 干 (do/shield), wrecking
#     names like 李承乾 and era names 乾元/乾封/乾化/乾符/乾寧/乾德 etc. Even
#     modern simplified text keeps 乾 in 乾隆/乾坤 by convention.
_PRESERVE_CHARS = "乾"
_PRESERVE_MAP = {ch: chr(0xE000 + i) for i, ch in enumerate(_PRESERVE_CHARS)}
_PRESERVE_TR = str.maketrans(_PRESERVE_MAP)
_RESTORE_TR = str.maketrans({v: k for k, v in _PRESERVE_MAP.items()})


def convert_text(s: str) -> str:
    return cc.convert(s.translate(_PRESERVE_TR)).translate(_RESTORE_TR)


def convert_obj(obj):
    if isinstance(obj, str):
        return convert_text(obj)
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
