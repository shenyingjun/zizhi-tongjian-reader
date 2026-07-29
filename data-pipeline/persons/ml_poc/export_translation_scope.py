from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def export_scope(mapping_path: Path, output_path: Path) -> dict:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    pair_jies: dict[str, dict[str, set[int]]] = {}
    for row in mapping.get("all_candidates", []):
        match = re.search(r"#pair-(\d+)$", str(row["source_page"]))
        if match is None:
            continue
        juan = str(int(row["juan"]))
        pair_index = str(int(match.group(1)) - 1)
        pair_jies.setdefault(juan, {}).setdefault(pair_index, set()).add(
            int(row["repo_jie_index"])
        )
    scope = {
        "schema_version": 1,
        "identity_fields_present": False,
        "source_mapping": mapping_path.name,
        "sources": mapping.get("sources", []),
        "pair_jies": {
            juan: {
                pair_index: sorted(jie_indexes)
                for pair_index, jie_indexes in sorted(
                    pairs.items(), key=lambda item: int(item[0])
                )
            }
            for juan, pairs in sorted(
                pair_jies.items(), key=lambda item: int(item[0])
            )
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(scope, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return scope


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export an identity-free translation-scope sidecar for Agent 1."
        )
    )
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scope = export_scope(args.mapping, args.output)
    print(json.dumps({
        "sources": len(scope["sources"]),
        "juans": len(scope["pair_jies"]),
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
