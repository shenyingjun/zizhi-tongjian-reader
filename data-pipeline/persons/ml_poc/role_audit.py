from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROLE_SURFACES = {
    "太皇太后", "皇太后", "皇太子", "御史大夫", "丞相司直",
    "骠骑将军", "车骑将军", "破羌将军", "前将军", "后将军",
    "左将军", "右将军", "卫将军", "大将军", "散骑常侍",
    "左谷蠡王", "右谷蠡王", "左奥鞬王", "右奥鞬王",
    "左贤王", "右贤王", "呼揭王", "长公主",
    "皇帝", "天子", "太子", "皇孙", "皇后", "太后", "王后", "帝",
    "公主", "夫人", "单于", "可汗", "赞普", "大昆弥", "小昆弥",
    "昆弥", "大司马", "丞相", "相国", "太尉", "太傅", "太师",
    "司徒", "司空", "将军", "都护", "廷尉", "侍御史", "御史",
    "刺史", "太守", "长史", "内史", "尚书", "侍中", "常侍",
    "谒者", "博士", "大夫", "校尉", "都尉", "中郎将", "郎中令",
    "中郎令", "司马", "司直", "大禄", "大监", "守丞",
}

ROLE_SUFFIXES = tuple(sorted(
    ROLE_SURFACES | {"王", "公", "侯", "后", "帝", "主"},
    key=len,
    reverse=True,
))
CONFERRAL_PREFIXES = (
    "自立为", "尊为", "立为", "拜为", "任为", "授为", "封为",
    "以为", "为", "拜", "任", "授", "除", "迁", "擢", "封", "立",
)
CLAUSE_BOUNDARIES = set("，。；：︰！？「」『』\n")
GENERIC_ROLE_LEFT = set("一二三四五六七八九十百千万两诸群众各凡数")
INSTITUTIONAL_ROLE_RIGHT = set("府宫庭位号职署印")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _paragraphs(task: dict) -> dict[int, str]:
    result = {}
    for jie in task["jies"]:
        for segment in jie["segments"]:
            start = int(segment["assembled_start"])
            end = int(segment["assembled_end"])
            result[int(segment["para_id"])] = jie["text"][start:end]
    return result


def _overlaps(
    para_id: int,
    start: int,
    end: int,
    annotations: list[dict],
) -> bool:
    return any(
        int(row["para_id"]) == para_id
        and start < int(row["end"])
        and int(row["start"]) < end
        for row in annotations
    )


def _adjacent_to_name_core(
    para_id: int,
    start: int,
    end: int,
    annotations: list[dict],
) -> bool:
    return any(
        int(row["para_id"]) == para_id
        and (
            int(row["end"]) == start
            or int(row["start"]) == end
        )
        and not _looks_role(str(row["surface"]))
        for row in annotations
    )


def _is_conferral(
    para_id: int,
    text: str,
    start: int,
    annotations: list[dict],
) -> bool:
    prefix = text[max(0, start - 3):start]
    if any(prefix.endswith(marker) for marker in CONFERRAL_PREFIXES):
        return True
    clause_start = max(
        (index + 1 for index, char in enumerate(text[:start])
         if char in CLAUSE_BOUNDARIES),
        default=0,
    )
    marker = text.rfind("为", clause_start, start)
    return marker >= 0 and start - marker <= 6


def _is_nonreferential_role_context(
    text: str,
    start: int,
    end: int,
) -> bool:
    return (
        start > 0 and text[start - 1] in GENERIC_ROLE_LEFT
    ) or (
        end < len(text) and text[end] in INSTITUTIONAL_ROLE_RIGHT
    )


def _looks_role(surface: str) -> bool:
    return any(surface.endswith(suffix) for suffix in ROLE_SUFFIXES)


def build_role_audit_pack(
    juan: int,
    blind_path: Path,
    recall_path: Path,
    state_path: Path,
) -> dict:
    task = _read(blind_path)
    recall = _read(recall_path)
    state = _read(state_path)
    if not state["recall"]["complete"]:
        raise ValueError(f"juan {juan} recall phase is incomplete")
    annotations = state["recall"]["annotations"]
    decisions = state["recall"]["decisions"]
    texts = _paragraphs(task)
    grouped: dict[tuple[int, int, int], dict] = {}

    def add(
        para_id: int,
        start: int,
        end: int,
        surface: str,
        source: str,
    ) -> None:
        if (
            _overlaps(para_id, start, end, annotations)
            or _adjacent_to_name_core(
                para_id, start, end, annotations
            )
            or _is_conferral(
                para_id, texts[para_id], start, annotations
            )
            or _is_nonreferential_role_context(
                texts[para_id], start, end
            )
        ):
            return
        key = (para_id, start, end)
        row = grouped.setdefault(key, {
            "id": f"role:{para_id}:{start}:{end}",
            "para_id": para_id,
            "start": start,
            "end": end,
            "surface": surface,
            "channels": [],
        })
        if row["surface"] != surface:
            raise ValueError(f"role candidate surface conflict at {key}")
        if source not in row["channels"]:
            row["channels"].append(source)

    for candidate in recall["candidates"]:
        if (
            decisions.get(candidate["id"]) == "reject"
            and _looks_role(str(candidate["surface"]))
        ):
            add(
                int(candidate["para_id"]),
                int(candidate["start"]),
                int(candidate["end"]),
                str(candidate["surface"]),
                "rejected_recall",
            )

    roles = sorted(ROLE_SURFACES, key=len, reverse=True)
    for para_id, text in texts.items():
        for start in range(len(text)):
            role = next(
                (surface for surface in roles if text.startswith(surface, start)),
                None,
            )
            if role is None:
                continue
            add(
                para_id,
                start,
                start + len(role),
                role,
                "role_lexicon",
            )

    rows = list(grouped.values())
    candidates = sorted(
        (
            row for row in rows
            if not any(
                other["para_id"] == row["para_id"]
                and other["start"] <= row["start"]
                and row["end"] <= other["end"]
                and (
                    other["start"] < row["start"]
                    or row["end"] < other["end"]
                )
                for other in rows
            )
        ),
        key=lambda row: (row["para_id"], row["start"], row["end"]),
    )
    for row in candidates:
        row["channels"].sort()
        text = texts[row["para_id"]]
        if text[row["start"]:row["end"]] != row["surface"]:
            raise ValueError(f"role candidate does not match text: {row}")
    return {
        "schema_version": 1,
        "phase": "role_audit",
        "juan": juan,
        "candidates": candidates,
        "note_evidence": [],
        "policy": (
            "Review uncovered role/title surfaces for specific human reference; "
            "reject generic, institutional, conferral, address, and author uses."
        ),
        "identity_fields_present": False,
    }


def prepare(
    blind_dir: Path,
    recall_dir: Path,
    state_dir: Path,
    output_dir: Path,
    reconcile_state: bool = False,
) -> list[Path]:
    manifest = _read(blind_dir / "manifest.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for selection in manifest["selected"]:
        juan = int(selection["juan"])
        output = output_dir / f"role_audit_juan_{juan:03d}.json"
        previous = _read(output) if output.is_file() else None
        payload = build_role_audit_pack(
            juan,
            blind_dir / f"blind_juan_{juan:03d}.json",
            recall_dir / f"recall_juan_{juan:03d}.json",
            state_dir / f"juan_{juan:03d}.json",
        )
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if reconcile_state and previous is not None:
            _reconcile_incomplete_state(
                state_dir / f"juan_{juan:03d}.json",
                previous,
                payload,
            )
        written.append(output)
    return written


def _reconcile_incomplete_state(
    state_path: Path,
    previous_pack: dict,
    current_pack: dict,
) -> None:
    state = _read(state_path)
    audit = state.get("role_audit")
    if not audit or not audit.get("initialized"):
        return
    if audit.get("complete"):
        raise ValueError(
            f"cannot reconcile completed role audit: {state_path}"
        )
    current_ids = {row["id"] for row in current_pack["candidates"]}
    previous_by_id = {
        row["id"]: row for row in previous_pack["candidates"]
    }
    stale_ids = set(audit["decisions"]) - current_ids
    stale_accepted = {
        (
            int(previous_by_id[candidate_id]["para_id"]),
            int(previous_by_id[candidate_id]["start"]),
            int(previous_by_id[candidate_id]["end"]),
        )
        for candidate_id in stale_ids
        if (
            candidate_id in previous_by_id
            and audit["decisions"][candidate_id] == "accept"
        )
    }
    recall_geometry = {
        (int(row["para_id"]), int(row["start"]), int(row["end"]))
        for row in state["recall"]["annotations"]
    }
    audit["annotations"] = [
        row for row in audit["annotations"]
        if (
            int(row["para_id"]), int(row["start"]), int(row["end"])
        ) not in stale_accepted
        or (
            int(row["para_id"]), int(row["start"]), int(row["end"])
        ) in recall_geometry
    ]
    audit["decisions"] = {
        candidate_id: decision
        for candidate_id, decision in audit["decisions"].items()
        if candidate_id in current_ids
    }
    temp = state_path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, state_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build identity-free specific-role audit packs."
    )
    parser.add_argument("--blind-dir", type=Path, required=True)
    parser.add_argument("--recall-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reconcile-incomplete-state",
        action="store_true",
        help="Drop obsolete decisions from an initialized, unlocked audit.",
    )
    args = parser.parse_args()
    for path in prepare(
        args.blind_dir,
        args.recall_dir,
        args.state_dir,
        args.output,
        args.reconcile_incomplete_state,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
