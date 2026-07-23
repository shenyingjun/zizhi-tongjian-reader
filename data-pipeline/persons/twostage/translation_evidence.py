"""Build and load paragraph-scoped translation identity evidence.

The evidence contains no source or translation prose. It records only source URLs,
NER identities, canonical-text offsets, risk labels, and hashes. Loading validates
every paragraph hash and candidate surface before Agent 1 can consume the evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TEXT = REPO / "web" / "public" / "text"
SCHEMA_VERSION = 2


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _safe_candidate(row: dict) -> bool:
    return not str(row.get("mapping_status", "")).startswith("flagged_")


def _jie_number(text: str) -> int | None:
    if not text:
        return None
    codepoint = ord(text[0])
    if 0x2460 <= codepoint <= 0x2473:
        return codepoint - 0x2460 + 1
    if 0x3251 <= codepoint <= 0x325F:
        return codepoint - 0x3251 + 21
    if 0x32B1 <= codepoint <= 0x32BF:
        return codepoint - 0x32B1 + 36
    return None


def _jie_by_paragraph(paragraphs: list[dict]) -> dict[int, tuple[int, int | None]]:
    result = {}
    jie_index = 0
    number = None
    for paragraph in paragraphs:
        text = paragraph.get("main", "") or ""
        next_number = _jie_number(text)
        if next_number is not None:
            jie_index += 1
            number = next_number
        result[int(paragraph["id"])] = (jie_index, number)
    return result


def build(mapping_path: Path, output_dir: Path) -> dict:
    mapping_bytes = mapping_path.read_bytes()
    mapping = json.loads(mapping_bytes)
    rows_by_juan: dict[int, list[dict]] = {}
    for source in mapping.get("sources", []):
        rows_by_juan.setdefault(int(source["juan"]), [])
    for row in mapping.get("all_candidates", []):
        rows_by_juan.setdefault(int(row["juan"]), []).append(row)
    if not rows_by_juan:
        raise ValueError("mapping JSON has no all_candidates rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    evidence_hashes = {}
    for juan, rows in sorted(rows_by_juan.items()):
        text_path = TEXT / f"juan_{juan:03d}.json"
        paragraphs = json.loads(text_path.read_text(encoding="utf-8"))["paragraphs"]
        text_by_pid = {
            int(paragraph["id"]): paragraph.get("main", "") or ""
            for paragraph in paragraphs
        }
        jie_by_pid = _jie_by_paragraph(paragraphs)
        grouped: dict[tuple[int, str], list[dict]] = {}
        for row in rows:
            pid = int(row["repo_para_id"])
            identity = str(row.get("identity_surface") or row["translation_ner_name"])
            grouped.setdefault((pid, identity), []).append(row)

        paragraph_payload: dict[str, dict] = {}
        for (pid, identity), identity_rows in sorted(grouped.items()):
            if pid not in text_by_pid:
                raise ValueError(f"juan {juan} has no paragraph {pid}")
            text = text_by_pid[pid]
            jie_index, jie_number = jie_by_pid[pid]
            candidates = []
            seen = set()
            handles = set()
            for row in identity_rows:
                row_jie_index = int(row["repo_jie_index"])
                if row_jie_index != jie_index:
                    raise ValueError(
                        f"juan {juan} paragraph {pid} belongs to jie {jie_index}, "
                        f"not mapped jie {row_jie_index}"
                    )
                row_jie_number = row.get("repo_jie_number")
                if row_jie_number is not None and int(row_jie_number) != jie_number:
                    raise ValueError(
                        f"juan {juan} paragraph {pid} has jie number {jie_number}, "
                        f"not mapped number {row_jie_number}"
                    )
                start, end = int(row["original_start"]), int(row["original_end"])
                surface = str(row["original_surface"])
                if not (0 <= start < end <= len(text)) or text[start:end] != surface:
                    raise ValueError(
                        f"juan {juan} paragraph {pid} candidate does not match "
                        f"canonical text at [{start},{end})"
                    )
                mode = str(row["transfer_mode"])
                status = str(row["mapping_status"])
                key = (start, end, mode, status)
                if key in seen:
                    continue
                seen.add(key)
                eligible = _safe_candidate(row)
                normalized_surface = str(
                    row.get("normalized_original_surface") or surface
                )
                if (
                    eligible
                    and mode in {"anchor_given", "title_given"}
                    and 1 <= len(normalized_surface) <= 2
                ):
                    handles.add(normalized_surface)
                candidates.append(
                    {
                        "start": start,
                        "end": end,
                        "surface": surface,
                        "normalized_surface": normalized_surface,
                        "transfer_mode": mode,
                        "mapping_status": status,
                        "eligible": eligible,
                        "source_kind": str(row.get("source_kind", "")),
                        "source_page": str(row.get("source_page", "")),
                        "ner_score": float(row["translation_ner_score"]),
                    }
                )
            entry = {
                "identity_surface": identity,
                "eligible_anchor": any(
                    candidate["eligible"] for candidate in candidates
                ),
                "handles": sorted(handles),
                "candidates": candidates,
            }
            key = str(pid)
            paragraph_payload.setdefault(
                key,
                {
                    "text_sha256": _sha256_text(text),
                    "jie_index": jie_index,
                    "jie_number": jie_number,
                    "identities": [],
                },
            )["identities"].append(entry)

        payload = {
            "schema_version": SCHEMA_VERSION,
            "juan": juan,
            "mapping_sha256": _sha256_bytes(mapping_bytes),
            "mapping_method": mapping.get("method", {}),
            "paragraphs": paragraph_payload,
        }
        output_path = output_dir / f"juan_{juan:03d}.json"
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        output_path.write_bytes(encoded)
        counts[str(juan)] = sum(
            len(paragraph["identities"])
            for paragraph in paragraph_payload.values()
        )
        evidence_hashes[str(juan)] = _sha256_bytes(encoded)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mapping_file": mapping_path.name,
        "mapping_sha256": _sha256_bytes(mapping_bytes),
        "juans": sorted(rows_by_juan),
        "identities_by_juan": counts,
        "evidence_sha256_by_juan": evidence_hashes,
        "policy": (
            "canonical numbered-jie-scoped identity anchors only; paragraph offsets "
            "and unique jie indexes are both validated; flagged candidates cannot "
            "authorize anchors; no source or translation prose persisted"
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_juan(
    evidence_dir: Path,
    juan: int,
    paragraphs: list[dict],
) -> dict[int, tuple[dict, ...]]:
    path = evidence_dir / f"juan_{juan:03d}.json"
    if not path.is_file():
        raise FileNotFoundError(f"translation evidence is missing: {path}")
    manifest_path = evidence_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"translation evidence manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported translation evidence manifest schema in {manifest_path}"
        )
    expected_hash = manifest.get("evidence_sha256_by_juan", {}).get(str(juan))
    if not isinstance(expected_hash, str):
        raise ValueError(
            f"translation evidence manifest has no hash for juan {juan}"
        )
    evidence_bytes = path.read_bytes()
    if _sha256_bytes(evidence_bytes) != expected_hash:
        raise ValueError(f"translation evidence hash mismatch: {path}")
    payload = json.loads(evidence_bytes.decode("utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported translation evidence schema in {path}")
    if int(payload.get("juan", -1)) != juan:
        raise ValueError(f"translation evidence juan mismatch in {path}")

    text_by_pid = {
        int(paragraph["id"]): paragraph.get("main", "") or ""
        for paragraph in paragraphs
    }
    jie_by_pid = _jie_by_paragraph(paragraphs)
    result = {}
    for pid_text, paragraph in payload.get("paragraphs", {}).items():
        pid = int(pid_text)
        text = text_by_pid.get(pid)
        if text is None:
            raise ValueError(f"translation evidence references unknown paragraph {pid}")
        if paragraph.get("text_sha256") != _sha256_text(text):
            raise ValueError(
                f"translation evidence paragraph hash mismatch: juan {juan} pid {pid}"
            )
        expected_jie_index, expected_jie_number = jie_by_pid[pid]
        if int(paragraph.get("jie_index", -1)) != expected_jie_index:
            raise ValueError(
                f"translation evidence jie mismatch: juan {juan} pid {pid}"
            )
        if paragraph.get("jie_number") != expected_jie_number:
            raise ValueError(
                f"translation evidence jie number mismatch: juan {juan} pid {pid}"
            )
        identities = []
        for identity in paragraph.get("identities", []):
            for candidate in identity.get("candidates", []):
                start, end = int(candidate["start"]), int(candidate["end"])
                if (
                    not (0 <= start < end <= len(text))
                    or text[start:end] != candidate["surface"]
                ):
                    raise ValueError(
                        f"translation evidence candidate mismatch: "
                        f"juan {juan} pid {pid} [{start},{end})"
                    )
            identities.append(identity)
        result[pid] = tuple(identities)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.mapping_json, args.output_dir)
    print(
        f"wrote {sum(result['identities_by_juan'].values())} identities for "
        f"{len(result['juans'])} juans to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
