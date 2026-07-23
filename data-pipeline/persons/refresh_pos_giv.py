"""Explicit, resumable cache-v3 refresh for Classical-Chinese POS output."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pos_giv


PERS = Path(__file__).resolve().parent
REPO = PERS.parents[1]
TEXT = REPO / "web" / "public" / "text"
POS_DIR = TEXT / "persons" / "pos_giv"


def _paragraphs(juan: int):
    source = TEXT / f"juan_{juan:03d}.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))["paragraphs"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly refresh resumable per-volume POS cache v3 files."
    )
    parser.add_argument(
        "--juans",
        nargs="*",
        type=int,
        default=list(range(1, 295)),
        help="volumes to refresh; defaults to all 294",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun volumes whose current source already has a valid v3 cache",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="retries after each failed volume (default: 1)",
    )
    args = parser.parse_args()
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    if any(juan < 1 or juan > 294 for juan in args.juans):
        parser.error("--juans values must be in the range 1..294")

    juans = list(dict.fromkeys(args.juans))
    started = time.perf_counter()
    refreshed = skipped = retries = 0
    failures: list[tuple[int, str]] = []
    total = len(juans)

    for index, juan in enumerate(juans, 1):
        volume_started = time.perf_counter()
        try:
            paragraphs = _paragraphs(juan)
        except Exception as exc:
            failures.append((juan, f"{type(exc).__name__}: {exc}"))
            print(
                f"[{index:03d}/{total:03d}] juan={juan:03d} FAILED "
                f"source={failures[-1][1]}",
                flush=True,
            )
            continue

        if (
            not args.force
            and pos_giv.cache_version(juan, paragraphs, POS_DIR)
            == pos_giv.CACHE_VERSION
        ):
            skipped += 1
            print(
                f"[{index:03d}/{total:03d}] juan={juan:03d} "
                "skipped=current-v3",
                flush=True,
            )
            continue

        for attempt in range(args.retries + 1):
            try:
                evidence = pos_giv.giv_for_juan(
                    juan, paragraphs, POS_DIR, refresh=True
                )
                refreshed += 1
                span_count = sum(len(item.spans) for item in evidence.values())
                elapsed = time.perf_counter() - volume_started
                print(
                    f"[{index:03d}/{total:03d}] juan={juan:03d} "
                    f"spans={span_count} attempts={attempt + 1} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                break
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    retries += 1
                    print(
                        f"[{index:03d}/{total:03d}] juan={juan:03d} "
                        f"retry={attempt + 1}/{args.retries} error={message}",
                        flush=True,
                    )
                    continue
                failures.append((juan, message))
                print(
                    f"[{index:03d}/{total:03d}] juan={juan:03d} "
                    f"FAILED attempts={attempt + 1} error={message}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    print(
        f"summary total={total} refreshed={refreshed} skipped={skipped} "
        f"retries={retries} failures={len(failures)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    for juan, message in failures:
        print(f"failure juan={juan:03d} error={message}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
