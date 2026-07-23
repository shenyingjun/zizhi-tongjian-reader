"""Two-stage orchestrator entry point (M2).

Drives the FULL pipeline from twostage by importing build_persons' extracted
phases: Phase-A enrichment (build_enriched_kb / via kb.load_enriched_kb) and
Phase-B emission (emit_mentions). At this milestone it is a faithful port — output
is byte-identical to running build_persons directly — which establishes the single
twostage entry point BEFORE M3 swaps the Phase-B identity layer (RULES card-gated
prebinding) for the occurrence-card -> person-card era-window merge.

Usage:
  python pipeline.py                 # write to the real persons/ output dir
  python pipeline.py <out_dir>       # write mentions to a parallel dir (NOTE: this
                                     # also redirects the pos_giv cache path, so a
                                     # cold parallel dir will load torch once)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import build_persons as bp  # noqa: E402
import kb as kb_mod          # noqa: E402


def run(out_dir: str | None = None):
    if out_dir:
        bp.OUT = Path(out_dir)
    state = kb_mod.load_enriched_kb()
    bp.emit_mentions(state)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
