"""M1b — self-contained enriched-KB provider for the two-stage pipeline.

Imports build_persons and runs its Phase-A enrichment (build_enriched_kb) so the
two-stage pipeline sources the FULL enriched KB (all enrichment-minted cards +
the per-juan RULES surface index) AT ORIGIN — not from the committed people.json
output. This closes the standalone -462 regression that build_v3 exhibited when it
built only from seed.build_seed (which lacks the gloss / narrow-R1 / R2 / LLM /
xref+variant / lookback / gloss_fill cards minted inside build_enriched_kb).

Importing build_persons runs its module-level seed build (~8s) but NOT main()
(guarded by __main__). load_enriched_kb() then runs Phase A only (no emission).
The POS-gate passes use the committed web/public/text/persons/pos_giv cache, so no
torch load on a warm cache.
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import build_persons as bp  # noqa: E402  (module-level build_seed runs on import)


def load_enriched_kb():
    """Run Phase-A enrichment and return the fully-enriched KB state.

    Returns a dict with:
      people        -> bp.PEOPLE_MERGED (enriched card list, mutated in place)
      rules         -> bp.RULES         (juan -> {surface: pid} literal index)
      anaphora_rules-> bp.ANAPHORA_RULES(juan -> {admitted given char})
      by_id, canon_to_pids, given_of, minted_admit, minted_anchor, llm_anchor
      counters      -> enrichment tallies (for reporting)
    """
    kb = bp.build_enriched_kb()
    return {
        "people": bp.PEOPLE_MERGED,
        "rules": bp.RULES,
        "anaphora_rules": bp.ANAPHORA_RULES,
        "by_id": kb["by_id"],
        "canon_to_pids": kb["canon_to_pids"],
        "given_of": kb["given_of"],
        "minted_admit": kb["minted_admit"],
        "minted_anchor": kb["minted_anchor"],
        "llm_anchor": kb["llm_anchor"],
        "counters": kb["counters"],
    }


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    kb = load_enriched_kb()
    people = kb["people"]
    rules = kb["rules"]
    n_rule_surfaces = sum(len(m) for m in rules.values())
    print(f"enriched people cards: {len(people)}")
    print(f"RULES juans: {len(rules)}   total surface->pid entries: {n_rule_surfaces}")
    print(f"canon_to_pids keys: {len(kb['canon_to_pids'])}")
    print(f"given_of (single-char antecedents): {len(kb['given_of'])}")
    print(f"minted_anchor juans: {len(kb['minted_anchor'])}   "
          f"llm_anchor juans: {len(kb['llm_anchor'])}")
    print("enrichment counters:", kb["counters"])
