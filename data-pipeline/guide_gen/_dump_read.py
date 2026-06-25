import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
recs = [json.loads(l) for l in open(here / "out" / "years.jsonl", encoding="utf-8")]
nos = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2, 3, 4]
for no in nos:
    out = here / "out" / f"read_{no}.txt"
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            if r["juan_no"] != no:
                continue
            sal = r.get("salience") or {}
            score = sal.get("score", 0)
            reasons = ",".join(sal.get("reasons") or [])
            f.write(
                f"##### anchor_pid={r['anchor_pid']} | {r['label']} | ce={r['ce_year']} "
                f"| range={r['source_range']} | events={r['event_paras']} "
                f"commentary={r['has_commentary']} | salience={score} {reasons}\n"
            )
            f.write(r["main_text"] + "\n\n")
print(f"wrote read files for {', '.join(str(n) for n in nos)}")
