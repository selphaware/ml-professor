"""CP10, A4: with `a` = the target plus a noise and `b` = that noise added to a gate table,
over 30 seeds: the headline skill, on how many seeds it is 0.98 or more, carries the
near-perfect warning, and names both columns (in the warning, and in the list).

    python -B report_a_two.py <a_two.json> [<a_two_full.json>]
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

print("| run | seeds | skill | at 0.98 or more | warned | both in the warning | both in the list |")
print("|---|---|---|---|---|---|---|")
for path in sys.argv[1:]:
    by = defaultdict(list)
    for r in lab10.load(path):
        by[r["run"]].append(r)
    for run, g in by.items():
        ok = [r for r in g if "error" not in r]
        near = [r for r in ok if r["skill"] is not None and r["skill"] >= 0.98]
        warned = [r for r in ok if any(c == "near_perfect_headline" for c, _ in r["warnings"])]
        both_w = [r for r in warned if all(lab10.fate(r, n) in ("named", "excluded", "warned")
                                           for n in ("pair_a", "pair_b"))]
        both_l = [r for r in ok if {"pair_a", "pair_b"} <= {e["column"] for e in r["rests_on"]}]
        sk = [r["skill"] for r in ok if r["skill"] is not None]
        print(f"| {run} | {len(g)} | {lab10.span(sk) if sk else '-'} | {len(near)} | "
              f"{len(warned)} | {len(both_w)} | {len(both_l)} |")
