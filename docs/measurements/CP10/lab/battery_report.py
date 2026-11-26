"""CP10: the pairs of a battery report that do not hold, and the outcomes of the pairs
that plant a carrier of the target.

    python -B battery_report.py <battery --json file> [--new]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
NEW = ("noise_rounded_1dp", "noise_score_1_to_40", "coded_copy_text", "coded_copy_number",
       "two_columns_together", "copy_5_cells_plus_1", "copy_5_cells_plus_20",
       "copy_5_cells_plus_1000", "copy_5_cells_times_1.5", "copy_5_cells_other_row",
       "feature_rounded_40", "feature_rounded_10")
pairs = doc["pairs"]
print("counts:", doc["counts"], "| pairs:", len(pairs))
print("\n## by change: holds / does not, and the outcomes of the planted columns\n")
print("| change | pairs | hold | do not hold | every planted column excluded | named by the "
      "near-perfect warning | warned by name | a column nothing names | ended in a stop |")
print("|---|---|---|---|---|---|---|---|---|")
by = defaultdict(list)
for p in pairs:
    by[p["change"]].append(p)
for change in NEW:
    group = by.get(change, [])
    outcome = Counter()
    for p in group:
        values = set(p.get("outcome", {}).values())
        if not values:
            outcome["stop" if p["kind"] == "stop" else "none"] += 1
        elif "silent" in values:
            outcome["silent"] += 1
        elif values == {"excluded"}:
            outcome["excluded"] += 1
        elif "named" in values and "warned" not in values:
            outcome["named"] += 1
        else:
            outcome["warned"] += 1
    holds = sum(p["class"] == "holds" for p in group)
    print(f"| {change} | {len(group)} | {holds} | {len(group) - holds} | {outcome['excluded']} | "
          f"{outcome['named']} | {outcome['warned']} | {outcome['silent']} | {outcome['stop']} |")
print("\n## every pair that does not hold\n")
for p in pairs:
    if p["class"] == "holds":
        continue
    seen = p["observed"]
    first = "; ".join(f"{name}: {message[:230]}" for name, message in p["fails"][:2])
    print(f"- [{p['class']}] {p['run']}:{p['change']} ({p['kind']}) skill {seen.get('skill')} "
          f"baseline {seen.get('baseline')} model {seen.get('model')} outcome {p.get('outcome')}\n"
          f"    {first}")
