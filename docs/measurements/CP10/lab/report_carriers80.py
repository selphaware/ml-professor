"""CP10, D80 point 2: every battery pair that plants a carrier of the target, its outcome
under the floor of D78 (0.05) and of D80, its lift over the unperturbed run, and whether
the claim holds under each reading of "within that run's floor".

    python -B report_carriers80.py <battery --json file>

The readings: the floor of the pair's own run or of the unperturbed run; the headline
within that floor either way (two-sided), or not above it by more (one-sided).
"""
import json
import sys
from collections import Counter
from pathlib import Path

doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
STRONG = ("excluded", "warned", "named", "listed", "within", "lowered", "silent")
rows = []
for p in doc["pairs"]:
    c = p.get("carrier")
    if not c:
        continue
    outcome = p.get("outcome") or {}
    four = [o for o in outcome.values() if o in ("excluded", "warned", "named", "listed")]
    best = min(outcome.values(), key=STRONG.index) if outcome else "?"
    old_listed = any(isinstance(f, (int, float)) and f >= 0.05 for f in c["falls"].values())
    old = best if best in ("excluded", "warned", "named") else ("listed" if old_listed else "none")
    lift, floor, base_floor = c.get("lift"), c.get("floor"), c.get("base_floor")
    reading = {}
    for name, fl in (("own", floor), ("base", base_floor)):
        for side in ("two", "one"):
            if four:
                reading[(name, side)] = True
            elif lift is None or not isinstance(fl, (int, float)):
                reading[(name, side)] = False
            else:
                reading[(name, side)] = abs(lift) <= fl if side == "two" else lift <= fl
    rows.append((p["run"], p["change"], p["class"], old, best, lift, floor, base_floor, reading))

print("counts of the outcome under D80 (a pair counts under its strongest planted column):",
      dict(Counter(r[4] for r in rows)))
print("counts under the floor of D78 (0.05):", dict(Counter(r[3] for r in rows)))
for key in (("own", "two"), ("own", "one"), ("base", "two"), ("base", "one")):
    bad = [r for r in rows if not r[8][key]]
    print(f"claim with the floor of the {key[0]} run, {key[1]}-sided: {len(rows) - len(bad)} "
          f"hold, {len(bad)} do not: " + ", ".join(f"{r[0]}:{r[1]}" for r in bad))
print("\n| pair | outcome under D78 | under D80 | lift over the run without it | floor of the "
      "pair's run | of the unperturbed run | class |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    f = lambda v: "-" if v is None else f"{v:+.4g}" if isinstance(v, float) else str(v)
    print(f"| {r[0]}:{r[1]} | {r[3]} | {r[4]} | {f(r[5])} | {f(r[6])} | {f(r[7])} | {r[2]} |")
