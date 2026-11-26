"""CP10, D80 point 1, on the reference itself: for each run of a set, on how many seeds the
list names any column under the floor of D78 (a fall of 0.05 or more) and under the floor
the reference writes (`headline.rests_on_floor`), with the floor's range and the columns
named most.

    python -B report_floor_ref.py <set.json> [<set.json> ...]
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

print("| run | seeds | floor: range | named at 0.05 (D78) | named at the floor (D80) | "
      "columns named most at the floor |")
print("|---|---|---|---|---|---|")
for path in sys.argv[1:]:
    by = defaultdict(list)
    for r in lab10.load(path):
        if "error" in r:
            continue
        key = r.get("run") or (r["make"] + " " + " ".join(f"{a}={b}" for a, b in
                                                          sorted(r["params"].items())))
        by[key].append(r)
    for key, g in by.items():
        old = sum(any(v[2] is not None and v[2] >= 0.05 for v in r["rests"].values()) for r in g)
        new = sum(bool(r["rests_on"]) for r in g)
        floors = [r["floor"] for r in g if r.get("floor") is not None]
        names = Counter(e["column"] for r in g for e in r["rests_on"])
        print(f"| {key} | {len(g)} | {lab10.span(floors) if floors else '-'} | {old} | {new} | "
              + ", ".join(f"{n} ({c})" for n, c in names.most_common(3)) + " |")
