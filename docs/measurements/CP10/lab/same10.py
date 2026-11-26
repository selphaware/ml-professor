"""CP10: do two packages give the same outputs, run by run? (the prototype against the
reference written from it)

    python -B same10.py <one.json> <other.json> [--a-free]

`--a-free` leaves out what the statement of section A writes (the shuffled keys, the
list, its floor and the near-perfect warning), for a prototype older than D80.
"""
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

a, b = lab10.load(sys.argv[1]), lab10.load(sys.argv[2])
free = "--a-free" in sys.argv
assert len(a) == len(b), (len(a), len(b))
FIELDS = ["error", "skill", "score", "model", "baseline", "baseline_score", "scheme", "baselines",
          "roles", "rows", "folds", "cv", "low", "high"]
if not free:
    FIELDS += ["rests", "rests_on", "floor"]
moved = Counter()
examples = {}
for x, y in zip(a, b):
    assert (x.get("run"), x.get("make"), x["k"], x.get("change")) == \
        (y.get("run"), y.get("make"), y["k"], y.get("change"))
    for f in FIELDS:
        if x.get(f) != y.get(f):
            moved[f] += 1
            examples.setdefault(f, (x.get("run") or x.get("make"), x["k"], x.get(f), y.get(f)))
    wa = sorted(map(tuple, x.get("warnings", [])))
    wb = sorted(map(tuple, y.get("warnings", [])))
    if free:
        wa = [w for w in wa if w[0] != "near_perfect_headline"]
        wb = [w for w in wb if w[0] != "near_perfect_headline"]
    if wa != wb:
        moved["warnings"] += 1
        examples.setdefault("warnings", (x.get("run") or x.get("make"), x["k"], wa, wb))
print(f"{len(a)} records compared; fields that differ: {dict(moved) or 'none'}")
for f, e in examples.items():
    print(f"  {f}: {str(e)[:400]}")
