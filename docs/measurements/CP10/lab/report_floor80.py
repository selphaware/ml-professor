"""CP10, D80 point 1: the rests-on floor, old against new, from logged runs.

    python -B report_floor80.py <gate.json> [<made-tables json>]

For each run: on how many seeds the list names any column, under
  old       a fall of at least 0.05
  new       at least 0.05 and at least the half-width of the headline skill's interval;
            where there is none, at least a stand-in: `rows` (the half-width of a
            bootstrap of the holdout rows as if independent) or a fixed number
and whether the lag of a walk stays named.
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

STANDINS = ("rows", 0.10, 0.15, 0.20)


def floors(r):
    f = r.get("log", {}).get("floor", {})
    half, rows_half = f.get("half"), f.get("rows_half")
    out = {"old": 0.05}
    for s in STANDINS:
        if half is not None:
            out[s] = max(0.05, half)
        elif s == "rows":
            out[s] = max(0.05, rows_half) if rows_half is not None else 0.05
        else:
            out[s] = max(0.05, s)
    return out, half is not None


def named(r, floor):
    return [n for n, v in r["rests"].items() if v[2] is not None and v[2] >= floor]


for path in sys.argv[1:]:
    recs = [r for r in lab10.load(path) if "error" not in r]
    by = defaultdict(list)
    for r in recs:
        key = r.get("run") or (r["make"] + " " + " ".join(f"{k}={v}" for k, v in r["params"].items()))
        by[key].append(r)
    print(f"\n## {Path(path).name}: seeds on which the list names any column\n")
    print("| run | seeds | with an interval | half-width: range | old floor 0.05 | "
          + " | ".join(f"new, stand-in {s}" for s in STANDINS) + " | columns named under the new rule |")
    print("|---|---|---|---|---|" + "---|" * len(STANDINS) + "---|")
    for key, g in by.items():
        counts = {k: 0 for k in ("old",) + STANDINS}
        with_interval, halves, names = 0, [], defaultdict(int)
        for r in g:
            fl, has = floors(r)
            with_interval += has
            h = r.get("log", {}).get("floor", {}).get("half")
            if h is not None:
                halves.append(h)
            for k, v in fl.items():
                counts[k] += bool(named(r, v))
            for n in named(r, fl["rows"]):
                names[n] += 1
        top = ", ".join(f"{n} ({c})" for n, c in sorted(names.items(), key=lambda x: -x[1])[:3])
        print(f"| {key} | {len(g)} | {with_interval} | {lab10.span(halves) if halves else '-'} | "
              f"{counts['old']} | " + " | ".join(str(counts[s]) for s in STANDINS) + f" | {top} |")
