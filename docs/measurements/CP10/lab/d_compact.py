"""CP10, section D, compact: on how many tables of 30 the altered copy is excluded, by the
size of the change, under each variant of the rule.

    python -B d_compact.py heavy <label>=<json> ...
    python -B d_compact.py lead  <label>=<json> ...
    python -B d_compact.py wild  <label>=<json> ...
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

what = sys.argv[1]
files = [a.split("=", 1) for a in sys.argv[2:]]
labels = [label for label, _ in files]
data = {label: lab10.load(path) for label, path in files}
SIZES = ["0.1", "1.0", "5.0", "20.0", "1000.0", "x1.5", "other", "+0.5", "+7"]


def excluded(recs):
    return sum(lab10.fate(r, "target_copy") == "excluded" for r in recs)


def other_excluded(recs):
    return sum(any(v[1] == "suspected_leak" and n != "target_copy" for n, v in r["roles"].items())
               for r in recs if "error" not in r)


if what in ("heavy", "lead"):
    groups = defaultdict(lambda: defaultdict(list))
    clean = defaultdict(lambda: defaultdict(list))
    for label in labels:
        for r in data[label]:
            p = r["params"]
            table = (r["make"], str(p.get("kind") or p.get("sigma")), p.get("scheme") or "time")
            if p.get("cells"):
                groups[table + (p["cells"],)][(label, str(p["size"]))].append(r)
            else:
                clean[table][label].append(r)
    sizes = [s for s in SIZES if any((labels[0], s) in g for g in groups.values())]
    print("the copy excluded, of 30 tables, for each size of change (" + " / ".join(labels) + ")\n")
    print("| table | kind or sigma | scheme | cells changed | " + " | ".join(sizes)
          + " | another column excluded, all sizes (" + ", ".join(labels) + ") |")
    print("|" + "---|" * (5 + len(sizes)))
    for key in sorted(groups, key=str):
        cells = [" / ".join(str(excluded(groups[key].get((label, s), []))) for label in labels)
                 for s in sizes]
        others = [sum(other_excluded(groups[key].get((label, s), [])) for s in sizes)
                  for label in labels]
        print("| " + " | ".join(str(k) for k in key) + " | " + " | ".join(cells) + " | "
              + ", ".join(map(str, others)) + " |")
    if clean:
        print("\nthe same tables with no copy: runs of 30 in which any column is excluded ("
              + ", ".join(labels) + "), and the headline under the last")
        for key in sorted(clean, key=str):
            last = [r["skill"] for r in clean[key].get(labels[-1], []) if "error" not in r
                    and r["skill"] is not None]
            print(f"  {' '.join(key)}: "
                  + ", ".join(str(other_excluded(clean[key].get(label, []))) for label in labels)
                  + f"; skill {lab10.span(last)}")
elif what == "wild":
    groups = defaultdict(lambda: defaultdict(list))
    for label in labels:
        for r in data[label]:
            p = r["params"]
            groups[(p["scheme"], p["value"])][(label, p["share"])].append(r)
    shares = sorted({s for g in groups.values() for _, s in g})
    print("runs of 30 in which some column of noise is excluded as a suspected leak or the run "
          "stops with no_features (" + " / ".join(labels) + "), by the share of target cells "
          "set to the wild value\n")
    print("| scheme | wild value | " + " | ".join(f"{100 * s:g}%" for s in shares) + " |")
    print("|" + "---|" * (2 + len(shares)))
    for key in sorted(groups):
        cells = []
        for s in shares:
            row = []
            for label in labels:
                recs = groups[key].get((label, s), [])
                bad = sum(("error" in r) or any(v[1] == "suspected_leak" for v in r["roles"].values())
                          for r in recs)
                row.append(str(bad))
            cells.append(" / ".join(row))
        print("| " + " | ".join(str(k) for k in key) + " | " + " | ".join(cells) + " |")
