"""CP10, section D: a copy of the target with a handful of cells changed, by the size of
the change, under each variant of the rule.

    python -B report_d.py cells  <label>=<d_cells json> ...       gate tables, by scheme
    python -B report_d.py heavy  <label>=<d_heavy json> ...       heavy-tailed tables
    python -B report_d.py wild   <label>=<d_wild json> ...        no column may be excluded
    python -B report_d.py runs   <label>=<d_cells json> ...       gate tables, run by run, where not 30 of 30
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
ORDER = ("excluded", "warned", "named", "silent")


def counts(recs, column="target_copy"):
    fates = Counter(lab10.fate(r, column) for r in recs)
    other = {k: v for k, v in fates.items() if k not in ORDER}
    return "/".join(str(fates.get(f, 0)) for f in ORDER) + (f" {other}" if other else "")


def others(recs, column="target_copy"):
    """Runs in which another column is excluded as a leak."""
    return sum(any(v[1] == "suspected_leak" and n != column for n, v in r["roles"].items())
               for r in recs if "error" not in r)


if what in ("cells", "runs"):
    groups = defaultdict(lambda: defaultdict(list))
    for label in labels:
        for r in data[label]:
            _, cells, size = r["change"].split(":")
            key = (r["tag"], int(cells), size) if what == "cells" else (r["tag"], r["run"], int(cells), size)
            groups[key][label].append(r)
    print("excluded/warned/named/silent of the copy; then the runs in which another column is excluded\n")
    print("| scheme | " + ("run | " if what == "runs" else "") + "cells | changed by | "
          + " | ".join(labels) + " | another column excluded (" + ", ".join(labels) + ") |")
    print("|" + "---|" * (4 + len(labels) + (what == "runs")))
    sizes = ["0.1", "1", "5", "20", "1000", "x1.5", "other"]
    for key in sorted(groups, key=lambda k: (k[:-2], k[-2], sizes.index(k[-1]))):
        cells_ = [counts(groups[key].get(label, [])) for label in labels]
        if what == "runs" and all(c.startswith("30/0/0/0") for c in cells_):
            continue
        print("| " + " | ".join(str(k) for k in key) + " | " + " | ".join(cells_) + " | "
              + ", ".join(str(others(groups[key].get(label, []))) for label in labels) + " |")
elif what == "heavy":
    groups = defaultdict(lambda: defaultdict(list))
    for label in labels:
        for r in data[label]:
            p = r["params"]
            if not p.get("cells"):
                key = (r["make"], str(p.get("kind") or p.get("sigma")), p.get("scheme") or p.get("where") or "", 0, "-")
            else:
                key = (r["make"], str(p.get("kind") or p.get("sigma")), p.get("scheme") or p.get("where") or "",
                       p["cells"], str(p["size"]))
            groups[key][label].append(r)
    print("| table | kind or sigma | scheme | cells | changed by | "
          + " | ".join(f"{label}: excluded/warned/named/silent" for label in labels)
          + " | another column excluded | share of training rows marked | headline where the copy is kept ("
          + labels[-1] + ") |")
    print("|" + "---|" * (8 + len(labels)))
    for key in sorted(groups, key=str):
        row = [counts(groups[key].get(label, [])) for label in labels]
        last = groups[key].get(labels[-1], [])
        marked = [r["log"]["marked"][0] / r["log"]["marked"][2] for r in last
                  if "error" not in r and r.get("log", {}).get("marked")]
        kept = [r["skill"] for r in last if lab10.fate(r, "target_copy") in ("warned", "silent", "named")]
        print("| " + " | ".join(str(k) for k in key) + " | " + " | ".join(row) + " | "
              + ", ".join(str(others(groups[key].get(label, []))) for label in labels)
              + f" | {min(marked):.3f} to {max(marked):.3f} | {lab10.span(kept) if kept else '-'} |")
elif what == "wild":
    groups = defaultdict(lambda: defaultdict(list))
    for label in labels:
        for r in data[label]:
            p = r["params"]
            groups[(p["scheme"], p["value"], p["share"])][label].append(r)
    print("runs of 30 in which a column is excluded as a suspected leak (every column is noise); "
          "then the runs that stop with no_features\n")
    print("| scheme | wild value | share of the target cells | " + " | ".join(labels) + " |")
    print("|" + "---|" * (3 + len(labels)))
    for key in sorted(groups):
        row = []
        for label in labels:
            recs = groups[key].get(label, [])
            ex = sum(any(v[1] == "suspected_leak" for v in r["roles"].values())
                     for r in recs if "error" not in r)
            stops = Counter(r["error"] for r in recs if "error" in r)
            row.append(f"{ex}" + (f" {dict(stops)}" if stops else ""))
        if all(c == "0" for c in row):
            continue
        print("| " + " | ".join(str(k) for k in key) + " | " + " | ".join(row) + " |")
    print("(rows with 0 under every variant are left out)")
