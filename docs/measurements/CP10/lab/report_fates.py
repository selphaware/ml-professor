"""CP10: what became of one column, counted over the seeds.

    python -B report_fates.py <column> <json> [<json> ...] [--by scheme|run] [--label a,b,..]
                              [--readers] [--order]

One row for each table (gate run and change, or maker and parameters), or summed over
the runs of a scheme with `--by scheme`. With several files, one block of counts each.
`--readers`: the range of each reader's reading of the column (rule 3: every row
scored). `--order`: the share against the order under each form.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("column")
parser.add_argument("files", nargs="+")
parser.add_argument("--by", default="run")
parser.add_argument("--label", default=None)
parser.add_argument("--readers", action="store_true")
parser.add_argument("--order", action="store_true")
parser.add_argument("--only", default=None, help="substring of the table's name")
args = parser.parse_args()
labels = args.label.split(",") if args.label else [Path(f).stem for f in args.files]
ORDER = ("excluded", "warned", "named", "silent")


def key_of(r):
    what = r.get("change") or ""
    if r.get("make"):
        params = dict(r.get("params") or {})
        scheme = params.pop("scheme", None)
        what = r["make"] + " " + " ".join(f"{k}={v}" for k, v in params.items())
        return (scheme or r.get("tag") or "", what) if args.by == "scheme" else (what, scheme or "")
    if args.by == "scheme":
        return (r.get("tag") or "", what)
    return (what, r["run"])


tables = defaultdict(lambda: defaultdict(list))
for label, path in zip(labels, args.files):
    for r in lab10.load(path):
        key = key_of(r)
        if args.only and args.only not in " ".join(key):
            continue
        tables[key][label].append(r)

head = "| table | " + " | ".join(f"{label}: excluded / warned / named / silent" for label in labels) \
    + " | headline skill | other columns excluded (runs) | other columns warned (runs) |"
if args.readers:
    head += " tree | line | levels | rows of the column's levels seen |"
if args.order:
    head += " share against the order: old | mean | median | random | pairs |"
print(head)
print("|" + "---|" * head.count("|"))
for key in sorted(tables, key=lambda k: (str(k[1]) if args.by == "run" else str(k[0]), str(k))):
    cells = []
    last = None
    for label in labels:
        recs = tables[key].get(label, [])
        last = recs or last
        fates = Counter(lab10.fate(r, args.column) for r in recs)
        other = {k: v for k, v in fates.items() if k not in ORDER}
        cells.append(" / ".join(str(fates.get(f, 0)) for f in ORDER)
                     + (f" {other}" if other else ""))
    recs = last or []
    done = [r for r in recs if "error" not in r]
    skills = [r["skill"] for r in done if r["skill"] is not None]
    others_ex = sum(any(v[1] == "suspected_leak" and n != args.column for n, v in r["roles"].items())
                    for r in done)
    others_warn = sum(any(c == "strong_single_column" and col != args.column
                          for c, col in r["warnings"]) for r in done)
    line = f"| {' '.join(str(k) for k in key if k)} | " + " | ".join(cells) \
        + f" | {lab10.span(skills) if skills else 'no value'} | {others_ex} | {others_warn} |"
    if args.readers:
        for reader in ("tree", "line", "levels"):
            vals = [r["log"]["screen"][args.column][reader][1] for r in done
                    if args.column in r.get("log", {}).get("screen", {})
                    and r["log"]["screen"][args.column].get(reader)
                    and r["log"]["screen"][args.column][reader][1] is not None]
            line += f" {lab10.span(vals)} |"
        seen = [r["log"]["screen"][args.column].get("seen") for r in done
                if args.column in r.get("log", {}).get("screen", {})]
        line += f" {lab10.span(seen, 2)} |"
    if args.order:
        for form in ("old", "mean", "median", "random", "pairs"):
            vals = [r["log"]["order"][args.column][form][0] for r in done
                    if args.column in r.get("log", {}).get("order", {})]
            none = sum(v is None for v in vals)
            line += f" {lab10.span(vals)}{f' ({none} no value)' if none else ''} |"
    print(line)
