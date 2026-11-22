"""CP8, section C2: what a run of verify/generated_tables.py found, by owner.

    python -B docs/measurements/CP8/generated_tables/summarise.py <run.json> [--seed S] [--tables N]

`<run.json>` is the file a run wrote with `--json`. Prints the failing tables by
the task of the seventh pass that owns the cause (T55 the date rule, T56 the stop
under `duplicates="group"` after a removed leak, T57 the fast profile's cap), and
any that belongs to none of them; then how many of the tables reach each thing
this pass decided, counted from the generator itself. Nothing is written.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[4]

OWNERS = [
    ("T55, the date rule (D69 point 1, D72 point 1)",
     ("dates written YYYYMMDD", "columns[daynum]", "date_column_undeclared, daynum",
      "columns[when] role", "date_column_undeclared, when")),
    ("T56, duplicates='group' after a removed leak (D69 point 2)",
     ("with duplicates='group' no combination",)),
    ("T57, the fast profile's cap with an entity column (D70, D72 point 5)",
     ("fast cap", "fast profile stops when an entity", "fast_profile_group_too_large")),
]
FAILS = ("raw", "outcome", "exact", "fit", "property")


def owner_of(entry):
    """The owner of one failing table: the first whose mark is in a failing finding.
    A table that carries a mark of T55 fails downstream too (the scores move with
    the column), so one mark is enough to own the table."""
    texts = [f"{f['field']} {f['application']} {f['reference']} {f['note']}"
             for f in entry["findings"] if f["class"] in FAILS]
    for owner, marks in OWNERS:
        if any(mark in text for mark in marks for text in texts):
            return owner
    return "NONE OF THE THREE"


def generator():
    path = REPO / "verify" / "generated_tables.py"
    spec = importlib.util.spec_from_file_location("generated_tables", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tables", type=int, default=None)
    args = parser.parse_args()
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    print(f"{run['package']} against {run['reference']}: {run['tables']} tables from seed "
          f"{run['seed']}, {run['wall_seconds']} s at {run['jobs']} workers")
    print(f"tables by class: {run['tables_by_class']}")
    by_owner = {}
    for entry in run["tables_with_findings"]:
        if entry["class"] == "fails":
            by_owner.setdefault(owner_of(entry), []).append(entry)
    print("\nfailing tables by owner")
    for owner, _ in OWNERS + [("NONE OF THE THREE", ())]:
        entries = by_owner.get(owner, [])
        print(f"  {owner}: {len(entries)} table(s): "
              + ", ".join(str(e["number"]) for e in entries))
        kinds = {}
        for entry in entries:
            for f in entry["findings"]:
                if f["class"] in FAILS:
                    key = f"[{f['class']}] " + f["field"].split(": ")[-1][:95]
                    kinds[key] = kinds.get(key, 0) + 1
        for key, count in sorted(kinds.items(), key=lambda kv: -kv[1])[:12]:
            print(f"      {count:>4}  {key}")
    print("\nopen:", {k: len(v) for k, v in run["open"].items()},
          "| stated limits:", {k: len(v) for k, v in run["stated_limits"].items()},
          "| not compared:", run["not_compared"])

    gt = generator()
    seed = run["seed"] if args.seed is None else args.seed
    n = run["tables"] if args.tables is None else args.tables
    reach = {}
    for number in range(n):
        case = gt.make_table(seed, number)
        text, a = "; ".join(case["notes"]), case["args"]
        marks = {
            "a YYYYMMDD column with one sentinel cell": "one sentinel cell" in text,
            "a YYYYMMDD column with a sentinel in a third of the rows": "a third of the rows" in text,
            "a YYYYMMDD column with two sentinels": "two sentinels" in text,
            "a YYYYMMDD time column with a value that is no date": "which is no date" in text,
            "a two-valued copy of the target beside few-valued columns": "beside few-valued" in text,
            "the same under duplicates='group'": "beside few-valued" in text
            and a["duplicates"] == "group",
            "an exact copy of the target (any table)": "y_copy: " in text,
        }
        who = case["oracle"]["who"]
        if a["group_col"] == "who" and a["time_col"] is False and who is not None:
            import numpy as np
            usable = ~case["oracle"]["y_missing"]
            sizes = np.bincount(who[usable]) if usable.any() else np.array([0])
            total, big = int(usable.sum()), int(sizes.max())
            marks["an entity of a quarter or more of the usable rows (grouped scheme)"] = \
                4 * big >= total
            marks["the same, past the fast cap under the fast profile"] = \
                4 * big >= total and total > gt.FAST_CAP and a["profile"] == "fast"
            marks["an entity of more rows than the fast cap"] = \
                big > gt.FAST_CAP and a["profile"] == "fast"
        for key, yes in marks.items():
            reach[key] = reach.get(key, 0) + bool(yes)
    print(f"\nwhat the {n} tables of seed {seed} reach")
    for key, count in reach.items():
        print(f"  {count:>4}  {key}")


if __name__ == "__main__":
    main()
