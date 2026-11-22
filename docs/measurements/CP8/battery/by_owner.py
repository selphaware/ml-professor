"""The failing pairs of a battery run, by the task of the seventh pass that owns them.

    python -B by_owner.py PAIRS.csv

PAIRS.csv is the table `pairs_csv.py` writes from a run's `--json` file. A pair
that fails outside the battery's two lists is given to a task by its run and its
change, which is where each decision of the pass has its tables:

    T55  the date rule (D69, D72): every pair of `day_sentinel`, and the three
         `ymd_sentinel_*` changes of `day_numbers`
    T56  the stop under duplicates="group" after a removed leak (D69, D72): every
         pair of `few_valued_leak`, and `few_valued:copy_two_valued`
    T57  the fast profile's cap (D70, D72): every pair of `large_entity` and of
         `quarter_entity`

Each line also shows what was seen (exit code, error code, skill), so that a pair
which fails for another reason than its task's shows. A failing pair that none of
the three rules names is printed under NONE OF THE THREE, and the script exits 1.
Standard library only; nothing is written.
"""
import csv
import sys
from collections import Counter, defaultdict

sys.dont_write_bytecode = True


def owner(run, change):
    if run == "day_sentinel" or (run == "day_numbers" and change.startswith("ymd_sentinel")):
        return "T55, the date rule"
    if run == "few_valued_leak" or (run, change) == ("few_valued", "copy_two_valued"):
        return "T56, the stop under duplicates='group' after a removed leak"
    if run in ("large_entity", "quarter_entity"):
        return "T57, the fast profile's cap"
    return "NONE OF THE THREE"


def main(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    classes = Counter(row["class"] for row in rows)
    print(f"{len(rows)} tool runs: " + ", ".join(f"{k} {v}" for k, v in sorted(classes.items())))
    owned = defaultdict(list)
    for row in rows:
        if row["class"] == "fails":
            owned[owner(row["run"], row["change"])].append(row)
    for name in sorted(owned):
        pairs = owned[name]
        print(f"{name}: {len(pairs)} pair(s)")
        seen = Counter((row["run"], row["exit_code"], row["error_code"] or "no error")
                       for row in pairs)
        for (run, code, error), count in sorted(seen.items()):
            names = [row["change"] for row in pairs if (row["run"], row["exit_code"],
                                                        row["error_code"] or "no error")
                     == (run, code, error)]
            print(f"    {count:>3}  {run}: exit {code}, {error}: " + ", ".join(names))
    for kind in ("open", "stated limit"):
        listed = [f"{row['run']}:{row['change']}" for row in rows if row["class"] == kind]
        print(f"{kind}: {len(listed)}" + (": " + ", ".join(listed) if listed else ""))
    return 1 if owned.get("NONE OF THE THREE") else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
