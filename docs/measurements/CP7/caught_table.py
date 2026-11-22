"""CP7, D65: for each broken copy, how many cases of the suite fail on it, before the
loosening and with it.

    python -B caught_table.py <label>=<mutation check output> [<label>=<output> ...]

Each file is what `verify/mutation_check.py` printed. One row for each broken
copy and one column for each file: the number of cases that pass on the good
reference and fail on that copy. A copy that a later column no longer catches
is marked, and the script exits 1.
"""
import re
import sys
from pathlib import Path

LINE = re.compile(r"^(F\d+[a-z]?|M\d+)\s+fails (\d+)")


def read(path):
    counts, good = {}, None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        found = LINE.match(line)
        if found:
            counts[found.group(1)] = int(found.group(2))
        elif line.startswith("good "):
            good = line.split("   [")[0].split(":")[0]
    return counts, good


def main(argv):
    columns = [(arg.split("=", 1)[0], *read(arg.split("=", 1)[1])) for arg in argv]
    cases = list(columns[0][1])
    for _, counts, _ in columns[1:]:
        cases += [c for c in counts if c not in cases]
    print("| Broken copy | " + " | ".join(label for label, _, _ in columns) + " |")
    print("|---|" + "---|" * len(columns))
    lost = []
    for case in cases:
        cells = []
        for k, (label, counts, _) in enumerate(columns):
            value = counts.get(case)
            cells.append("" if value is None else str(value))
            if value == 0 and any(c[1].get(case) for c in columns[:k]):
                lost.append((case, label))
        print(f"| {case} | " + " | ".join(cells) + " |")
    for label, _, good in columns:
        print(f"{label}: {good}")
    for case, label in lost:
        print(f"LOST: no case fails on {case} any more under `{label}`")
    return 1 if lost else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
