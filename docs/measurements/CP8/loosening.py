"""CP8: the one existing test file as the tests stage is to loosen it (D59, D69, D72).

    python -B loosening.py <folder>     write the loosened file into <folder>
    python -B loosening.py --table      print the replacement, as a table

`tests/test_date_numbers.py` is copied from the repo with the one replacement
below, which must match exactly once; every other line stays as it is. <folder>
is laid over a copy of tests/ outside the repo (`verify/mutation_check.py --overlay
<folder>`), so that the loosening can be proved before the tests stage makes it.
Nothing is written inside the repo.

Why: `test_values_that_are_no_dates_leave_the_column_what_it_was` holds fourteen
columns to "one value that is no date, and the rule does not apply" (D66). D69
reverses that for a column of dates with a sentinel: eleven of the fourteen are 200
days with one value that is no date, which D72's rule calls a date column. For those
eleven the assertion accepts the column as built (a numeric feature) or as decided
(excluded as a date), so the test passes before the change is built and after it.
`day_first`, `yyyymm` and `one_date_throughout` stay exact. The new behaviour is
pinned exactly by `tests/test_date_sentinels.py`, which is red until it is built.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = next(folder for folder in (Path(__file__).resolve().parents[3], Path(r"E:\CODE\ml-professor"))
            if (folder / "tests" / "test_date_numbers.py").is_file())

EDITS = {}


def edit(name, old, new, count=1):
    EDITS.setdefault(name, []).append((old, new, count))


edit("test_date_numbers.py", '''    record = _roles(_no_date_case(case))["day"]
    if case == "one_date_throughout":
        assert record["role"] == "excluded" and record["reason"] == "constant", record
    else:
        assert record["role"] == "feature" and record["kind"] == "numeric", record
''', '''    record = _roles(_no_date_case(case))["day"]
    if case == "one_date_throughout":
        assert record["role"] == "excluded" and record["reason"] == "constant", record
    elif case in ("day_first", "yyyymm"):
        assert record["role"] == "feature" and record["kind"] == "numeric", record
    else:
        # D59, D69: 200 days and one value that is no date. A number as built, and a
        # date column once D69 is built; `test_date_sentinels.py` holds the second.
        as_built = record["role"] == "feature" and record["kind"] == "numeric"
        as_decided = record["role"] == "excluded" and record["reason"] == "datetime"
        assert as_built or as_decided, record
''')

WHERE = {"test_date_numbers.py": ["test_values_that_are_no_dates_leave_the_column_what_it_was"]}


def loosened(name):
    """The file with its replacement made, and with the line endings it has in the repo."""
    raw = (REPO / "tests" / name).read_bytes().decode("utf-8")
    windows = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    for old, new, count in EDITS[name]:
        assert text.count(old) == count, (name, text.count(old), old[:70])
        text = text.replace(old, new)
    return text.replace("\n", "\r\n") if windows else text


def main(argv):
    assert {k: len(v) for k, v in WHERE.items()} == {k: len(v) for k, v in EDITS.items()}
    if argv == ["--table"]:
        print("| File | In | Lines before | Lines after |")
        print("|---|---|---|---|")
        for name, edits in EDITS.items():
            for (old, new, _), where in zip(edits, WHERE[name]):

                def cell(block):
                    return "<br>".join("`" + line.strip() + "`"
                                       for line in block.strip("\n").split("\n") if line.strip())
                print(f"| `{name}` | `{where}` | {cell(old)} | {cell(new)} |")
        return
    overlay = Path(argv[0])
    overlay.mkdir(parents=True, exist_ok=True)
    for name in EDITS:
        (overlay / name).write_bytes(loosened(name).encode("utf-8"))
    print(f"{len(EDITS)} file, {sum(len(v) for v in EDITS.values())} replacement -> {overlay}")


if __name__ == "__main__":
    main(sys.argv[1:])
