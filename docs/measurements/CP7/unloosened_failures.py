"""CP7, proof 4: is every existing case that the changed reference fails, before the
loosening, a case the loosening covers?

    python -B unloosened_failures.py <calls.jsonl> <failed.txt>

<calls.jsonl> is the log of count_calls.py; <failed.txt> lists, one a line as
`file.py::test[case]`, the cases of the existing suite that fail against the
changed reference with no test file touched. A failing case is accounted for
when one of these holds:

  statement   it makes a call that leaves a statement unsaid and that D63 stops
              (the log shows the call reaching step 5 without that stop);
  fixture     it takes a fixture that makes such a call once for many tests, so
              that the log names only the first of them: `full_runs` of
              conftest.py (the whole session), `runs` of test_result_schema.py
              and `finished` of test_review4_small.py (the module);
  old name    it reads `n_rows_with_copies` or `copies_grouped`, the two names
              D66 changed.

Prints the count by file and by reason, and every case that none explains.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
log, failed = Path(sys.argv[1]), Path(sys.argv[2])


def new_stop(call):
    if call["time_col"] == "None":
        return "time_order_undeclared"
    if call["group_col"] == "None":
        return "entities_undeclared"
    return None


changed = set()
for line in log.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    call = json.loads(line)
    stop = new_stop(call)
    if stop and call["reached_step5"] and call["outcome"] != stop:
        name = (call["test"] or "?").rsplit(" (", 1)[0]
        path, _, case = name.partition("::")
        changed.add((path.split("/")[-1], case))

source = {p.name: p.read_text(encoding="utf-8") for p in (REPO / "tests").glob("test_*.py")}


def function_text(file, test):
    """The source of one test function."""
    found = re.search(rf"^def {re.escape(test)}\(.*?(?=^(?:def |@pytest|class )|\Z)",
                      source[file], re.S | re.M)
    return found.group(0) if found else ""


SHARED = {"full_runs", "runs", "finished"}
reasons, unexplained = defaultdict(Counter), []
for line in failed.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    file, _, case = line.partition("::")
    test = case.split("[")[0]
    body = function_text(file, test)
    if (file, case) in changed:
        why = "statement"
    elif SHARED & set(re.findall(r"\w+", body.split("):", 1)[0])):
        why = "fixture"
    elif "n_rows_with_copies" in body or "copies_grouped" in body:
        why = "old name"
    else:
        why = None
    if why is None:
        unexplained.append(line)
    else:
        reasons[file][why] += 1

total = Counter()
print("| File | fail | a statement unsaid | through a fixture | an old name |")
print("|---|---|---|---|---|")
for file in sorted(reasons):
    row = reasons[file]
    total.update(row)
    print(f"| `{file}` | {sum(row.values())} | {row['statement']} | {row['fixture']} | "
          f"{row['old name']} |")
print(f"| total | {sum(total.values())} | {total['statement']} | {total['fixture']} | "
      f"{total['old name']} |")
print(f"\nnot explained: {len(unexplained)}")
for line in unexplained:
    print("  ", line)
