"""Read the log of count_calls.py: which tests leave time order or entities unsaid,
and which of those calls D63 would stop (they get as far as step 5 today)."""
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

log, junit = Path(sys.argv[1]), Path(sys.argv[2])
detail = len(sys.argv) > 3 and sys.argv[3]

cases = defaultdict(set)            # file -> test ids
outcome_of = {}
for item in ET.parse(junit).iter("testcase"):
    module = item.get("classname", "").split(".")[-1] + ".py"
    kinds = {child.tag for child in item}
    cases[module].add(item.get("name"))
    outcome_of[(module, item.get("name"))] = "FAILED" if kinds & {"failure", "error", "skipped"} else "passed"
print("suite:", sum(len(v) for v in cases.values()), "cases in", len(cases), "files;",
      sum(v == "FAILED" for v in outcome_of.values()), "failing in the lab")

calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
print("calls of profile():", len(calls))


def test_of(call):
    text = call["test"] or "?"
    text = text.rsplit(" (", 1)[0]
    path, _, name = text.partition("::")
    return path.split("/")[-1], name, (call["test"] or "").rsplit("(", 1)[-1].rstrip(")")


def new_stop(call):
    """The code D63 would raise for this call, or None when both statements are made."""
    if call["time_col"] == "None":
        return "time_order_undeclared"
    if call["group_col"] == "None":
        return "entities_undeclared"
    return None


def entry(call):
    if call["cli"] is not None:
        return "command line (h.run_cli or a subprocess)"
    file, func, _ = call["stack"][-1]
    return f"{file}::{func}"


by_file = defaultdict(lambda: defaultdict(set))
entries = defaultdict(set)
entry_calls = Counter()
same_stop_tests = defaultdict(set)
kinds = Counter()
per_test = defaultdict(list)
for call in calls:
    file, name, phase = test_of(call)
    stop = new_stop(call)
    by_file[file]["calls"].add(name)
    if stop is None:
        kinds["both said"] += 1
        continue
    by_file[file]["unsaid"].add(name)
    if not call["reached_step5"]:
        kinds["unsaid, ends before step 5 (unchanged by D63)"] += 1
        by_file[file]["before5"].add(name)
        continue
    if call["outcome"] == stop:
        kinds["unsaid, stops today with the same code"] += 1
        by_file[file]["same_stop"].add(name)
        same_stop_tests[file].add(name)
        per_test[(file, name)].append(("same-stop", call))
        continue
    kinds["unsaid, D63 changes the outcome"] += 1
    by_file[file]["changed"].add(name)
    entries[entry(call)].add((file, name))
    entry_calls[entry(call)] += 1
    per_test[(file, name)].append(("changed", call))

print()
for kind, n in kinds.items():
    print(f"  {n:5d} calls: {kind}")
print()
print("| File | cases | call profile()/CLI | leave something unsaid | D63 changes the outcome | stop today with the same code |")
print("|---|---|---|---|---|---|")
tot = Counter()
for file in sorted(cases):
    row = by_file.get(file, {})
    n = [len(cases[file])] + [len(row.get(k, ())) for k in ("calls", "unsaid", "changed", "same_stop")]
    for k, v in zip(("cases", "calls", "unsaid", "changed", "same_stop"), n):
        tot[k] += v
    if n[1]:
        print(f"| {file} | " + " | ".join(str(v) for v in n) + " |")
print(f"| total | {tot['cases']} | {tot['calls']} | {tot['unsaid']} | {tot['changed']} | {tot['same_stop']} |")
extra = sorted(set(by_file) - set(cases))
if extra:
    print("calls outside a test case:", extra, {f: {k: sorted(v) for k, v in by_file[f].items()} for f in extra})

print()
print("where the calls that D63 changes are made (the function that calls profile()):")
for name, tests in sorted(entries.items(), key=lambda kv: -len(kv[1])):
    files = Counter(f for f, _ in tests)
    print(f"  {len(tests):5d} cases, {entry_calls[name]:5d} calls: {name}   "
          + ", ".join(f"{f} {n}" for f, n in sorted(files.items())))

if detail == "same":
    print()
    for (file, name), items in sorted(per_test.items()):
        for kind, call in items:
            if kind == "same-stop":
                print(f"same-stop: {file}::{name}  time={call['time_col']} group={call['group_col']} "
                      f"-> {call['outcome']}  via {entry(call)}")
if detail == "changed":
    print()
    for (file, name), items in sorted(per_test.items()):
        for kind, call in items:
            if kind == "changed":
                print(f"{file}::{name}  time={call['time_col']} group={call['group_col']} "
                      f"-> {call['outcome']}  via {entry(call)}")
