"""One line for each pair of a battery run, from the file its `--json` wrote.

    python -B pairs_csv.py RESULTS.json OUT.csv

Columns: run, change, kind, class, exit code, error code, skill, baseline, model,
seconds. `class` is holds, stated limit, open or fails; for a file written by
the prototype of 2026-11-22, which had no classes, it is holds or fails.
Standard library only; nothing else is written.
"""
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
results, out = Path(sys.argv[1]), Path(sys.argv[2])
data = json.loads(results.read_text(encoding="utf-8"))
with open(out, "w", newline="\n", encoding="utf-8") as fh:
    writer = csv.writer(fh, lineterminator="\n")
    writer.writerow(["run", "change", "kind", "class", "exit_code", "error_code", "skill",
                     "baseline", "model", "seconds"])
    for pair in data["pairs"]:
        seen = pair.get("observed") or {}
        verdict = pair.get("class") or ("holds" if pair.get("ok") else "fails")
        writer.writerow([pair["run"], pair["change"], pair["kind"], verdict, seen.get("rc"),
                         seen.get("error") or "", "" if seen.get("skill") is None
                         else seen["skill"], seen.get("baseline") or "", seen.get("model") or "",
                         pair.get("seconds")])
count = {}
for pair in data["pairs"]:
    verdict = pair.get("class") or ("holds" if pair.get("ok") else "fails")
    count[verdict] = count.get(verdict, 0) + 1
print(f"{out}: {len(data['pairs'])} pairs, {count}; the run took {data.get('wall_seconds')} s "
      f"at {data.get('jobs')} side by side")
