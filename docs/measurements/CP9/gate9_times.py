"""CP9: the time of a gate run, per edit and in full, three runs each, in turn.

    python -B gate9_times.py <tree that holds verify/smoke_test.py> <deliverable> [rounds]
"""
import subprocess
import sys
import time
from pathlib import Path

tree, deliverable = Path(sys.argv[1]).resolve(), sys.argv[2]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
for i in range(rounds):
    for label, extra in (("per edit (--fast)", ["--fast"]), ("full", [])):
        started = time.perf_counter()
        done = subprocess.run([sys.executable, "-B", "verify/smoke_test.py", deliverable] + extra,
                              cwd=str(tree), capture_output=True, text=True)
        seconds = time.perf_counter() - started
        last = (done.stdout.strip().splitlines() or ["no output"])[-1]
        print(f"round {i + 1}  {label:18s} {seconds:6.1f} s  {last}", flush=True)
