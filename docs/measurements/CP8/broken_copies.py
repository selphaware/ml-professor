"""CP8, proofs 1 and 2: which gate checks fail on which package.

    python -B broken_copies.py copies [name ...]         the gate in full on copies of the
                                                         reference: good, F12b, F14b, F15,
                                                         F16 ... F22 by default
    python -B broken_copies.py package <package folder> [check,check]
                                                         the gate (or some checks) on one
                                                         package, e.g. the reference as it
                                                         stood, or the application

The gate is run from a copy in the lab, so its artifacts are there and
verify/artifacts/runs/ in the repo is left as it is. For `copies`, each copy is the
repo's reference with one switch set in `_variant.py`. Each line: the package, the
number of checks, and the checks that fail, with the start of each message.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab                                              # noqa: E402

NEW = ["good", "F12b", "F14b", "F15"] + [f"F{i}" for i in range(16, 23)]


def run_gate(gate, package, only=None):
    cmd = [sys.executable, "-B", str(gate), str(package), "--json"]
    if only:
        cmd += ["--only", only]
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=lab.child_env())
    took = time.perf_counter() - started
    summary = json.loads([line for line in proc.stdout.splitlines() if line.startswith("{")][-1])
    failed = {c["id"]: c["message"] for c in summary["checks"] if not c["ok"]}
    return proc.returncode, len(summary["checks"]), failed, took


def show(label, outcome):
    code, n, failed, took = outcome
    print(f"{label}: exit {code}, {n} checks, {len(failed)} failing [{took:.0f} s]: "
          f"{', '.join(failed) or '-'}", flush=True)
    for check, message in failed.items():
        print(f"    {check}: {' '.join(message.split())[:260]}")


if __name__ == "__main__":
    gate = lab.lab_gate()
    if sys.argv[1] == "copies":
        for name in sys.argv[2:] or NEW:
            show(name, run_gate(gate, lab.fixture(name)))
    else:
        package = Path(sys.argv[2]).resolve()
        show(str(package), run_gate(gate, package, sys.argv[3] if len(sys.argv) > 3 else None))
