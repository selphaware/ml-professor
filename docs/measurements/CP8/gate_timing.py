"""CP8, item 8 of the plan (D72): how long one gate run takes at four launch settings.

    python -B gate_timing.py <gate file> <package folder> [--repeats 3]

Run alone, with nothing else loading the machine. <gate file> is a gate script
(the repo's verify/smoke_test.py, or a copy of an earlier one); <package folder>
a known-good package `mlprofessor`, for instance a copy of the reference made by
`lab.fixture("good")`. For each setting a copy of the gate with START_SPACING and
PARALLEL rewritten is placed in the lab, run once to write its datasets, then
`--repeats` times in full and with --fast. Nothing is written inside the repo.
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab                                              # noqa: E402

SETTINGS = ((0.5, 8), (0.25, 8), (0.25, 16), (0.1, 16))


def variant(gate, spacing, parallel):
    text = Path(gate).read_bytes().decode("utf-8")
    text, a = re.subn(r"^START_SPACING = [\d.]+$", f"START_SPACING = {spacing}", text, flags=re.M)
    text, b = re.subn(r"^PARALLEL = \d+", f"PARALLEL = {parallel}", text, flags=re.M)
    assert a == 1 and b == 1, (a, b)
    copy = lab.lab_gate(gate, label=f"timing_{str(spacing).replace('.', '')}_{parallel}")
    copy.write_bytes(text.encode("utf-8"))
    return copy


def seconds(gate, package, fast):
    cmd = [sys.executable, "-B", str(gate), str(package)] + (["--fast"] if fast else [])
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=lab.child_env())
    took = time.perf_counter() - started
    return took, proc.stdout.strip().splitlines()[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gate")
    parser.add_argument("package")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(f"gate: {args.gate}\npackage: {args.package}")
    print("| start spacing, side by side | full: seconds | per edit (--fast): seconds | verdicts |")
    print("|---|---|---|---|")
    for spacing, parallel in SETTINGS:
        copy = variant(args.gate, spacing, parallel)
        seconds(copy, args.package, False)              # writes the datasets under this stamp
        full, fast, said = [], [], set()
        for _ in range(args.repeats):
            took, last = seconds(copy, args.package, False)
            full.append(took), said.add(last)
            took, last = seconds(copy, args.package, True)
            fast.append(took), said.add(last)
        print(f"| {spacing} s, {parallel} | " + ", ".join(f"{v:.1f}" for v in full) + " | "
              + ", ".join(f"{v:.1f}" for v in fast) + f" | {'; '.join(sorted(said))} |",
              flush=True)
