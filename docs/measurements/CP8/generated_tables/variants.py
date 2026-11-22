"""Does the generated-table test see a fault? Each broken copy of the reference
against the good reference. Prototype evidence (CP8 section C2).

    python -B variants.py <tables> <jobs per run> <runs side by side> [case,case,...]

For every case of verify/mutation_check.py it builds a copy of the reference with
that one switch set, under variants/<case>/mlprofessor, and runs
generated_tables.py with that copy as the package and the good reference as the
reference. Prints one line a case: the tables with a counted finding, and the
findings by class. Nothing is written inside the repo.
"""
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "before" / "verify" / "reference" / "mlprofessor"
CASES = ([f"F{i}" for i in range(1, 10)] + ["F6b", "F6c", "F6d", "F7b", "F7c"]
         + ["F10", "F7d", "F8b"] + [f"F{i}" for i in range(11, 15)]
         + [f"M{i}" for i in range(1, 31)])


def one(case, tables, jobs):
    target = HERE / "variants" / case / "mlprofessor"
    if target.parent.exists():
        shutil.rmtree(target.parent)
    shutil.copytree(REFERENCE, target, ignore=shutil.ignore_patterns("__pycache__"))
    (target / "_variant.py").write_text(f"BROKEN = {case!r}\n", encoding="utf-8")
    out = HERE / "variants" / f"{case}.json"
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-B", str(HERE / "generated_tables.py"), str(target), "--reference",
         str(REFERENCE), "--tables", str(tables), "--jobs", str(jobs), "--tmp",
         str(HERE / "work"), "--json", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    (HERE / "variants" / f"{case}.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    seconds = time.perf_counter() - started
    try:
        doc = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return case, proc.returncode, None, None, seconds
    hit = [e["number"] for e in doc["tables_with_findings"]
           if any(f["class"] not in ("split", "open") for f in e["findings"])]
    return case, proc.returncode, hit, doc["findings_by_class"], seconds


def main():
    tables, jobs, side = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    cases = sys.argv[4].split(",") if len(sys.argv) > 4 else CASES
    (HERE / "variants").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=side) as pool:
        for case, rc, hit, classes, seconds in pool.map(lambda c: one(c, tables, jobs), cases):
            if hit is None:
                print(f"{case:<5} exit {rc}: no report", flush=True)
                continue
            print(f"{case:<5} exit {rc}: {len(hit):>3} of {tables} tables with a counted finding "
                  f"{hit[:6]}{'...' if len(hit) > 6 else ''}  {classes}  [{seconds:.0f} s]",
                  flush=True)


if __name__ == "__main__":
    main()
