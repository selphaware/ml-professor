"""Does the test suite catch what it is there to catch? (D39)

The gate's self-test proves the gate against the reference implementation
broken in thirty-one ways (F1 to F22, and the other forms F6b, F6c, F6d, F7b,
F7c, F7d, F8b, F12b and F14b: D46, D49, D50, D57, D58, D62 to D66, D69 to
D72). This script proves the test suite the same way, against those thirty-one
and against thirty-nine smaller faults no gate check is relied on to catch (M1
to M39; verify/reference/mlprofessor/_variant.py lists them all).

For the known-good reference and for each variant it builds a throwaway
project outside the repo and runs the whole suite there:

    tests/              a copy of the repo's
    mlprofessor/        the reference implementation with that one switch set
    README.md, requirements.txt, and every file the README names

and prints one line per case: how many tests fail, and up to five of them.

    python verify/mutation_check.py [--jobs N] [--only good,M1,...] [--keep]
                                    [--overlay FOLDER]

`--overlay FOLDER` copies every `.py` file of FOLDER into each project's copy of
tests/ after the repo's own, replacing a file of the same name. It is for a
harness session, which may not write under tests/: the tests it plans can be
proved as prototypes before the tests stage writes the real files. The verdict
that counts is the one without it.

Exit codes
    0  the good reference fails nothing, and every variant fails at least one
       test that the good reference passes
    1  otherwise: a variant no test catches, or a test the good reference fails
    3  HARNESS ERROR: a project could not be built or its suite could not run

Four things to know.

- The suite must judge the copy, not the application. tests/conftest.py puts
  the folder above tests/ first on the import path, and at the repo root that
  folder holds the application, so PYTHONPATH cannot point the suite at the
  reference. Each project is therefore a folder of its own, and one extra test
  is added to its copy of tests/ that fails unless `mlprofessor` was imported
  from that project with the expected switch. It is not counted.
- The tests of README.md (files named tests/test_readme*.py) are run and
  shown on a line of their own, and do not decide the exit code: the README is
  the application's file, the same in every case, so those tests say nothing
  about a variant.
- A data file the README names is copied from the repo. One under
  verify/artifacts/data/ is written by the gate's own generator instead,
  because that folder exists only after the gate has run.
- tests/test_judge_tools.py is not run here (D72). Its tests run the judge's
  two other tools, verify/battery.py and verify/generated_tables.py, on the
  package beside them: they need verify/, which a throwaway project does not
  have, and they judge the application against the reference, not a variant
  of the reference. To see what those tools make of a variant, run each on a
  copy of it by hand.

Run it one at a time. It writes nothing inside the repo. Standard library only.
"""
import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REFERENCE = HERE / "reference" / "mlprofessor"
GATE_FILE = HERE / "smoke_test.py"
GATE_DATA = "verify/artifacts/data/"
CASES = (["good"] + [f"F{i}" for i in range(1, 10)] + ["F6b", "F6c", "F6d", "F7b", "F7c"]
         + ["F10", "F7d", "F8b"] + [f"F{i}" for i in range(11, 15)]
         + ["F12b", "F14b"] + [f"F{i}" for i in range(15, 23)]
         + [f"M{i}" for i in range(1, 40)])
PROBE = "test_mutation_check_probe.py"
# Run by the loop against the application, never against a variant (see the docstring).
NOT_RUN_HERE = "tests/test_judge_tools.py"
README_TESTS = re.compile(r"^test_readme\w*\.py::")
RUN_SECONDS = 1800
EXIT_OK, EXIT_FAIL, EXIT_HARNESS = 0, 1, 3

PROBE_TEXT = '''"""Added by the mutation check to its own copy of tests/. Not part of the suite."""
from pathlib import Path


def test_the_suite_is_judging_this_copy():
    import mlprofessor
    from mlprofessor import _variant

    here = Path(__file__).resolve().parents[1] / "mlprofessor"
    assert Path(mlprofessor.__file__).resolve().parent == here, mlprofessor.__file__
    assert _variant.BROKEN == {broken!r}
'''


class Harness(Exception):
    """The check itself could not run. Never a verdict on the suite."""


def _ignore(folder, names):
    return [n for n in names if n in ("__pycache__", ".pytest_cache")]


def readme_files(text):
    """Relative paths the README names, each once, in the order they appear."""
    found = []
    for raw in re.findall(r"(?<![\w./\\:-])[\w.-]+(?:[/\\][\w.-]+)+", text):
        rel = raw.replace("\\", "/").rstrip(".")
        if rel not in found:
            found.append(rel)
    return found


def _gate_module():
    """The gate script as a module, for its dataset generator."""
    sys.path.insert(0, str(HERE))               # gatekit
    spec = importlib.util.spec_from_file_location("mlprof_gate", str(GATE_FILE))
    module = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [str(GATE_FILE)]                 # gatekit reads the project root from it
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


def place_readme_files(project, notes):
    """Copy what the README names into `project`, at the same relative paths."""
    gate = None
    for rel in readme_files((REPO / "README.md").read_text(encoding="utf-8")):
        top = rel.split("/", 1)[0].lower()
        if top in ("mlprofessor", "tests"):     # the project has its own
            continue
        target = project / rel
        if rel.lower().startswith(GATE_DATA) and rel.lower().endswith(".csv"):
            name = Path(rel).stem
            gate = gate or _gate_module()
            if name in gate.DATASETS:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(gate.dataset_text(name).encode("utf-8"))
                notes.add(f"{rel}: written by the gate's generator (the README names a file "
                          "that exists only after the gate has run)")
                continue
        source = REPO / rel
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            notes.add(f"{rel}: copied from the repo")


def build(root, case, notes, overlay=None):
    """One throwaway project. Returns its folder."""
    project = root / case
    shutil.copytree(REPO / "tests", project / "tests", ignore=_ignore)
    if overlay is not None:
        for path in sorted(Path(overlay).glob("*.py")):
            replaced = (project / "tests" / path.name).exists()
            shutil.copyfile(path, project / "tests" / path.name)
            notes.add(f"overlay: tests/{path.name} "
                      + ("replaced by" if replaced else "added from") + f" {overlay}")
    shutil.copytree(REFERENCE, project / "mlprofessor", ignore=_ignore)
    broken = None if case == "good" else case
    (project / "mlprofessor" / "_variant.py").write_text(f"BROKEN = {broken!r}\n",
                                                         encoding="utf-8")
    (project / "tests" / PROBE).write_text(PROBE_TEXT.format(broken=broken), encoding="utf-8")
    for name in ("README.md", "requirements.txt"):
        shutil.copyfile(REPO / name, project / name)
    place_readme_files(project, notes)
    return project


def run_suite(project, case):
    """Run the whole suite in `project`. Returns {test id: 'passed' | 'failed'}."""
    report = project.parent / f"{case}.junit.xml"
    env = dict(os.environ)
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONSTARTUP", "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-m", "pytest", "-c", "tests/pytest.ini", "--rootdir", ".",
           "--confcutdir", "tests", "-p", "no:cacheprovider", "-q", "--junitxml", str(report),
           "--ignore", NOT_RUN_HERE, "tests"]
    try:
        proc = subprocess.run(cmd, cwd=str(project), env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=RUN_SECONDS)
    except subprocess.TimeoutExpired:
        raise Harness(f"{case}: the suite did not finish within {RUN_SECONDS} seconds")
    except OSError as exc:
        raise Harness(f"{case}: could not start {sys.executable}: {exc}")
    tail = (proc.stdout or "")[-1500:] + (proc.stderr or "")[-500:]
    if proc.returncode not in (0, 1) or not report.is_file():
        raise Harness(f"{case}: pytest exited {proc.returncode} without judging the suite\n{tail}")
    outcome = {}
    for item in ET.parse(report).iter("testcase"):
        module = item.get("classname", "").split(".")[-1]
        test_id = f"{module}.py::{item.get('name', '?')}"
        kinds = {child.tag for child in item}
        # a skipped test proves nothing, so it counts as failing
        outcome[test_id] = "failed" if kinds & {"failure", "error", "skipped"} else "passed"
    probe = [t for t in outcome if t.startswith(PROBE)]
    if not probe or any(outcome[t] != "passed" for t in probe):
        raise Harness(f"{case}: the suite did not import `mlprofessor` from {project} with the "
                      f"switch {case!r}, so its result says nothing about that variant\n{tail}")
    return {t: o for t, o in outcome.items() if not t.startswith(PROBE)}


def one_case(root, case, notes, overlay=None):
    started = time.perf_counter()
    try:
        outcome = run_suite(build(root, case, notes, overlay), case)
    except Harness as exc:
        return case, None, str(exc), time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001  (a copy failing, a report that will not parse)
        return case, None, f"{case}: {type(exc).__name__}: {exc}", time.perf_counter() - started
    return case, outcome, "", time.perf_counter() - started


def shown(tests, limit=5):
    names = sorted(tests)
    more = f" (and {len(names) - limit} more)" if len(names) > limit else ""
    return ", ".join(names[:limit]) + more


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="Run the suite against the reference "
                                                 "implementation and each broken variant of it.")
    parser.add_argument("--jobs", type=int, default=4, help="cases run side by side (default 4)")
    parser.add_argument("--only", default=None, help="comma-separated cases, e.g. good,M1,M2")
    parser.add_argument("--keep", action="store_true", help="keep the throwaway projects")
    parser.add_argument("--overlay", default=None,
                        help="a folder of .py files copied over each project's copy of tests/ "
                             "(prototypes of planned tests; see the docstring)")
    args = parser.parse_args(argv)
    if args.overlay is not None and not Path(args.overlay).is_dir():
        print(f"HARNESS ERROR: --overlay {args.overlay} is not a folder")
        return EXIT_HARNESS
    cases = CASES if not args.only else [c.strip() for c in args.only.split(",") if c.strip()]
    unknown = [c for c in cases if c not in CASES]
    if unknown or args.jobs < 1:
        print(f"HARNESS ERROR: unknown case(s) {unknown}; the cases are {', '.join(CASES)}"
              if unknown else "HARNESS ERROR: --jobs must be at least 1")
        return EXIT_HARNESS
    if "good" not in cases:
        cases = ["good"] + cases                # every variant is read against it
    if not (REFERENCE / "core.py").is_file() or not (REPO / "tests").is_dir():
        print(f"HARNESS ERROR: {REFERENCE} or {REPO / 'tests'} is missing")
        return EXIT_HARNESS

    root = Path(tempfile.mkdtemp(prefix="mlprof-mutation-")).resolve()
    if root == REPO or REPO in root.parents:
        print(f"HARNESS ERROR: the temporary folder {root} is inside the repo; set TEMP to a "
              "folder outside it")
        return EXIT_HARNESS
    notes, results, problems = set(), {}, []
    print(f"mutation check: {len(cases)} case(s), {args.jobs} at a time, projects under {root}")
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for case, outcome, problem, seconds in pool.map(
                    lambda c: one_case(root, c, notes, args.overlay), cases):
                if outcome is None:
                    problems.append(problem)
                else:
                    results[case] = (outcome, seconds)
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
    for note in sorted(notes):
        print(f"  note: {note}")
    if (REPO / NOT_RUN_HERE).is_file() or (args.overlay is not None
                                           and (Path(args.overlay) / Path(NOT_RUN_HERE).name)
                                           .is_file()):
        print(f"  note: {NOT_RUN_HERE} is not run here: it runs the judge's other tools on "
              "the application")
    if problems:
        for problem in problems:
            print(f"HARNESS ERROR - this says nothing about the suite: {problem}")
        return EXIT_HARNESS

    good, _ = results["good"]
    readme = {t for t in good if README_TESTS.match(t)}
    counted = set(good) - readme
    good_failing = {t for t in counted if good[t] == "failed"}
    verdict = EXIT_OK
    print(f"good   fails {len(good_failing)} of {len(counted)}"
          + (f": {shown(good_failing)}" if good_failing else "")
          + f"   [{results['good'][1]:.0f} s]")
    if good_failing:
        verdict = EXIT_FAIL
    for case in cases:
        if case == "good":
            continue
        outcome, seconds = results[case]
        # caught: a test that passes on the good reference and fails on this variant
        caught = {t for t in counted if good[t] == "passed" and outcome.get(t) != "passed"}
        line = f"{case:<6} fails {len(caught)}"
        if caught:
            line += f": {shown(caught)}"
        else:
            line += "   NOT CAUGHT: no test fails on this variant"
            verdict = EXIT_FAIL
        print(f"{line}   [{seconds:.0f} s]")
    readme_failing = sorted(t for t in readme if good[t] == "failed")
    print(f"README tests (not counted: they judge README.md, the same file in every case): "
          f"{len(readme_failing)} of {len(readme)} failing"
          + (f": {shown(readme_failing)}" if readme_failing else ""))
    if verdict == EXIT_OK:
        print(f"MUTATION CHECK: PASS (the good reference fails nothing; each of "
              f"{len(cases) - 1} variant(s) fails at least one test)")
    else:
        print("MUTATION CHECK: FAIL")
    return verdict


if __name__ == "__main__":
    sys.exit(main())
