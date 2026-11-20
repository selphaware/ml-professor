"""CP4: are the tests the plan demands red or green today, and do the guards bite?

    python -B prototypes.py <folder> [case ...]

<folder> holds prototype test files (and helper modules) written from the
"Third pass" section of plan/test-plan.md. They are not kept in the repo: the
tests stage writes the real ones. Each case is a package to judge them against:

    app                 the application, the folder `mlprofessor` at the repo root
    good                the reference implementation, verify/reference
    F6b, M6, ...        the reference with that one switch set (see its _variant.py)

For each case a throwaway project is built outside the repo, as
verify/mutation_check.py builds its own: a copy of tests/ with the prototype
files added, the package as the folder `mlprofessor`, README.md and
requirements.txt. Only the prototype files are run. One line is printed per
test, with its outcome under every case. Nothing is written inside the repo.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]


def _mutation_check():
    spec = importlib.util.spec_from_file_location(
        "mutation_check", str(REPO / "verify" / "mutation_check.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _mutation_check()


def build(root, case, folder):
    if case == "app":
        project = root / "app"
        shutil.copytree(REPO / "tests", project / "tests", ignore=M._ignore)
        shutil.copytree(REPO / "mlprofessor", project / "mlprofessor", ignore=M._ignore)
        for name in ("README.md", "requirements.txt"):
            shutil.copyfile(REPO / name, project / name)
        M.place_readme_files(project, set())
    else:
        project = M.build(root, case, set())    # the reference, with its switch and a probe
    names = []
    for path in sorted(Path(folder).glob("*.py")):
        shutil.copyfile(path, project / "tests" / path.name)
        if path.name.startswith("test_"):
            names.append(path.name)
    return project, names


def run(root, case, folder):
    project, names = build(root, case, folder)
    report = root / f"{case}.junit.xml"
    env = dict(os.environ)
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONSTARTUP", "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    targets = [f"tests/{name}" for name in names]
    if case != "app":
        targets.append(f"tests/{M.PROBE}")      # fails unless this copy is the one judged
    cmd = [sys.executable, "-m", "pytest", "-c", "tests/pytest.ini", "--rootdir", ".",
           "--confcutdir", "tests", "-p", "no:cacheprovider", "-q", "--junitxml", str(report),
           *targets]
    proc = subprocess.run(cmd, cwd=str(project), env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=1800)
    (root / f"{case}.stdout.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    outcome = {}
    for item in ET.parse(report).iter("testcase"):
        module = item.get("classname", "").split(".")[-1]
        kinds = {child.tag for child in item}
        bad = kinds & {"failure", "error", "skipped"}
        outcome[f"{module}.py::{item.get('name', '?')}"] = "FAILED" if bad else "passed"
    probe = [t for t in outcome if t.startswith(M.PROBE)]
    if case != "app" and (not probe or any(outcome[t] != "passed" for t in probe)):
        raise SystemExit(f"{case}: the prototypes did not judge the copy meant for them")
    return {t: o for t, o in outcome.items() if not t.startswith(M.PROBE)}


def main(folder, cases):
    root = Path(tempfile.mkdtemp(prefix="mlprof-prototypes-")).resolve()
    if root == REPO or REPO in root.parents:
        raise SystemExit(f"the temporary folder {root} is inside the repo")
    print(f"prototypes from {folder}; projects under {root}")
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = dict(zip(cases, pool.map(lambda case: run(root, case, folder), cases)))
    tests = sorted({t for outcome in results.values() for t in outcome})
    width = max(len(t) for t in tests)
    for test in tests:
        print(f"{test:<{width}}  " + "  ".join(
            f"{case}: {results[case].get(test, '(absent)'):<6}" for case in cases))
    for case in cases:
        failed = sum(o == "FAILED" for o in results[case].values())
        print(f"{case}: {failed} of {len(results[case])} failing")
    print(f"pytest output of each case: {root}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:] or ["app", "good"])
