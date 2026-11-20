"""CP5, section A (D49, D53): which locked tests profile a table in which holdout rows
have a copy among the training rows?

    python -B copies_in_tests.py <work folder outside the repo>

Builds a throwaway project there (a copy of tests/, the reference implementation
as the package, the README and what it names), with two changes to the package:
the rule on copies never stops a run, and each count it makes is appended to a
log with the name of the test that is running. Then it runs the whole suite and
prints every test in which a holdout row had a copy, with the count, the holdout
rows and whether the rule of D53 (more than 1%, and at least 10 rows) is met.

This is what D53 rests on: a rule on the share alone would stop two locked
tests that expect a finished run. Nothing is written inside the repo.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
WORK = Path(sys.argv[1]).resolve()
assert REPO not in WORK.parents and WORK != REPO, "the work folder must be outside the repo"

ANCHOR = "        copy_share = n_copies / len(holdout)\n"
LOG_LINES = (
    "        if os.environ.get('CP5_COPIES_LOG'):\n"
    "            with open(os.environ['CP5_COPIES_LOG'], 'a', encoding='utf-8') as _fh:\n"
    "                _fh.write(json.dumps({'test': os.environ.get('PYTEST_CURRENT_TEST', ''),\n"
    "                                      'copies': n_copies, 'holdout': int(len(holdout))})\n"
    "                          + '\\n')\n"
)
STOP = '            if duplicates == "error" and BROKEN != "F9":'


def build():
    project = WORK / "copies_in_tests"
    if project.exists():
        shutil.rmtree(project)
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache")
    shutil.copytree(REPO / "tests", project / "tests", ignore=ignore)
    shutil.copytree(REPO / "verify" / "reference" / "mlprofessor", project / "mlprofessor",
                    ignore=ignore)
    for name in ("README.md", "requirements.txt"):
        shutil.copyfile(REPO / name, project / name)
    shutil.copytree(REPO / "examples", project / "examples", ignore=ignore)
    core = project / "mlprofessor" / "core.py"
    text = core.read_text(encoding="utf-8")
    assert text.count(ANCHOR) == 1 and text.count(STOP) == 1, "core.py is not as this script expects"
    text = text.replace(ANCHOR, ANCHOR + LOG_LINES).replace(STOP, "            if False:")
    core.write_text(text, encoding="utf-8")
    return project


def main():
    project = build()
    log = WORK / "copies_in_tests.jsonl"
    if log.exists():
        log.unlink()
    env = dict(os.environ)
    for name in ("PYTEST_ADDOPTS", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", CP5_COPIES_LOG=str(log))
    proc = subprocess.run([sys.executable, "-B", "-m", "pytest", "-c", "tests/pytest.ini",
                           "--rootdir", ".", "--confcutdir", "tests", "-p", "no:cacheprovider",
                           "-q", "tests"], cwd=str(project), env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    print("pytest:", (proc.stdout or "").strip().splitlines()[-1])
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    print(f"{len(events)} runs under the random scheme reached the count")
    worst = {}
    for event in events:
        if event["copies"]:
            name = event["test"].split(" ")[0].replace("tests/", "")
            if name not in worst or event["copies"] / event["holdout"] > \
                    worst[name]["copies"] / worst[name]["holdout"]:
                worst[name] = event
    print(f"{len(worst)} test(s) in which a holdout row had a copy among the training rows:")
    for name, event in sorted(worst.items(), key=lambda kv: -kv[1]["copies"] / kv[1]["holdout"]):
        share = event["copies"] / event["holdout"]
        met = event["copies"] >= 10 and share > 0.01
        print(f"  {event['copies']:4d} of {event['holdout']:4d} ({100 * share:6.2f}%)  "
              f"{'MEETS THE RULE' if met else 'below the rule'}  {name}")


if __name__ == "__main__":
    main()
