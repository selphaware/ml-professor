"""CP7, D59 and D65: which existing tests call profile() or the command line with
nothing said about time order or about entities?

    python -B count_calls.py <lab folder> <log.jsonl> [pytest args...]

Builds a throwaway project in <lab folder> (outside the repo): a copy of tests/, a
copy of the application whose `profile()` logs every call, README.md,
requirements.txt and examples/. Runs the suite there. One JSON line per call:
the test (PYTEST_CURRENT_TEST, inherited by a command-line subprocess), what was
passed as time_col and group_col, whether the run got as far as step 5, and how
it ended. Nothing is written inside the repo.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]

INIT = '''"""mlprofessor, instrumented copy for the CP7 count (not the application)."""
from mlprofessor.result import ProfileResult

_REACHED = []


def profile(data, target, task="auto", time_col=None, gap=0,
            profile="full", seed=0, out_dir=None, n_jobs=-1, duplicates="error",
            group_col=None):
    """Measure how predictable `target` is in `data`. Returns a ProfileResult."""
    import json
    import os
    import sys
    import traceback

    from mlprofessor.pipeline import run

    here = os.sep + "tests" + os.sep
    stack = [[os.path.basename(f.filename), f.name, f.lineno]
             for f in traceback.extract_stack() if here in f.filename]
    record = {"test": os.environ.get("PYTEST_CURRENT_TEST"),
              "time_col": repr(time_col), "group_col": repr(group_col),
              "duplicates": repr(duplicates), "profile": repr(profile),
              "cli": None if stack else sys.argv[1:], "stack": stack}
    del _REACHED[:]
    try:
        result = run(data, target, task=task, time_col=time_col, gap=gap, profile=profile,
                     seed=seed, out_dir=out_dir, n_jobs=n_jobs, duplicates=duplicates,
                     group_col=group_col)
        record["outcome"] = "returned"
        record["scheme"] = result.to_dict()["validation"]["scheme"]
        record["warnings"] = sorted({w["code"] for w in result.to_dict()["warnings"]})
        return result
    except BaseException as exc:
        record["outcome"] = getattr(exc, "code", None) or type(exc).__name__
        raise
    finally:
        record["reached_step5"] = bool(_REACHED)
        with open(os.environ["MLPROF_COUNT_LOG"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\\n")


__all__ = ["profile", "ProfileResult"]
'''

MARK_OLD = ("    dates = _date_columns(roles)\n"
            "    if time_col is None and not time_order_stated and dates:\n")
MARK_NEW = ("    dates = _date_columns(roles)\n"
            "    import mlprofessor as _m\n"
            "    _m._REACHED.append(1)\n"
            "    if time_col is None and not time_order_stated and dates:\n")


def _ignore(folder, names):
    return [n for n in names if n in ("__pycache__", ".pytest_cache")]


def main(lab, log, extra):
    lab, log = Path(lab).resolve(), Path(log).resolve()
    assert REPO not in lab.parents and lab != REPO
    if lab.exists():
        shutil.rmtree(lab)
    shutil.copytree(REPO / "tests", lab / "tests", ignore=_ignore)
    shutil.copytree(REPO / "mlprofessor", lab / "mlprofessor", ignore=_ignore)
    shutil.copytree(REPO / "examples", lab / "examples", ignore=_ignore)
    for name in ("README.md", "requirements.txt"):
        shutil.copyfile(REPO / name, lab / name)
    (lab / "mlprofessor" / "__init__.py").write_text(INIT, encoding="utf-8")
    pipeline = lab / "mlprofessor" / "pipeline.py"
    text = pipeline.read_text(encoding="utf-8")
    assert text.count(MARK_OLD) == 1
    pipeline.write_text(text.replace(MARK_OLD, MARK_NEW), encoding="utf-8")
    if log.exists():
        log.unlink()
    env = dict(os.environ)
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONSTARTUP", "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", MLPROF_COUNT_LOG=str(log))
    cmd = [sys.executable, "-B", "-m", "pytest", "-c", "tests/pytest.ini", "--rootdir", ".",
           "--confcutdir", "tests", "-p", "no:cacheprovider", "-q",
           "--junitxml", str(lab / "junit.xml"), *(extra or ["tests"])]
    proc = subprocess.run(cmd, cwd=str(lab), env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    print(proc.stdout[-3000:])
    print(proc.stderr[-1500:])
    print("exit", proc.returncode)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
