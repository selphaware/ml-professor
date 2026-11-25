"""T60: the judge's two other tools run on the application (spec 8.2; D56, D71, D72, D73,
D74).

Each test runs one script of `verify/` from the repo root in a process of its own, on
the package `mlprofessor`, with `--json FILE` (FILE in a temporary folder of the
session). Until D74 each test asserted that its tool exits 0. Since D74 the judge
asserts rules the application of the seventh pass does not have yet, so each test
accepts exit 0, or exit 1 when every pair (every table) that fails is one the tasks of
the eighth pass own (`cp9_helpers.BATTERY_OWNED`, `cp9_helpers.TABLES_OWNED`); any
other exit code fails, and so does a failure that is not on the list (D59). A tool is
run once in a session: `run_tool` keeps what the run gave, and the strict form,
`tests/test_judge_tools_pass8.py` (T68), asks it for the same run. Nothing is imported
from `verify/`; the scripts write under `verify/artifacts/` only, as the gate does.
"""
import json
import os
import subprocess
from pathlib import Path

import cp9_helpers as q
import mlp_helpers as h

BATTERY = ("verify/battery.py", 3600)
TABLES = ("verify/generated_tables.py", 3600)
_RUNS = {}


def run_tool(script, timeout, tmp_path_factory):
    """`(exit code, report, last lines)` of one tool run with `--json`, on the package
    `mlprofessor`, from the repo root. The report is None when the tool wrote none. The
    tool is run once in a session, whichever test asks first."""
    if script not in _RUNS:
        report = tmp_path_factory.mktemp("judge_tools") / (Path(script).stem + ".json")
        env = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME")}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [h.PYTHON, "-B", script, "mlprofessor", "--json", str(report)],
            cwd=str(h.REPO_ROOT), env=env, capture_output=True, text=True, timeout=timeout,
        )
        printed = (proc.stdout or "") + (proc.stderr or "")
        tail = "\n".join(printed.splitlines()[-40:])
        doc = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else None
        _RUNS[script] = (proc.returncode, doc, tail)
    return _RUNS[script]


def _owned_or_clean(script, timeout, tmp_path_factory):
    """None when the tool exits 0; its report when it exits 1; any other code fails."""
    code, doc, tail = run_tool(script, timeout, tmp_path_factory)
    assert code in (0, 1), f"{script} exited {code}; its last lines:\n{tail}"
    if code == 0:
        return None
    assert doc is not None, f"{script} exited 1 and wrote no report; its last lines:\n{tail}"
    return doc


def test_the_perturbation_battery_holds(tmp_path_factory):
    doc = _owned_or_clean(*BATTERY, tmp_path_factory)
    if doc is None:
        return
    failing = {f"{pair['run']}:{pair['change']}" for pair in doc["pairs"]
               if pair["class"] == "fails"}
    assert failing, "verify/battery.py exited 1 and its report names no failing pair"
    other = sorted(failing - q.BATTERY_OWNED)
    assert not other, (f"{len(other)} pair(s) fail that the eighth pass does not own "
                       f"(of {len(failing)} failing): {other[:20]}")


def test_the_generated_tables_agree_with_the_reference(tmp_path_factory):
    doc = _owned_or_clean(*TABLES, tmp_path_factory)
    if doc is None:
        return
    failing = {int(entry["number"]) for entry in doc["tables_with_findings"]
               if entry["class"] == "fails"}
    assert failing, "verify/generated_tables.py exited 1 and its report names no failing table"
    assert len(failing) == doc["tables_by_class"]["fails"]
    other = sorted(failing - q.TABLES_OWNED)
    assert not other, (f"{len(other)} table(s) fail that the eighth pass does not own "
                       f"(of {len(failing)} failing): {other[:20]}")
