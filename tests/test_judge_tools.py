"""T60: the judge's two other tools run on the application (spec 8.2; D56, D71, D72, D73).

Each test runs one script of `verify/` from the repo root in a process of its own, on
the package `mlprofessor`, and asserts that it exits 0. Nothing is imported from
`verify/`; the scripts write under `verify/artifacts/` only, as the gate does.
"""
import os
import subprocess

import mlp_helpers as h


def _run_tool(script, timeout):
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [h.PYTHON, "-B", script, "mlprofessor"], cwd=str(h.REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    printed = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(printed.splitlines()[-40:])
    assert proc.returncode == 0, f"{script} exited {proc.returncode}; its last lines:\n{tail}"


def test_the_perturbation_battery_holds():
    _run_tool("verify/battery.py", timeout=1500)


def test_the_generated_tables_agree_with_the_reference():
    _run_tool("verify/generated_tables.py", timeout=900)
