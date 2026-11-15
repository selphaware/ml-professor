"""Run the judge: the gate and the test suite.

One implementation, used by the Stop hook, the PostToolUse hook and the
driver, so all three always agree on what "passing" means.
Standard library only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

GATE_SCRIPT = "verify/smoke_test.py"
PLAN_FILE = "plan/tasks.json"
PYTEST_INI = "tests/pytest.ini"
TAIL_CHARS = 6000

PASS, FAIL, HARNESS = "pass", "fail", "harness_error"


@dataclass
class Result:
    status: str
    failing: list = field(default_factory=list)
    passed: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    output: str = ""

    @property
    def ok(self) -> bool:
        return self.status == PASS


def tail(text: str, limit: int = TAIL_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else "...[truncated]...\n" + text[-limit:]


def load_plan(root) -> dict | None:
    """The task plan, or None if it does not exist or cannot be read."""
    path = Path(root) / PLAN_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _clean_env() -> dict:
    env = dict(os.environ)
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONSTARTUP"):
        env.pop(name, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(cmd, root, timeout):
    try:
        proc = subprocess.run(
            cmd, cwd=str(root), env=_clean_env(), capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        out = out.decode("utf-8", "replace") if isinstance(out, bytes) else out
        return None, out + f"\nTIMEOUT after {timeout}s"
    except OSError as exc:
        return None, f"could not start {cmd[0]}: {exc}"


def _last_json_line(text: str) -> dict | None:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
    return None


def gate_list(root, python=None, timeout=60):
    """Check ids the gate declares. Returns (ids_with_docs, error_text)."""
    python = python or sys.executable
    rc, out = _run([python, GATE_SCRIPT, "--list", "--json"], root, timeout)
    data = _last_json_line(out)
    if rc != 0 or not data or "checks" not in data:
        return None, tail(out, 1500) or "gate --list produced no output"
    return {c["id"]: c for c in data["checks"]}, ""


def run_gate(root, deliverable, only=None, python=None, timeout=300, fast=False) -> Result:
    """Run the gate. ``only`` is a list of check ids, or None for all of them.

    ``fast`` leaves out the checks the gate marks slow (the per-edit profile).
    """
    python = python or sys.executable
    root = Path(root)
    if not (root / GATE_SCRIPT).is_file():
        return Result(HARNESS, output=f"HARNESS ERROR: {GATE_SCRIPT} does not exist")
    if only is not None and not only:
        return Result(PASS, output="no gate checks selected")
    cmd = [python, GATE_SCRIPT, str(deliverable), "--json"]
    if only:
        cmd += ["--only", ",".join(only)]
    if fast:
        cmd.append("--fast")
    rc, out = _run(cmd, root, timeout)
    data = _last_json_line(out)
    if rc is None:
        return Result(HARNESS, output="HARNESS ERROR: gate did not finish. " + tail(out))
    if rc not in (0, 1) or data is None:
        prefix = "" if "HARNESS ERROR" in out else "HARNESS ERROR: gate exited %s. " % rc
        return Result(HARNESS, output=prefix + tail(out))
    checks = data.get("checks", [])
    failing = [c["id"] for c in checks if not c.get("ok")]
    passed = [c["id"] for c in checks if c.get("ok")]
    status = PASS if rc == 0 and not failing else FAIL
    # Drop the machine-readable summary line: it repeats what the lines above say.
    human = "\n".join(l for l in out.splitlines() if not l.lstrip().startswith('{"gate"'))
    return Result(status, failing=failing, passed=passed, output=tail(human))


def run_tests(root, targets=None, python=None, timeout=900) -> Result:
    """Run pytest on ``targets`` (default: the whole tests/ directory).

    The run is pinned to tests/pytest.ini and cut off from any conftest.py
    outside tests/, so application code cannot reconfigure it.
    """
    python = python or sys.executable
    root = Path(root)
    if targets is not None and not targets:
        return Result(PASS, output="no tests selected")
    if not (root / PYTEST_INI).is_file():
        return Result(HARNESS, output=f"HARNESS ERROR: {PYTEST_INI} is missing (kit file)")
    targets = list(targets) if targets else ["tests"]
    missing = [t for t in targets if not (root / t.split("::")[0]).exists()]
    if missing:
        return Result(FAIL, failing=missing,
                      output="test target(s) do not exist: " + ", ".join(missing))
    fd, xml_path = tempfile.mkstemp(prefix="loop-junit-", suffix=".xml")
    os.close(fd)
    try:
        cmd = [python, "-m", "pytest", "-c", PYTEST_INI, "--rootdir", ".",
               "--confcutdir", "tests", "-p", "no:cacheprovider", "-q",
               "--junitxml", xml_path, *targets]
        rc, out = _run(cmd, root, timeout)
        if rc is None:
            return Result(HARNESS, output="HARNESS ERROR: tests did not finish. " + tail(out))
        if "No module named pytest" in out or rc in (3, 4):
            return Result(HARNESS, output="HARNESS ERROR: pytest could not run. " + tail(out))
        passed, failing, skipped = _read_junit(xml_path)
    finally:
        try:
            os.unlink(xml_path)
        except OSError:
            pass
    if rc == 5 or not (passed or failing or skipped):
        return Result(FAIL, failing=["<no tests collected>"],
                      output="no tests were collected. " + tail(out))
    if skipped:
        # A skipped test verifies nothing. Treat it as a failure, loudly.
        return Result(FAIL, failing=failing + [f"SKIPPED {s}" for s in skipped],
                      passed=passed, skipped=skipped,
                      output="skipped or xfailed tests are not allowed.\n" + tail(out))
    status = PASS if rc == 0 and not failing else FAIL
    if status == FAIL and not failing:
        failing = [f"<pytest exited {rc}>"]
    return Result(status, failing=failing, passed=passed, output=tail(out))


def _read_junit(xml_path):
    passed, failing, skipped = [], [], []
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, OSError):
        return passed, failing, skipped
    for case in tree.iter("testcase"):
        name = case.get("name", "?")
        cls = case.get("classname", "")
        test_id = f"{cls}::{name}" if cls else name
        kinds = {child.tag for child in case}
        if kinds & {"failure", "error"}:
            failing.append(test_id)
        elif "skipped" in kinds:
            skipped.append(test_id)
        else:
            passed.append(test_id)
    return passed, failing, skipped


def limit(root, name, default):
    """A value from limits in loop/loop.json, or ``default`` if it cannot be read."""
    try:
        cfg = json.loads((Path(root) / "loop" / "loop.json").read_text(encoding="utf-8"))
        return cfg.get("limits", {}).get(name, default)
    except (OSError, ValueError, AttributeError):
        return default


def read_accept():
    """Acceptance set the driver exported for this run, or None if unset.

    None means "no driver": hooks then judge against the full gate and suite.
    """
    raw = os.environ.get("LOOP_ACCEPT")
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return {"gate": list(data.get("gate", [])), "tests": list(data.get("tests", []))}
    except (ValueError, AttributeError):
        return None


def signature(gate: Result, tests: Result) -> str:
    """Stable fingerprint of a failure, used for the three-strikes rule."""
    return "gate:" + ",".join(sorted(gate.failing)) + "|tests:" + ",".join(sorted(tests.failing))
