"""Shared by the CP8 measurement scripts: a lab folder outside the repo.

Nothing here writes inside the repo. The lab is MLPROF_LAB_WORK, or `mlprof-lab`
in the system's temporary folder, then `cp8/`. A copy of the judge placed there
runs as the judge does, with its artifacts in the lab, so that
verify/artifacts/runs/ in the repo goes on holding the last gate run on the build.
"""
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
LAB = Path(os.environ.get("MLPROF_LAB_WORK") or Path(tempfile.gettempdir()) / "mlprof-lab") / "cp8"
SKIP = shutil.ignore_patterns("__pycache__", ".pytest_cache", "artifacts")


def fixture(name, reference=None):
    """A copy of the reference (the repo's, or the folder given) carrying one named
    fault, or none for `good`. Returns the package folder."""
    source = Path(reference) if reference else REPO / "verify" / "reference" / "mlprofessor"
    target = LAB / "fixtures" / name / "mlprofessor"
    if target.parent.exists():
        shutil.rmtree(target.parent)
    shutil.copytree(source, target, ignore=SKIP)
    broken = None if name == "good" else name
    (target / "_variant.py").write_text(f"BROKEN = {broken!r}\n", encoding="utf-8")
    return target


def lab_gate(gate_file=None, label="gate"):
    """A copy of a gate script, with gatekit beside it, under the lab. Returns the copy."""
    source = Path(gate_file) if gate_file else REPO / "verify" / "smoke_test.py"
    folder = LAB / label / "verify"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, folder / "smoke_test.py")
    shutil.copyfile(REPO / "verify" / "gatekit.py", folder / "gatekit.py")
    if not (folder / "reference").exists():
        shutil.copytree(REPO / "verify" / "reference", folder / "reference", ignore=SKIP)
    return folder / "smoke_test.py"


def gate_module():
    """The repo's gate script as a module, for its dataset generator."""
    sys.path.insert(0, str(REPO / "verify"))
    path = REPO / "verify" / "smoke_test.py"
    spec = importlib.util.spec_from_file_location("mlprof_gate", str(path))
    module = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [str(path)]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


def child_env():
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    return env
