"""Shared fixtures. Helpers live in `mlp_helpers.py` beside this file."""
import sys

# Nothing may be written inside tests/ at run time.
sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
for _p in (str(TESTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlp_helpers as h  # noqa: E402


@pytest.fixture
def run_cli():
    """Run the command line in a new process, one at a time."""
    return h.run_cli


@pytest.fixture(scope="session")
def full_runs(tmp_path_factory):
    """One full-profile run per task, shared by the full-profile tests.

    The regression table is read from a CSV so the command line can be
    compared with it.
    """
    import mlprofessor

    base = tmp_path_factory.mktemp("full_runs")
    clf = h.clf_frame(n=150, seed=11)
    reg = h.reg_frame(n=150, seed=12)
    reg_csv = base / "reg.csv"
    reg.to_csv(reg_csv, index=False)
    out = {
        "clf_frame": clf,
        "reg_frame": reg,
        "reg_csv": str(reg_csv),
        "clf_dir": base / "clf_out",
        "reg_dir": base / "reg_out",
    }
    out["clf"] = h.stated_profile(clf, "y", profile="full", n_jobs=1, seed=0,
                                     out_dir=out["clf_dir"])
    out["reg"] = h.stated_profile(str(reg_csv), "y", profile="full", n_jobs=1, seed=0,
                                     out_dir=out["reg_dir"])
    return out
