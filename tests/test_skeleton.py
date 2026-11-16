"""T1: the package imports and its surface has the pinned shape (spec 2)."""
import importlib
import importlib.metadata
import inspect
import re

import pytest

import mlp_helpers as h

ERRORS = [
    ("InputError", "input_error"),
    ("ColumnNotFoundError", "column_not_found"),
    ("DuplicateColumnsError", "duplicate_columns"),
    ("TargetConstantError", "target_constant"),
    ("TaskAmbiguousError", "task_ambiguous"),
    ("TaskUnsupportedError", "task_unsupported"),
    ("TimeColumnError", "time_column_invalid"),
    ("TooFewRowsError", "too_few_rows"),
    ("FastProfileError", "fast_profile_too_few_timestamps"),
    ("GapError", "gap_too_large"),
    ("NoFeaturesError", "no_features"),
    ("OneClassError", "one_class"),
    ("AllModelsFailedError", "all_models_failed"),
]


def test_package_surface_exists():
    import mlprofessor
    from mlprofessor import errors, extensions

    assert callable(mlprofessor.profile)
    assert isinstance(mlprofessor.ProfileResult, type)
    assert callable(extensions.use)
    stages = importlib.import_module("mlprofessor.stages")
    assert inspect.ismodule(stages)
    assert issubclass(errors.ProfileError, Exception)


def test_profile_signature_and_defaults():
    import mlprofessor

    params = inspect.signature(mlprofessor.profile).parameters
    expected = [
        ("data", inspect.Parameter.empty),
        ("target", inspect.Parameter.empty),
        ("task", "auto"),
        ("time_col", None),
        ("gap", 0),
        ("profile", "full"),
        ("seed", 0),
        ("out_dir", None),
        ("n_jobs", -1),
    ]
    assert [name for name, _ in expected] == list(params)
    for name, default in expected:
        assert params[name].default == default, name


@pytest.mark.parametrize("name,code", ERRORS)
def test_error_classes_and_codes(name, code):
    from mlprofessor import errors

    cls = getattr(errors, name)
    assert issubclass(cls, errors.ProfileError)
    assert cls.code == code


def test_help_exits_zero_and_lists_every_option():
    top = h.run_cli(["--help"])
    assert top.returncode == 0, top.stderr
    sub = h.run_cli(["profile", "--help"])
    assert sub.returncode == 0, sub.stderr
    for option in ["--data", "--target", "--out", "--task", "--time-col", "--gap",
                   "--profile", "--seed", "--n-jobs"]:
        assert option in sub.stdout, option


def test_requirements_pin_the_four_libraries():
    text = (h.REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    pins = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)\s*==\s*([A-Za-z0-9_.\-]+)", line)
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    for dist in ["pandas", "numpy", "scikit-learn", "threadpoolctl"]:
        assert dist in pins, f"{dist} is not pinned with =="
        assert pins[dist] == importlib.metadata.version(dist), dist
