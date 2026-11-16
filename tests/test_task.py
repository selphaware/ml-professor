"""T4: `stages.decide_task` (spec step 4, D4, D21)."""
import numpy as np
import pandas as pd
import pytest


def _loaded_column(tmp_path, lines, name="y"):
    """Write the column beside a counter (so a blank is a blank cell, not a
    skipped blank line) and load it as the tool does."""
    from mlprofessor import stages

    path = tmp_path / "y.csv"
    body = "\n".join(f"{v},{i}" for i, v in enumerate(lines))
    path.write_text(f"{name},k\n{body}\n", encoding="utf-8", newline="")
    frame, _ = stages.load_table(str(path))
    return frame[name]


TWO_VALUE_SHAPES = {
    "zero_one": (["0", "1", "1", "0", "1", "0"] * 5, 1),
    "zero_one_blanks": (["0", "1", "", "0", "1", ""] * 5, 1),
    "yes_no": (["yes", "no", "no", "yes"] * 5, "yes"),
    "true_false": (["True", "False", "True", "False"] * 5, True),
    "two_floats": (["0.5", "2.25", "2.25", "0.5"] * 5, 2.25),
}


@pytest.mark.parametrize("shape", sorted(TWO_VALUE_SHAPES))
def test_f5_two_values_are_classification_whatever_the_type(tmp_path, shape):
    """F5: two distinct values are classification, even when loaded as float or bool."""
    from mlprofessor import stages

    lines, positive = TWO_VALUE_SHAPES[shape]
    column = _loaded_column(tmp_path, lines)
    if shape == "zero_one_blanks":
        assert column.isna().sum() == 10 and column.dtype.kind == "f"
    res = stages.decide_task(column)
    assert res["type"] == "classification"
    assert res["decided_by"] == "rule"
    assert res["n_distinct"] == 2
    assert len(res["classes"]) == 2
    assert res["positive_class"] == positive
    assert res["classes"][1] == positive
    assert res["warnings"] == []


def test_classes_sorted_positive_second():
    from mlprofessor import stages

    res = stages.decide_task(pd.Series(["yes", "no", "yes"]))
    assert list(res["classes"]) == ["no", "yes"] and res["positive_class"] == "yes"
    res = stages.decide_task(pd.Series([1, 0, 1, 0]))
    assert [int(c) for c in res["classes"]] == [0, 1] and res["positive_class"] == 1
    res = stages.decide_task(pd.Series([True, False, True]))
    assert [bool(c) for c in res["classes"]] == [False, True] and res["positive_class"] is not False
    assert bool(res["positive_class"]) is True


def test_many_fractional_values_are_regression():
    from mlprofessor import stages

    values = pd.Series(np.random.default_rng(0).normal(size=80))
    res = stages.decide_task(values)
    assert res["type"] == "regression"
    assert res["decided_by"] == "rule"
    assert res["n_distinct"] == values.nunique()
    assert res["classes"] is None and res["positive_class"] is None
    assert res["warnings"] == []


def test_whole_number_regression_warning(tmp_path):
    from mlprofessor import stages

    ints = np.random.default_rng(1).integers(0, 200, size=150)
    column = _loaded_column(tmp_path, [str(v) for v in ints])
    res = stages.decide_task(column)
    assert res["type"] == "regression"
    assert res["n_distinct"] == len(set(ints.tolist()))
    assert res["warnings"] == ["whole_number_regression_target"]


@pytest.mark.parametrize("k", [3, 10, 20])
def test_f5_few_distinct_numbers_are_ambiguous(k):
    """F5: 3 to 20 distinct numbers stop the run rather than guess."""
    from mlprofessor import errors, stages

    values = pd.Series(np.arange(200) % k)
    with pytest.raises(errors.TaskAmbiguousError):
        stages.decide_task(values)


def test_more_than_two_text_values_ambiguous():
    from mlprofessor import errors, stages

    with pytest.raises(errors.TaskAmbiguousError):
        stages.decide_task(pd.Series(["a", "b", "c"] * 30))


def test_declared_task_honoured():
    from mlprofessor import stages

    res = stages.decide_task(pd.Series([0, 1] * 20), task="regression")
    assert res["type"] == "regression" and res["decided_by"] == "declared"
    assert res["warnings"] == []
    res = stages.decide_task(pd.Series(["cat", "dog"] * 20), task="classification")
    assert res["type"] == "classification" and res["decided_by"] == "declared"
    assert res["positive_class"] == "dog"
    many = pd.Series(np.arange(100))
    res = stages.decide_task(many, task="regression")
    assert res["decided_by"] == "declared" and res["warnings"] == []


@pytest.mark.parametrize("values,task", [
    (["a", "b", "c"] * 10, "classification"),
    ([0, 1, 2] * 10, "classification"),
    (list(range(60)), "classification"),
    (["a", "b"] * 10, "regression"),
])
def test_declared_task_refused(values, task):
    from mlprofessor import errors, stages

    with pytest.raises(errors.TaskUnsupportedError):
        stages.decide_task(pd.Series(values), task=task)


@pytest.mark.parametrize("task", ["auto", "classification", "regression"])
def test_one_value_is_constant(task):
    from mlprofessor import errors, stages

    with pytest.raises(errors.TargetConstantError):
        stages.decide_task(pd.Series([3.0, np.nan, 3.0, 3.0, np.nan]), task=task)


def test_missing_values_ignored():
    from mlprofessor import stages

    res = stages.decide_task(pd.Series([0.0, 1.0, np.nan, 1.0, np.nan, 0.0]))
    assert res["type"] == "classification" and res["n_distinct"] == 2


def test_invalid_task_value():
    from mlprofessor import errors, stages

    with pytest.raises(errors.InputError):
        stages.decide_task(pd.Series([0, 1] * 10), task="multiclass")
