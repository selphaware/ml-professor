"""T63: the line beside the leak screen's tree, and the direct reading of a class (spec
step 8, 4.3, 7; D74 point 2).

A copy of a regression target is excluded whatever its scale and under every scheme,
time order included: the tree cannot follow a target beyond the rows it was fitted on,
and sees no split in a column whose values differ by less than about 1e-7. A copy of
a class is excluded when a fold holds fewer than five rows of one class. Every run is
`cp9_helpers.screened`.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

import cp7_helpers as s
import cp9_helpers as q
import mlp_helpers as h

COPIES = {
    "exact": lambda y: y.copy(),
    "rescaled": lambda y: 3.7 * y + 12.0,
    "negative": lambda y: -2.0 * y + 5.0,
    "scale_1e-9": lambda y: 1e-9 * y,
}


def _assert_copy_is_out(result, name="copy"):
    record = q.column(result, name)
    assert record["role"] == "excluded" and record["reason"] == "suspected_leak", record
    assert record["leak_rule"] == "predicts_target", record
    assert len(q.told(result, "suspected_leak", name)) == 1, s.codes(result)
    assert not q.told(result, "strong_single_column", name)
    assert q.is_feature(q.column(result, "n1")), q.column(result, "n1")
    assert q.leaks(result) == [name], q.leaks(result)


def _target(frame):
    return frame["y"].to_numpy(dtype=float)


@pytest.mark.parametrize("scheme,copy", list(itertools.product(q.SCHEMES, COPIES)))
def test_a_copy_of_the_target_is_excluded_whatever_its_scale(scheme, copy):
    """F24: a rescaled copy of a trending target is a feature under time order, and a copy
    scaled by 1e-9 under every scheme."""
    frame, statements = q.scheme_table(scheme, seed=8301)
    frame = q.with_column(frame, "copy", COPIES[copy](_target(frame)))
    _assert_copy_is_out(q.screened(frame, statements))


@pytest.mark.parametrize("scheme,cell", list(itertools.product(q.SCHEMES, [999999.0, -999.0])))
def test_one_cell_far_off_in_the_copy_does_not_keep_it(scheme, cell):
    """M41: the screen's line is fitted so that one cell far off cannot turn it."""
    frame, statements = q.scheme_table(scheme, seed=8311)
    clean = q.screened(frame, statements)
    row = q.training_rows(clean, frame)[40]
    values = COPIES["rescaled"](_target(frame))
    values[row] = cell
    _assert_copy_is_out(q.screened(q.with_column(frame, "copy", values), statements))


@pytest.mark.parametrize("scheme", q.SCHEMES)
def test_a_copy_with_empty_cells_is_excluded(scheme):
    """A rescaled copy with two empty cells is excluded under every scheme."""
    frame, statements = q.scheme_table(scheme, seed=8321)
    values = COPIES["rescaled"](_target(frame))
    values[np.random.default_rng(8321).choice(len(frame), size=2, replace=False)] = np.nan
    result = q.screened(q.with_column(frame, "copy", values), statements)
    _assert_copy_is_out(result)


@pytest.mark.parametrize("scheme", q.SCHEMES)
def test_columns_that_do_not_follow_the_target_stay_features(scheme):
    """A table with no copy: no leak and no `strong_single_column`."""
    frame, statements = q.scheme_table(scheme, seed=8331)
    result = q.screened(frame, statements)
    assert q.leaks(result) == []
    assert not q.told(result, "strong_single_column")
    assert q.is_feature(q.column(result, "n1"))


def test_a_driver_known_before_the_outcome_is_kept_under_time_order():
    """A walk as the feature and the walk plus noise of 0.5 as the target: a feature."""
    rng = np.random.default_rng(8341)
    driver = np.cumsum(rng.normal(size=400))
    frame = pd.DataFrame({"t": np.arange(400, dtype=np.int64), "x": driver,
                          "n1": rng.normal(size=400),
                          "y": driver + 0.5 * rng.normal(size=400)})
    result = q.screened(frame, {"time_col": "t", "group_col": False})
    assert q.is_feature(q.column(result, "x")), q.column(result, "x")
    assert q.leaks(result) == []


def _few_of_one_class(seed=8351):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=600)
    y[:100] = 0
    y[[20, 60]] = 1
    return pd.DataFrame({"t": np.arange(600, dtype=np.int64), "n1": rng.normal(size=600),
                         "y": y.astype(np.int64)})


CLASS_COPIES = {
    "number": lambda y: y.astype(float),
    "number_reversed": lambda y: 5.0 - 2.0 * y,
    "text": lambda y: pd.array(["closed" if v == 1 else "open" for v in y], dtype="str"),
}


@pytest.mark.parametrize("case", list(CLASS_COPIES))
def test_a_copy_of_the_class_is_excluded_when_a_fold_holds_few_of_one_class(case):
    """M47: a copy of a class with fewer than five rows in a fold's training rows."""
    frame = _few_of_one_class()
    y = frame["y"].to_numpy()
    assert int(y[:100].sum()) == 2
    frame = q.with_column(frame, "copy", CLASS_COPIES[case](y))
    result = q.screened(frame, {"time_col": "t", "group_col": False})
    assert result.to_dict()["task"]["type"] == "classification"
    _assert_copy_is_out(result)


@pytest.mark.parametrize("scheme", ["random", "time"])
def test_a_column_that_drives_the_class_is_not_excluded(scheme):
    """A column that drives a class, and does not copy it, is no leak."""
    frame = h.clf_frame(n=600, seed=8361)
    statements = dict(s.SAID)
    if scheme == "time":
        frame.insert(0, "t", np.arange(600, dtype=np.int64))
        statements["time_col"] = "t"
    result = q.screened(frame, statements)
    assert q.leaks(result) == []
    assert q.is_feature(q.column(result, "x1"))
