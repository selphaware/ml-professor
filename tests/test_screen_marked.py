"""T73: rule 2 where more training rows are marked than may be set aside (spec step 8, 7;
D77 point 5, D78 point 7).

A training row is marked when its target lies more than 10 median deviations from the
median (or 10 median steps from its previous value). When more than 1% of the training
rows are marked none is set aside; rule 2 still leaves unscored the 1% of rows the
column misses by most and the 1% the baseline misses by most, and is also read against
each fold's training median. So a copy of a long-tailed target with five cells moved is
excluded, and noise beside a target with sentinel cells is not.
"""
import numpy as np
import pytest

import cp10_helpers as t
import cp7_helpers as s
import cp9_helpers as q


def _marked(y, training):
    """The training rows whose target lies more than 10 median deviations from the
    median of the training targets."""
    values = np.asarray([float(y[r]) for r in training])
    middle = float(np.median(values))
    spread = float(np.median(np.abs(values - middle)))
    return [r for r in training if abs(float(y[r]) - middle) > q.SCREEN_FAR * spread]


def _copy_run(seed, spreads):
    frame = t.lognormal_table(n=600, seed=seed)
    training = t.training_of(frame)
    marked = _marked(frame["y"].to_numpy(), training)
    assert len(marked) > q.SCREEN_SET_ASIDE * len(training), len(marked)
    rng = np.random.default_rng(seed)
    moved = sorted(int(r) for r in rng.choice(training, size=5, replace=False))
    frame = q.with_column(frame, "copy", t.moved_copy(frame["y"], moved, spreads))
    result = t.run(frame)
    assert set(moved) <= set(t.training_rows_of(result, frame))
    return result


def _assert_copy_is_the_only_leak(result):
    assert t.leaks(result) == ["copy"], (t.leaks(result), s.codes(result))
    assert t.record(result, "copy")["leak_rule"] == "predicts_target"
    for name in ("n1", "n2"):
        assert t.record(result, name)["role"] == "feature", name


@pytest.mark.parametrize("seed", [9501, 9502, 9503])
def test_a_copy_with_five_cells_moved_far_is_excluded(seed):
    """F34: a log-normal target, a copy 1.07 y + 3 with five cells moved up by 20 spreads."""
    _assert_copy_is_the_only_leak(_copy_run(seed, 20.0))


@pytest.mark.parametrize("spreads", [1.0, 1000.0])
def test_whatever_the_size_of_the_change(spreads):
    """F34: five cells moved by one spread, or by a thousand: excluded all the same."""
    _assert_copy_is_the_only_leak(_copy_run(9504, spreads))


def _noise(scheme, seed):
    rng = np.random.default_rng(seed)
    if scheme == "random":
        frame = q.walk(n=600, seed=seed).drop(columns=["t"])
        frame["y"] = rng.normal(size=len(frame))
        return frame, dict(t.RANDOM)
    frame = s.sized_entities([5] * 120, seed=seed)
    return frame, {"time_col": False, "group_col": "who"}


@pytest.mark.parametrize("scheme,sentinel,share", [
    ("random", 999999.0, 0.013), ("random", -999.0, 0.013), ("random", 999999.0, 0.015),
    ("grouped", 999999.0, 0.013), ("grouped", -999.0, 0.015)])
def test_wild_target_cells_beyond_the_share_make_no_column_a_leak(scheme, sentinel, share):
    """M63: noise, with 1.3% or 1.5% of the training targets a sentinel: no column excluded."""
    seed = 9510 + int(1000 * share) + (7 if sentinel < 0 else 0)
    frame, statements = _noise(scheme, seed)
    training = t.training_of(frame, statements)
    k = int(round(share * len(training)))
    assert k > q.SCREEN_SET_ASIDE * len(training)
    rows = np.random.default_rng(seed).choice(training, size=k, replace=False)
    frame.loc[rows, "y"] = sentinel
    result = t.run(frame, statements)
    assert set(int(r) for r in rows) <= set(t.training_rows_of(result, frame))
    assert t.leaks(result) == [], (t.leaks(result), s.codes(result))
