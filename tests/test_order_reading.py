"""T63: the second reading of the `strong_single_column` warning, the column's order
(spec step 8, 2.5, 4.3, 7, 11; D36, D76 point 1).

For a numeric column of a regression that rule 2 did not exclude and whose skill has a
value below 0.90: when the target, put in the column's order, moves the other way
between at most 10% of neighbouring training rows, and does not follow its own previous
value as closely, the column is named. It stays a feature. The shares are counted by
hand with `cp9_helpers.against_order`.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

import cp7_helpers as s
import cp9_helpers as q
import mlp_helpers as h

TIME = {"time_col": "t", "group_col": False}
STRONG = "strong_single_column"


def _shares(result, frame, name):
    """Over the training rows of a run at unique timestamps, in time order: the share of
    the column against the order, and the share of the row before in its place."""
    order = frame.sort_values("t", kind="stable").index.tolist() if "t" in frame else None
    training = set(q.training_rows(result, frame))
    y = frame["y"].to_numpy(dtype=float)
    x = frame[name].to_numpy(dtype=float)
    rows = [r for r in (order or range(len(frame))) if r in training]
    column_share = q.against_order([x[r] for r in rows], [y[r] for r in rows])
    if order is None:
        return column_share, None
    previous = q.against_order([y[a] for a in rows[:-1]], [y[b] for b in rows[1:]])
    return column_share, previous


def _assert_named_and_kept(result, name):
    assert q.is_feature(q.column(result, name)), q.column(result, name)
    assert len(q.told(result, STRONG, name)) == 1, s.codes(result)
    assert not q.told(result, "suspected_leak", name)


def _copy(frame):
    return 3.7 * frame["y"].to_numpy(dtype=float) + 12.0


@pytest.mark.parametrize("seed", [8501, 8502, 8503])
def test_a_copy_with_some_cells_moved_is_named_under_time_order(seed):
    """M44: a copy with 4% of its cells moved is named, and stays a feature."""
    frame = q.walk(n=600, seed=seed)
    frame = q.with_column(frame, "copy", q.rotated(_copy(frame), 0.04, seed=seed,
                                                   rows=range(450)))
    result = q.screened(frame, TIME)
    share, previous = _shares(result, frame, "copy")
    assert 0 < share <= q.ORDER_AGAINST, share
    assert previous > q.ORDER_AGAINST, previous
    _assert_named_and_kept(result, "copy")
    assert q.holds_share(q.told(result, STRONG, "copy")[0], share)


CURVES = {"exponential": lambda y: np.exp(y / 4.0), "cube": lambda y: y ** 3}


@pytest.mark.parametrize("curve", list(CURVES))
def test_a_curve_of_the_target_is_named_under_time_order(curve):
    """A curve of the target is excluded, or named and kept."""
    frame = q.walk(n=600, seed=8511)
    frame = q.with_column(frame, "curve", CURVES[curve](frame["y"].to_numpy(dtype=float)))
    result = q.screened(frame, TIME)
    share, _ = _shares(result, frame, "curve")
    assert share == 0
    if q.is_leak(q.column(result, "curve")):
        assert q.leaks(result) == ["curve"]
    else:
        _assert_named_and_kept(result, "curve")


MOVED = (0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.16)


def test_the_first_condition_is_a_share_of_at_most_one_tenth():
    """M44: seven copies with 5% to 16% of their cells moved: none excluded, each named
    exactly when its share against the order is at most 0.10."""
    frame = q.walk(n=600, seed=8521)
    for k, moved in enumerate(MOVED):
        frame = q.with_column(frame, f"c{k}", q.rotated(_copy(frame), moved, seed=8521 + k,
                                                        rows=range(450)))
    result = q.screened(frame, TIME)
    assert q.leaks(result) == []
    outcomes = set()
    for k in range(len(MOVED)):
        name = f"c{k}"
        share, previous = _shares(result, frame, name)
        assert previous > q.ORDER_AGAINST
        named = share <= q.ORDER_AGAINST
        outcomes.add(named)
        assert q.is_feature(q.column(result, name))
        assert len(q.told(result, STRONG, name)) == int(named), (name, share)
    assert outcomes == {True, False}


def test_the_second_condition_keeps_the_lag_of_a_running_total_quiet():
    """M43: where the target follows its own previous value as closely, nothing is named."""
    rng = np.random.default_rng(8531)
    total = np.cumsum(rng.exponential(size=600))
    frame = pd.DataFrame({"t": np.arange(600, dtype=np.int64),
                          "lag": np.concatenate([[np.nan], total[:-1]]),
                          "counter": 0.37 * np.arange(600) + 1.3,
                          "n1": rng.normal(size=600), "y": total})
    result = q.screened(frame, TIME)
    for name in ("lag", "counter"):
        share, previous = _shares(result, frame, name)
        assert share == 0 and previous == 0, (name, share, previous)
        assert q.is_feature(q.column(result, name)), q.column(result, name)
        assert not q.told(result, STRONG, name)


def test_with_no_previous_value_the_second_condition_has_nothing_to_ask():
    """M44: under the random scheme a copy with ten wild training targets is named."""
    frame = h.reg_frame(n=600, seed=8541)
    clean = q.screened(frame, s.SAID)
    training = q.training_rows(clean, frame)
    frame = q.with_column(frame, "copy", _copy(frame))
    frame.loc[[training[40 * k + 5] for k in range(10)], "y"] = 999999.0
    result = q.screened(frame, s.SAID)
    share, _ = _shares(result, frame, "copy")
    assert share <= q.ORDER_AGAINST
    _assert_named_and_kept(result, "copy")


def test_fewer_than_twenty_pairs_give_no_reading():
    """A column of three bands of the target has two pairs: no order reading."""
    frame = h.reg_frame(n=600, seed=8551)
    y = frame["y"].to_numpy(dtype=float)
    bands = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3])).astype(float)
    frame = q.with_column(frame, "band", bands)
    result = q.screened(frame, s.SAID)
    assert _shares(result, frame, "band")[0] is None
    assert q.is_feature(q.column(result, "band"))
    assert not q.told(result, STRONG, "band")


def _driver(n, r2):
    rng = np.random.default_rng([8561, n, int(round(r2 * 100))])
    x = np.cumsum(rng.normal(size=n))
    noise = rng.normal(size=n)
    noise *= np.std(x) * np.sqrt((1.0 - r2) / r2) / np.std(noise)
    return pd.DataFrame({"t": np.arange(n, dtype=np.int64), "x": x,
                         "n1": rng.normal(size=n), "y": x + noise})


@pytest.mark.parametrize("n,r2", list(itertools.product([100, 200, 400], [0.9, 0.95, 0.99])))
def test_a_strong_driver_known_before_the_outcome_is_not_named(n, r2):
    """A walk that explains 90% to 99% of the target is a feature, not named at 0.90 and
    0.95."""
    frame = _driver(n, r2)
    result = q.screened(frame, TIME)
    share, _ = _shares(result, frame, "x")
    assert share > q.ORDER_AGAINST, share
    assert q.is_feature(q.column(result, "x"))
    assert q.leaks(result) == []
    if r2 < 0.99:
        assert not q.told(result, STRONG, "x")


def test_a_column_warned_for_its_skill_gets_one_record():
    """A column named by both readings gets one record (4% of its training cells moved:
    with 2.5% the reference excludes it by rule 2 on this table)."""
    frame = h.reg_frame(n=600, seed=8571)
    clean = q.screened(frame, s.SAID)
    training = q.training_rows(clean, frame)
    frame = q.with_column(frame, "copy", q.rotated(frame["y"].to_numpy(dtype=float), 0.04,
                                                   seed=8571, rows=training))
    result = q.screened(frame, s.SAID)
    assert _shares(result, frame, "copy")[0] <= q.ORDER_AGAINST
    _assert_named_and_kept(result, "copy")


def test_a_column_whose_skill_cannot_be_computed_is_named_by_neither_reading():
    """D36: the table of test_skill_undefined.py: `lead` is in the target's order and is
    named by no rule, because its skill has no value."""
    n = 240
    rng = np.random.default_rng(76)
    y = rng.normal(size=n) * 3.0
    y[20:180] = 0.0
    frame = pd.DataFrame({"t": np.arange(n, dtype=float), "x1": rng.normal(size=n),
                          "lead": 2.0 * y + 1.0, "y": y})
    result = q.screened(frame, TIME, task="regression")
    assert h.holdout_rows(result) == list(range(180, 240))
    assert _shares(result, frame, "lead")[0] == 0
    assert q.is_feature(q.column(result, "lead"))
    assert not q.told(result, "suspected_leak", "lead")
    assert not q.told(result, STRONG, "lead")


@pytest.mark.parametrize("seed", [1, 2])
def test_a_classification_has_no_order_reading(seed):
    """The order reading is for a regression only."""
    frame = h.clf_frame(n=600, seed=seed, signal=5.0)
    result = q.screened(frame, s.SAID)
    assert result.to_dict()["task"]["type"] == "classification"
    assert _shares(result, frame, "x1")[0] <= q.ORDER_AGAINST
    assert not q.told(result, STRONG)


def test_a_categorical_column_has_no_order_reading():
    """Twenty-four bands of a walk's target as text: in the target's order, and not read
    by its order, because it is categorical."""
    frame = q.walk(n=600, seed=8581)
    y = frame["y"].to_numpy(dtype=float)
    edges = np.linspace(y.min(), y.max(), 25)[1:-1]
    index = np.digitize(y, edges)
    frame = q.with_column(frame, "band", pd.array([f"b{k:02d}" for k in index], dtype="str"))
    result = q.screened(frame, TIME)
    training = sorted(q.training_rows(result, frame))
    assert q.against_order(index[training].astype(float), y[training]) <= q.ORDER_AGAINST
    record = q.column(result, "band")
    assert record["kind"] == "categorical" and q.is_feature(record), record
    assert not q.told(result, STRONG, "band")
