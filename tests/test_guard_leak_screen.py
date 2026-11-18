"""T18 guards: the leak screen sees training rows only (spec 3 step 8, 8.1, M3).

Green today. A failure here means the application is wrong, not the test.
Time order with unique numeric timestamps 0..399: 300 training rows in six
blocks of 50, and rows 300 to 399 as the holdout.
"""
import numpy as np
import pandas as pd

import cp3_helpers as c
import mlp_helpers as h

N = 400
HOLDOUT = list(range(300, 400))


def _run(frame):
    log = h.Log()
    result = c.run(frame, families=[h.family("rec", log)], time_col="t")
    assert h.holdout_rows(result) == HOLDOUT
    return result


def _block(row):
    """Training block 1..6 of a training row."""
    return row // 50 + 1


def test_m3_missing_by_class_counts_training_rows_only():
    """M3 (small F1): the missing-by-class rule counts training rows, never holdout rows."""
    rng = np.random.default_rng(51)
    y = np.ones(N, dtype=int)
    train_class0 = [5, 30, 55, 80, 105, 130, 160, 190, 220, 260]
    y[train_class0] = 0
    y[300:390] = 0
    filled = [80, 130, 260]
    late = rng.normal(size=N)
    late[y == 0] = np.nan
    late[filled] = rng.normal(size=len(filled))
    frame = pd.DataFrame({"t": np.arange(N, dtype=float), "x1": rng.normal(size=N),
                          "late": late, "y": y})

    blocks = [_block(r) for r in train_class0]
    assert blocks.count(1) == 2 and set(blocks) == {1, 2, 3, 4, 5, 6}, "every fold is scored"
    assert all(2 <= _block(r) <= 6 for r in filled)
    assert (y[300:] == 0).sum() == 90
    assert not np.isnan(late[y == 1]).any()
    training_share = np.isnan(late[:300][y[:300] == 0]).mean()
    overall_share = np.isnan(late[y == 0]).mean()
    assert training_share == 0.7
    assert overall_share == 0.97

    result = _run(frame)
    record = h.columns_of(result)["late"]
    assert record["role"] == "feature"
    assert record["reason"] is None
    assert record["leak_rule"] is None
    assert ("suspected_leak", "late") not in h.warnings_of(result)


def test_m3_single_column_skill_is_fitted_on_fold_training_rows_only():
    """M3 (small F1): the screen's single-column trees are fitted on each fold's training rows only."""
    rng = np.random.default_rng(52)
    y = np.tile([0, 1], N // 2)
    y[:300] = y[np.concatenate([rng.permutation(50) + 50 * b for b in range(6)])]
    block = np.r_[np.repeat(np.arange(1, 7), 50), np.repeat(np.arange(2, 7), 20)]
    high = np.where(block % 2 == 1, y, 1 - y)
    frame = pd.DataFrame({"t": np.arange(N, dtype=float), "x1": rng.normal(size=N),
                          "drift": block + 0.25 + 0.5 * high, "y": y})

    for b in range(1, 7):
        rows = slice(50 * (b - 1), 50 * b)
        assert (y[rows] == 1).sum() == 25
    for k, b in enumerate(range(2, 7)):
        rows = slice(300 + 20 * k, 300 + 20 * (k + 1))
        assert (y[rows] == 1).sum() == 10
        assert set(frame["drift"].to_numpy()[rows]) == {b + 0.25, b + 0.75}

    result = _run(frame)
    record = h.columns_of(result)["drift"]
    assert record["role"] == "feature"
    assert record["reason"] is None
    assert record["leak_rule"] is None
    warnings = h.warnings_of(result)
    assert ("suspected_leak", "drift") not in warnings
    assert ("strong_single_column", "drift") not in warnings
