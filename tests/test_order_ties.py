"""T72: the order reading between rows of neighbouring values (spec step 8, 4.2, 4.3, 11;
D76, D77 point 4, D78 point 7).

For each two neighbouring distinct values of a numeric column every row of the one is
compared with every row of the next; the shares in which the target rises and falls are
each averaged over the pairs of values, and the share against the order is the smaller
average. Rows of equal column value are never put in the target's order, so noise with
repeated values is not named; one point for each value (M62) is not the reading.
"""
import numpy as np
import pandas as pd
import pytest

import cp10_helpers as t
import cp7_helpers as s
import cp9_helpers as q


@pytest.mark.parametrize("seed", [9601, 9602])
def test_noise_whose_values_repeat_is_not_named(seed):
    """F33: a whole number from 1 to 40 and noise rounded to one place: features, no warning."""
    rng = np.random.default_rng(seed)
    n = 600
    x1 = rng.normal(size=n)
    whole, rounded = t.repeated_noise(n, rng)
    frame = pd.DataFrame({"x1": x1, "whole": whole, "rounded": rounded,
                          "y": 2.0 * x1 + rng.normal(size=n)})
    result = t.run(frame)
    for name in ("whole", "rounded"):
        assert t.quiet_feature(result, name), (name, s.codes(result))
        assert t.record(result, name)["shuffled_score"] is not None


@pytest.mark.parametrize("seed", [9603, 9605])
def test_a_column_whose_means_rise_with_the_target_is_not_named(seed):
    """M62: 22 values whose means are in the target's order, and the rows not: not named."""
    rng = np.random.default_rng(seed)
    n = 2000
    value = rng.integers(0, 22, size=n)
    y = 0.3 * value + rng.normal(size=n)
    frame = pd.DataFrame({"v": value.astype(np.int64), "z1": rng.normal(size=n), "y": y})
    result = t.run(frame)
    training = t.training_rows_of(result, frame)
    v, target = frame.loc[training, "v"].tolist(), frame.loc[training, "y"].tolist()
    assert t.against_by_pairs(v, target) > 2 * q.ORDER_AGAINST
    assert t.against_by_means(v, target) <= q.ORDER_AGAINST
    assert t.quiet_feature(result, "v"), s.codes(result)
    assert t.record(result, "v")["shuffled_score"] is not None


def test_a_column_in_the_targets_order_with_repeated_values_is_named():
    """F33, M62: a copy of a walk with 4% of its training cells moved round, each value twice."""
    frame = q.walk(n=400, seed=9604, per=2)
    statements = {"time_col": "t", "group_col": False}
    training = [r for r in range(len(frame)) if frame.loc[r, "t"] < 300]
    copy = q.rotated(frame["y"], 0.04, seed=9604, rows=training)
    frame = q.with_column(frame, "copy", copy)
    result = t.run(frame, statements)
    assert sorted(h_rows for h_rows in t.training_rows_of(result, frame)) == training
    assert t.leaks(result) == [], t.leaks(result)
    assert t.record(result, "copy")["role"] == "feature"
    messages = t.told(result, t.STRONG, "copy")
    assert len(messages) == 1, s.codes(result)
    share = t.against_by_pairs(frame.loc[training, "copy"].tolist(),
                               frame.loc[training, "y"].tolist())
    assert 0.0 < share <= q.ORDER_AGAINST, share
    assert q.holds_share(messages[0], share), (share, messages[0])
