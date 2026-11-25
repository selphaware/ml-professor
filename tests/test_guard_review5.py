"""T48: two guards of the sixth pass (spec step 8a, step 11, 4.3; D23, D49, D62).

Both hold behaviour the application already has. A guard that fails means the
application is wrong, not the test.
"""
import numpy as np
import pandas as pd
import pytest

import cp5_helpers as p
import cp7_helpers as s
import mlp_helpers as h

N_TIMES, PER_TIME, N_TRAIN_TIMES = 80, 3, 60


def _panel(phi, seed):
    """Three rows on each of 80 numeric timestamps `t`: the level of a timestamp follows
    the one before with weight `phi`, times 3, and each row adds normal noise of 0.2."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=N_TIMES)
    level = np.zeros(N_TIMES)
    level[0] = shocks[0]
    for k in range(1, N_TIMES):
        level[k] = phi * level[k - 1] + shocks[k]
    level = 3.0 * level
    n = N_TIMES * PER_TIME
    frame = pd.DataFrame({
        "t": np.repeat(np.arange(N_TIMES, dtype=float), PER_TIME),
        "x": rng.normal(size=n),
    })
    frame["y"] = np.repeat(level, PER_TIME) + 0.2 * rng.normal(size=n)
    return frame


def _both_estimators(frame):
    """Lag-1 autocorrelation of the per-timestamp means over the 60 training timestamps,
    by the two usual estimators."""
    means = frame.groupby("t")["y"].mean().sort_index().to_numpy()[:N_TRAIN_TIMES]
    ratio = h.lag1_autocorr(means)
    pearson = float(np.corrcoef(means[:-1], means[1:])[0, 1])
    return ratio, pearson


@pytest.mark.parametrize("phi, seed, low, high, warned", [
    pytest.param(0.57, 41, 0.54, 0.58, True, id="just_above"),
    pytest.param(0.45, 59, 0.42, 0.46, False, id="just_below"),
])
def test_m23_the_panel_warning_starts_just_above_one_half(phi, seed, low, high, warned):
    """M23: the autocorrelated_panel_target warning is raised above 0.5, not only above 0.8."""
    frame = _panel(phi, seed)
    ratio, pearson = _both_estimators(frame)
    assert low <= ratio <= high and low <= pearson <= high, (ratio, pearson)

    result = s.run(frame, families=[h.family("rec", h.Log())], time_col="t", group_col=False)
    found = s.warning_records(result, "autocorrelated_panel_target")
    if warned:
        assert len(found) == 1, s.codes(result)
        assert found[0]["column"] == "y"
    else:
        assert found == [], s.codes(result)

    doc = result.to_dict()
    v = doc["validation"]
    assert v["scheme"] == "time_ordered"
    assert v["n_train_timestamps"] == N_TRAIN_TIMES
    names = [b["name"] for b in doc["baselines"]]
    assert names in (["training_mean"], ["training_mean", "previous_value"])
    assert doc["headline"]["baseline"] in names


def _features(frame):
    return [tuple(row) for row in frame[["n1", "n2", "n3", "n4"]].itertuples(index=False,
                                                                             name=None)]


def test_m30_a_copy_at_step_8a_still_needs_the_same_target():
    """M30: a copy at step 8a agrees in every feature and the target, not in the features alone."""
    base = p.noise_frame(400, seed=1301)
    flipped = base.iloc[:120].copy()
    flipped["y"] = 1 - flipped["y"]
    again = base.iloc[120:150].copy()
    frame = pd.concat([base, flipped, again], ignore_index=True)
    frame = frame.iloc[np.random.default_rng(1302).permutation(len(frame))]
    frame = frame.reset_index(drop=True)

    result, log = p.run_kept(frame, duplicates="keep", **s.SAID)
    v = result.to_dict()["validation"]
    holdout = h.holdout_rows(result)
    assert v["n_holdout_copies"] == p.copies_by_hand(result, log)
    assert v["holdout_copy_share"] == pytest.approx(v["n_holdout_copies"] / len(holdout))

    keys = _features(frame)
    held = set(holdout)
    training_keys = {keys[r] for r in range(len(frame)) if r not in held}
    same_features = sum(keys[r] in training_keys for r in holdout)
    assert v["n_holdout_copies"] < same_features, (v["n_holdout_copies"], same_features)
