"""T13: interval and p-value from the holdout predictions (spec step 12, D9, D23, D27)."""
import numpy as np
import pandas as pd
import pytest

import mlp_helpers as h

INTERVAL_FIELDS = ["interval_low", "interval_high", "skill_low", "skill_high"]


def _run(frame, target="y", families="rec", seed=0, **kw):
    import mlprofessor
    from mlprofessor import extensions

    if families == "rec":
        families = [h.family("rec", h.Log())]
    args = dict(h.FAST)
    args.update(kw)
    with extensions.use(families=families):
        return h.stated_profile(frame, target, seed=seed, **args)


def _codes(res):
    return [code for code, _ in h.warnings_of(res)]


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_interval_contains_the_score(task):
    frame = h.clf_frame(300, seed=1) if task == "classification" else h.reg_frame(300, seed=1)
    a = _run(frame)
    hl = a.headline
    assert hl["interval_method"] == "bootstrap_rows"
    assert hl["interval_level"] == pytest.approx(0.95)
    assert hl["interval_low"] <= hl["score"] <= hl["interval_high"]
    assert hl["interval_low"] < hl["interval_high"]
    assert hl["skill_low"] <= hl["skill"] <= hl["skill_high"]
    assert hl["n_resamples"] == 200
    assert 100 <= hl["n_resamples_used"] <= 200
    assert hl["block_length"] is None and hl["interval_reason"] is None
    assert hl["n_permutations"] == 200 and hl["p_value_reason"] is None
    assert "interval_approximate" not in _codes(a)
    b = _run(frame)
    assert h.plain(dict(a.headline)) == h.plain(dict(b.headline))


def test_full_profile_uses_a_thousand_resamples():
    res = _run(h.clf_frame(200, seed=2), profile="full")
    assert res.headline["n_resamples"] == 1000
    assert res.headline["n_permutations"] == 1000
    p = res.headline["p_value"]
    assert abs(p * 1001 - round(p * 1001)) < 1e-6


def test_p_value_floor_with_a_perfect_holdout():
    frame = h.clf_frame(240, seed=3)
    hold = h.holdout_rows(_run(frame))
    y = frame["y"].to_numpy()
    values = np.random.default_rng(0).random(len(y))
    values[hold] = y[hold]
    res = _run(frame, families=[h.family("oracle", h.Log(), h.LookupEstimator, values=values)])
    assert res.headline["score"] == pytest.approx(1.0)
    assert res.headline["p_value"] == pytest.approx(1 / 201)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_p_value_is_a_multiple_of_one_over_r_plus_one(seed):
    res = _run(h.clf_frame(200, seed=10 + seed), seed=seed)
    p = res.headline["p_value"]
    assert 0 < p <= 1
    assert abs(p * 201 - round(p * 201)) < 1e-6


def test_shuffled_target_p_value_not_small():
    large = 0
    for seed in range(5):
        frame = h.clf_frame(200, seed=20 + seed)
        frame["y"] = np.random.default_rng(seed).permutation(frame["y"].to_numpy())
        p = _run(frame, seed=seed).headline["p_value"]
        large += p > 0.05
    assert large >= 3


@pytest.mark.parametrize("n,block", [(256, 4), (260, 5)])
def test_block_bootstrap_under_time_order(n, block):
    frame = h.time_frame(n, seed=4)
    res = _run(frame, time_col="date")
    hl = res.headline
    assert res.to_dict()["validation"]["n_holdout_timestamps"] == -(-n // 4)
    assert hl["interval_method"] == "bootstrap_blocks"
    assert hl["block_length"] == block
    assert hl["interval_low"] <= hl["score"] <= hl["interval_high"]
    assert hl["p_value"] is None and hl["n_permutations"] is None
    assert isinstance(hl["p_value_reason"], str) and hl["p_value_reason"]
    approx = [(c, col) for c, col in h.warnings_of(res) if c == "interval_approximate"]
    assert approx == [("interval_approximate", None)]
    assert "no_interval" not in _codes(res)


def test_no_interval_with_ten_holdout_timestamps():
    rng = np.random.default_rng(5)
    when = np.repeat(h.dates(40).to_numpy(), 3)
    frame = pd.DataFrame({"date": when, "x1": rng.normal(size=120)})
    frame["y"] = frame["x1"] + rng.normal(size=120)
    res = _run(frame, time_col="date")
    hl = res.headline
    assert res.to_dict()["validation"]["n_holdout_timestamps"] == 10
    assert hl["interval_method"] is None
    assert all(hl[k] is None for k in INTERVAL_FIELDS)
    assert isinstance(hl["interval_reason"], str) and hl["interval_reason"]
    assert "no_interval" in _codes(res)
    assert "interval_approximate" not in _codes(res)


def test_interval_with_two_hundred_holdout_timestamps():
    res = _run(h.time_frame(800, seed=6), time_col="date")
    hl = res.headline
    assert res.to_dict()["validation"]["n_holdout_timestamps"] == 200
    assert hl["interval_method"] == "bootstrap_blocks"
    assert all(hl[k] is not None for k in INTERVAL_FIELDS)
    assert _codes(res).count("interval_approximate") == 1
    assert "no_interval" not in _codes(res)


@pytest.mark.parametrize("task,n,warned", [
    ("classification", 200, True), ("classification", 600, False),
    ("regression", 120, True), ("regression", 400, False),
])
def test_small_holdout_warning(task, n, warned):
    frame = h.clf_frame(n, seed=7, signal=0.5) if task == "classification" else h.reg_frame(n, seed=7)
    res = _run(frame)
    found = [(c, col) for c, col in h.warnings_of(res) if c == "small_holdout"]
    assert found == ([("small_holdout", None)] if warned else [])
