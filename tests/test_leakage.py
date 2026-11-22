"""T12: the leak screen and its warning tier (spec step 8, 4.3, 10.2; D11, D24, D30, D33)."""
import numpy as np
import pandas as pd
import pytest

import mlp_helpers as h


def _run(frame, target="y", **kw):
    """Run with a recording fake model and step; return (result, log)."""
    import mlprofessor
    from mlprofessor import extensions

    log = h.Log()
    args = dict(h.FAST)
    args.update(kw)
    with extensions.use(families=[h.family("rec", log)], steps=[h.step_factory(log)]):
        res = h.stated_profile(frame, target, seed=0, **args)
    return res, log


def _seen_by_steps(log):
    cols = set()
    for o in log.objects("step"):
        cols.update(o["fit_columns"])
    return cols


def _codes_for(res, column):
    return [code for code, col in h.warnings_of(res) if col == column]


def _flip(y, n_flip, seed, among=None):
    """A copy of a 0/1 target with exactly n_flip labels flipped."""
    rng = np.random.default_rng(seed)
    pool = np.arange(len(y)) if among is None else np.asarray(among)
    rows = rng.choice(pool, size=n_flip, replace=False)
    out = y.copy()
    out[rows] = 1 - out[rows]
    return out


def test_f2_copies_and_class_filled_columns_excluded():
    """F2: a copy of the target, a rescaled copy and a column filled for one class never reach a model."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, size=n)
    partly = rng.normal(size=n)
    partly[rng.random(n) < np.where(y == 0, 0.6, 0.4)] = np.nan
    frame = pd.DataFrame({
        "strong": y + rng.normal(0, 0.85, size=n),
        "copy": y,
        "rescaled": y * 3.7 + 12.0,
        "by_class": np.where(y == 1, rng.normal(size=n), np.nan),
        "partly_missing": partly,
        "z1": rng.normal(size=n),
        "y": y,
    })
    res, log = _run(frame)
    cols = h.columns_of(res)
    for name, rule in [("copy", "predicts_target"), ("rescaled", "predicts_target"),
                       ("by_class", "missing_by_class")]:
        assert (cols[name]["role"], cols[name]["reason"], cols[name]["leak_rule"]) == \
               ("excluded", "suspected_leak", rule), name
        assert _codes_for(res, name).count("suspected_leak") == 1, name
        assert "strong_single_column" not in _codes_for(res, name), name
    seen = _seen_by_steps(log)
    assert not seen & {"copy", "rescaled", "by_class"}
    for name in ["strong", "partly_missing"]:
        assert (cols[name]["role"], cols[name]["reason"], cols[name]["leak_rule"]) == \
               ("feature", None, None), name
        assert not {"suspected_leak", "strong_single_column"} & set(_codes_for(res, name)), name
        assert name in seen


def test_f2_regression_rescaled_copy_excluded():
    """F2: a rescaled copy of a numeric target is caught."""
    frame = h.reg_frame(1500, seed=1)
    frame.insert(0, "leak", frame["y"] * 3.7 + 12.0)
    res, log = _run(frame)
    cols = h.columns_of(res)
    assert (cols["leak"]["reason"], cols["leak"]["leak_rule"]) == ("suspected_leak", "predicts_target")
    assert "suspected_leak" in _codes_for(res, "leak")
    assert cols["x1"]["role"] == "feature" and cols["x1"]["leak_rule"] is None
    assert "leak" not in _seen_by_steps(log)


def test_f2_warning_tier_between_the_cut_offs():
    """F2: a near-copy (2% flipped, 1,500 rows) is kept and warned; the copy is excluded; a predictive column gets neither."""
    rng = np.random.default_rng(2)
    n = 1500
    y = rng.integers(0, 2, size=n)
    frame = pd.DataFrame({
        "near": _flip(y, 30, seed=3),
        "copy": y.copy(),
        "strong": y + rng.normal(0, 0.85, size=n),
        "z1": rng.normal(size=n),
        "y": y,
    })
    res, log = _run(frame)
    cols = h.columns_of(res)
    assert (cols["near"]["role"], cols["near"]["reason"], cols["near"]["leak_rule"]) == ("feature", None, None)
    assert _codes_for(res, "near").count("strong_single_column") == 1
    assert "suspected_leak" not in _codes_for(res, "near")
    assert "near" in _seen_by_steps(log)
    assert cols["copy"]["reason"] == "suspected_leak"
    assert "strong_single_column" not in _codes_for(res, "copy")
    assert not {"suspected_leak", "strong_single_column"} & set(_codes_for(res, "strong"))
    assert sum(code == "strong_single_column" for code, _ in h.warnings_of(res)) == 1


def test_f2_regression_warning_tier():
    """F2: the target plus noise of 20% of its deviation is kept and warned."""
    rng = np.random.default_rng(4)
    n = 1500
    x1 = rng.normal(size=n)
    y = 2.0 * x1 + rng.normal(size=n)
    frame = pd.DataFrame({"near": y + rng.normal(0, 0.2 * y.std(), size=n), "x1": x1, "y": y})
    res, log = _run(frame)
    cols = h.columns_of(res)
    assert (cols["near"]["role"], cols["near"]["leak_rule"]) == ("feature", None)
    assert _codes_for(res, "near").count("strong_single_column") == 1
    assert "suspected_leak" not in _codes_for(res, "near")
    assert "near" in _seen_by_steps(log)


def _time_clf(n, seed):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"date": h.dates(n), "x1": rng.normal(size=n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-1.5 * frame["x1"].to_numpy()))).astype(int)
    return frame, y, rng


def test_f1_leak_rule_judged_on_training_rows_only():
    """F1: a column that copies the target only on the holdout is not flagged; only on training rows, it is."""
    n = 400
    frame, y, rng = _time_clf(n, 5)
    hold = np.arange(n) >= 300           # unique timestamps: the last quarter
    noise = rng.integers(0, 2, size=n)
    frame["copy_holdout"] = np.where(hold, y, noise)
    frame["copy_train"] = np.where(hold, noise, y)
    frame["y"] = y
    res, log = _run(frame, time_col="date")
    assert h.holdout_rows(res) == list(range(300, 400))
    cols = h.columns_of(res)
    assert (cols["copy_train"]["reason"], cols["copy_train"]["leak_rule"]) == ("suspected_leak", "predicts_target")
    assert (cols["copy_holdout"]["role"], cols["copy_holdout"]["leak_rule"]) == ("feature", None)
    assert "suspected_leak" not in _codes_for(res, "copy_holdout")


def test_f1_warning_tier_judged_on_training_rows_only():
    """F1: the warning tier, too, is measured on training rows only."""
    n = 1500
    frame, y, rng = _time_clf(n, 6)
    hold = np.arange(n) >= 1125
    noise = rng.integers(0, 2, size=n)
    near_train = _flip(y, 28, seed=7, among=np.flatnonzero(~hold))
    frame["near_train"] = np.where(hold, noise, near_train)
    frame["copy_holdout"] = np.where(hold, y, noise)
    frame["y"] = y
    res, log = _run(frame, time_col="date")
    assert h.holdout_rows(res) == list(range(1125, 1500))
    cols = h.columns_of(res)
    assert (cols["near_train"]["role"], cols["near_train"]["leak_rule"]) == ("feature", None)
    assert _codes_for(res, "near_train").count("strong_single_column") == 1
    assert (cols["copy_holdout"]["role"], cols["copy_holdout"]["leak_rule"]) == ("feature", None)
    assert not {"suspected_leak", "strong_single_column"} & set(_codes_for(res, "copy_holdout"))


def test_lagged_target_on_random_walk_not_flagged_under_time_order():
    frame = h.time_frame(400, seed=8, walk=True)
    frame["lag1"] = frame["y"].shift(1)
    res, log = _run(frame, time_col="date")
    cols = h.columns_of(res)
    assert (cols["lag1"]["role"], cols["lag1"]["leak_rule"]) == ("feature", None)
    assert "suspected_leak" not in _codes_for(res, "lag1")
    assert "lag1" in _seen_by_steps(log)


def test_only_feature_a_leak_raises_no_features():
    from mlprofessor import errors

    y = np.random.default_rng(9).integers(0, 2, size=200)
    frame = pd.DataFrame({"copy": y, "row_id": np.arange(200), "y": y})
    with pytest.raises(errors.NoFeaturesError):
        _run(frame)
