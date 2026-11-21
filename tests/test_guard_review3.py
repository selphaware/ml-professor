"""T31: six behaviours the third review could break unseen (spec 8.1, M8 to M13; D52).

Green today, every test: each passes against the application and fails on the
variant it is named for. A guard that fails means the application is wrong.
Nothing here asserts a particular resample or kept row a seed produces: those
are open (spec 11).
"""
import statistics

import numpy as np
import pandas as pd

import cp3_helpers as c
import mlp_helpers as h


def _lookup(log, values):
    return [h.family("lookup", log, estimator=h.LookupEstimator, values=values)]


def test_m8_the_interval_is_the_middle_95_percent():
    """M8: the interval is the 2.5th to the 97.5th percentile of the resamples (spec step 12)."""
    gen = np.random.default_rng(81)
    n = 400
    frame = pd.DataFrame({"x1": gen.normal(size=n), "y": gen.normal(size=n)})
    values = frame["y"].to_numpy() + gen.normal(size=n)
    own = np.random.default_rng(82)
    low_gaps, high_gaps = [], []
    for seed in range(60):
        result = c.run(frame, families=_lookup(h.Log(), values), seed=seed, task="regression")
        head = result.headline
        assert head["interval_method"] == "bootstrap_rows"
        assert head["interval_level"] == 0.95
        assert head["n_resamples_used"] == 200
        y_true = result.predictions["y_true"].to_numpy(dtype=float)
        y_pred = result.predictions["y_pred"].to_numpy(dtype=float)
        errors = (y_true - y_pred) ** 2
        assert len(errors) == 100
        assert abs(errors.mean() - head["score"]) <= 1e-9 * max(1.0, abs(head["score"]))
        means = errors[own.integers(0, len(errors), size=(4000, len(errors)))].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        sd = means.std()
        low_gaps.append((head["interval_low"] - lo) / sd)
        high_gaps.append((head["interval_high"] - hi) / sd)
    mean_low, mean_high = float(np.mean(low_gaps)), float(np.mean(high_gaps))
    assert abs(mean_low) < 0.18, f"interval_low sits {mean_low:+.3f} sd from the 2.5th percentile"
    assert abs(mean_high) < 0.18, f"interval_high sits {mean_high:+.3f} sd from the 97.5th percentile"


def test_m9_the_leak_screen_uses_the_previous_value():
    """M9: under time order the leak screen judges a column against the previous value too (step 8)."""
    gen = np.random.default_rng(91)
    n = 600
    t = np.arange(n, dtype=float)
    y = 30.0 * np.sin(2.0 * np.pi * t / 100.0) + 0.1 * gen.normal(size=n)
    lag = np.concatenate([[y[0]], y[:-1]])
    frame = pd.DataFrame({"t": t, "lag": lag, "x1": gen.normal(size=n), "y": y})
    result = c.run(frame, time_col="t", task="regression")
    lag_record = h.columns_of(result)["lag"]
    assert lag_record["role"] == "feature", lag_record
    assert lag_record["reason"] is None and lag_record["leak_rule"] is None, lag_record
    for code, column in h.warnings_of(result):
        assert not (code in ("suspected_leak", "strong_single_column") and column == "lag"), code
    doc = result.to_dict()
    assert [b["name"] for b in doc["baselines"]] == ["training_mean", "previous_value"]
    assert result.headline["baseline"] == "previous_value"


def test_m10_the_fast_cap_is_a_sample_from_the_seed():
    """M10: the fast profile's 2,000 rows are a sample drawn from the seed, not the first rows (step 6)."""
    frame = h.clf_frame(n=3000)
    kept_sets = []
    for seed in (0, 1):
        log = h.Log()
        result = c.run(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)],
                       seed=seed)
        assert h.rows_of(result)["dropped_row_cap"] == 1000
        holdout = set(h.holdout_rows(result))
        kept = log.all_fit_rows() | holdout
        assert len(kept) == 2000
        late = sum(1 for r in kept if r >= 2000)
        assert 550 <= late <= 780, f"seed {seed}: {late} kept rows lie at position 2,000 or beyond"
        assert any(r >= 2000 for r in holdout)
        kept_sets.append(kept)
    assert kept_sets[0] != kept_sets[1]


def test_m11_cv_std_is_the_sample_deviation():
    """M11: cv_std is the sample standard deviation of the fold scores (step 9, 5.2)."""
    for frame in (h.clf_frame(n=300), h.reg_frame(n=300)):
        result = c.run(frame)
        models = h.models_of(result)
        assert models
        for model in models:
            scores = model["fold_scores"]
            assert len(scores) == 5 and all(s is not None for s in scores), model
            assert len(set(scores)) > 1, model
            sample = statistics.stdev(scores)
            population = statistics.pstdev(scores)
            assert abs(model["cv_std"] - sample) <= 1e-9 * abs(sample), model
            assert abs(sample - population) > 1e-6 * abs(sample)
        chosen = [m for m in models if m["chosen"]]
        assert len(chosen) == 1
        assert result.to_dict()["cv"]["chosen_std"] == chosen[0]["cv_std"]


def test_m12_tied_predictions_in_a_resample():
    """M12: in every resample tied predictions share their rank (steps 11, 12)."""
    gen = np.random.default_rng(121)
    n = 400
    frame = pd.DataFrame({"x1": gen.normal(size=n), "z1": gen.normal(size=n),
                          "y": gen.integers(0, 2, size=n)})

    result = c.run(frame, families=_lookup(h.Log(), np.full(n, 0.5)))
    head = result.headline
    assert head["score"] == 0.5
    assert head["skill"] == 0.0
    assert head["n_resamples_used"] == 200
    assert head["interval_low"] == 0.5 and head["interval_high"] == 0.5
    assert head["skill_low"] == 0.0 and head["skill_high"] == 0.0

    values = np.where(frame["x1"].to_numpy() > 0, 0.75, 0.25)
    result = c.run(frame, families=_lookup(h.Log(), values))
    head = result.headline
    own = h.auc_by_hand(result.predictions["y_true"], result.predictions["y_pred"])
    assert abs(head["score"] - own) <= 1e-12
    assert head["interval_low"] <= head["score"] <= head["interval_high"]
    assert head["score"] - head["interval_low"] < 0.2
    assert head["interval_high"] - head["score"] < 0.2


def test_m13_n_imputed_counts_rows_that_reach_a_model():
    """M13: n_imputed counts missing cells in the rows that reach a model only (step 8, 5.2)."""
    frame = h.clf_frame(n=260)
    frame["colour"] = pd.Series(["red" if k % 2 else "blue" for k in range(260)], dtype=object)
    frame["y"] = frame["y"].astype(float)
    frame.loc[0:39, "y"] = np.nan
    frame.loc[20:69, "x1"] = np.nan
    frame.loc[30:44, "colour"] = None
    result = c.run(frame)
    rows = h.rows_of(result)
    assert rows["dropped_missing_target"] == 40
    assert rows["training"] + rows["holdout"] == 220
    cols = h.columns_of(result)
    assert (cols["x1"]["n_missing"], cols["x1"]["n_imputed"]) == (50, 30)
    assert cols["colour"]["kind"] == "categorical"
    assert (cols["colour"]["n_missing"], cols["colour"]["n_imputed"]) == (15, 5)
    assert (cols["z1"]["n_missing"], cols["z1"]["n_imputed"]) == (0, 0)
