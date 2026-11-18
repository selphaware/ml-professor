"""T19: skill when a baseline's error is zero (spec 3 steps 8, 11, 12; 2.5, 2.7, 5.2; D36, D40, D42).

Red today, apart from `test_single_column_skill_that_cannot_be_computed_flags_nothing`.
"""
import numpy as np
import pandas as pd
import pytest

import cp3_helpers as c
import mlp_helpers as h


# ------------------------------------------------------------------ tables

def _flat_holdout_series(n=240, seed=71):
    """Unique numeric timestamps; the target is exactly 0.0 from row 150 on.

    The holdout is rows 180 to 239 and the last training row is flat too, so
    the previous value is exact on every holdout row.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    y = 2.0 * x1 + rng.normal(size=n)
    y[150:] = 0.0
    return pd.DataFrame({"t": np.arange(n, dtype=float), "x1": x1,
                         "z1": rng.normal(size=n), "y": y})


def _random_scheme_zero_mean_table(seed=0, n=240):
    """Training rows half 1.0 and half -1.0 (mean exactly 0); holdout rows 0.0."""
    from mlprofessor import stages

    split = stages.split_random(n, None, seed)
    training = [int(r) for r in split["training"]]
    holdout = [int(r) for r in split["holdout"]]
    assert len(training) == 180 and len(holdout) == 60
    rng = np.random.default_rng(72)
    y = np.zeros(n)
    signs = np.array([1.0] * 90 + [-1.0] * 90)
    y[training] = rng.permutation(signs)
    frame = pd.DataFrame({"x1": rng.normal(size=n), "z1": rng.normal(size=n), "y": y})
    return frame, training, holdout


def _one_wrong_previous_value_series(n=256, seed=73):
    """Target about 5 over the training rows, exactly 5.0 on the last one, 0.0 on the holdout.

    The previous value is wrong on the first holdout timestamp only.
    """
    rng = np.random.default_rng(seed)
    y = 5.0 + 0.1 * rng.normal(size=n)
    y[191] = 5.0
    y[192:] = 0.0
    frame = pd.DataFrame({"t": np.arange(n, dtype=float), "x1": rng.normal(size=n), "y": y})
    return frame, y + 0.5


def _constant_family(value, n):
    log = h.Log()
    return [h.family("const", log, h.LookupEstimator, values=np.full(n, value))]


def _run_one_wrong(frame, values):
    log = h.Log()
    return c.run(frame, families=[h.family("plus_half", log, h.LookupEstimator, values=values)],
                 time_col="t", task="regression")


# ------------------------------------------------------------------- tests

def test_flat_holdout_under_time_order_reports_null_skill():
    """A target flat across the holdout: skill is null, everything else is reported."""
    frame = _flat_holdout_series()
    result = c.run(frame, time_col="t")
    rows = h.holdout_rows(result)
    assert rows == list(range(180, 240))
    y = frame["y"].to_numpy()

    doc = result.to_dict()
    baselines = {b["name"]: b for b in doc["baselines"]}
    assert set(baselines) == {"training_mean", "previous_value"}
    mean_score = h.mse_by_hand(y[180:], np.full(60, y[:180].mean()))
    assert baselines["training_mean"]["score"] == pytest.approx(mean_score, rel=1e-9)
    assert baselines["training_mean"]["used"] is False
    assert baselines["previous_value"]["score"] == 0
    assert baselines["previous_value"]["used"] is True

    head = result.headline
    assert head["baseline"] == "previous_value"
    assert head["baseline_score"] == 0
    preds = result.predictions
    score = h.mse_by_hand(preds["y_true"], preds["y_pred"])
    assert head["score"] == pytest.approx(score, rel=1e-9, abs=1e-12)
    assert head["score"] > 0
    assert head["skill"] is None
    assert head["skill_low"] is None
    assert head["skill_high"] is None

    assert h.warnings_of(result).count(("skill_undefined", None)) == 1
    assert c.count_code(result, "skill_undefined") == 1
    assert head["interval_method"] == "bootstrap_blocks"
    assert head["interval_low"] <= head["score"] <= head["interval_high"]
    assert head["n_resamples_used"] == 200
    assert head["n_resamples_skill_used"] == 0
    assert head["p_value"] is None
    assert isinstance(head["p_value_reason"], str) and head["p_value_reason"].strip()
    assert "no_interval" not in c.codes(result)
    assert "interval_approximate" in c.codes(result)
    c.assert_strict_json(result)


def test_zero_error_of_the_training_mean_under_the_random_scheme():
    """The training mean is the value of every holdout row: skill is null."""
    frame, training, holdout = _random_scheme_zero_mean_table()
    result = c.run(frame, families=_constant_family(0.25, len(frame)), task="regression")
    assert h.holdout_rows(result) == holdout

    head = result.headline
    assert head["baseline"] == "training_mean"
    assert head["baseline_score"] == 0
    assert head["score"] == 0.0625
    assert head["skill"] is None
    assert head["skill_low"] is None
    assert head["skill_high"] is None
    assert h.warnings_of(result).count(("skill_undefined", None)) == 1
    assert c.count_code(result, "skill_undefined") == 1
    assert head["interval_method"] == "bootstrap_rows"
    assert head["interval_low"] == 0.0625
    assert head["interval_high"] == 0.0625
    assert head["n_resamples_used"] == 200
    assert head["n_resamples_skill_used"] == 0
    assert head["p_value"] is None
    assert head["n_permutations"] is None
    assert isinstance(head["p_value_reason"], str) and head["p_value_reason"].strip()
    c.assert_strict_json(result)


def test_resample_with_a_zero_baseline_is_left_out_of_the_skill_interval_only():
    """A resample whose baseline error is zero keeps its score and loses only its skill."""
    frame, training, holdout = _random_scheme_zero_mean_table()
    frame.loc[holdout[0], "y"] = 2.0
    frame.loc[holdout[1], "y"] = -2.0
    result = c.run(frame, families=_constant_family(0.25, len(frame)), task="regression")
    assert h.holdout_rows(result) == holdout

    y_hold = frame["y"].to_numpy()[holdout]
    baseline_score = h.mse_by_hand(y_hold, np.zeros(60))
    assert baseline_score == pytest.approx(8 / 60)
    score = h.mse_by_hand(y_hold, np.full(60, 0.25))
    head = result.headline
    assert head["baseline"] == "training_mean"
    assert head["baseline_score"] == pytest.approx(baseline_score, rel=1e-12)
    assert head["score"] == pytest.approx(score, rel=1e-12)
    assert head["skill"] == pytest.approx(1 - score / baseline_score, rel=1e-9)

    assert head["n_resamples_used"] == 200
    assert 100 <= head["n_resamples_skill_used"] < 200
    assert isinstance(head["skill_low"], float) and isinstance(head["skill_high"], float)
    assert head["interval_low"] <= head["score"] <= head["interval_high"]
    assert "skill_undefined" not in c.codes(result)
    assert "no_skill_interval" not in c.codes(result)
    assert isinstance(head["p_value"], float)
    c.assert_strict_json(result)


def test_no_skill_interval_when_fewer_than_half_the_resamples_have_a_baseline_error():
    """Skill has a value but fewer than half the resamples can compute it: `no_skill_interval`."""
    frame, values = _one_wrong_previous_value_series()
    result = _run_one_wrong(frame, values)
    assert h.holdout_rows(result) == list(range(192, 256))

    head = result.headline
    assert head["baseline"] == "previous_value"
    assert head["baseline_score"] == pytest.approx(25 / 64, rel=1e-12)
    assert head["score"] == 0.25
    assert head["skill"] == pytest.approx(1 - 0.25 / (25 / 64), rel=1e-9)
    assert head["interval_method"] == "bootstrap_blocks"
    assert head["interval_low"] == 0.25
    assert head["interval_high"] == 0.25
    assert head["n_resamples_used"] == 200
    assert 0 < head["n_resamples_skill_used"] < 100
    assert head["skill_low"] is None
    assert head["skill_high"] is None
    assert h.warnings_of(result).count(("no_skill_interval", None)) == 1
    assert c.count_code(result, "no_skill_interval") == 1
    assert "skill_undefined" not in c.codes(result)
    c.assert_strict_json(result)


def test_skill_undefined_and_no_skill_interval_are_two_codes():
    """D42: `skill_undefined` is for a null headline skill only; `no_skill_interval` for a missing skill interval only."""
    flat = c.run(_flat_holdout_series(), time_col="t")
    assert flat.headline["skill"] is None
    assert h.warnings_of(flat).count(("skill_undefined", None)) == 1
    assert c.count_code(flat, "skill_undefined") == 1
    assert "no_skill_interval" not in c.codes(flat)

    frame, values = _one_wrong_previous_value_series()
    one_wrong = _run_one_wrong(frame, values)
    head = one_wrong.headline
    assert isinstance(head["skill"], float)
    assert head["interval_method"] is not None
    assert head["skill_low"] is None
    assert h.warnings_of(one_wrong).count(("no_skill_interval", None)) == 1
    assert c.count_code(one_wrong, "no_skill_interval") == 1
    assert "skill_undefined" not in c.codes(one_wrong)

    panel = h.panel_frame(n_times=40, per_time=3, seed=74, persistent=False)
    short = c.run(panel, time_col="date", task="regression")
    head = short.headline
    assert isinstance(head["skill"], float)
    assert head["interval_method"] is None
    assert "no_interval" in c.codes(short)
    assert "skill_undefined" not in c.codes(short)
    assert "no_skill_interval" not in c.codes(short)

    ordinary = c.run(h.time_frame(n=200, seed=75), time_col="date")
    assert isinstance(ordinary.headline["skill"], float)
    assert "skill_undefined" not in c.codes(ordinary)
    assert "no_skill_interval" not in c.codes(ordinary)


def test_single_column_skill_that_cannot_be_computed_flags_nothing():
    """D36 in the leak screen: a column whose single-column skill has no value stays a feature, unflagged."""
    n = 240
    rng = np.random.default_rng(76)
    y = rng.normal(size=n) * 3.0
    y[20:180] = 0.0
    frame = pd.DataFrame({"t": np.arange(n, dtype=float), "x1": rng.normal(size=n),
                          "lead": 2.0 * y + 1.0, "y": y})
    result = c.run(frame, time_col="t", task="regression")
    assert h.holdout_rows(result) == list(range(180, 240))

    record = h.columns_of(result)["lead"]
    assert record["role"] == "feature"
    assert record["reason"] is None
    assert record["leak_rule"] is None
    warnings = h.warnings_of(result)
    assert ("suspected_leak", "lead") not in warnings
    assert ("strong_single_column", "lead") not in warnings
    assert isinstance(result.headline["skill"], float)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_every_resample_counts_for_skill_when_no_baseline_error_is_zero(task):
    """With no zero baseline error, every resample that has a score has a skill too."""
    frame = h.clf_frame(n=240, seed=77) if task == "classification" else h.reg_frame(n=240, seed=77)
    result = c.run(frame)
    head = result.headline
    assert head["n_resamples_skill_used"] == head["n_resamples_used"]
    if task == "regression":
        assert head["n_resamples_used"] == 200
    assert head["skill_low"] <= head["skill"] <= head["skill_high"]
    assert "skill_undefined" not in c.codes(result)
    assert "no_skill_interval" not in c.codes(result)


def test_stages_skill_with_a_zero_baseline_is_none():
    """`stages.skill` returns None, not an error, for a zero squared-error baseline."""
    from mlprofessor import stages

    assert stages.skill("mse", 1.0, 0.0) is None
    assert stages.skill("mse", 0.0, 0.0) is None
    assert stages.skill("mse", 1.0, 4.0) == pytest.approx(0.75)


def test_command_line_finishes_on_a_flat_holdout(tmp_path):
    """The command line on a flat holdout: exit 0, strict JSON, null skill, the warning printed."""
    path = tmp_path / "flat.csv"
    _flat_holdout_series().to_csv(path, index=False)
    out = tmp_path / "out"
    proc = h.run_cli(["profile", "--data", path, "--target", "y", "--time-col", "t",
                      "--out", out, "--profile", "fast", "--n-jobs", "1"])
    assert proc.returncode == 0, proc.stderr
    doc = h.load_json_strict(out / "profile.json")
    assert doc["headline"]["skill"] is None
    assert any("skill_undefined" in line for line in proc.stdout.splitlines())
