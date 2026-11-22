"""T25: no number that is not finite is reported, or turned into null (D44); D48.

Spec: section 3 ("Numbers that are not finite", steps 4, 5, 8 to 13), 2.4,
2.5, 2.7, 5.1, 5.2. Regression unless stated; fast profile.

Red today: every test but one. Green today:
`test_a_prediction_that_is_not_finite_counts_as_raising` (it fails on M6).

Numbers are read from `result.to_dict()` and `result.headline` as the
application gives them (`k.assert_raw_numbers`): a helper that reads NaN as
None first would hide the very fault this file exists for.
"""
import numpy as np
import pandas as pd
import pytest

import cp3_helpers as c
import cp4_helpers as k
import mlp_helpers as h

BASELINE_NAMES = ("training_mean", "previous_value")


def _names_a_baseline(error):
    return any(name in str(error) for name in BASELINE_NAMES)


def test_error_class_and_code():
    from mlprofessor import errors

    assert issubclass(errors.NonFiniteScoreError, errors.ProfileError)
    assert errors.NonFiniteScoreError.code == "non_finite_score"


# ------------------------------------------------------ the target as the cause

def _extreme(case):
    frame = k.reg_table(400, seed=41)
    if case == "largest_double_at_row_5":
        frame.loc[5, "y"] = k.LARGEST_DOUBLE
    elif case == "largest_double_at_row_395":
        frame.loc[395, "y"] = k.LARGEST_DOUBLE
    else:
        frame["y"] = frame["y"] * 1e160
    return frame


EXTREME = ["largest_double_at_row_5", "largest_double_at_row_395", "target_times_1e160"]


@pytest.mark.parametrize("time_col", [None, "t"], ids=["random", "time_ordered"])
@pytest.mark.parametrize("case", EXTREME)
def test_extreme_target_stops_with_one_named_error_before_any_fit(tmp_path, case, time_col):
    """A target too large for squared error stops with `non_finite_score`, naming the baseline, before any fit."""
    from mlprofessor.errors import NonFiniteScoreError

    frame = _extreme(case)
    assert np.isfinite(frame["y"].to_numpy()).all()
    log = h.Log()
    out = tmp_path / "out"
    with pytest.raises(NonFiniteScoreError) as caught:
        c.run(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)],
              time_col=time_col, out_dir=out)
    assert _names_a_baseline(caught.value), str(caught.value)
    assert log.events == [], "a step or a model was fitted before the baselines were scored"
    assert h.files_in(out) == []


def test_extreme_target_through_the_command_line(tmp_path):
    frame = _extreme("largest_double_at_row_395")
    path = tmp_path / "extreme.csv"
    frame.to_csv(path, index=False)
    read_back = pd.read_csv(path, low_memory=False)["y"].to_numpy()
    assert np.isfinite(read_back).all() and read_back[395] > 1e308, "the CSV holds the value as it is"

    out = tmp_path / "out"
    proc = h.run_cli(["profile", "--data", path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1"])
    assert proc.returncode == 3, proc.stdout + proc.stderr
    lines = [line for line in proc.stderr.splitlines() if line.strip()]
    assert len(lines) == 1, proc.stderr
    assert lines[0].startswith("error: non_finite_score:"), proc.stderr
    assert h.files_in(out) == []


def test_a_target_cell_that_spoils_only_the_folds_stops_before_any_fit(tmp_path):
    """A cell of 1e155 in a training block leaves both holdout baselines finite; the leak screen must stop the run."""
    from mlprofessor.errors import NonFiniteScoreError

    frame = k.reg_table(400, seed=42)
    frame.loc[120, "y"] = 1e155
    y = frame["y"].to_numpy(dtype=float)
    holdout = np.arange(300, 400)

    with np.errstate(over="ignore", invalid="ignore"):
        train_mean = y[:300].mean()
        mean_error = ((y[holdout] - train_mean) ** 2).sum()
        previous_error = ((y[holdout] - y[holdout - 1]) ** 2).sum()
        fold_error = (y[120] - y[:100].mean()) ** 2
    assert np.isfinite(mean_error) and mean_error < 1e308
    assert np.isfinite(previous_error)
    assert not np.isfinite(fold_error)

    log = h.Log()
    out = tmp_path / "out"
    with pytest.raises(NonFiniteScoreError) as caught:
        c.run(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)],
              time_col="t", out_dir=out)
    assert _names_a_baseline(caught.value), str(caught.value)
    assert log.events == [], "a step or a model was fitted before the leak screen stopped"
    assert h.files_in(out) == []


# ---------------------------------------------- a configuration as the cause

def _sound_values(frame, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    y = frame["y"].to_numpy(dtype=float)
    return scale * y + 0.1 * rng.normal(size=len(y))


def test_a_fold_score_that_is_not_finite_fails_the_configuration():
    frame = h.reg_frame(n=240, seed=43)
    log = h.Log()
    families = [
        h.family("far", log, h.LookupEstimator, values=np.full(len(frame), 1e200)),
        h.family("sound", log, h.LookupEstimator, values=_sound_values(frame, 44)),
    ]
    result = c.run(frame, families=families)
    doc = result.to_dict()
    models = {m["config"]: m for m in doc["models"]}

    far = models["far-0"]
    assert far["status"] == "failed"
    assert isinstance(far["error"], str) and far["error"].strip()
    assert far["fold_scores"] == [None] * 5
    assert far["cv_mean"] is None and far["cv_std"] is None
    assert far["chosen"] is False

    sound = models["sound-0"]
    assert sound["status"] == "fitted" and sound["chosen"] is True
    assert result.headline["model"] == "sound-0"
    assert [w["code"] for w in doc["warnings"]].count("model_failed") == 1
    k.assert_raw_numbers(result)


def test_every_fold_score_not_finite_is_all_models_failed(tmp_path):
    from mlprofessor.errors import AllModelsFailedError

    frame = h.reg_frame(n=240, seed=43)
    out = tmp_path / "out"
    with pytest.raises(AllModelsFailedError):
        c.run(frame, families=[h.family("far", h.Log(), h.LookupEstimator,
                                        values=np.full(len(frame), 1e200))],
              out_dir=out)
    assert h.files_in(out) == []


def test_a_holdout_score_that_is_not_finite_stops_the_run_without_a_fallback(tmp_path):
    """A finite holdout prediction whose score is not finite stops the run; the next configuration is not tried."""
    from mlprofessor.errors import NonFiniteScoreError

    frame = h.reg_frame(n=240, seed=45)
    sound = _sound_values(frame, 46)
    first = c.run(frame, families=[h.family("sound", h.Log(), h.LookupEstimator, values=sound)])
    holdout = h.holdout_rows(first)
    hold = set(holdout)

    wild = frame["y"].to_numpy(dtype=float).copy()
    wild[holdout] = 1e200
    log = h.Log()
    families = [
        h.family("sound", log, h.LookupEstimator, values=sound),
        h.family("wild", log, h.LookupEstimator, values=wild),
    ]
    out = tmp_path / "out"
    with pytest.raises(NonFiniteScoreError) as caught:
        c.run(frame, families=families, out_dir=out)
    assert "wild-0" in str(caught.value), str(caught.value)
    assert h.files_in(out) == []
    reached = [ev["family"] for ev in log.events
               if ev["action"] == "predict" and set(ev["rows"]) & hold]
    assert reached and set(reached) == {"wild"}, f"holdout rows reached {reached}"


def test_a_prediction_that_is_not_finite_counts_as_raising():
    """M6: a prediction that is not finite fails its configuration, before or after the choice, and is never written."""
    frame = h.clf_frame(n=240, seed=47)
    y = frame["y"].to_numpy(dtype=float)
    rng = np.random.default_rng(48)
    strong = 0.8 * y + rng.uniform(0.0, 0.2, size=len(y))
    sound = 0.3 * y + rng.uniform(0.0, 0.7, size=len(y))

    first = c.run(frame, families=[h.family("sound", h.Log(), h.LookupEstimator, values=sound)])
    holdout = h.holdout_rows(first)
    training = sorted(set(range(len(frame))) - set(holdout))

    clean = c.run(frame, families=[h.family("holdout_nan", h.Log(), h.LookupEstimator, values=strong)])
    clean_model = clean.to_dict()["models"][0]
    assert clean_model["status"] == "fitted" and clean_model["chosen"] is True
    assert all(k.is_number(s) for s in clean_model["fold_scores"])

    holdout_nan = strong.copy()
    holdout_nan[holdout[:3]] = np.nan
    folds_nan = sound.copy()
    folds_nan[training[:3]] = np.nan
    log = h.Log()
    families = [
        h.family("folds_nan", log, h.LookupEstimator, values=folds_nan),
        h.family("sound", log, h.LookupEstimator, values=sound),
        h.family("holdout_nan", log, h.LookupEstimator, values=holdout_nan),
    ]
    result = c.run(frame, families=families)
    doc = result.to_dict()
    models = {m["config"]: m for m in doc["models"]}

    early = models["folds_nan-0"]
    assert early["status"] == "failed"
    assert early["fold_scores"] == [None] * 5 and early["cv_mean"] is None

    late = models["holdout_nan-0"]
    assert late["status"] == "failed" and late["chosen"] is False
    assert isinstance(late["error"], str) and late["error"].strip()
    assert late["fold_scores"] == pytest.approx(clean_model["fold_scores"], rel=1e-12)
    assert late["cv_mean"] == pytest.approx(clean_model["cv_mean"], rel=1e-12)

    assert models["sound-0"]["chosen"] is True
    assert result.headline["model"] == "sound-0"
    assert [w["code"] for w in doc["warnings"]].count("model_failed") == 2

    rows = h.holdout_rows(result)
    assert rows == holdout
    y_pred = result.predictions["y_pred"].to_numpy(dtype=float)
    assert np.isfinite(y_pred).all()
    assert np.allclose(y_pred, sound[rows], rtol=0, atol=1e-12)
    k.assert_raw_numbers(result)


# ------------------------------------------ feature values too large (D48)

BIG_CELLS = {10: 1e39, 77: -1e200, 150: k.LARGEST_DOUBLE, 201: np.inf}
EDGE_CELLS = {11: 1e38, 78: -1e38}


def _big_table():
    frame = h.clf_frame(n=300, seed=49)
    rng = np.random.default_rng(50)
    big = rng.normal(size=len(frame))
    edge = rng.normal(size=len(frame))
    for r, v in BIG_CELLS.items():
        big[r] = v
    for r, v in EDGE_CELLS.items():
        edge[r] = v
    frame.insert(3, "big", big)
    frame.insert(4, "edge", edge)
    return frame


def test_feature_values_too_large_for_the_tree_models_are_read_as_missing():
    frame = _big_table()
    log = c.KeepLog()
    result = c.run(frame, families=[h.family("mat", log, c.MatrixEstimator)],
                   steps=[c.keep_step_factory(log)])
    doc = result.to_dict()
    columns = {col["name"]: col for col in doc["columns"]}

    assert columns["big"]["role"] == "feature" and columns["big"]["kind"] == "numeric"
    assert columns["big"]["n_infinite"] == len(BIG_CELLS)
    assert columns["big"]["n_imputed"] == len(BIG_CELLS)
    assert columns["edge"]["role"] == "feature"
    assert columns["edge"]["n_infinite"] == 0
    assert [w["column"] for w in doc["warnings"] if w["code"] == "infinite_values"] == ["big"]

    step_fits = log.calls("step", "fit")
    assert step_fits, "the step was fitted"
    for ev in step_fits:
        seen = ev["frame"]
        for r in BIG_CELLS:
            if r in seen.index:
                assert np.isnan(seen.loc[r, "big"]), f"big at row {r} reached a step as {seen.loc[r, 'big']!r}"
        for r, v in EDGE_CELLS.items():
            if r in seen.index:
                assert seen.loc[r, "edge"] == v, f"edge at row {r} reached a step as {seen.loc[r, 'edge']!r}"

    model_fits = log.calls("model", "fit")
    assert model_fits, "the model was fitted"
    for ev in model_fits:
        assert np.isfinite(ev["values"]).all(), "a matrix that is not finite was handed to a model"
    k.assert_raw_numbers(result)


def test_column_roles_count_too_large_values_with_the_infinite_ones():
    from mlprofessor import stages

    roles = stages.column_roles(_big_table(), "y")
    by_name = {r["name"]: r for r in h.records(roles)}
    assert by_name["big"]["role"] == "feature"
    assert by_name["big"]["n_infinite"] == len(BIG_CELLS)
    assert by_name["edge"]["role"] == "feature"
    assert by_name["edge"]["n_infinite"] == 0


def test_preprocessor_reads_infinite_and_too_large_cells_as_missing():
    from mlprofessor import stages

    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 1e200, np.inf, -1e39]})
    prep = stages.Preprocessor({"a": "numeric"}).fit(frame)
    out = prep.transform(frame)
    got = out["a"].to_numpy(dtype=float)
    assert np.isfinite(got).all()

    filled = np.array([1.0, 2.0, 3.0, 4.0, 2.5, 2.5, 2.5])
    readings = [(filled - filled.mean()) / filled.std(ddof=d) for d in (0, 1)]
    assert any(np.allclose(got, want, rtol=0, atol=1e-12) for want in readings), got


# ----------------------------------------------- a complex target (D48)

def test_a_complex_target_is_not_numeric():
    import mlprofessor
    from mlprofessor import stages
    from mlprofessor.errors import TaskAmbiguousError, TaskUnsupportedError

    many = pd.Series(np.arange(40) + 2j)
    with pytest.raises(TaskUnsupportedError):
        stages.decide_task(many, task="regression")
    with pytest.raises(TaskAmbiguousError):
        stages.decide_task(many, task="auto")
    two = stages.decide_task(pd.Series([1 + 2j, 3 - 1j] * 20), task="auto")
    assert two["type"] == "classification"
    assert two["n_distinct"] == 2

    frame = h.reg_frame(n=160, seed=51)
    frame["y"] = frame["y"].to_numpy() + 0.5j
    assert frame["y"].dtype.kind == "c"
    with pytest.raises(TaskUnsupportedError):
        mlprofessor.profile(frame, "y", task="regression", **h.FAST)
    with pytest.raises(TaskAmbiguousError):
        mlprofessor.profile(frame, "y", task="auto", **h.FAST)


# ----------------------------------------------- where the outcome is open

def test_a_target_on_a_very_large_scale_gives_finite_numbers_or_the_named_error():
    """At 1e80 a run may finish with finite numbers or stop with `non_finite_score`; never a null, never all_models_failed."""
    import mlprofessor
    from mlprofessor.errors import NonFiniteScoreError

    frame = h.reg_frame(n=240, seed=52)
    frame["y"] = frame["y"] * 1e80
    result = None
    try:
        result = h.stated_profile(frame, "y", **h.FAST)
    except NonFiniteScoreError:
        pass
    if result is not None:
        k.assert_raw_numbers(result)
