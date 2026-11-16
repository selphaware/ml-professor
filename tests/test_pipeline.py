"""T9: `profile()` in memory, random scheme, fast profile, through the seam (spec 2.6, 3).

Only `predictions`, `models`, `rows`, `columns`, `warnings` and the keys
metric, score, model, baseline, baseline_score, skill of `headline` are used.
"""
import math

import numpy as np
import pandas as pd
import pytest

import mlp_helpers as h


def _run(frame, target="y", families=None, steps=None, seed=0, **kw):
    import mlprofessor
    from mlprofessor import extensions

    args = dict(h.FAST)
    args.update(kw)
    with extensions.use(families=families, steps=steps):
        return mlprofessor.profile(frame, target, seed=seed, **args)


def _with_missing_targets(frame, rows):
    frame = frame.copy()
    frame["y"] = frame["y"].astype(float)
    frame.loc[rows, "y"] = np.nan
    return frame


def _table(task, n=200, seed=1):
    return h.clf_frame(n, seed=seed) if task == "classification" else h.reg_frame(n, seed=seed)


# ------------------------------------------------------------------ F1

@pytest.mark.parametrize("task", ["classification", "regression"])
def test_f1_no_fit_receives_a_holdout_row(task):
    """F1: no model fit and no preprocessing fit ever sees a holdout row, nor its own fold's validation rows."""
    missing = [3, 30, 77, 150, 199]
    frame = _with_missing_targets(_table(task), missing)
    log = h.Log()
    res = _run(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)])

    hold = set(h.holdout_rows(res))
    usable = set(range(len(frame))) - set(missing)
    training = usable - hold
    assert hold and hold <= usable
    for ev in log.events:
        if ev["action"] == "fit":
            assert ev["is_frame"]
            assert not set(ev["rows"]) & hold
            assert not set(ev["rows"]) & set(missing)
            assert ev["y_index_ok"]
            if task == "classification":
                assert set(ev["y"]) <= {0, 1}

    for kind in ["model", "step"]:
        objs = log.objects(kind)
        assert len(objs) == 6, f"{kind}: five folds and the final fit, one fresh object each"
        assert all(o["n_fit"] == 1 for o in objs)
        finals = [o for o in objs if set(o["fit"]) == training]
        assert len(finals) == 1
        folds = [o for o in objs if o is not finals[0]]
        validations = []
        for o in folds:
            fit = set(o["fit"])
            assert fit < training
            validations.append(training - fit)
        assert sorted(r for v in validations for r in v) == sorted(training)
    fold_models = [o for o in log.objects("model") if set(o["fit"]) != training]
    for o in fold_models:
        assert set(o["predict"]) == training - set(o["fit"])
    step_vals = {frozenset(training - set(o["fit"])) for o in log.objects("step") if set(o["fit"]) != training}
    model_vals = {frozenset(o["predict"]) for o in fold_models}
    assert step_vals == model_vals


def test_steps_receive_feature_columns_only():
    frame = h.clf_frame(200, seed=3)
    log = h.Log()
    _run(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)])
    for o in log.objects("step"):
        assert set(o["fit_columns"]) == {"x1", "z1", "z2"}


def test_f1_builtin_preprocessing_never_sees_holdout_values():
    """F1: replacing holdout feature values with extremes changes no fold score."""
    frame = h.clf_frame(240, seed=2)
    first = _run(frame)
    hold = h.holdout_rows(first)
    changed = frame.copy()
    changed.loc[hold, "x1"] = 1e6
    changed.loc[hold, "z1"] = -1e6
    changed.loc[hold, "z2"] = 5e5
    second = _run(changed)
    assert h.holdout_rows(second) == hold
    before = {m["config"]: m["fold_scores"] for m in h.models_of(first)}
    after = {m["config"]: m["fold_scores"] for m in h.models_of(second)}
    assert before == after


def test_f1_memorising_model_is_not_perfect_on_the_holdout():
    """F1: a model refitted on all rows would recall the holdout targets exactly."""
    frame = h.clf_frame(240, seed=4)
    log = h.Log()
    res = _run(frame, families=[h.family("mem", log, h.MemorizeEstimator)])
    y = frame["y"].to_numpy()
    rows = h.holdout_rows(res)
    assert res.headline["score"] == pytest.approx(h.auc_by_hand(y[rows], res.predictions["y_pred"]))
    assert res.headline["score"] < 0.85


def test_only_the_chosen_configuration_predicts_the_holdout():
    frame = h.clf_frame(200, seed=5)
    log = h.Log()
    noise = np.random.default_rng(0).random(len(frame))
    fams = [h.family("a", log), h.family("b", log, h.LookupEstimator, values=noise)]
    res = _run(frame, families=fams)
    hold = set(h.holdout_rows(res))
    touching = [o for o in log.objects("model") if set(o["predict"]) & hold]
    assert len(touching) == 1
    assert set(touching[0]["predict"]) == hold
    assert f"{touching[0]['family']}-0" == res.headline["model"]
    assert not set(touching[0]["fit"]) & hold


# ------------------------------------------------------------------ F3, F4

def test_f3_headline_is_the_holdout_score_not_cv():
    """F3: a family perfect on the holdout and random elsewhere: headline 1.0, cv about 0.5."""
    frame = h.clf_frame(240, seed=6)
    probe = _run(frame, families=[h.family("probe", h.Log())])
    hold = h.holdout_rows(probe)
    y = frame["y"].to_numpy()
    values = np.random.default_rng(1).random(len(frame))
    values[hold] = y[hold]
    res = _run(frame, families=[h.family("oracle", h.Log(), h.LookupEstimator, values=values)])

    assert h.holdout_rows(res) == hold
    recomputed = h.auc_by_hand(y[hold], res.predictions["y_pred"].to_numpy())
    assert recomputed == pytest.approx(1.0)
    assert res.headline["score"] == pytest.approx(recomputed)
    chosen = [m for m in h.models_of(res) if m["chosen"]][0]
    assert chosen["cv_mean"] < 0.8


def test_f4_classification_metric_by_hand():
    """F4: the headline is ROC AUC against chance, worked out here from chosen predictions."""
    frame = h.clf_frame(200, seed=7)
    y = frame["y"].to_numpy()
    rng = np.random.default_rng(2)
    values = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.2, size=len(y)), 0.01, 0.99)
    res = _run(frame, families=[h.family("pick", h.Log(), h.LookupEstimator, values=values)])
    rows = h.holdout_rows(res)
    auc = h.auc_by_hand(y[rows], values[rows])
    hl = res.headline
    assert hl["metric"] == "roc_auc"
    assert hl["baseline"] == "chance"
    assert hl["baseline_score"] == pytest.approx(0.5)
    assert hl["model"] == "pick-0"
    assert hl["score"] == pytest.approx(auc, abs=1e-12)
    assert hl["skill"] == pytest.approx(2 * auc - 1, abs=1e-12)
    assert np.allclose(res.predictions["y_pred"].to_numpy(dtype=float), values[rows])


def test_regression_metric_by_hand():
    missing = [0, 44, 120]
    frame = _with_missing_targets(h.reg_frame(220, seed=8), missing)
    y = frame["y"].to_numpy()
    values = np.where(np.isnan(y), 0.0, y + np.random.default_rng(3).normal(0, 1.0, size=len(y)))
    res = _run(frame, families=[h.family("pick", h.Log(), h.LookupEstimator, values=values)])
    rows = h.holdout_rows(res)
    training = sorted(set(range(len(y))) - set(missing) - set(rows))
    score = h.mse_by_hand(y[rows], values[rows])
    base = h.mse_by_hand(y[rows], np.full(len(rows), y[training].mean()))
    hl = res.headline
    assert hl["metric"] == "mse"
    assert hl["baseline"] == "training_mean"
    assert hl["score"] == pytest.approx(score, rel=1e-9)
    assert hl["baseline_score"] == pytest.approx(base, rel=1e-9)
    assert hl["skill"] == pytest.approx(1 - score / base, rel=1e-9)


# ------------------------------------------------------------------ selection

def test_selection_highest_auc_lowest_mse():
    frame = h.clf_frame(200, seed=9)
    y = frame["y"].to_numpy()
    rng = np.random.default_rng(4)
    weak = rng.random(len(y))
    strong = np.clip(y * 0.6 + rng.random(len(y)) * 0.4, 0, 1)
    res = _run(frame, families=[h.family("weak", h.Log(), h.LookupEstimator, values=weak),
                                h.family("strong", h.Log(), h.LookupEstimator, values=strong)])
    models = {m["config"]: m for m in h.models_of(res)}
    assert models["strong-0"]["cv_mean"] > models["weak-0"]["cv_mean"]
    assert res.headline["model"] == "strong-0"
    assert [m["config"] for m in h.models_of(res) if m["chosen"]] == ["strong-0"]

    rframe = h.reg_frame(200, seed=10)
    ry = rframe["y"].to_numpy()
    good = ry + rng.normal(0, 0.5, size=len(ry))
    bad = ry + rng.normal(0, 3.0, size=len(ry))
    res = _run(rframe, families=[h.family("bad", h.Log(), h.LookupEstimator, values=bad),
                                 h.family("good", h.Log(), h.LookupEstimator, values=good)])
    models = {m["config"]: m for m in h.models_of(res)}
    assert models["good-0"]["cv_mean"] < models["bad-0"]["cv_mean"]
    assert res.headline["model"] == "good-0"


def test_ties_go_to_earlier_family_then_earlier_grid_position():
    frame = h.clf_frame(200, seed=11)
    same = np.random.default_rng(5).random(len(frame))
    res = _run(frame, families=[h.family("zeta", h.Log(), h.LookupEstimator, values=same),
                                h.family("alpha", h.Log(), h.LookupEstimator, values=same)])
    models = {m["config"]: m for m in h.models_of(res)}
    assert models["zeta-0"]["cv_mean"] == models["alpha-0"]["cv_mean"]
    assert res.headline["model"] == "zeta-0"
    res = _run(frame, families=[h.family("one", h.Log(), h.LookupEstimator, values=same,
                                         grid=[{"k": 1}, {"k": 2}])])
    assert [m["config"] for m in h.models_of(res)] == ["one-0", "one-1"]
    assert res.headline["model"] == "one-0"


# ------------------------------------------------------------------ rows

def test_f6_rows_add_up():
    """F6: every loaded row is accounted for."""
    missing = list(range(0, 200, 15))
    frame = _with_missing_targets(h.clf_frame(200, seed=12), missing)
    res = _run(frame, families=[h.family("rec", h.Log())])
    rows = h.rows_of(res)
    assert list(res.rows["item"]) == h.ROWS_ITEMS
    usable = 200 - len(missing)
    assert rows["loaded"] == 200
    assert rows["dropped_missing_target"] == len(missing)
    assert rows["dropped_missing_time"] == rows["dropped_gap"] == rows["dropped_row_cap"] == 0
    assert rows["holdout"] == math.ceil(usable / 4) == len(res.predictions)
    assert rows["training"] == usable - rows["holdout"]


def test_f6_row_cap_counted():
    """F6: the fast profile's cap is counted, never silent."""
    missing = [5, 600, 1200, 2400]
    frame = _with_missing_targets(h.clf_frame(2600, seed=13), missing)
    res = _run(frame, families=[h.family("rec", h.Log())])
    rows = h.rows_of(res)
    assert rows["dropped_missing_target"] == len(missing)
    assert rows["dropped_row_cap"] == 2600 - len(missing) - 2000
    assert rows["training"] + rows["holdout"] == 2000
    assert rows["holdout"] == 500
    assert sum(rows[k] for k in h.ROWS_ITEMS[1:]) == rows["loaded"] == 2600
    assert not set(h.holdout_rows(res)) & set(missing)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_holdout_and_folds_are_split_random_of_usable_rows(task):
    from mlprofessor import stages

    missing = [2, 9, 50, 51, 133, 170, 219]
    frame = _with_missing_targets(_table(task, n=220, seed=14), missing)
    log = h.Log()
    res = _run(frame, families=[h.family("rec", log)], seed=5)
    usable = np.array([i for i in range(len(frame)) if i not in missing])
    classes = frame["y"].to_numpy()[usable] if task == "classification" else None
    split = stages.split_random(len(usable), classes=classes, seed=5)
    assert h.holdout_rows(res) == usable[np.asarray(split["holdout"], dtype=int)].tolist()
    expected_vals = {frozenset(usable[np.asarray(f["validation"], dtype=int)].tolist())
                     for f in split["folds"]}
    hold = set(h.holdout_rows(res))
    got_vals = {frozenset(o["predict"]) for o in log.objects("model") if not set(o["predict"]) & hold}
    assert got_vals == expected_vals


def test_predictions_row_is_source_position():
    frame = h.clf_frame(210, seed=15, labels=("no", "yes"))
    frame.index = np.random.default_rng(6).permutation(np.arange(5000, 5210))
    frame["y"] = frame["y"].astype(object)
    frame.iloc[[4, 8, 100], frame.columns.get_loc("y")] = None
    res = _run(frame, families=[h.family("rec", h.Log())])
    pred = res.predictions
    assert list(pred.columns)[:3] == ["row", "y_true", "y_pred"]
    rows = [int(r) for r in pred["row"]]
    assert rows == sorted(rows)
    assert [str(v) for v in pred["y_true"]] == [str(frame["y"].iloc[r]) for r in rows]
    assert not {4, 8, 100} & set(rows)


@pytest.mark.parametrize("labels", [(0, 1), ("yes", "no")])
def test_y_pred_is_probability_of_positive_class(labels):
    """With labels ("yes", "no") the positive class "yes" sits where the signal is low."""
    frame = h.clf_frame(300, seed=16, labels=labels, signal=3.0)
    res = _run(frame)
    pred = res.predictions
    positive = 1 if labels == (0, 1) else "yes"
    y_pred = pred["y_pred"].to_numpy(dtype=float)
    assert np.all((y_pred >= 0) & (y_pred <= 1))
    is_pos = np.array([v == positive for v in frame["y"].iloc[pred["row"].to_numpy()]])
    assert y_pred[is_pos].mean() > y_pred[~is_pos].mean() + 0.1


def test_shuffled_target_has_no_skill():
    frame = h.clf_frame(400, seed=17, signal=3.0)
    real = _run(frame)
    shuffled = frame.copy()
    shuffled["y"] = np.random.default_rng(7).permutation(frame["y"].to_numpy())
    noise = _run(shuffled)
    assert real.headline["skill"] > 0.45
    assert abs(noise.headline["skill"]) < 0.35


def test_constant_column_changes_no_score():
    frame = h.clf_frame(200, seed=18)
    with_const = frame.copy()
    with_const.insert(1, "k", 3.0)
    a, b = _run(frame), _run(with_const)
    assert h.records(a.models) == h.records(b.models)
    assert a.headline["score"] == b.headline["score"]
    assert h.records(a.predictions) == h.records(b.predictions)
    assert h.columns_of(b)["k"]["reason"] == "constant"


def test_same_seed_same_tables_other_seed_other_holdout():
    frame = h.reg_frame(200, seed=19)
    a, b = _run(frame, seed=3), _run(frame, seed=3)
    for part in ["predictions", "models", "rows", "columns", "warnings"]:
        assert h.records(getattr(a, part)) == h.records(getattr(b, part)), part
    keys = ["metric", "score", "model", "baseline", "baseline_score", "skill"]
    assert h.plain({k: a.headline[k] for k in keys}) == h.plain({k: b.headline[k] for k in keys})
    c = _run(frame, seed=4)
    assert h.holdout_rows(c) != h.holdout_rows(a)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_blas_held_to_one_thread_inside_fit(n_jobs):
    frame = h.clf_frame(200, seed=20)
    log = h.Log()
    _run(frame, families=[h.family("rec", log, capture_threads=True)], n_jobs=n_jobs)
    assert log.threads
    blas = [pool for info in log.threads for pool in info if pool.get("user_api") == "blas"]
    assert blas, "numpy's BLAS should be visible to threadpoolctl"
    assert all(pool["num_threads"] == 1 for pool in blas)


def test_default_fast_families():
    res = _run(h.clf_frame(200, seed=21))
    models = h.models_of(res)
    assert [m["config"] for m in models] == ["linear-0", "tree-0"]
    assert [m["family"] for m in models] == ["linear", "tree"]
    assert all(m["status"] == "fitted" for m in models)
    assert models[1]["params"] == {}
    assert sum(bool(m["chosen"]) for m in models) == 1
