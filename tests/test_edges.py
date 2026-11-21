"""T16: every edge of spec section 7, invalid arguments (2.1), the fixed texts (5.4), D20."""
import numpy as np
import pandas as pd
import pytest

import mlp_helpers as h


def _run(frame, target="y", families="rec", **kw):
    import mlprofessor
    from mlprofessor import extensions

    if families == "rec":
        families = [h.family("rec", h.Log())]
    args = dict(h.FAST)
    args.update(kw)
    with extensions.use(families=families):
        return mlprofessor.profile(frame, target, **args)


# --------------------------------------------------------------- named errors

def _constant():
    f = h.clf_frame(150, seed=1)
    f["y"] = 1
    return f, {}


def _ambiguous():
    f = h.clf_frame(150, seed=2)
    f["y"] = np.arange(150) % 5
    return f, {}


def _no_features():
    y = np.random.default_rng(3).integers(0, 2, size=150)
    return pd.DataFrame({"k": 4.0, "row_id": np.arange(150), "y": y}), {}


def _few_of_a_class():
    f = h.clf_frame(200, seed=4)
    f["y"] = 0
    f.loc[[3, 50, 90, 120, 170], "y"] = 1
    return f, {}


def _training_one_class():
    f = h.time_frame(200, seed=5, task="classification")
    f["y"] = np.r_[np.zeros(150, dtype=int), np.arange(50) % 2]
    return f, {"time_col": "date"}


def _holdout_one_class():
    f = h.time_frame(200, seed=6, task="classification")
    f["y"] = np.r_[np.arange(150) % 2, np.zeros(50, dtype=int)]
    return f, {"time_col": "date"}


def _day_first():
    f = h.time_frame(150, seed=7)
    f["date"] = f["date"].dt.strftime("%d/%m/%Y")
    return f, {"time_col": "date"}


def _few_timestamps():
    f = h.reg_frame(100, seed=8)
    f["date"] = np.repeat(h.dates(5).to_numpy(), 20)
    return f, {"time_col": "date"}


def _duplicate_columns():
    f = h.clf_frame(150, seed=9)
    f.columns = ["x1", "x1", "z2", "y"]
    return f, {}


ERROR_CASES = {
    "target_absent": (lambda: (h.clf_frame(150), {"target": "nope"}), "ColumnNotFoundError"),
    "time_col_absent": (lambda: (h.clf_frame(150), {"time_col": "nope"}), "ColumnNotFoundError"),
    "target_constant": (_constant, "TargetConstantError"),
    "task_unsupported": (lambda: (h.reg_frame(150), {"task": "classification"}), "TaskUnsupportedError"),
    "task_ambiguous": (_ambiguous, "TaskAmbiguousError"),
    "no_features": (_no_features, "NoFeaturesError"),
    "too_few_rows": (lambda: (h.clf_frame(60), {}), "TooFewRowsError"),
    "too_few_of_a_class": (_few_of_a_class, "TooFewRowsError"),
    "training_one_class": (_training_one_class, "OneClassError"),
    "holdout_one_class": (_holdout_one_class, "OneClassError"),
    "time_column_invalid": (_day_first, "TimeColumnError"),
    "gap_too_large": (lambda: (h.time_frame(200), {"time_col": "date", "gap": 150}), "GapError"),
    "too_few_timestamps": (_few_timestamps, "TooFewRowsError"),
    "duplicate_columns": (_duplicate_columns, "DuplicateColumnsError"),
}


@pytest.mark.parametrize("case", sorted(ERROR_CASES))
def test_edge_raises_its_named_error(case, tmp_path):
    from mlprofessor import errors

    build, name = ERROR_CASES[case]
    frame, kw = build()
    target = kw.pop("target", "y")
    out = tmp_path / "out"
    with pytest.raises(getattr(errors, name)):
        _run(frame, target=target, out_dir=out, **kw)
    assert h.files_in(out) == []


def test_every_configuration_failing(tmp_path):
    from mlprofessor import errors

    with pytest.raises(errors.AllModelsFailedError):
        _run(h.clf_frame(150), families=[h.family("broken", h.Log(), h.FailingEstimator)],
             out_dir=tmp_path / "out")
    assert h.files_in(tmp_path / "out") == []


# ------------------------------------------------------------ invalid arguments

@pytest.mark.parametrize("kw", [
    {"task": "multiclass"}, {"profile": "medium"}, {"seed": -1}, {"seed": 1.5},
    {"gap": -1}, {"gap": 1.5}, {"gap": 1}, {"time_col": "y"},
    {"n_jobs": 0}, {"n_jobs": -2}, {"n_jobs": 1.5},
], ids=lambda kw: "-".join(f"{k}={v}" for k, v in kw.items()))
def test_invalid_argument_raises_input_error(kw, tmp_path):
    import mlprofessor
    from mlprofessor import errors

    args = {"profile": "fast", "n_jobs": 1}
    args.update(kw)
    with pytest.raises(errors.InputError):
        mlprofessor.profile(h.clf_frame(150), "y", out_dir=tmp_path / "out", **args)
    assert h.files_in(tmp_path / "out") == []


@pytest.mark.parametrize("data", [123, None, [1, 2, 3]])
def test_invalid_data_raises_input_error(data):
    import mlprofessor
    from mlprofessor import errors

    with pytest.raises(errors.InputError):
        mlprofessor.profile(data, "y", profile="fast", n_jobs=1)


def test_invalid_argument_reported_before_loading():
    import mlprofessor
    from mlprofessor import errors

    frame, _ = _duplicate_columns()
    with pytest.raises(errors.InputError):
        mlprofessor.profile(frame, "y", profile="medium", n_jobs=1)


# ------------------------------------------------------------ reported outcomes

def test_fold_with_one_class_not_scored():
    f = h.time_frame(300, seed=10, task="classification")
    y = f["y"].to_numpy().copy()
    y[:40] = 0
    f["y"] = y
    assert 0 < y[40:76].sum() < 36
    res = _run(f, time_col="date")
    doc = res.to_dict()
    assert ("fold_not_scored", None) in h.warnings_of(res)
    scored = [fold["scored"] for fold in doc["validation"]["folds"]]
    assert scored == [False, True, True, True, True]
    assert doc["cv"]["n_scored_folds"] == 4
    for m in h.models_of(res):
        assert m["fold_scores"][0] is None
        assert all(s is not None for s in m["fold_scores"][1:])


def test_negative_skill_reported_as_measured():
    frame = h.clf_frame(200, seed=11)
    y = frame["y"].to_numpy()
    rng = np.random.default_rng(0)
    worse = np.clip(1 - y * 0.5 + rng.normal(0, 0.3, size=len(y)), 0, 1)
    res = _run(frame, families=[h.family("bad", h.Log(), h.LookupEstimator, values=worse)])
    rows = h.holdout_rows(res)
    auc = h.auc_by_hand(y[rows], worse[rows])
    assert auc < 0.5
    assert res.headline["skill"] == pytest.approx(2 * auc - 1, abs=1e-12)

    rframe = h.reg_frame(200, seed=12)
    ry = rframe["y"].to_numpy()
    noisy = ry + rng.normal(0, 3 * ry.std(), size=len(ry))
    res = _run(rframe, families=[h.family("bad", h.Log(), h.LookupEstimator, values=noisy)])
    rows = h.holdout_rows(res)
    training = sorted(set(range(len(ry))) - set(rows))
    score = h.mse_by_hand(ry[rows], noisy[rows])
    base = h.mse_by_hand(ry[rows], np.full(len(rows), ry[training].mean()))
    assert 1 - score / base < 0
    assert res.headline["skill"] == pytest.approx(1 - score / base, rel=1e-9)


def test_dataframe_outcomes_under_time_order():
    rng = np.random.default_rng(13)
    f = h.time_frame(200, seed=13)
    late = rng.normal(size=200)
    late[:60] = np.nan                       # all missing in fold 1's training rows
    late[[100, 180]] = np.inf
    f["late"] = late
    f["cat"] = np.where(np.arange(200) < 150, np.where(np.arange(200) % 2, "a", "b"), "c")
    f["lists"] = [[i] for i in range(200)]
    f.loc[[70, 85], "date"] = pd.NaT
    res = _run(f, time_col="date")
    cols = h.columns_of(res)
    rows = h.rows_of(res)
    assert rows["dropped_missing_time"] == 2
    assert (cols["lists"]["role"], cols["lists"]["reason"]) == ("excluded", "unsupported_type")
    assert cols["late"]["role"] == "feature" and cols["late"]["n_infinite"] == 2
    assert cols["late"]["n_imputed"] == 62
    assert ("infinite_values", "late") in h.warnings_of(res)
    assert cols["cat"]["role"] == "feature" and cols["cat"]["kind"] == "categorical"
    assert len(res.predictions) == rows["holdout"]
    assert np.isfinite(res.predictions["y_pred"].to_numpy(dtype=float)).all()


def test_duplicate_rows_kept_and_reported():
    frame = h.clf_frame(150, seed=14)
    frame = pd.concat([frame, frame.iloc[[0, 5]]], ignore_index=True)
    res = _run(frame)
    doc = res.to_dict()
    assert doc["input"]["duplicate_rows"] == 2
    assert ("duplicate_rows", None) in h.warnings_of(res)
    assert h.rows_of(res)["loaded"] == 152


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("scheme", ["random", "time"])
def test_limitations_and_caveat_are_the_fixed_texts(task, scheme):
    frame = h.time_frame(160, seed=15, task=task)
    kw = {"time_col": "date"} if scheme == "time" else {}
    from mlprofessor import errors
    try:
        doc = _run(frame, **kw).to_dict()
    except errors.ProfileError as exc:
        if scheme != "random" or exc.code != "time_order_undeclared":
            raise
        doc = _run(frame, time_col=False).to_dict()
    assert doc["limitations"] == h.LIMITATIONS
    if scheme == "random":
        assert doc["validation"]["caveat"] == h.CAVEAT
    else:
        assert doc["validation"]["caveat"] is None


def test_roles_and_task_same_under_fast_and_full():
    rng = np.random.default_rng(16)
    frame = h.clf_frame(2400, seed=16)
    frame.insert(0, "row_id", np.arange(2400))
    frame.insert(1, "k", 1.0)
    frame.insert(2, "colour", [["red", "green", "blue"][i] for i in rng.integers(0, 3, size=2400)])
    frame.insert(3, "note", [f"note {i}" for i in rng.integers(0, 5000, size=2400)])
    fast = _run(frame, profile="fast").to_dict()
    full = _run(frame, profile="full").to_dict()
    assert fast["meta"]["profile"] == "fast" and full["meta"]["profile"] == "full"
    cap = lambda doc: {r["item"]: r["count"] for r in doc["rows"]}["dropped_row_cap"]  # noqa: E731
    assert (cap(fast), cap(full)) == (400, 0)
    pick = lambda doc: [(c["name"], c["role"], c["reason"], c["kind"] if c["role"] == "feature" else None)
                        for c in doc["columns"]]  # noqa: E731
    assert pick(fast) == pick(full)
    assert fast["task"] == full["task"]
