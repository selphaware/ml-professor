"""T11: `profile()` with `time_col` (spec steps 3, 6, 7, 11; D7, D10, D18, D20)."""
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
        return h.stated_profile(frame, target, seed=seed, **args)


def _iso(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


def test_scheme_follows_time_col():
    frame = h.time_frame(200, seed=1, task="classification")
    log = h.Log()
    from mlprofessor import errors
    try:
        plain_result = _run(frame, families=[h.family("rec", log)])
    except errors.ProfileError as exc:
        if exc.code != "time_order_undeclared":
            raise
        plain_result = _run(frame, families=[h.family("rec", log)], time_col=False)
    plain_run = plain_result.to_dict()["validation"]
    assert plain_run["scheme"] == "random"
    for key in ["time_col", "time_form", "train_end", "holdout_start", "holdout_end",
                "n_train_timestamps", "n_holdout_timestamps", "timestamps_unique"]:
        assert plain_run[key] is None, key
    assert isinstance(plain_run["caveat"], str) and plain_run["caveat"]

    v = _run(frame, families=[h.family("rec", log)], time_col="date").to_dict()["validation"]
    d = frame["date"]
    assert v["scheme"] == "time_ordered"
    assert v["time_col"] == "date" and v["time_form"] == "datetime"
    assert v["stratified"] is False
    assert v["n_train_timestamps"] == 150 and v["n_holdout_timestamps"] == 50
    assert v["timestamps_unique"] is True
    assert v["train_end"] == _iso(d.iloc[149])
    assert v["holdout_start"] == _iso(d.iloc[150])
    assert v["holdout_end"] == _iso(d.iloc[199])
    assert v["caveat"] is None


def test_numeric_and_iso_time_forms_reported():
    frame = h.time_frame(160, seed=2)
    num = frame.assign(date=np.arange(160, dtype=float) * 2.0)
    v = _run(num, families=[h.family("rec", h.Log())], time_col="date").to_dict()["validation"]
    assert v["time_form"] == "numeric"
    assert v["holdout_start"] == pytest.approx(120 * 2.0) and v["train_end"] == pytest.approx(119 * 2.0)
    iso = frame.assign(date=frame["date"].dt.strftime("%Y-%m-%d"))
    v = _run(iso, families=[h.family("rec", h.Log())], time_col="date").to_dict()["validation"]
    assert v["time_form"] == "iso8601"
    assert v["holdout_start"] == _iso(frame["date"].iloc[120])


def test_fixture_csv_with_iso_time_column():
    path = h.REPO_ROOT / "tests" / "fixtures" / "example.csv"
    source = pd.read_csv(path)
    res = _run(str(path), target="churned", families=[h.family("rec", h.Log())],
               time_col="signup_date")
    doc = res.to_dict()
    assert doc["validation"]["time_form"] == "iso8601"
    assert h.rows_of(res)["loaded"] == len(source)
    assert [c["name"] for c in doc["columns"]] == list(source.columns)
    assert h.holdout_rows(res) == list(range(len(source) - len(source) // 4, len(source)))


def test_f7_run_uses_split_time_of_usable_keys():
    """F7: the run's holdout, gap and folds are split_time of the usable rows' keys."""
    from mlprofessor import stages

    frame = h.time_frame(240, seed=3)
    frame.loc[[5, 17, 33], "y"] = np.nan
    frame.loc[[17, 40, 41, 90], "date"] = pd.NaT
    log = h.Log()
    res = _run(frame, families=[h.family("rec", log)], time_col="date", gap=2)
    doc = res.to_dict()

    usable = np.asarray(stages.usable_rows(frame, "y", time_col="date")["usable"], dtype=int)
    keys, _ = stages.read_time(frame["date"].iloc[usable])
    split = stages.split_time(keys, gap=2)
    pos = lambda part: usable[np.asarray(part, dtype=int)]  # noqa: E731

    assert h.holdout_rows(res) == pos(split["holdout"]).tolist()
    rows = h.rows_of(res)
    assert rows["dropped_missing_target"] == 3
    assert rows["dropped_missing_time"] == 3
    assert rows["dropped_gap"] == len(split["gap"])
    assert rows["training"] == len(split["training"])
    assert rows["holdout"] == len(split["holdout"])
    for rec, fold in zip(doc["validation"]["folds"], split["folds"]):
        assert rec["gap_rows"] == len(fold["gap"]) > 0
        assert rec["validation_rows"] == len(fold["validation"])
        assert rec["training_rows"] == len(fold["training"])
    hold = set(h.holdout_rows(res))
    fold_objs = [o for o in log.objects("model") if not set(o["predict"]) & hold]
    assert {frozenset(o["predict"]) for o in fold_objs} == {frozenset(pos(f["validation"]).tolist())
                                                             for f in split["folds"]}
    assert {frozenset(o["fit"]) for o in fold_objs} == {frozenset(pos(f["training"]).tolist())
                                                         for f in split["folds"]}
    final = [o for o in log.objects("model") if set(o["predict"]) & hold]
    assert len(final) == 1 and set(final[0]["fit"]) == set(pos(split["training"]).tolist())


def test_f7_shuffled_table_gives_same_holdout_rows():
    """F7: the holdout is chosen by time, not by file position."""
    frame = h.time_frame(200, seed=4)
    perm = np.random.default_rng(0).permutation(len(frame))
    shuffled = frame.iloc[perm].reset_index(drop=True)
    a = _run(frame, families=[h.family("rec", h.Log())], time_col="date")
    b = _run(shuffled, families=[h.family("rec", h.Log())], time_col="date")
    assert sorted(perm[h.holdout_rows(b)].tolist()) == h.holdout_rows(a)
    assert h.holdout_rows(a) == list(range(150, 200))


def test_f7_no_fit_at_or_after_holdout_or_inside_gap():
    """F7: no fit of any kind sees a row at or after the first holdout timestamp, or in a gap."""
    frame = h.time_frame(220, seed=5)
    gap = 3
    log = h.Log()
    res = _run(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)],
               time_col="date", gap=gap)
    when = frame["date"].to_numpy()
    first_hold = when[h.holdout_rows(res)].min()
    stamps = np.unique(when)
    gap_stamps = set(stamps[stamps < first_hold][-gap:].tolist())
    for ev in log.events:
        if ev["action"] == "fit":
            fit_when = when[ev["rows"]]
            assert fit_when.max() < first_hold
            assert not set(fit_when.tolist()) & gap_stamps
    for o in log.objects("model"):
        if not o["predict"]:
            continue
        start = when[o["predict"]].min()
        before = set(stamps[stamps < start][-gap:].tolist())
        assert when[o["fit"]].max() < start
        assert not set(when[o["fit"]].tolist()) & before


def test_fast_cap_keeps_most_recent_timestamps():
    frame = h.time_frame(2300, seed=6)
    log = h.Log()
    res = _run(frame, families=[h.family("rec", log)], time_col="date")
    rows = h.rows_of(res)
    assert rows["dropped_row_cap"] == 300
    assert rows["training"] + rows["holdout"] + rows["dropped_gap"] == 2000
    assert min(h.holdout_rows(res)) >= 300
    assert min(r for ev in log.events for r in ev["rows"]) >= 300


def test_fast_cap_keeps_whole_timestamps_in_a_panel():
    rng = np.random.default_rng(7)
    when = np.repeat(h.dates(50).to_numpy(), 50)
    frame = pd.DataFrame({"date": when, "x1": rng.normal(size=2500)})
    frame["y"] = frame["x1"] + rng.normal(size=2500)
    log = h.Log()
    res = _run(frame, families=[h.family("rec", log)], time_col="date")
    rows = h.rows_of(res)
    assert rows["dropped_row_cap"] == 500
    kept = set(np.unique(when)[-40:].tolist())
    for ev in log.events:
        assert set(when[ev["rows"]].tolist()) <= kept


def test_wide_panel_under_fast_raises():
    from mlprofessor import errors

    rng = np.random.default_rng(8)
    when = np.repeat(h.dates(30).to_numpy(), 300)
    frame = pd.DataFrame({"date": when, "x1": rng.normal(size=len(when))})
    frame["y"] = frame["x1"] + rng.normal(size=len(when))
    with pytest.raises(errors.FastProfileError):
        _run(frame, families=[h.family("rec", h.Log())], time_col="date")


def test_f7_day_first_time_column_stops():
    """F7: a day-first date is never guessed at."""
    from mlprofessor import errors

    frame = h.time_frame(150, seed=9)
    frame["date"] = frame["date"].dt.strftime("%d/%m/%Y")
    with pytest.raises(errors.TimeColumnError):
        _run(frame, families=[h.family("rec", h.Log())], time_col="date")


def test_date_column_undeclared():
    frame = h.time_frame(160, seed=10)
    frame["day_text"] = frame["date"].dt.strftime("%Y-%m-%d")
    from mlprofessor import errors
    try:
        res = _run(frame, families=[h.family("rec", h.Log())])
    except errors.ProfileError as exc:
        if exc.code != "time_order_undeclared":
            raise
        res = _run(frame, families=[h.family("rec", h.Log())], time_col=False)
    found = [c for code, c in h.warnings_of(res) if code == "date_column_undeclared"]
    assert sorted(found) == ["date", "day_text"]
    assert isinstance(res.to_dict()["validation"]["caveat"], str)
    res = _run(frame, families=[h.family("rec", h.Log())], time_col="date")
    assert not [c for code, c in h.warnings_of(res) if code == "date_column_undeclared"]


def _walk_frame(n=300, seed=11):
    frame = h.time_frame(n, seed=seed, walk=True)
    frame["lag1"] = frame["y"].shift(1)
    return frame


def test_f8_previous_value_baseline_on_a_random_walk():
    """F8: on a random walk the headline baseline is the previous value, and skill is near zero."""
    frame = _walk_frame()
    res = _run(frame, time_col="date")
    doc = res.to_dict()
    y = frame["y"].to_numpy()
    hold = h.holdout_rows(res)
    training = [r for r in range(len(y)) if r < min(hold)]
    mean_score = h.mse_by_hand(y[hold], np.full(len(hold), y[training].mean()))
    prev_score = h.mse_by_hand(y[hold], y[np.array(hold) - 1])
    base = {b["name"]: b for b in doc["baselines"]}
    assert set(base) == {"training_mean", "previous_value"}
    assert base["training_mean"]["score"] == pytest.approx(mean_score, rel=1e-9)
    assert base["previous_value"]["score"] == pytest.approx(prev_score, rel=1e-9)
    assert base["previous_value"]["used"] is True and base["training_mean"]["used"] is False
    assert res.headline["baseline"] == "previous_value"
    assert res.headline["baseline_score"] == pytest.approx(prev_score, rel=1e-9)
    assert res.headline["skill"] < 0.3


def test_f8_previous_value_respects_the_gap():
    """F8: with a gap of 2 the previous value is three timestamps back."""
    frame = _walk_frame(260, seed=12)
    y = frame["y"].to_numpy()
    res = _run(frame, families=[h.family("rec", h.Log())], time_col="date", gap=2)
    hold = np.array(h.holdout_rows(res))
    prev_score = h.mse_by_hand(y[hold], y[hold - 3])
    base = {b["name"]: b for b in res.to_dict()["baselines"]}
    assert base["previous_value"]["score"] == pytest.approx(prev_score, rel=1e-9)


@pytest.mark.parametrize("persistent", [True, False])
def test_panel_uses_training_mean_and_flags_persistence(persistent):
    frame = h.panel_frame(60, 5, seed=13, persistent=persistent)
    means = frame.groupby("date")["y"].mean().to_numpy()[:45]
    r = h.lag1_autocorr(means)
    assert (r > 0.8) if persistent else (r < 0.3)
    from mlprofessor import errors
    try:
        res = _run(frame, families=[h.family("rec", h.Log())], time_col="date")
    except errors.ProfileError as exc:
        if exc.code != "entities_undeclared" or not persistent:
            raise
        res = _run(frame, families=[h.family("rec", h.Log())], time_col="date",
                   group_col=False)
    doc = res.to_dict()
    assert [b["name"] for b in doc["baselines"]] in (
        ["training_mean"], ["training_mean", "previous_value"])
    assert doc["validation"]["timestamps_unique"] is False
    flagged = [c for code, c in h.warnings_of(res) if code == "autocorrelated_panel_target"]
    assert flagged == (["y"] if persistent else [])
