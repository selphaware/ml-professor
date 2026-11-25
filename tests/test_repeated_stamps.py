"""T62: the previous value at repeated timestamps (spec 2.7, steps 8 and 11, 5.2, 5.4, 7,
11; D73, D74 point 1).

Under time order with no entity column the previous value of a row is the mean target
of the rows at the timestamp `gap + 1` distinct timestamps before its own. Both
baselines are scored on the holdout and the lower error is the headline's, a tie going
to the training mean. Runs use a recording family and `task="regression"`.
"""
import math

import numpy as np
import pandas as pd
import pytest

import cp7_helpers as s
import cp9_helpers as q
import mlp_helpers as h

PANEL = "autocorrelated_panel_target"


def _run(frame, time="t", gap=0, **kw):
    _, families = s.rec()
    return s.run(frame, families=families, task="regression", time_col=time,
                 group_col=False, gap=gap, **kw)


def _keys(frame, time):
    values = frame[time]
    if pd.api.types.is_datetime64_any_dtype(values):
        return [int(v) for v in values.astype("int64").tolist()]
    return [float(v) for v in values.tolist()]


def _sides(result, frame, gap, time):
    """(training rows, holdout rows) from the holdout of the result and the timestamps of
    the table: the training rows are the rows at a timestamp no later than the last
    training timestamp; the rows of the gap are on neither side."""
    holdout = h.holdout_rows(result)
    keys = _keys(frame, time)
    stamps = sorted(set(keys))
    first_held = min(keys[r] for r in holdout)
    held_rank = stamps.index(first_held)
    last_training = stamps[held_rank - gap - 1]
    training = [r for r in range(len(frame)) if keys[r] <= last_training]
    assert not set(training) & set(holdout)
    return training, holdout


def _assert_both(result, frame, gap=0, time="t"):
    """Both baselines, each as the test scores it; the lower is the headline's."""
    doc = result.to_dict()
    names = [b["name"] for b in doc["baselines"]]
    assert names == ["training_mean", "previous_value"], doc["baselines"]
    training, holdout = _sides(result, frame, gap, time)
    y = [float(v) for v in frame["y"].tolist()]
    mean = math.fsum(y[r] for r in training) / len(training)
    mean_score = math.fsum((y[r] - mean) ** 2 for r in holdout) / len(holdout)
    previous = q.previous_by_hand(y, _keys(frame, time), gap)
    assert all(previous[r] is not None for r in holdout)
    previous_score = math.fsum((y[r] - previous[r]) ** 2 for r in holdout) / len(holdout)
    scores = {b["name"]: b["score"] for b in doc["baselines"]}
    assert math.isclose(scores["training_mean"], mean_score, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(scores["previous_value"], previous_score, rel_tol=1e-9,
                        abs_tol=1e-12)
    used = "training_mean" if mean_score <= previous_score else "previous_value"
    assert [b["name"] for b in doc["baselines"] if b["used"]] == [used]
    head = doc["headline"]
    assert head["baseline"] == used
    assert math.isclose(head["baseline_score"], scores[used], rel_tol=1e-12)
    assert math.isclose(head["skill"], 1.0 - head["score"] / head["baseline_score"],
                        rel_tol=1e-9, abs_tol=1e-12)
    assert doc["validation"]["timestamps_unique"] is False
    return used


def _twelve_stamps(seed=8201):
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 5, size=12)
    keys = np.repeat(np.arange(12, dtype=float) * 10.0 + 3.0, counts)
    y = rng.normal(size=len(keys)) * 5.0
    order = rng.permutation(len(keys))
    return y[order], keys[order]


@pytest.mark.parametrize("gap", [0, 1, 2])
def test_the_step_function_gives_the_mean_of_the_timestamp_before(gap):
    """`stages.previous_value` at repeated keys: the mean target of the timestamp before."""
    from mlprofessor import stages

    y, keys = _twelve_stamps()
    assert len(set(keys.tolist())) == 12 and len(keys) > 12
    got = np.asarray(stages.previous_value(y, keys, gap=gap), dtype=float)
    want = q.previous_by_hand(y.tolist(), keys.tolist(), gap)
    assert got.shape == (len(y),)
    for k, (a, b) in enumerate(zip(got.tolist(), want)):
        if b is None:
            assert math.isnan(a), (k, a)
        else:
            assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (k, a, b)


def test_the_step_function_is_unchanged_at_unique_keys():
    """At unique keys the previous value is the target of the row `gap + 1` before."""
    from mlprofessor import stages

    keys = np.array([40.0, 10.0, 60.0, 20.0, 50.0, 30.0])
    y = np.array([4.5, 1.5, 6.5, 2.5, 5.5, 3.5])
    by_key = dict(zip(keys.tolist(), y.tolist()))
    ordered = sorted(by_key)
    for gap in (0, 1):
        got = np.asarray(stages.previous_value(y, keys, gap=gap), dtype=float)
        for k, key in enumerate(keys.tolist()):
            rank = ordered.index(key) - gap - 1
            if rank < 0:
                assert math.isnan(got[k])
            else:
                assert got[k] == by_key[ordered[rank]]


def _uneven(seed=8211):
    frame = q.walk(n=200, seed=seed, per=3, noise=0.5)
    keep = np.random.default_rng([seed, 1]).random(len(frame)) > 0.3
    return frame[keep].reset_index(drop=True)


BOTH = {
    "three_a_timestamp": (lambda: q.walk(n=200, seed=8211, per=3, noise=0.5), 0),
    "rows_shuffled": (lambda: q.walk(n=200, seed=8211, per=3, noise=0.5, shuffle=True), 0),
    "uneven": (_uneven, 0),
    "gap_2": (lambda: q.walk(n=200, seed=8211, per=3, noise=0.5), 2),
}


@pytest.mark.parametrize("case", list(BOTH))
def test_both_baselines_are_scored_at_repeated_timestamps(case):
    """F23 (M40): at repeated timestamps the previous value is a baseline, the mean target
    of the timestamp before."""
    make, gap = BOTH[case]
    frame = make()
    if case == "uneven":
        held = frame.groupby("t").size()
        assert {1, 2, 3} <= set(held.tolist()) and len(held) < 200
    result = _run(frame, gap=gap)
    assert _assert_both(result, frame, gap=gap) == "previous_value"


def test_one_repeated_row_leaves_the_previous_value_in_use():
    """F23: one row written twice leaves the previous value the headline's baseline."""
    clean = q.walk(n=400, seed=8221)
    frame = pd.concat([clean.iloc[:100], clean.iloc[[99]], clean.iloc[100:]],
                      ignore_index=True)
    result = _run(frame)
    assert _assert_both(result, frame) == "previous_value"
    clean_result = _run(clean)
    clean_scores = {b["name"]: b["score"] for b in clean_result.to_dict()["baselines"]}
    score = {b["name"]: b["score"] for b in result.to_dict()["baselines"]}["previous_value"]
    assert abs(score - clean_scores["previous_value"]) <= 0.05 * clean_scores["previous_value"]
    assert "duplicate_rows" in s.codes(result)


def test_the_training_mean_is_the_headline_baseline_where_its_error_is_lower():
    """A panel of levels drawn afresh: the training mean has the lower error and is used."""
    frame = h.panel_frame(n_times=80, per_time=4, seed=8231, persistent=False)
    result = _run(frame, time="date")
    assert _assert_both(result, frame, time="date") == "training_mean"
    assert PANEL not in s.codes(result)


def test_unique_timestamps_are_scored_as_before():
    """At unique timestamps the previous value is the target of the row before."""
    frame = q.walk(n=300, seed=8241)
    result = _run(frame)
    doc = result.to_dict()
    holdout = h.holdout_rows(result)
    y = frame["y"].to_numpy(dtype=float)
    by_hand = math.fsum((y[r] - y[r - 1]) ** 2 for r in holdout) / len(holdout)
    scores = {b["name"]: b["score"] for b in doc["baselines"]}
    assert math.isclose(scores["previous_value"], by_hand, rel_tol=1e-9)
    assert doc["validation"]["timestamps_unique"] is True


def test_the_leak_screen_measures_a_lag_against_the_timestamp_before():
    """F23: the mean target of the timestamp before, offered as a feature, is no leak."""
    frame = q.walk(n=400, seed=8251, per=3, noise=0.1)
    lag = q.previous_by_hand(frame["y"].tolist(), frame["t"].tolist())
    frame = q.with_column(frame, "lag", [np.nan if v is None else v for v in lag])
    result = _run(frame)
    assert q.is_feature(q.column(result, "lag")), q.column(result, "lag")
    assert not q.told(result, "suspected_leak", "lag")
    assert not q.told(result, "strong_single_column", "lag")


def _late_noise(seed=8262):
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(size=90))
    late = walk.mean() + 2.0 * rng.normal(size=30)
    level = np.repeat(np.concatenate([walk, late]), 4)
    n = len(level)
    return pd.DataFrame({"t": np.repeat(np.arange(120, dtype=np.int64), 4),
                         "x1": rng.normal(size=n),
                         "y": level + 0.3 * rng.normal(size=n)})


PANELS = {
    "persistent": (lambda: h.panel_frame(n_times=80, per_time=4, seed=8261, persistent=True),
                   "date", "previous_value"),
    "late_noise": (_late_noise, "t", "training_mean"),
}


@pytest.mark.parametrize("case", list(PANELS))
def test_the_panel_warning_names_the_baseline_the_run_used(case):
    """M51 on `persistent`: the panel warning names the baseline the run used."""
    make, time, used = PANELS[case]
    frame = make()
    result = _run(frame, time=time)
    assert _assert_both(result, frame, time=time) == used
    records = s.warning_records(result, PANEL)
    assert len(records) == 1, s.codes(result)
    assert records[0]["column"] == "y"
    other = "training_mean" if used == "previous_value" else "previous_value"
    message = str(records[0]["message"])
    assert q.panel_passage(used) in message, message
    assert q.panel_passage(other) not in message, message


def test_the_command_line_reports_both_baselines(tmp_path):
    """The first table through the command line: both baselines and the panel passage."""
    frame = q.walk(n=200, seed=8211, per=3, noise=0.5)
    csv_path = tmp_path / "walk.csv"
    frame.to_csv(csv_path, index=False)
    out = tmp_path / "out"
    proc = s.run_cli(["profile", "--data", csv_path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1", "--task", "regression",
                      "--time-col", "t", "--no-entities"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = h.load_json_strict(out / "profile.json")
    assert [b["name"] for b in doc["baselines"]] == ["training_mean", "previous_value"]
    used = [b["name"] for b in doc["baselines"] if b["used"]]
    assert len(used) == 1 and used[0] == doc["headline"]["baseline"]
    panel = [w for w in doc["warnings"] if w["code"] == PANEL]
    assert len(panel) == 1
    assert q.panel_passage(used[0]) in panel[0]["message"]
