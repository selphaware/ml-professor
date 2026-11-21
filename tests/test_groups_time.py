"""T44: entities under time order (spec 2.7 `previous_value`, steps 7, 8, 11,
5.2; D57). Red today.

The split is by time as ever; with an entity column the second baseline is
each entity's own previous value, in the headline and in the leak screen.
"""
import numpy as np
import pandas as pd
import pytest

import cp6_helpers as g
import mlp_helpers as h


def _lookup(frame, log=None):
    values = frame["prev"].to_numpy() + 0.25
    return [h.family("lookup", log if log is not None else h.Log(),
                     estimator=h.LookupEstimator, values=values)]


def _rel(a, b):
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b))


@pytest.mark.parametrize("gap", [0, 2])
def test_each_entity_has_its_own_previous_value(gap):
    """F8b: with the entity column declared, the baseline is each entity's own previous value."""
    frame = g.walks(6, 60, seed=901)
    result = g.run(frame, families=_lookup(frame), time_col="t", group_col="who", gap=gap)
    doc = result.to_dict()
    holdout = h.holdout_rows(result)
    t = frame["t"].to_numpy()
    y = frame["y"].to_numpy()
    assert sorted(set(t[holdout].tolist())) == [float(k) for k in range(45, 60)]

    training = [r for r in range(len(frame)) if t[r] < 45 - gap]
    mean = y[training].mean()
    previous = g.previous_by_hand(frame, gap)
    assert all(previous[r] is not None for r in holdout)
    mean_score = h.mse_by_hand(y[holdout], np.full(len(holdout), mean))
    prev_score = h.mse_by_hand(y[holdout], [previous[r] for r in holdout])
    model_score = h.mse_by_hand(y[holdout], frame["prev"].to_numpy()[holdout] + 0.25)

    base = {b["name"]: b for b in doc["baselines"]}
    assert set(base) == {"training_mean", "previous_value"}
    assert _rel(base["training_mean"]["score"], mean_score), (base, mean_score)
    assert _rel(base["previous_value"]["score"], prev_score), (base, prev_score)
    assert prev_score < mean_score
    assert base["previous_value"]["used"] is True and base["training_mean"]["used"] is False
    head = result.headline
    assert head["baseline"] == "previous_value"
    assert _rel(head["baseline_score"], prev_score)
    assert _rel(head["score"], model_score)
    assert _rel(head["skill"], 1.0 - model_score / prev_score)

    v = doc["validation"]
    assert v["scheme"] == "time_ordered"
    assert v["group_col"] == "who"
    assert (v["n_groups_training"], v["n_groups_holdout"]) == (6, 6)
    assert v["n_holdout_without_previous"] == 0
    assert v["timestamps_unique"] is False
    assert h.columns_of(result)["who"]["role"] == "group"
    assert g.count_code(result, "autocorrelated_panel_target") == 0
    assert g.count_code(result, "no_previous_value") == 0


def test_the_split_is_the_one_time_order_gives_without_entities():
    frame = g.walks(6, 60, seed=902)
    rng = np.random.default_rng(903)
    frame["n1"] = rng.normal(size=len(frame))
    frame["y"] = (frame["n1"] + 0.5 * rng.normal(size=len(frame)) > 0).astype(int)
    with_entities = g.run(frame, families=[h.family("rec", h.Log())], time_col="t",
                          group_col="who")
    without = g.run(frame, families=[h.family("rec", h.Log())], time_col="t")
    assert sorted(h.holdout_rows(with_entities)) == sorted(h.holdout_rows(without))
    assert h.rows_of(with_entities) == h.rows_of(without)
    assert [b["name"] for b in with_entities.to_dict()["baselines"]] == ["chance"]


def test_an_entity_with_no_earlier_row_gets_the_training_mean():
    five = g.walks(5, 60, seed=904)
    late = g.walks(1, 10, seed=905, start=50)
    late["who"] = "late"
    frame = pd.concat([five, late], ignore_index=True)
    frame = frame.iloc[np.random.default_rng(906).permutation(len(frame))].reset_index(drop=True)

    result = g.run(frame, families=_lookup(frame), time_col="t", group_col="who")
    doc = result.to_dict()
    holdout = h.holdout_rows(result)
    held = set(holdout)
    assert h.rows_of(result)["dropped_gap"] == 0
    y = frame["y"].to_numpy()
    training = [r for r in range(len(frame)) if r not in held]
    mean = y[training].mean()
    previous = g.previous_by_hand(frame, 0)
    missing = [r for r in holdout if previous[r] is None]
    assert len(missing) == 1 and frame["who"].iloc[missing[0]] == "late"
    stand_in = [mean if previous[r] is None else previous[r] for r in holdout]
    expected = h.mse_by_hand(y[holdout], stand_in)

    base = {b["name"]: b for b in doc["baselines"]}
    assert _rel(base["previous_value"]["score"], expected), (base, expected)
    v = doc["validation"]
    assert v["n_holdout_without_previous"] == 1
    assert g.count_code(result, "no_previous_value") == 1
    assert (v["n_groups_training"], v["n_groups_holdout"]) == (5, 6)


def test_a_repeated_timestamp_inside_one_entity_is_an_error(tmp_path):
    from mlprofessor import errors

    frame = g.walks(6, 60, seed=907)
    twice = pd.concat([frame, frame.iloc[[33]]], ignore_index=True)
    out = tmp_path / "out"
    err = g.raised(twice, families=_lookup(twice), time_col="t", group_col="who",
                   out_dir=str(out))
    assert isinstance(err, errors.GroupColumnError), f"{type(err).__name__}: {err}"
    assert err.code == "group_column_invalid"
    assert h.files_in(out) == []

    noise = twice.copy()
    noise["y"] = np.random.default_rng(908).integers(0, 2, size=len(noise))
    result = g.run(noise, families=[h.family("rec", h.Log())], time_col="t", group_col="who")
    assert result.to_dict()["validation"]["scheme"] == "time_ordered"


def test_the_leak_screen_measures_a_lag_against_the_entity_baseline():
    """M16: the lag of a persistent panel is measured against each entity's previous value."""
    frame = g.walks(6, 60, seed=909)
    result = g.run(frame, families=[h.family("rec", h.Log())], time_col="t", group_col="who")
    record = h.columns_of(result)["prev"]
    assert (record["role"], record["reason"]) == ("feature", None), record
    assert g.count_code(result, "suspected_leak") == 0


def test_the_command_line_takes_both_columns(tmp_path):
    path = tmp_path / "walks.csv"
    g.walks(6, 60, seed=910).to_csv(path, index=False)
    out = tmp_path / "out"
    proc = h.run_cli(["profile", "--data", path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1", "--time-col", "t",
                      "--group-col", "who"])
    assert proc.returncode == 0, proc.stderr
    doc = h.load_json_strict(out / "profile.json")
    assert doc["headline"]["baseline"] == "previous_value"
