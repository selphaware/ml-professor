"""T50: `duplicates="group"` keeps together the rows that agree in every feature column,
whatever their target (spec 2.1, 2.5, steps 7 and 8a, 5.2, 5.4, 11; D62, D66).

The count is `validation.n_rows_same_features` and the warning `same_features_grouped`;
a copy at step 8a still agrees in the target too.
"""
import json

import numpy as np
import pandas as pd
import pytest

import cp5_helpers as p
import cp6_helpers as g
import cp7_helpers as s
import mlp_helpers as h

NOISE = ["n1", "n2", "n3", "n4"]


def _keys(frame, columns):
    return [tuple(row) for row in frame[columns].itertuples(index=False, name=None)]


def _sides(result, keys):
    holdout = set(h.holdout_rows(result))
    held = {keys[r] for r in holdout}
    rest = {keys[r] for r in range(len(keys)) if r not in holdout}
    return rest, held


def test_rows_that_agree_in_every_feature_stay_together_whatever_their_target():
    """F12: under duplicates="group" no combination of feature values is on both sides or in two folds."""
    frame = s.few_valued(600, (4, 3), seed=1501)
    log = h.Log()
    result = s.run(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)],
                   duplicates="group", **s.SAID)
    keys = _keys(frame, ["c1", "c2"])
    assert len(set(keys)) == 12
    rest, held = _sides(result, keys)
    assert not rest & held, sorted(rest & held)
    entries = g.fits(log)
    assert entries
    for fit_rows, predicted_rows in entries:
        assert not {keys[r] for r in fit_rows} & {keys[r] for r in predicted_rows}

    v = result.to_dict()["validation"]
    assert v["scheme"] == "grouped"
    assert (v["n_groups_training"], v["n_groups_holdout"]) == (len(rest), len(held))
    assert len(rest) + len(held) == 12
    assert v["n_rows_same_features"] == 600
    assert v["n_holdout_copies"] == 0


def test_the_count_is_of_rows_that_agree_in_every_feature():
    """M25: the count of duplicates="group" is of rows equal in every feature, whatever the target."""
    base = p.noise_frame(300, seed=1502)
    flipped = base.iloc[:40].copy()
    flipped["y"] = 1 - flipped["y"]
    frame = pd.concat([base, flipped], ignore_index=True)
    frame = frame.iloc[np.random.default_rng(1503).permutation(len(frame))]
    frame = frame.reset_index(drop=True)

    result = s.run(frame, families=[h.family("rec", h.Log())], duplicates="group", **s.SAID)
    v = result.to_dict()["validation"]
    assert v["n_rows_same_features"] == 80
    assert v["n_groups_training"] + v["n_groups_holdout"] == 300
    assert v["n_holdout_copies"] == 0
    assert s.codes(result).count("same_features_grouped") == 1
    rest, held = _sides(result, _keys(frame, NOISE))
    assert not rest & held


def test_the_count_and_the_warning_have_their_new_names():
    doubled = p.twice(p.noise_frame(300, seed=1504))
    result = s.run(doubled, families=[h.family("rec", h.Log())], duplicates="group", **s.SAID)
    v = result.to_dict()["validation"]
    assert v["n_rows_same_features"] == 600
    assert "n_rows_with_copies" not in v
    found = s.codes(result)
    assert found.count("same_features_grouped") == 1 and "copies_grouped" not in found
    assert s.warning_records(result, "same_features_grouped")[0]["column"] is None

    single = p.noise_frame(300, seed=1505)
    result = s.run(single, families=[h.family("rec", h.Log())], duplicates="group", **s.SAID)
    v = result.to_dict()["validation"]
    assert v["scheme"] == "random"
    assert v["n_rows_same_features"] == 0
    assert "same_features_grouped" not in s.codes(result)

    for value in ("error", "keep"):
        result = s.run(single, families=[h.family("rec", h.Log())], duplicates=value, **s.SAID)
        v = result.to_dict()["validation"]
        assert "n_rows_same_features" in v and v["n_rows_same_features"] is None, value
        assert "n_rows_with_copies" not in v, value

    timed = doubled.copy()
    timed.insert(0, "t", np.arange(len(timed), dtype=float))
    result = s.run(timed, families=[h.family("rec", h.Log())], duplicates="group",
                   time_col="t", group_col=False)
    v = result.to_dict()["validation"]
    assert v["scheme"] == "time_ordered"
    assert "n_rows_same_features" in v and v["n_rows_same_features"] is None


def test_a_set_that_spans_two_entities_joins_them():
    base = g.entity_frame(60, sizes=5, seed=1506)
    names = sorted(set(base["who"]))
    extra = []
    for k in range(20):
        first, second = names[2 * k], names[2 * k + 1]
        source = base[base["who"] == first].iloc[0]
        extra.append({"who": second, "m1": source["m1"], "m2": source["m2"],
                      "y": 1 - int(source["y"])})
    frame = pd.concat([base, pd.DataFrame(extra)], ignore_index=True)
    frame = frame.iloc[np.random.default_rng(1507).permutation(len(frame))]
    frame = frame.reset_index(drop=True)

    result = s.run(frame, families=[h.family("rec", h.Log())], duplicates="group",
                   time_col=False, group_col="who")
    training, held = g.sides(result, frame)
    assert not training & held
    for k in range(20):
        pair = {names[2 * k], names[2 * k + 1]}
        assert pair <= training or pair <= held, pair
    v = result.to_dict()["validation"]
    assert v["n_rows_same_features"] == 40
    assert v["n_groups_training"] + v["n_groups_holdout"] == 40


def test_the_copies_stop_says_when_each_answer_fits():
    from mlprofessor import errors

    frame = s.few_valued(600, seed=1508)
    err = s.raised(frame, families=p.flat_family(h.Log(), len(frame)), **s.SAID)
    assert isinstance(err, errors.HoldoutCopiesError), f"{type(err).__name__}: {err}"
    text = str(err)
    assert s.COPIES_ADVICE in text, text
    assert text.index("duplicates='keep'") < text.index("duplicates='group'"), text


def test_too_few_combinations_cannot_be_split_and_a_few_give_no_interval():
    from mlprofessor import errors

    four = s.few_valued(400, (2, 2), seed=1509)
    four["y"] = (four["c2"] == "v2_1").astype(int)
    err = s.raised(four, families=[h.family("rec", h.Log())], duplicates="group", **s.SAID)
    assert isinstance(err, errors.TooFewGroupsError), f"{type(err).__name__}: {err}"

    twelve = s.few_valued(600, (4, 3), seed=1510)
    result = s.run(twelve, families=[h.family("rec", h.Log())], duplicates="group", **s.SAID)
    head = result.headline
    assert result.to_dict()["validation"]["scheme"] == "grouped"
    assert head["interval_method"] is None
    assert head["interval_low"] is None and head["interval_high"] is None
    assert "30" in head["interval_reason"], head["interval_reason"]
    assert head["p_value"] is None
    assert "independent rows" in head["p_value_reason"], head["p_value_reason"]
    assert "no_interval" in s.codes(result)


def test_the_command_line_uses_the_new_names(tmp_path):
    path = tmp_path / "doubled.csv"
    p.twice(p.noise_frame(300, seed=1511)).to_csv(path, index=False)
    out = tmp_path / "out"
    proc = s.run_cli(["profile", "--data", path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1", "--duplicates", "group",
                      "--no-time-order", "--no-entities"])
    assert proc.returncode == 0, proc.stderr
    doc = json.loads((out / "profile.json").read_text(encoding="utf-8"))
    assert doc["validation"]["n_rows_same_features"] == 600
    assert "same_features_grouped" in proc.stdout, proc.stdout
    assert "copies_grouped" not in proc.stdout, proc.stdout
