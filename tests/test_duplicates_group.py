"""T45: `duplicates="group"` keeps each set of exact copies on one side of the
split and of every fold (spec 2.1, 2.2, 2.5, steps 7 and 8a, 5.2; D57, D60).

A copy is an exact copy, as in `test_duplicates.py`; this file does not test
rows that differ.
"""
import numpy as np
import pandas as pd
import pytest

import cp5_helpers as p
import cp6_helpers as g
import mlp_helpers as h

FEATURES = ["n1", "n2", "n3", "n4"]


def _tuples(frame, columns):
    return [tuple(row) for row in frame[columns].itertuples(index=False, name=None)]


def _doubled(seed=1101):
    return p.twice(p.noise_frame(300, seed=seed))


def test_each_set_of_copies_stays_on_one_side_and_in_one_fold():
    """M22: with duplicates="group" no set of copies is on both sides of the split or of a fold."""
    frame = _doubled()
    log = h.Log()
    result = g.run(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)],
                   duplicates="group")
    keys = _tuples(frame, FEATURES + ["y"])
    holdout = set(h.holdout_rows(result))
    rest = set(range(len(frame))) - holdout
    held_keys = {keys[r] for r in holdout}
    rest_keys = {keys[r] for r in rest}
    assert not held_keys & rest_keys
    entries = g.fits(log)
    assert entries
    for fit_rows, predicted_rows in entries:
        assert not {keys[r] for r in fit_rows} & {keys[r] for r in predicted_rows}

    doc = result.to_dict()
    v = doc["validation"]
    assert v["scheme"] == "grouped"
    assert v["caveat"] == g.GROUPED_CAVEAT
    assert v["n_rows_with_copies"] == 600
    assert v["n_holdout_copies"] == 0 and v["holdout_copy_share"] == 0.0
    assert (v["n_groups_training"], v["n_groups_holdout"]) == (len(rest_keys), len(held_keys))
    assert v["group_col"] is None
    assert doc["input"]["duplicates"] == "group"
    assert g.count_code(result, "copies_grouped") == 1
    assert g.count_code(result, "holdout_copies_kept") == 0
    assert g.count_code(result, "duplicate_rows") == 1
    assert result.headline["interval_method"] == "bootstrap_groups"
    assert result.headline["p_value"] is None


def test_a_table_with_no_copies_is_split_as_by_default():
    frame = h.clf_frame(200, seed=1102)
    grouped = g.run(frame, families=[h.family("rec", h.Log())], duplicates="group")
    plain = g.run(frame, families=[h.family("rec", h.Log())])
    assert h.holdout_rows(grouped) == h.holdout_rows(plain)
    v = grouped.to_dict()["validation"]
    assert v["scheme"] == "random"
    assert v["n_rows_with_copies"] == 0
    assert g.count_code(grouped, "copies_grouped") == 0
    assert plain.to_dict()["validation"]["n_rows_with_copies"] is None


def test_a_few_copies_are_grouped_and_counted():
    base = p.noise_frame(300, seed=1103)
    frame = pd.concat([base, base.iloc[:7]], ignore_index=True)
    result = g.run(frame, families=[h.family("rec", h.Log())], duplicates="group")
    keys = _tuples(frame, FEATURES + ["y"])
    holdout = set(h.holdout_rows(result))
    held_keys = {keys[r] for r in holdout}
    rest_keys = {keys[r] for r in range(len(frame)) if r not in holdout}
    assert not held_keys & rest_keys
    v = result.to_dict()["validation"]
    assert v["n_rows_with_copies"] == 14
    assert v["scheme"] == "grouped"
    assert v["n_groups_training"] + v["n_groups_holdout"] == 300


def test_copies_and_declared_entities_are_kept_together_both():
    base = g.entity_frame(60, sizes=5, seed=1104)
    chosen = sorted(set(base["who"]))[:10]
    again = base[base["who"].isin(chosen)].copy()
    again["who"] = ["f" + name[1:] for name in again["who"]]
    frame = pd.concat([base, again], ignore_index=True)
    frame = frame.iloc[np.random.default_rng(1105).permutation(len(frame))].reset_index(drop=True)

    result = g.run(frame, families=[h.family("rec", h.Log())], duplicates="group",
                   group_col="who")
    training, held = g.sides(result, frame)
    assert not training & held
    keys = _tuples(frame, ["m1", "m2", "y"])
    holdout = set(h.holdout_rows(result))
    held_keys = {keys[r] for r in holdout}
    rest_keys = {keys[r] for r in range(len(frame)) if r not in holdout}
    assert not held_keys & rest_keys
    v = result.to_dict()["validation"]
    assert v["n_rows_with_copies"] == 100
    assert v["n_groups_training"] + v["n_groups_holdout"] == 60
    assert v["n_holdout_copies"] == 0


def test_under_time_order_it_changes_nothing():
    frame = p.twice(p.noise_frame(150, seed=1106))
    frame.insert(0, "t", np.arange(len(frame), dtype=float))
    grouped = g.run(frame, families=[h.family("rec", h.Log())], time_col="t",
                    duplicates="group").to_dict()
    plain = g.run(frame, families=[h.family("rec", h.Log())], time_col="t").to_dict()
    assert grouped["validation"]["n_rows_with_copies"] is None
    assert grouped["validation"]["scheme"] == "time_ordered"
    assert grouped["input"]["duplicates"] == "group"
    assert h.without(grouped, "meta.timing", "input.duplicates") == \
        h.without(plain, "meta.timing", "input.duplicates")


def test_copies_that_appear_when_the_leak_screen_excludes_a_column_stop_the_run():
    from mlprofessor import errors

    frame = _doubled()
    frame["leak"] = frame["y"] + 0.001 * np.arange(len(frame))
    err = g.raised(frame, families=p.flat_family(h.Log(), len(frame)), duplicates="group")
    assert isinstance(err, errors.HoldoutCopiesError), f"{type(err).__name__}: {err}"
    assert "leak" in str(err) and "group" in str(err), str(err)


def test_the_default_message_names_the_third_value():
    from mlprofessor import errors

    frame = _doubled()
    err = g.raised(frame, families=p.flat_family(h.Log(), len(frame)))
    assert isinstance(err, errors.HoldoutCopiesError), f"{type(err).__name__}: {err}"
    assert "group" in str(err) and "keep" in str(err), str(err)


@pytest.mark.parametrize("value", ["groups", "GROUP", "", None, True],
                         ids=["groups", "GROUP", "empty", "None", "True"])
def test_other_values_are_refused(value):
    from mlprofessor import errors

    err = g.raised(h.clf_frame(120, seed=1107), families=[h.family("rec", h.Log())],
                   duplicates=value)
    assert isinstance(err, errors.InputError), f"{type(err).__name__}: {err}"
    assert err.code == "input_error"


def test_the_command_line_takes_group(tmp_path):
    path = tmp_path / "twice.csv"
    _doubled().to_csv(path, index=False)
    out = tmp_path / "out"
    proc = h.run_cli(["profile", "--data", path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1", "--duplicates", "group"])
    assert proc.returncode == 0, proc.stderr
    doc = h.load_json_strict(out / "profile.json")
    assert doc["validation"]["scheme"] == "grouped"
    assert doc["validation"]["n_holdout_copies"] == 0
    assert "copies_grouped" in proc.stdout, proc.stdout
