"""T43: a declared entity column under the random scheme (spec 2.1, 2.2, 2.4,
2.7, steps 2, 5, 6, 7, 8a, 12, 5.2, 5.4; D57). Red today.

Which entities are held out, which fold each is in and which the fast
profile's cap keeps are open (spec 11): the tests assert the properties of
steps 6 and 7, and find a run's rows out from the run itself.
"""
import inspect
import json

import numpy as np
import pandas as pd
import pytest

import cp5_helpers as p
import cp6_helpers as g
import mlp_helpers as h

TASKS = ["classification", "regression"]


def _rec(log=None):
    return [h.family("rec", log if log is not None else h.Log())]


def test_profile_has_eleven_parameters():
    import mlprofessor

    params = inspect.signature(mlprofessor.profile).parameters
    assert list(params) == ["data", "target", "task", "time_col", "gap", "profile", "seed",
                            "out_dir", "n_jobs", "duplicates", "group_col"]
    assert params["duplicates"].default == "error"
    assert params["group_col"].default is None


@pytest.mark.parametrize("task", TASKS)
def test_no_entity_on_both_sides_or_in_two_folds(task):
    """F10: the rows of one entity are on one side of the holdout and in one fold's validation part."""
    frame = g.entity_frame(60, sizes=5, seed=801, task=task)
    log = h.Log()
    result = g.run(frame, families=_rec(log), steps=[h.step_factory(log)], group_col="who")
    training, held = g.sides(result, frame)
    assert not training & held, sorted(training & held)

    names = frame["who"].tolist()
    holdout = set(h.holdout_rows(result))
    rest = set(range(len(frame))) - holdout
    entries = g.fits(log)
    assert len(entries) == 6, [(len(a), len(b)) for a, b in entries]
    predicted = []
    for fit_rows, predicted_rows in entries[:5]:
        assert not fit_rows & holdout and not predicted_rows & holdout
        fit_entities = {names[r] for r in fit_rows}
        predicted_entities = {names[r] for r in predicted_rows}
        assert not fit_entities & predicted_entities, sorted(fit_entities & predicted_entities)
        predicted.append(predicted_rows)
    for i in range(5):
        for j in range(i + 1, 5):
            assert not predicted[i] & predicted[j], (i, j)
    assert set().union(*predicted) == rest
    assert entries[5] == (rest, holdout)

    holdout_entities = {names[r] for r in holdout}
    holdout_entity_rows = {r for r in range(len(frame)) if names[r] in holdout_entities}
    for ev in log.events:
        if ev["action"] == "fit":
            assert not set(ev["rows"]) & holdout_entity_rows, (ev["kind"], ev["oid"])


@pytest.mark.parametrize("task", TASKS)
def test_holdout_is_a_quarter_of_the_rows_in_whole_entities(task):
    sizes = np.random.default_rng(802).integers(1, 9, size=90)
    frame = g.entity_frame(90, sizes=sizes, seed=803, task=task)
    result = g.run(frame, families=_rec(), group_col="who")
    training, held = g.sides(result, frame)
    assert not training & held

    size = frame["who"].value_counts().to_dict()
    holdout = set(h.holdout_rows(result))
    if task == "regression":
        pools = {"all": sorted(size)}
    else:
        by_class = frame.groupby("who")["y"].mean()
        pools = {
            label: sorted(e for e in size if (by_class[e] >= 0.5) == (label == 1))
            for label in (0, 1)
        }
    for label, entities in pools.items():
        rows = sum(size[e] for e in entities)
        held_rows = sum(size[e] for e in entities if e in held)
        largest = max(size[e] for e in entities)
        assert 4 * held_rows >= rows, (label, held_rows, rows)
        assert held_rows < rows / 4 + largest, (label, held_rows, rows, largest)
    share = result.to_dict()["validation"]["holdout_share"]
    assert share == pytest.approx(len(holdout) / len(frame), rel=1e-12)


def test_the_output_reports_the_grouped_scheme():
    frame = g.entity_frame(60, sizes=5, seed=804)
    result = g.run(frame, families=_rec(), group_col="who")
    doc = result.to_dict()
    v = doc["validation"]
    training, held = g.sides(result, frame)
    assert v["scheme"] == "grouped"
    assert v["caveat"] == g.GROUPED_CAVEAT
    assert v["group_col"] == "who" and doc["input"]["group_col"] == "who"
    assert (v["n_groups_training"], v["n_groups_holdout"]) == (len(training), len(held))
    assert v["stratified"] is True
    assert v["n_holdout_without_previous"] is None
    for key in ("time_col", "time_form", "train_end", "holdout_start", "holdout_end",
                "n_train_timestamps", "n_holdout_timestamps", "timestamps_unique"):
        assert v[key] is None, key
    record = h.columns_of(result)["who"]
    assert (record["role"], record["kind"], record["reason"]) == ("group", None, None), record
    assert v["n_holdout_copies"] == 0 and v["holdout_copy_share"] == 0.0
    json.dumps(h.plain(doc), allow_nan=False)


def test_without_an_entity_column_the_keys_are_null():
    doc = g.run(h.clf_frame(200, seed=805), families=_rec()).to_dict()
    v = doc["validation"]
    assert v["scheme"] == "random"
    assert doc["input"]["group_col"] is None
    for key in ("group_col", "n_groups_training", "n_groups_holdout", "n_rows_with_copies"):
        assert v[key] is None, key


def test_group_col_false_is_recorded_and_changes_nothing_else():
    frame = h.clf_frame(200, seed=806)
    plain = g.run(frame, families=_rec()).to_dict()
    stated = g.run(frame, families=_rec(), group_col=False).to_dict()
    assert plain["input"]["group_col"] is None
    assert stated["input"]["group_col"] is False
    assert h.without(plain, "meta.timing", "input.group_col") == \
        h.without(stated, "meta.timing", "input.group_col")


def _right_or_wrong_entities(n_entities=160, size=10, seed=807):
    rng = np.random.default_rng(seed)
    labels = np.arange(n_entities) % 2
    right = np.random.default_rng(808).random(n_entities) < 0.6
    centre = rng.normal(size=(n_entities, 2))
    rows = []
    for k in range(n_entities):
        for _ in range(size):
            rows.append({"who": f"e{k:04d}",
                         "m1": centre[k, 0] + 0.05 * rng.normal(),
                         "m2": centre[k, 1] + 0.05 * rng.normal(),
                         "y": int(labels[k]),
                         "_p": 0.9 if (labels[k] == 1) == right[k] else 0.1})
    frame = pd.DataFrame(rows)
    frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
    values = frame.pop("_p").to_numpy()
    return frame, values


def test_the_interval_resamples_whole_entities_and_there_is_no_p_value():
    """M21: under the grouped scheme each resample draws whole entities."""
    frame, values = _right_or_wrong_entities()
    log = h.Log()
    families = [h.family("lookup", log, estimator=h.LookupEstimator, values=values)]
    result = g.run(frame, families=families, group_col="who")
    head = result.headline
    assert head["interval_method"] == "bootstrap_groups"
    assert head["block_length"] is None
    assert head["n_resamples_used"] == 200
    assert head["p_value"] is None and head["n_permutations"] is None
    assert "independent rows" in head["p_value_reason"]
    assert head["interval_low"] <= head["score"] <= head["interval_high"]
    width = head["interval_high"] - head["interval_low"]
    assert width > 0.18, f"interval {head['interval_low']} to {head['interval_high']}"


def test_fewer_than_thirty_holdout_entities_no_interval():
    frame = g.entity_frame(60, sizes=5, seed=809)
    result = g.run(frame, families=_rec(), group_col="who")
    head = result.headline
    held = result.to_dict()["validation"]["n_groups_holdout"]
    assert held < 30
    for key in ("interval_method", "interval_low", "interval_high", "skill_low", "p_value"):
        assert head[key] is None, key
    assert str(held) in head["interval_reason"] and "30" in head["interval_reason"]
    assert g.count_code(result, "no_interval") == 1


@pytest.mark.parametrize("n_entities,size,stops", [(5, 40, True), (6, 40, True), (8, 30, False)],
                         ids=["five", "six", "eight"])
def test_too_few_entities(n_entities, size, stops):
    from mlprofessor import errors

    frame = g.entity_frame(n_entities, sizes=size, seed=810, task="regression")
    if stops:
        err = g.raised(frame, families=_rec(), group_col="who")
        assert isinstance(err, errors.TooFewGroupsError), f"{type(err).__name__}: {err}"
        assert err.code == "too_few_groups"
    else:
        result = g.run(frame, families=_rec(), group_col="who")
        assert result.to_dict()["validation"]["scheme"] == "grouped"


def test_an_empty_entity_cell_in_a_usable_row_is_an_error(tmp_path):
    from mlprofessor import errors

    base = g.entity_frame(60, sizes=5, seed=811)
    base["who"] = base["who"].astype(object)
    for k, cell in enumerate([None, "   "]):
        frame = base.copy()
        frame.loc[17, "who"] = cell
        out = tmp_path / f"out{k}"
        err = g.raised(frame, families=_rec(), group_col="who", out_dir=str(out))
        assert isinstance(err, errors.GroupColumnError), f"{type(err).__name__}: {err}"
        assert err.code == "group_column_invalid"
        assert "who" in str(err), str(err)
        assert h.files_in(out) == []

    frame = base.copy()
    frame["y"] = frame["y"].astype(float)
    frame.loc[17, "who"] = None
    frame.loc[17, "y"] = np.nan
    result = g.run(frame, families=_rec(), group_col="who")
    assert h.rows_of(result)["dropped_missing_target"] == 1


@pytest.mark.parametrize("kw,expected", [
    ({"group_col": "y"}, "InputError"),
    ({"group_col": True}, "InputError"),
    ({"group_col": 3}, "InputError"),
    ({"group_col": "t", "time_col": "t"}, "InputError"),
    ({"group_col": "nobody"}, "ColumnNotFoundError"),
], ids=["target", "true", "number", "time_col", "nobody"])
def test_arguments_that_cannot_be_an_entity_column(kw, expected):
    from mlprofessor import errors

    frame = g.entity_frame(60, sizes=5, seed=812)
    frame["t"] = np.arange(len(frame), dtype=float)
    err = g.raised(frame, families=_rec(), **kw)
    assert isinstance(err, getattr(errors, expected)), f"{type(err).__name__}: {err}"
    assert err.code == getattr(errors, expected).code


def test_every_row_its_own_entity_is_allowed():
    frame = h.clf_frame(200, seed=813)
    frame.insert(0, "who", np.arange(200))
    result = g.run(frame, families=_rec(), group_col="who")
    v = result.to_dict()["validation"]
    assert v["scheme"] == "grouped"
    assert v["n_groups_training"] + v["n_groups_holdout"] == 200
    assert h.columns_of(result)["who"]["role"] == "group"


def test_a_class_held_by_one_entity_is_one_class():
    from mlprofessor import errors

    sizes = [20] + [5] * 59
    frame = g.entity_frame(60, sizes=sizes, seed=814)
    frame["y"] = (frame["who"] == "e0000").astype(int)
    err = g.raised(frame, families=_rec(), group_col="who")
    assert isinstance(err, errors.OneClassError), f"{type(err).__name__}: {err}"
    assert err.code == "one_class"


def test_the_fast_cap_keeps_whole_entities():
    """M17: under the fast profile's cap every entity is kept whole or left out whole."""
    sizes = np.random.default_rng(815).integers(1, 12, size=520)
    frame = g.entity_frame(520, sizes=sizes, seed=816)
    assert len(frame) > 2500
    size = frame["who"].value_counts().to_dict()
    names = frame["who"].tolist()

    log = h.Log()
    result = g.run(frame, families=_rec(log), group_col="who")
    rows = h.rows_of(result)
    kept_count = rows["training"] + rows["holdout"]
    assert kept_count <= 2000
    assert rows["dropped_row_cap"] == len(frame) - kept_count

    holdout = h.holdout_rows(result)
    kept = set(holdout)
    for fit_rows, predicted_rows in g.fits(log):
        kept |= fit_rows | predicted_rows
    assert len(kept) == kept_count
    kept_entities = {names[r] for r in kept}
    assert sum(size[e] for e in kept_entities) == kept_count
    left_out = [size[e] for e in size if e not in kept_entities]
    assert left_out, "the cap left no entity out"
    assert kept_count + min(left_out) > 2000

    other = g.run(frame, families=_rec(), group_col="who", seed=1)
    assert sorted(h.holdout_rows(other)) != sorted(holdout)


def test_copies_across_entities_are_still_counted():
    from mlprofessor import errors

    first = g.entity_frame(60, sizes=5, seed=817)
    second = first.copy()
    second["who"] = ["f" + name[1:] for name in second["who"]]
    frame = pd.concat([first, second], ignore_index=True)
    n = len(frame)
    err = g.raised(frame, families=p.flat_family(h.Log(), n), group_col="who")
    assert isinstance(err, errors.HoldoutCopiesError), f"{type(err).__name__}: {err}"

    result = g.run(frame, families=p.flat_family(h.Log(), n), group_col="who",
                   duplicates="keep")
    v = result.to_dict()["validation"]
    assert v["scheme"] == "grouped"
    assert v["n_holdout_copies"] >= 10
    assert g.count_code(result, "holdout_copies_kept") == 1


def test_profile_and_split_random_agree():
    """Spec 2.7: the run's holdout is `stages.split_random(n, classes, seed, groups)`."""
    from mlprofessor import stages

    frame = g.entity_frame(60, sizes=5, seed=818)
    frame["y"] = frame["y"].astype(float)
    frame.loc[[4, 90, 211], "y"] = np.nan
    result = g.run(frame, families=_rec(), group_col="who")
    usable = np.flatnonzero(frame["y"].notna().to_numpy())
    classes = frame["y"].to_numpy()[usable]
    who = frame["who"].to_numpy()[usable]
    renamed = np.array([f"renamed-{name[::-1]}" for name in who], dtype=object)
    got = sorted(h.holdout_rows(result))
    for groups in (who, renamed):
        split = stages.split_random(len(usable), classes, seed=0, groups=groups)
        assert sorted(int(r) for r in usable[np.asarray(split["holdout"], dtype=int)]) == got


def test_the_command_line_takes_the_entity_column(tmp_path):
    path = tmp_path / "table.csv"
    g.entity_frame(60, sizes=5, seed=819).to_csv(path, index=False)
    common = ["profile", "--data", path, "--target", "y", "--profile", "fast", "--n-jobs", "1"]

    out = tmp_path / "grouped"
    proc = h.run_cli([*common, "--out", out, "--group-col", "who"])
    assert proc.returncode == 0, proc.stderr
    doc = h.load_json_strict(out / "profile.json")
    assert doc["validation"]["scheme"] == "grouped"
    assert doc["input"]["group_col"] == "who"

    out = tmp_path / "none"
    proc = h.run_cli([*common, "--out", out, "--no-entities"])
    assert proc.returncode == 0, proc.stderr
    doc = h.load_json_strict(out / "profile.json")
    assert doc["input"]["group_col"] is False
    assert doc["validation"]["scheme"] == "random"

    out = tmp_path / "both"
    proc = h.run_cli([*common, "--out", out, "--group-col", "who", "--no-entities"])
    assert proc.returncode == 2, proc.stderr
    assert h.files_in(out) == []

    out = tmp_path / "missing"
    proc = h.run_cli([*common, "--out", out, "--group-col", "nobody"])
    assert proc.returncode == 3, proc.stderr
    assert "column_not_found" in proc.stderr, proc.stderr
