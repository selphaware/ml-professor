"""T49: both statements on every call (spec 2.1, 2.2, 2.4, step 5, 5.2, 11; D63, D66).

Every call says whether the rows are ordered in time (`time_col`) and whether rows
belong together (`group_col`); `None` is "not said", and a call that leaves either
unsaid stops at step 5 with a named error, whatever the table holds.
"""
import inspect
import json

import numpy as np
import pandas as pd
import pytest

import cp6_helpers as g
import cp7_helpers as s
import mlp_helpers as h

TIME_CODE = "time_order_undeclared"
ENTITY_CODE = "entities_undeclared"


def _stopped(frame, tmp_path, **kw):
    """The error of the call as made, with a recording family and step; nothing written,
    nothing fitted or transformed."""
    log = h.Log()
    out = tmp_path / "out"
    err = s.raised(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)],
                   out_dir=str(out), **kw)
    assert h.files_in(out) == [], h.files_in(out)
    assert log.events == [], f"{len(log.events)} calls reached the seam before the stop"
    return err


def _holds_all(text, texts):
    missing = [t for t in texts if t not in text]
    assert not missing, f"missing {missing} in: {text}"


# ------------------------------------------------ nothing said about time order

@pytest.mark.parametrize("case", ["classification", "regression", "entities_named",
                                  "no_entities_said"])
def test_nothing_said_about_time_order_stops_whatever_the_table_holds(case, tmp_path):
    """F11: with nothing said about time order the run stops, though no column holds a date."""
    from mlprofessor import errors

    if case == "classification":
        frame, kw = h.clf_frame(200, seed=1401), {}
    elif case == "regression":
        frame, kw = h.reg_frame(200, seed=1402), {}
    elif case == "entities_named":
        frame, kw = g.entity_frame(60, sizes=5, seed=1403), {"group_col": "who"}
    else:
        frame, kw = h.clf_frame(200, seed=1404), {"group_col": False}
    err = _stopped(frame, tmp_path, **kw)
    assert err.code == TIME_CODE, f"{type(err).__name__}: {err}"
    assert isinstance(err, errors.TimeOrderUndeclaredError)
    _holds_all(str(err), s.TIME_TEXTS)


# ------------------------------------------------ nothing said about entities

def _entities_case(case):
    if case == "no_time_order":
        return h.clf_frame(200, seed=1411), {"time_col": False}
    if case == "one_series":
        return h.time_frame(200, seed=1412), {"time_col": "date"}
    if case == "classification":
        return h.time_frame(200, seed=1413, task="classification"), {"time_col": "date"}
    return h.panel_frame(60, 5, seed=13, persistent=False), {"time_col": "date"}


@pytest.mark.parametrize("case", ["no_time_order", "one_series", "classification",
                                  "a_panel_that_does_not_persist"])
def test_nothing_said_about_entities_stops_whatever_the_table_holds(case, tmp_path):
    """F11, M19: with time order said and nothing said about entities the run stops, whatever the table."""
    from mlprofessor import errors

    frame, kw = _entities_case(case)
    err = _stopped(frame, tmp_path, **kw)
    assert err.code == ENTITY_CODE, f"{type(err).__name__}: {err}"
    assert isinstance(err, errors.EntitiesUndeclaredError)
    _holds_all(str(err), s.ENTITY_TEXTS)


def test_one_message_asks_for_everything_that_is_missing(tmp_path):
    """M24: with both statements unsaid the one message asks for both."""
    err = _stopped(h.clf_frame(200, seed=1421), tmp_path)
    assert err.code == TIME_CODE, f"{type(err).__name__}: {err}"
    _holds_all(str(err), s.TIME_TEXTS + s.ENTITY_TEXTS)


def test_none_passed_explicitly_is_not_said(tmp_path):
    frame = h.clf_frame(200, seed=1422)
    err = _stopped(frame, tmp_path / "a", time_col=None, group_col=None)
    assert err.code == TIME_CODE, f"{type(err).__name__}: {err}"
    err = _stopped(frame, tmp_path / "b", time_col=False, group_col=None)
    assert err.code == ENTITY_CODE, f"{type(err).__name__}: {err}"


def test_the_entities_message_says_that_timestamps_repeat(tmp_path):
    frame = h.panel_frame(60, 5, seed=1423, persistent=False)
    frame.loc[[0, 1, 2, 3], "y"] = np.nan
    usable = frame["y"].notna()
    n_usable = int(usable.sum())
    n_stamps = int(frame.loc[usable, "date"].nunique())
    assert (n_usable, n_stamps) == (296, 60)
    err = _stopped(frame, tmp_path, time_col="date")
    assert err.code == ENTITY_CODE, f"{type(err).__name__}: {err}"
    numbers = s.numbers_in(str(err))
    assert float(n_usable) in numbers and float(n_stamps) in numbers, str(err)


# ------------------------------------------------ the order of the errors

def _early_case(case):
    """(frame, keywords, the code) for an error of steps 1 to 4, a statement unsaid."""
    if case == "invalid_seed":
        return h.clf_frame(200, seed=1431), {"seed": -1}, "input_error"
    if case == "no_such_target":
        return h.clf_frame(200, seed=1432), {"target": "nothing_here"}, "column_not_found"
    if case == "too_few_rows":
        return h.clf_frame(60, seed=1433), {}, "too_few_rows"
    if case == "task_ambiguous":
        frame = h.clf_frame(200, seed=1434)
        frame["y"] = np.arange(200) % 5
        return frame, {}, "task_ambiguous"
    if case == "day_first_time":
        frame = h.clf_frame(200, seed=1435)
        frame.insert(0, "d", [d.strftime("%d/%m/%Y") for d in h.dates(200)])
        return frame, {"time_col": "d"}, "time_column_invalid"
    frame = g.entity_frame(60, sizes=5, seed=1436)
    frame["who"] = frame["who"].astype(object)
    frame.loc[7, "who"] = None
    return frame, {"group_col": "who"}, "group_column_invalid"


@pytest.mark.parametrize("case", ["invalid_seed", "no_such_target", "too_few_rows",
                                  "task_ambiguous", "day_first_time", "empty_entity_cell"])
def test_an_error_of_steps_one_to_four_comes_first(case):
    frame, kw, code = _early_case(case)
    target = kw.pop("target", "y")
    err = s.raised(frame, target, families=[h.family("rec", h.Log())], **kw)
    assert err.code == code, f"{type(err).__name__}: {err}"


def _later_case(case):
    """(frame, keywords, the stop's code, the code once both are said, families)."""
    log = h.Log()
    families = [h.family("rec", log)]
    if case == "name_collision":
        frame = h.clf_frame(200, seed=1441)
        frame["colour"] = ["red", "blue"] * 100
        frame["colour=red"] = np.random.default_rng(1441).normal(size=200)
        return frame, {}, TIME_CODE, "feature_name_collision", families
    if case == "every_row_twice":
        import cp5_helpers as p
        return (p.twice(p.noise_frame(200, seed=1442)), {}, TIME_CODE,
                "holdout_copies_in_training", families)
    if case == "every_family_fails":
        return (h.clf_frame(200, seed=1443), {}, TIME_CODE, "all_models_failed",
                [h.family("broken", log, estimator=h.FailingEstimator)])
    if case == "no_features":
        frame = h.clf_frame(200, seed=1444)[["y"]].copy()
        frame.insert(0, "flat", 1.0)
        return frame, {}, TIME_CODE, "no_features", families
    rng = np.random.default_rng(1445)
    frame = pd.DataFrame({"t": np.repeat(np.arange(5, dtype=float), 60),
                          "x": rng.normal(size=300), "y": rng.normal(size=300)})
    return frame, {"time_col": "t"}, ENTITY_CODE, "too_few_rows", families


@pytest.mark.parametrize("case", ["name_collision", "every_row_twice", "every_family_fails",
                                  "no_features", "five_timestamps"])
def test_the_stop_comes_before_every_later_error(case):
    frame, kw, stop, later, families = _later_case(case)
    err = s.raised(frame, families=families, **kw)
    assert err.code == stop, f"{type(err).__name__}: {err}"
    both = {**s.SAID, **kw}
    err = s.raised(frame, families=families, **both)
    assert err.code == later, f"{type(err).__name__}: {err}"


# ------------------------------------------------ what a finished run records

def test_the_four_combinations_and_what_is_recorded():
    log, fam = s.rec()

    doc = s.run(h.clf_frame(200, seed=1451), families=fam, time_col=False,
                group_col=False).to_dict()
    assert doc["validation"]["scheme"] == "random"
    assert doc["input"]["time_col"] is False and doc["input"]["group_col"] is False
    assert doc["validation"]["time_col"] is None and doc["validation"]["group_col"] is None

    doc = s.run(g.entity_frame(60, sizes=5, seed=1452), families=fam, time_col=False,
                group_col="who").to_dict()
    assert doc["validation"]["scheme"] == "grouped"
    assert doc["input"]["time_col"] is False and doc["input"]["group_col"] == "who"
    assert doc["validation"]["group_col"] == "who"

    doc = s.run(h.time_frame(200, seed=1453, walk=True), families=fam, time_col="date",
                group_col=False).to_dict()
    assert doc["validation"]["scheme"] == "time_ordered"
    assert doc["input"]["time_col"] == "date" and doc["input"]["group_col"] is False
    assert sorted(b["name"] for b in doc["baselines"]) == ["previous_value", "training_mean"]
    assert doc["validation"]["n_holdout_without_previous"] is None

    doc = s.run(g.walks(6, 60, seed=1454), families=fam, time_col="t",
                group_col="who").to_dict()
    assert doc["validation"]["scheme"] == "time_ordered"
    assert doc["input"]["time_col"] == "t" and doc["input"]["group_col"] == "who"
    assert sorted(b["name"] for b in doc["baselines"]) == ["previous_value", "training_mean"]
    assert doc["validation"]["n_holdout_without_previous"] == 0


def test_the_signature_keeps_its_eleven_parameters_and_their_defaults():
    import mlprofessor

    params = inspect.signature(mlprofessor.profile).parameters
    assert list(params) == ["data", "target", "task", "time_col", "gap", "profile", "seed",
                            "out_dir", "n_jobs", "duplicates", "group_col"]
    assert params["time_col"].default is None
    assert params["group_col"].default is None


def test_the_step_functions_take_what_they_are_given():
    from mlprofessor import stages

    frame = h.clf_frame(200, seed=1461)
    frame.insert(0, "when", h.dates(200))
    roles = stages.column_roles(frame, "y")
    by_name = {r["name"]: r for r in h.records(roles)}
    assert by_name["when"]["role"] == "excluded" and by_name["when"]["reason"] == "datetime"
    assert by_name["x1"]["role"] == "feature"

    counts = stages.usable_rows(frame, "y")
    assert len(counts["usable"]) == 200

    split = stages.split_random(200, seed=0)
    assert len(split["holdout"]) == 50 and len(split["folds"]) == 5


# ------------------------------------------------ the command line

def test_the_command_line_asks_for_each_missing_pair(tmp_path):
    path = tmp_path / "table.csv"
    h.clf_frame(200, seed=1471).to_csv(path, index=False)
    common = ["profile", "--data", path, "--target", "y", "--profile", "fast",
              "--n-jobs", "1"]

    def stops(name, more, code, texts):
        out = tmp_path / name
        proc = s.run_cli([*common, "--out", out, *more])
        assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
        lines = proc.stderr.strip().splitlines()
        assert len(lines) == 1, proc.stderr
        assert lines[0].startswith(f"error: {code}:"), lines[0]
        _holds_all(lines[0], texts)
        assert h.files_in(out) == []

    stops("nothing", [], TIME_CODE,
          ["--time-col", "--no-time-order", "--group-col", "--no-entities"])
    stops("time_only", ["--no-time-order"], ENTITY_CODE, ["--group-col", "--no-entities"])
    stops("entities_only", ["--no-entities"], TIME_CODE, ["--time-col", "--no-time-order"])

    out = tmp_path / "both"
    proc = s.run_cli([*common, "--out", out, "--no-time-order", "--no-entities"])
    assert proc.returncode == 0, proc.stderr
    assert h.files_in(out) == ["predictions.csv", "profile.json"]
    doc = json.loads((out / "profile.json").read_text(encoding="utf-8"))
    assert doc["input"]["time_col"] is False and doc["input"]["group_col"] is False
