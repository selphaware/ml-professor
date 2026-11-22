"""T56: `duplicates="group"` and a column the leak screen removes (spec 2.4, 2.5, steps 7
and 8a, 5.4, 11; D69, D72).

Under `"group"` the units are formed before the leak screen. When the screen excludes a
column, a holdout row that then agrees with a training row in every feature that is
left stops the run with `holdout_copies_in_training`, with no threshold, naming the
column. And the `same_features_grouped` warning says what `"group"` measures.
"""
import pandas as pd
import pytest

import cp3_helpers as c
import cp5_helpers as p
import cp7_helpers as s
import cp8_helpers as e
import mlp_helpers as h

CODE = "holdout_copies_in_training"
MEASURES = ["n1", "n2", "n3", "n4"]


def _few_valued_with_status(seed):
    return e.with_copy_of_target(s.few_valued(600, (4, 4), seed=seed))


def _assert_copies_error(err):
    from mlprofessor import errors

    assert type(err) is errors.HoldoutCopiesError, f"{type(err).__name__}: {err}"
    assert err.code == CODE


def test_a_copy_of_the_target_under_group_stops_the_run(tmp_path):
    """F12b: under duplicates="group" a column the leak screen removes leaves combinations on both sides."""
    frame = _few_valued_with_status(1901)
    log = h.Log()
    out = tmp_path / "out"
    err = s.raised(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)],
                   duplicates="group", out_dir=out, **s.SAID)
    _assert_copies_error(err)
    message = str(err)
    for needed in ("status", "group", "keep", s.COPIES_ADVICE):
        assert needed in message, f"{needed!r} not in: {message}"
    assert e.holds_count_and_share(message, len(frame)), message
    assert log.events == [], "a step or a model was handed rows before the stop"
    assert h.files_in(out) == []


def test_the_stop_names_a_date_column_that_was_excluded_too():
    frame = _few_valued_with_status(1910)
    frame.insert(0, "seen on", pd.array(
        [d.strftime("%Y-%m-%d") for d in pd.date_range("2022-01-03", periods=len(frame))],
        dtype="str"))
    err = s.raised(frame, families=[h.family("rec", h.Log())], duplicates="group", **s.SAID)
    _assert_copies_error(err)
    message = str(err)
    for needed in ("status", "seen on", "time_col"):
        assert needed in message, f"{needed!r} not in: {message}"


def test_one_combination_on_both_sides_is_enough():
    """M33: the stop under "group" after a removed leak has no threshold: one such row is enough."""
    frame = e.one_row_alike_in_every_entity(24, 20)
    assert not frame.duplicated(MEASURES + ["leak"]).any()
    alike = frame[MEASURES].eq(list(e.ALIKE)).all(axis=1)
    assert int(alike.sum()) == 24
    assert frame.loc[alike, "who"].nunique() == 24
    assert sorted(frame.loc[alike, "y"].value_counts().tolist()) == [12, 12]
    assert int(frame.duplicated(MEASURES, keep=False).sum()) == 24

    err = s.raised(frame, families=[h.family("rec", h.Log())], duplicates="group",
                   time_col=False, group_col="who")
    _assert_copies_error(err)
    message = str(err)
    assert "leak" in message and "group" in message, message
    assert any(1 <= n <= 9 for n in e.whole_numbers_in(message)), message
    assert e.holds_count_and_share(message, len(frame)), message


def _finished_cases():
    return {
        "few_valued": (lambda: s.few_valued(600, (4, 4), seed=1920), "grouped", False),
        "noise_with_status": (lambda: e.with_copy_of_target(p.noise_frame(400, seed=1921)),
                              "random", True),
        "twice_with_status": (lambda: e.with_copy_of_target(p.twice(p.noise_frame(300, seed=1922))),
                              "grouped", True),
    }


@pytest.mark.parametrize("case", list(_finished_cases()))
def test_no_combination_of_the_features_is_on_both_sides_in_a_finished_run(case):
    make, scheme, has_status = _finished_cases()[case]
    frame = make()
    log, families = s.rec()
    result = s.run(frame, families=families, duplicates="group", **s.SAID)
    doc = result.to_dict()
    assert doc["validation"]["scheme"] == scheme
    if has_status:
        record = h.columns_of(result)["status"]
        assert (record["role"], record["reason"]) == ("excluded", "suspected_leak"), record
    assert e.rows_reached(result, log) == set(range(len(frame)))
    keys = e.feature_keys(result, frame)
    holdout = set(h.holdout_rows(result))
    held = {keys[r] for r in holdout}
    rest = {keys[r] for r in range(len(frame)) if r not in holdout}
    assert not held & rest, sorted(held & rest)[:5]
    assert doc["validation"]["n_holdout_copies"] == 0


def test_the_grouped_warning_says_what_group_measures():
    """M37: the same_features_grouped warning says what duplicates="group" measures."""
    frame = s.few_valued(600, (4, 4), seed=1930)
    result = s.run(frame, families=[h.family("rec", h.Log())], duplicates="group", **s.SAID)
    found = s.messages(result, "same_features_grouped")
    assert len(found) == 1, s.codes(result)
    assert e.GROUP_MEASURES in found[0], found[0]


def test_under_keep_and_error_nothing_changes():
    frame = _few_valued_with_status(1940)
    result = s.run(frame, families=[h.family("rec", h.Log())], duplicates="keep", **s.SAID)
    record = h.columns_of(result)["status"]
    assert (record["role"], record["reason"]) == ("excluded", "suspected_leak"), record
    v = result.to_dict()["validation"]
    assert v["scheme"] == "random"
    assert v["n_rows_same_features"] is None
    assert s.codes(result).count("holdout_copies_kept") == 1, s.codes(result)

    err = s.raised(frame, families=[h.family("rec", h.Log())], **s.SAID)
    _assert_copies_error(err)
    assert s.COPIES_ADVICE in str(err), str(err)


def test_the_command_line_stops_too(tmp_path):
    frame = _few_valued_with_status(1901)
    csv_path = tmp_path / "table.csv"
    frame.to_csv(csv_path, index=False)
    out = tmp_path / "out"
    proc = s.run_cli(["profile", "--data", csv_path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1", "--duplicates", "group",
                      "--no-time-order", "--no-entities"])
    assert proc.returncode == 3, proc.stdout + proc.stderr
    lines = proc.stderr.strip().splitlines()
    assert len(lines) == 1, proc.stderr
    assert lines[0].startswith(f"error: {CODE}: "), lines[0]
    assert "status" in lines[0], lines[0]
    assert h.files_in(out) == []


def test_without_the_leak_the_same_table_runs():
    frame = _few_valued_with_status(1901).drop(columns=["status"])
    result = s.run(frame, families=[h.family("rec", h.Log())], duplicates="group", **s.SAID)
    v = result.to_dict()["validation"]
    assert v["scheme"] == "grouped"
    assert v["n_holdout_copies"] == 0
    c.assert_strict_json(result)
