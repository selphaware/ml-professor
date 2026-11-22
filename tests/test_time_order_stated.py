"""T42: a date column with nothing said about time order stops the run;
`time_col=False` and `--no-time-order` (spec 2.1, 2.2, 2.4, 2.5, step 5,
step 8a, 5.2, 11; D58).
"""
import pandas as pd
import pytest

import cp5_helpers as p
import cp6_helpers as g
import mlp_helpers as h

CODE = "time_order_undeclared"


def _table(as_text=False, seed=701):
    frame = h.time_frame(200, seed=seed, task="classification")
    if as_text:
        frame["date"] = pd.Series(frame["date"].dt.strftime("%Y-%m-%d").tolist(), dtype=object)
    return frame


@pytest.mark.parametrize("as_text", [False, True], ids=["datetime", "iso_text"])
def test_a_date_column_and_no_word_about_time_stops_the_run(as_text, tmp_path):
    """F7d: a table with a date column and nothing said about time order stops at step 5."""
    from mlprofessor import errors

    log = h.Log()
    out = tmp_path / "out"
    err = g.raised(_table(as_text), families=[h.family("rec", log)],
                   steps=[h.step_factory(log)], out_dir=str(out), as_called=True)
    assert err.code == CODE, f"{type(err).__name__}: {err}"
    assert isinstance(err, errors.TimeOrderUndeclaredError)
    text = str(err)
    for needle in ("'date'", "time_col", "--time-col", "time_col=False", "--no-time-order"):
        assert needle in text, (needle, text)
    assert h.files_in(out) == []
    assert log.events == [], "a step or a model was fitted before the stop"


def test_the_message_names_every_date_column():
    frame = _table()
    frame["shipped"] = frame["date"] + pd.Timedelta(days=3)
    err = g.raised(frame, families=[h.family("rec", h.Log())])
    assert err.code == CODE, f"{type(err).__name__}: {err}"
    assert "'date'" in str(err) and "'shipped'" in str(err), str(err)


def test_the_statement_gives_the_random_scheme_and_is_recorded():
    result = g.run(_table(), families=[h.family("rec", h.Log())], time_col=False)
    doc = result.to_dict()
    assert doc["input"]["time_col"] is False
    assert doc["validation"]["time_col"] is None
    assert doc["validation"]["scheme"] == "random"
    assert doc["validation"]["caveat"] == h.CAVEAT
    pairs = h.warnings_of(result)
    assert [c for code, c in pairs if code == "date_column_undeclared"] == ["date"], pairs
    record = h.columns_of(result)["date"]
    assert (record["role"], record["reason"]) == ("excluded", "datetime"), record


def test_the_statement_on_a_table_with_no_dates_changes_only_its_record():
    frame = h.clf_frame(200, seed=702)
    plain = g.run(frame, families=[h.family("rec", h.Log())]).to_dict()
    stated = g.run(frame, families=[h.family("rec", h.Log())], time_col=False).to_dict()
    assert plain["input"]["time_col"] is None or plain["input"]["time_col"] is False
    assert stated["input"]["time_col"] is False
    assert h.without(plain, "meta.timing", "input.time_col") == \
        h.without(stated, "meta.timing", "input.time_col")


def test_a_declared_time_column_lets_other_date_columns_be_excluded():
    frame = h.time_frame(200, seed=703)
    frame["shipped"] = frame["date"] + pd.Timedelta(days=2)
    result = g.run(frame, families=[h.family("rec", h.Log())], time_col="date")
    assert h.columns_of(result)["shipped"]["reason"] == "datetime"
    assert g.count_code(result, "date_column_undeclared") == 0


@pytest.mark.parametrize("kw", [{"time_col": True}, {"time_col": False, "gap": 2}],
                         ids=["true", "false_with_gap"])
def test_values_of_time_col_that_are_refused(kw):
    from mlprofessor import errors

    err = g.raised(_table(), families=[h.family("rec", h.Log())], **kw)
    assert isinstance(err, errors.InputError), f"{type(err).__name__}: {err}"
    assert err.code == "input_error"


def _seen_twice():
    frame = p.twice(p.noise_frame(300, seed=704))
    frame.insert(0, "seen", h.dates(len(frame)))
    return frame


def test_the_date_error_comes_before_the_copies_error():
    from mlprofessor import errors

    frame = _seen_twice()
    n = len(frame)
    err = g.raised(frame, families=p.flat_family(h.Log(), n), as_called=True)
    assert err.code == CODE, f"{type(err).__name__}: {err}"

    err = g.raised(frame, families=p.flat_family(h.Log(), n), time_col=False)
    assert isinstance(err, errors.HoldoutCopiesError), f"{type(err).__name__}: {err}"
    assert "seen" in str(err) and "time_col" in str(err), str(err)


def test_the_command_line_states_it(tmp_path):
    path = tmp_path / "table.csv"
    _table().to_csv(path, index=False)
    common = ["profile", "--data", path, "--target", "y", "--profile", "fast", "--n-jobs", "1"]

    out = tmp_path / "plain"
    proc = h.run_cli([*common, "--out", out], as_given=True)
    assert proc.returncode == 3, proc.stderr
    assert CODE in proc.stderr, proc.stderr
    assert h.files_in(out) == []

    out = tmp_path / "stated"
    proc = h.run_cli([*common, "--out", out, "--no-time-order"])
    assert proc.returncode == 0, proc.stderr
    doc = h.load_json_strict(out / "profile.json")
    assert doc["input"]["time_col"] is False
    assert doc["validation"]["scheme"] == "random"
    assert "date_column_undeclared" in proc.stdout, proc.stdout

    out = tmp_path / "both"
    proc = h.run_cli([*common, "--out", out, "--no-time-order", "--time-col", "date"])
    assert proc.returncode == 2, proc.stderr
    assert h.files_in(out) == []
