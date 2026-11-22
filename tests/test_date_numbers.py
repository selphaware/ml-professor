"""T52: dates written as whole numbers, YYYYMMDD, are a date form (spec 2.7, steps 3, 5
and 8a, 4.5, 11; D66).

Rule 2a of step 5: every non-missing value among the usable rows reads as a calendar
date written YYYYMMDD, and at least two differ. Such a column is excluded as `datetime`,
named under `time_col=False`, and accepted as the time column.
"""
import datetime as dt
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import cp5_helpers as p
import cp7_helpers as s
import mlp_helpers as h


def _roles(frame, target="y", **kw):
    from mlprofessor import stages

    return {r["name"]: r for r in h.records(stages.column_roles(frame, target, **kw))}


def _as_form(frame, form):
    """`day` of `s.day_numbers()` in another form of column."""
    out = frame.copy()
    values = out["day"].tolist()
    if form == "int64":
        return out
    if form == "float_with_gaps":
        column = np.array(values, dtype=float)
        column[::17] = np.nan
        out["day"] = column
    elif form == "text":
        out["day"] = pd.array([f" {v} " if k % 9 == 0 else str(v)
                               for k, v in enumerate(values)], dtype="str")
    elif form == "nullable_int":
        column = pd.array(values, dtype="Int64")
        column[::13] = pd.NA
        out["day"] = column
    elif form == "python_ints":
        out["day"] = pd.Series([None if k % 11 == 0 else int(v)
                                for k, v in enumerate(values)], dtype=object)
    elif form == "decimal":
        out["day"] = pd.Series([Decimal(int(v)) for v in values], dtype=object)
    else:
        raise AssertionError(form)
    return out


FORMS = ["int64", "float_with_gaps", "text", "nullable_int", "python_ints", "decimal"]


@pytest.mark.parametrize("form", FORMS)
def test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date(form):
    """F14: a column of dates written YYYYMMDD is excluded as a date, not read as a number."""
    frame = _as_form(s.day_numbers(seed=1701), form)
    log = h.Log()
    result = s.run(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)],
                   **s.SAID)
    record = h.columns_of(result)["day"]
    assert record["role"] == "excluded" and record["reason"] == "datetime", record
    assert (record["n_coerced"], record["n_infinite"], record["n_imputed"]) == (0, 0, 0)
    found = s.warning_records(result, "date_column_undeclared")
    assert [r["column"] for r in found] == ["day"], s.codes(result)
    assert result.to_dict()["validation"]["scheme"] == "random"
    handed = {c for ev in log.events if ev["kind"] == "step" for c in ev["columns"]}
    assert handed and "day" not in handed, sorted(handed)


def _two_values(first, second, n=200, seed=1702):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"day": np.array([first, second] * (n // 2), dtype=np.int64),
                         "x": rng.normal(size=n), "y": rng.integers(0, 2, size=n)})


@pytest.mark.parametrize("first, second", [
    pytest.param(19000101, 21001231, id="ends_of_the_range"),
    pytest.param(20240229, 20240301, id="leap_day_2024"),
    pytest.param(20000229, 20000301, id="leap_day_2000"),
    pytest.param(19991231, 20000101, id="turn_of_the_century"),
])
def test_the_ends_of_the_range_and_the_calendar_are_dates(first, second):
    record = _roles(_two_values(first, second))["day"]
    assert record["role"] == "excluded" and record["reason"] == "datetime", record


def _no_date_case(case):
    """`s.day_numbers()` with one value that is no date, or a column of another kind."""
    frame = s.day_numbers(seed=1703)
    values = frame["day"].tolist()
    singles = {"before_1900": 18991231, "after_2100": 21010101, "29_feb_2023": 20230229,
               "29_feb_1900": 19000229, "month_13": 20231301, "day_0": 20230100,
               "31_april": 20230431}
    if case in singles:
        values[5] = singles[case]
        frame["day"] = np.array(values, dtype=np.int64)
    elif case == "a_fraction":
        column = np.array(values, dtype=float)
        column[5] = 20230101.5
        frame["day"] = column
    elif case == "an_infinite_cell":
        column = np.array(values, dtype=float)
        column[5] = np.inf
        frame["day"] = column
    elif case == "day_first":
        frame["day"] = np.array([(v % 100) * 1_000_000 + (v // 100 % 100) * 10_000 + v // 10_000
                                 for v in values], dtype=np.int64)
    elif case == "yyyymm":
        frame["day"] = np.array([v // 100 for v in values], dtype=np.int64)
    elif case in ("nine_digit_text", "seven_digit_text"):
        text = [str(v) for v in values]
        text[5] = "0" + text[5] if case == "nine_digit_text" else text[5][:7]
        frame["day"] = pd.array(text, dtype="str")
    elif case == "one_date_throughout":
        frame["day"] = np.full(len(frame), 20230101, dtype=np.int64)
    else:
        raise AssertionError(case)
    return frame


@pytest.mark.parametrize("case", [
    "before_1900", "after_2100", "29_feb_2023", "29_feb_1900", "month_13", "day_0",
    "31_april", "a_fraction", "an_infinite_cell", "day_first", "yyyymm", "nine_digit_text",
    "seven_digit_text", "one_date_throughout",
])
def test_values_that_are_no_dates_leave_the_column_what_it_was(case):
    record = _roles(_no_date_case(case))["day"]
    if case == "one_date_throughout":
        assert record["role"] == "excluded" and record["reason"] == "constant", record
    elif case in ("day_first", "yyyymm"):
        assert record["role"] == "feature" and record["kind"] == "numeric", record
    else:
        # D59, D69: 200 days and one value that is no date. A number as built, and a
        # date column once D69 is built; `test_date_sentinels.py` holds the second.
        as_built = record["role"] == "feature" and record["kind"] == "numeric"
        as_decided = record["role"] == "excluded" and record["reason"] == "datetime"
        assert as_built or as_decided, record


def test_it_is_judged_on_the_usable_rows():
    frame = s.day_numbers(seed=1704)
    frame["y"] = frame["y"].astype(float)
    frame.loc[5, "day"] = 20231301
    frame.loc[5, "y"] = np.nan
    record = _roles(frame)["day"]
    assert record["role"] == "excluded" and record["reason"] == "datetime", record


def _one_a_day(form, n=200, seed=1705):
    rng = np.random.default_rng(seed)
    first = dt.date(2021, 3, 1)
    days = [s.date_number(first + dt.timedelta(days=k)) for k in range(n)]
    column = (np.array(days, dtype=np.int64) if form == "int64"
              else pd.array([str(v) for v in days], dtype="str"))
    return pd.DataFrame({"day": column, "x": rng.normal(size=n),
                         "y": rng.integers(0, 2, size=n)})


@pytest.mark.parametrize("form", ["int64", "text"])
def test_the_date_rule_comes_before_the_id_like_rule(form):
    """M29: one row a day in date order is a date, not an identifier."""
    result = s.run(_one_a_day(form), families=[h.family("rec", h.Log())], **s.SAID)
    record = h.columns_of(result)["day"]
    assert record["role"] == "excluded" and record["reason"] == "datetime", record
    found = s.warning_records(result, "date_column_undeclared")
    assert [r["column"] for r in found] == ["day"], s.codes(result)


def test_it_is_named_when_nothing_is_said_about_time_order():
    err = s.raised(s.day_numbers(seed=1706), families=[h.family("rec", h.Log())])
    assert err.code == "time_order_undeclared", f"{type(err).__name__}: {err}"
    assert "'day'" in str(err), str(err)


def _expected_holdout(frame):
    """The rows of the latest whole days that together hold a quarter of the usable rows."""
    usable = frame["day"].notna().to_numpy()
    days = [int(v) if ok else None for v, ok in zip(frame["day"].tolist(), usable)]
    n = int(usable.sum())
    counts = pd.Series([d for d in days if d is not None]).value_counts()
    held_days, total = [], 0
    for day in sorted(counts.index, reverse=True):
        held_days.append(int(day))
        total += int(counts[day])
        if 4 * total >= n:
            break
    first_held = min(held_days)
    rows = [r for r, d in enumerate(days) if d is not None and d >= first_held]
    train_end = max(d for d in days if d is not None and d < first_held)
    return rows, train_end, first_held, max(held_days)


def _assert_time_run(frame, result):
    rows, train_end, start, end = _expected_holdout(frame)
    assert sorted(h.holdout_rows(result)) == rows
    v = result.to_dict()["validation"]
    assert v["scheme"] == "time_ordered" and v["time_form"] == "numeric"
    assert v["train_end"] == train_end and not isinstance(v["train_end"], str)
    assert v["holdout_start"] == start and not isinstance(v["holdout_start"], str)
    assert v["holdout_end"] == end and not isinstance(v["holdout_end"], str)
    assert h.columns_of(result)["day"]["role"] == "time"


@pytest.mark.parametrize("form", ["int64", "nullable_int"])
def test_a_date_number_column_is_a_time_column_like_any_number(form):
    frame = _as_form(s.day_numbers(seed=1707), form)
    result = s.run(frame, families=[h.family("rec", h.Log())], time_col="day", group_col=False)
    _assert_time_run(frame, result)


def test_eight_digit_text_is_accepted_as_the_time_column():
    """M28: eight-digit text in which every value is a calendar date is a numeric time column."""
    from mlprofessor import errors, stages

    base = s.day_numbers(seed=1708)
    numbers = base["day"].tolist()
    frame = base.copy()
    frame["day"] = pd.array([str(v) for v in numbers], dtype="str")
    result = s.run(frame, families=[h.family("rec", h.Log())], time_col="day", group_col=False)
    _assert_time_run(base, result)

    keys, form = stages.read_time(frame["day"])
    assert form == "numeric"
    keys = np.asarray(keys)
    assert keys.dtype == np.float64
    assert keys.tolist() == [float(v) for v in numbers]

    text = [str(v) for v in numbers]
    text[7] = "20231301"
    frame["day"] = pd.array(text, dtype="str")
    err = s.raised(frame, families=[h.family("rec", h.Log())], time_col="day", group_col=False)
    assert isinstance(err, errors.TimeColumnError), f"{type(err).__name__}: {err}"


def test_under_a_time_column_another_date_number_column_is_excluded_without_a_word():
    frame = h.time_frame(200, seed=1709)
    rng = np.random.default_rng(1709)
    first = dt.date(2015, 1, 1)
    frame["day2"] = np.array([s.date_number(first + dt.timedelta(days=int(k)))
                              for k in rng.integers(0, 3000, size=200)], dtype=np.int64)
    result = s.run(frame, families=[h.family("rec", h.Log())], time_col="date", group_col=False)
    record = h.columns_of(result)["day2"]
    assert record["role"] == "excluded" and record["reason"] == "datetime", record
    assert "date_column_undeclared" not in s.codes(result)


def test_the_copies_stop_names_a_date_number_column():
    from mlprofessor import errors

    base = p.noise_frame(300, seed=1710)
    rng = np.random.default_rng(1710)
    first = dt.date(2015, 1, 1)
    base.insert(0, "visit_day", np.array(
        [s.date_number(first + dt.timedelta(days=int(k))) for k in rng.integers(0, 3000, 300)],
        dtype=np.int64))
    frame = p.twice(base, seed=1711)
    err = s.raised(frame, families=p.flat_family(h.Log(), len(frame)), **s.SAID)
    assert isinstance(err, errors.HoldoutCopiesError), f"{type(err).__name__}: {err}"
    assert "visit_day" in str(err) and "time_col" in str(err), str(err)
