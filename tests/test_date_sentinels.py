"""T55: a column of dates written YYYYMMDD with values that are no dates (spec 2.4, 2.5,
2.7, steps 3 and 5 rule 2a, 4.2, 4.3, 11; D69, D72).

Rule 2a is judged by the column's distinct values over the usable rows: with d distinct
values that read as dates and x that do not, the column is a date column when d >= 2 and
(d >= 9 x or (x <= 2 and d >= 5)). As the time column, a column of dates by that count
that holds a value that is no date stops the run with `time_column_invalid`.
"""
import numpy as np
import pandas as pd
import pytest

import cp5_helpers as p
import cp7_helpers as s
import cp8_helpers as e
import mlp_helpers as h
import test_date_numbers as dn


def _days(seed, form="int64"):
    """`s.day_numbers(seed=seed)`: 600 rows, three for each of 200 days, `day` as int64,
    as floats or as text of eight digits."""
    frame = s.day_numbers(seed=seed)
    if form == "float":
        frame["day"] = frame["day"].to_numpy(dtype=float)
    elif form == "text":
        frame["day"] = pd.array([str(v) for v in frame["day"].tolist()], dtype="str")
    else:
        assert form == "int64"
    return frame


def _assert_date(record):
    assert e.is_date(record), record
    assert (record["n_coerced"], record["n_infinite"], record["n_imputed"]) == (0, 0, 0), record


# ------------------------------------------------------------ the rule (step 5)

def _floats_with_gaps_and_an_infinite_cell(seed):
    frame = _days(seed, "float")
    column = frame["day"].to_numpy(dtype=float).copy()
    column[::17] = np.nan
    column[5] = np.inf
    frame["day"] = column
    return frame


SENTINEL_CASES = {
    "one_cell_of_99991231": lambda: e.planted(_days(5501), [e.SENTINEL], seed=1),
    "one_cell_of_0": lambda: e.planted(_days(5502), [0], seed=2),
    "one_cell_of_20240230": lambda: e.planted(_days(5503), [20240230], seed=3),
    "5_percent_of_the_rows_99991231": lambda: e.planted(_days(5504), [e.SENTINEL],
                                                        share=0.05, seed=4),
    "30_percent_of_the_rows_99991231": lambda: e.planted(_days(5505), [e.SENTINEL],
                                                         share=0.30, seed=5),
    "60_percent_of_the_rows_99991231": lambda: e.planted(_days(5506), [e.SENTINEL],
                                                         share=0.60, seed=6),
    "one_cell_each_of_99991231_and_0": lambda: e.planted(_days(5507), [e.SENTINEL, 0], seed=7),
    "one_cell_each_of_the_three_others": lambda: e.planted(_days(5508), e.OTHERS, seed=8),
    "floats_with_gaps_and_an_infinite_cell": lambda: _floats_with_gaps_and_an_infinite_cell(5509),
    "text_with_one_cell_0": lambda: e.planted(_days(5510, "text"), [0], seed=10),
}
OLD_CASES = ["before_1900", "after_2100", "29_feb_2023", "29_feb_1900", "month_13", "day_0",
             "31_april", "a_fraction", "an_infinite_cell", "nine_digit_text",
             "seven_digit_text"]


@pytest.mark.parametrize("case", list(SENTINEL_CASES) + OLD_CASES)
def test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column(case):
    """F14b: one value that is no date, and a column of dates written YYYYMMDD is a numeric feature."""
    frame = SENTINEL_CASES[case]() if case in SENTINEL_CASES else dn._no_date_case(case)
    _assert_date(e.roles(frame)["day"])


def test_a_sentinel_counts_once_however_many_cells_hold_it():
    """M31: the date rule counts cells, not distinct values."""
    rng = np.random.default_rng(5521)
    dated = np.repeat(np.array(e.days(20), dtype=np.int64), 3)
    n = 600
    y = rng.integers(0, 2, size=n)
    x = rng.normal(size=n)
    order = rng.permutation(n)

    one = np.concatenate([dated, np.full(n - 60, e.SENTINEL, dtype=np.int64)])[order]
    frame = pd.DataFrame({"day": one, "x": x, "y": y})
    assert frame["day"].nunique() == 21
    _assert_date(e.roles(frame)["day"])

    many = np.concatenate([dated, np.repeat(np.arange(30_000_000, 30_000_060,
                                                      dtype=np.int64), 9)])[order]
    frame = pd.DataFrame({"day": many, "x": x, "y": y})
    assert frame["day"].nunique() == 80
    record = e.roles(frame)["day"]
    assert e.is_number(record), record


@pytest.mark.parametrize("n_dates, n_others, is_date", [
    pytest.param(2, 0, True, id="two_dates_and_nothing_else"),
    pytest.param(5, 1, True, id="five_dates_and_one_other"),
    pytest.param(5, 2, True, id="five_dates_and_two_others"),
    pytest.param(27, 3, True, id="twenty_seven_dates_and_three_others"),
    pytest.param(4, 1, False, id="four_dates_and_one_other"),
    pytest.param(4, 2, False, id="four_dates_and_two_others"),
    pytest.param(26, 3, False, id="twenty_six_dates_and_three_others"),
])
def test_the_rule_from_both_sides(n_dates, n_others, is_date):
    frame = e.dates_with(n_dates, e.OTHERS[:n_others], seed=5530 + n_dates + n_others)
    record = e.roles(frame)["day"]
    if is_date:
        _assert_date(record)
    else:
        assert e.is_number(record), record


DATE_CODES = (20230115, 20240229, 19991231, 20110704)
OTHER_CODES = (31415926, 20231301, 12345678, 99999999)


@pytest.mark.parametrize("n_codes, n_dates, form", [
    pytest.param(2, 1, "int64", id="two_codes_one_date"),
    pytest.param(3, 1, "int64", id="three_codes_one_date"),
    pytest.param(3, 2, "int64", id="three_codes_two_dates"),
    pytest.param(5, 1, "int64", id="five_codes_one_date"),
    pytest.param(5, 2, "int64", id="five_codes_two_dates"),
    pytest.param(5, 3, "int64", id="five_codes_three_dates"),
    pytest.param(5, 4, "int64", id="five_codes_four_dates"),
    pytest.param(3, 2, "text", id="three_codes_two_dates_as_text"),
    pytest.param(5, 4, "text", id="five_codes_four_dates_as_text"),
])
def test_a_few_codes_of_which_some_read_as_dates_are_no_date_column(n_codes, n_dates, form):
    rng = np.random.default_rng(5540 + 10 * n_codes + n_dates)
    codes = list(DATE_CODES[:n_dates]) + list(OTHER_CODES[:n_codes - n_dates])
    n = 300
    column = np.array([codes[k % n_codes] for k in range(n)], dtype=np.int64)
    column = column[rng.permutation(n)]
    frame = pd.DataFrame({"day": column, "x": rng.normal(size=n),
                          "y": rng.integers(0, 2, size=n)})
    if form == "text":
        frame["day"] = pd.array([str(v) for v in column.tolist()], dtype="str")
    assert frame["day"].nunique() == n_codes
    record = e.roles(frame)["day"]
    assert record["role"] == "feature", record


def _days_with_values_of_their_own(n_values, seed):
    """200 days, with one of the three cells of `n_values` of them set to a value of its
    own that is no date: 99991231, 99991230, ..."""
    frame = _days(seed)
    rng = np.random.default_rng(seed + 1)
    chosen = rng.choice(sorted(set(frame["day"].tolist())), size=n_values, replace=False)
    column = frame["day"].to_numpy().copy()
    for j, day in enumerate(chosen):
        row = int(np.flatnonzero(column == day)[0])
        column[row] = e.SENTINEL - j
    frame["day"] = column
    return frame


def test_too_many_values_that_are_no_dates_leave_the_column_a_number():
    for n_values in (23, 70):
        frame = _days_with_values_of_their_own(n_values, seed=5550 + n_values)
        assert frame["day"].nunique() == 200 + n_values
        record = e.roles(frame)["day"]
        assert e.is_number(record), (n_values, record)


def test_the_date_rule_is_judged_on_the_usable_rows():
    frame = e.planted(_days(5561), [e.SENTINEL], seed=61)
    frame["y"] = frame["y"].astype(float)
    candidates = np.flatnonzero(frame["day"].to_numpy() != e.SENTINEL)
    lose = np.random.default_rng(5562).choice(candidates, size=60, replace=False)
    column = frame["day"].to_numpy().copy()
    target = frame["y"].to_numpy().copy()
    for j, row in enumerate(sorted(int(r) for r in lose)):
        column[row] = 30_000_000 + j
        target[row] = np.nan
    frame["day"] = column
    frame["y"] = target
    usable = frame[frame["y"].notna()]["day"]
    dates_left = {v for v in usable.tolist() if v != e.SENTINEL}
    assert len(dates_left) >= 190 and (usable == e.SENTINEL).sum() == 1
    _assert_date(e.roles(frame)["day"])


def test_it_is_excluded_named_and_handed_to_no_step():
    frame = e.planted(_days(5571), [e.SENTINEL], share=0.30, seed=71)
    log = h.Log()
    result = s.run(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)],
                   **s.SAID)
    _assert_date(h.columns_of(result)["day"])
    found = s.warning_records(result, "date_column_undeclared")
    assert [r["column"] for r in found] == ["day"], s.codes(result)
    assert result.to_dict()["validation"]["scheme"] == "random"
    handed = {c for ev in log.events if ev["kind"] == "step" for c in ev["columns"]}
    assert handed and "day" not in handed, sorted(handed)


def test_it_is_named_when_nothing_is_said_and_by_the_copies_stop():
    from mlprofessor import errors

    frame = e.planted(_days(5581), [e.SENTINEL], share=0.30, seed=81)
    err = s.raised(frame, families=[h.family("rec", h.Log())])
    assert err.code == "time_order_undeclared", (err.code, str(err))
    assert "'day'" in str(err), str(err)

    table = _noise_with_visit_day(5582)
    err = s.raised(table, families=[h.family("rec", h.Log())], **s.SAID)
    assert type(err) is errors.HoldoutCopiesError, (type(err), str(err))
    assert "visit_day" in str(err) and "time_col" in str(err), str(err)


def _noise_with_visit_day(seed):
    frame = p.noise_frame(300, seed=seed)
    visit = e.days(300)
    for k in range(0, 300, 4):
        visit[k] = e.SENTINEL
    frame.insert(0, "visit_day", np.array(visit, dtype=np.int64))
    return p.twice(frame, seed=seed)


def test_under_another_time_column_it_is_excluded_without_a_word():
    frame = h.time_frame(200, seed=5591)
    closed = e.days(200)
    for k in range(0, 200, 3):
        closed[k] = e.SENTINEL
    frame.insert(1, "closed", np.array(closed, dtype=np.int64))
    frame = frame.iloc[np.random.default_rng(5592).permutation(200)].reset_index(drop=True)
    log = h.Log()
    result = s.run(frame, families=[h.family("rec", log)], time_col="date", group_col=False)
    _assert_date(h.columns_of(result)["closed"])
    assert "date_column_undeclared" not in s.codes(result), s.codes(result)


# ------------------------------------------------------ as the time column (step 3)

def _time_column_with_sentinel(form, seed, n_sentinel=37, n_zero=0):
    """`day` in `form` with 99991231 in `n_sentinel` cells and 0 in `n_zero`; as floats,
    three more cells empty."""
    frame = _days(seed, "int64")
    rng = np.random.default_rng(seed + 1)
    rows = rng.choice(len(frame), size=n_sentinel + n_zero + 3, replace=False)
    values = frame["day"].to_numpy(dtype=object).copy()
    for row in rows[:n_sentinel]:
        values[row] = e.SENTINEL
    for row in rows[n_sentinel:n_sentinel + n_zero]:
        values[row] = 0
    if form == "int64":
        frame["day"] = np.array(values.tolist(), dtype=np.int64)
    elif form == "float":
        column = np.array(values.tolist(), dtype=float)
        column[rows[n_sentinel + n_zero:]] = np.nan
        frame["day"] = column
    else:
        frame["day"] = pd.array([str(v) for v in values.tolist()], dtype="str")
    return frame, rows[:n_sentinel]


def _assert_names_the_sentinel(err, count=37):
    from mlprofessor import errors

    assert type(err) is errors.TimeColumnError, (type(err), str(err))
    assert err.code == "time_column_invalid"
    assert str(e.SENTINEL) in str(err), str(err)
    if count is not None:
        assert count in e.whole_numbers_in(str(err)), str(err)


FORMS = ["int64", "float", "text"]


@pytest.mark.parametrize("form", FORMS)
def test_as_the_time_column_a_value_that_is_no_date_stops_the_run(form, tmp_path):
    """M32: as the time column, a value that is no date among dates written YYYYMMDD is ordered as a number."""
    frame, _ = _time_column_with_sentinel(form, seed=5601 + FORMS.index(form))
    out = tmp_path / "out"
    err = s.raised(frame, families=[h.family("rec", h.Log())], time_col="day",
                   group_col=False, out_dir=out)
    _assert_names_the_sentinel(err)
    assert h.files_in(out) == []


def test_the_message_names_the_value_that_most_rows_hold():
    frame, _ = _time_column_with_sentinel("int64", seed=5611, n_zero=5)
    assert (frame["day"] == 0).sum() == 5 and (frame["day"] == e.SENTINEL).sum() == 37
    err = s.raised(frame, families=[h.family("rec", h.Log())], time_col="day",
                   group_col=False)
    _assert_names_the_sentinel(err)


def test_the_time_column_is_judged_over_every_value_whatever_its_row():
    frame, rows = _time_column_with_sentinel("int64", seed=5621)
    target = frame["y"].to_numpy(dtype=float).copy()
    target[rows] = np.nan
    frame["y"] = target
    err = s.raised(frame, families=[h.family("rec", h.Log())], time_col="day",
                   group_col=False)
    _assert_names_the_sentinel(err, count=None)


@pytest.mark.parametrize("form", FORMS)
def test_read_time_refuses_it_too(form):
    from mlprofessor import errors, stages

    frame, _ = _time_column_with_sentinel(form, seed=5631 + FORMS.index(form))
    with pytest.raises(errors.ProfileError) as caught:
        stages.read_time(frame["day"])
    _assert_names_the_sentinel(caught.value)

    clean = _days(5641, "int64" if form == "int64" else form)
    if form == "float":
        column = clean["day"].to_numpy(dtype=float).copy()
        column[::17] = np.nan
        clean["day"] = column
    keys, time_form = stages.read_time(clean["day"])
    keys = np.asarray(keys)
    assert time_form == "numeric"
    assert keys.dtype == np.float64, keys.dtype
    expected = pd.to_numeric(pd.Series(clean["day"].tolist()), errors="coerce").to_numpy(float)
    assert np.array_equal(np.isnan(keys), np.isnan(expected))
    filled = ~np.isnan(expected)
    assert np.array_equal(keys[filled], expected[filled])


def test_a_time_column_that_is_no_date_column_is_a_number_as_before():
    frame = _days(5651)
    rng = np.random.default_rng(5652)
    rows = rng.choice(len(frame), size=30, replace=False)
    column = frame["day"].to_numpy().copy()
    for j, row in enumerate(rows):
        column[row] = 30_000_000 + j
    frame["day"] = column
    result = s.run(frame, families=[h.family("rec", h.Log())], time_col="day",
                   group_col=False)
    validation = result.to_dict()["validation"]
    assert validation["scheme"] == "time_ordered"
    assert validation["time_form"] == "numeric"
    assert validation["holdout_end"] == 30_000_029, validation["holdout_end"]
    assert h.columns_of(result)["day"]["role"] == "time"

    counted = _days(5653)
    column = np.arange(len(counted), dtype=np.int64)
    column[int(np.random.default_rng(5654).integers(0, len(counted)))] = e.SENTINEL
    counted["day"] = column
    result = s.run(counted, families=[h.family("rec", h.Log())], time_col="day",
                   group_col=False)
    validation = result.to_dict()["validation"]
    assert validation["scheme"] == "time_ordered"
    assert validation["holdout_end"] == e.SENTINEL, validation["holdout_end"]
