"""T64: the `extreme_value` warning (spec step 11, 2.5, 4.3, 7, 11; D74 point 3, D76
point 3).

For every numeric feature that reaches the models, and the target of a regression, over
the rows that are split: the distinct finite values in order; from each end the three
outermost are looked at, and a value is far outside when its gap to the next value
inward is more than 30 times the spread of the ten distinct values after that one. One
warning for each value named; its message holds `holds the value X in N row`. The rule
is written again, value by value, in `cp9_helpers.far_outside`.
"""
import re

import numpy as np
import pandas as pd
import pytest

import cp7_helpers as s
import cp9_helpers as q
import mlp_helpers as h

CODE = "extreme_value"
NAMED = re.compile(r"holds the value (\S+) in (\d+) row")


def _table(n=300, seed=8601):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": 50.0 + 10.0 * rng.normal(size=n), "b": rng.normal(size=n),
                         "flag": rng.integers(0, 2, size=n).astype(np.int64),
                         "y": rng.normal(size=n)})


def _named(result, name):
    """`(value, rows)` of every `extreme_value` record about `name`."""
    out = []
    for record in s.warning_records(result, CODE):
        if record["column"] != name:
            continue
        found = NAMED.search(str(record["message"]))
        assert found, record["message"]
        out.append((float(found.group(1)), int(found.group(2))))
    return out


def _assert_as_by_hand(result, frame, name):
    """The values named about `name` are those of `far_outside` over the rows that have a
    target, each with its count of rows, in the pinned form."""
    present = frame["y"].notna().to_numpy()
    want = q.far_outside(frame[name].to_numpy(dtype=float)[present])
    got = _named(result, name)
    assert sorted(set(got)) == sorted({(v, k) for v, k, _ in want}), (got, want)
    messages = [str(r["message"]) for r in s.warning_records(result, CODE)
                if r["column"] == name]
    for value, rows, _ in want:
        text = f"holds the value {value!r} in {rows} row"
        assert any(text in m for m in messages), (text, messages)
    return want


def _run(frame, **kw):
    _, families = s.rec()
    return s.run(frame, families=families, **{**s.SAID, **kw})


@pytest.mark.parametrize("name,value", [("a", -999.0), ("a", 999999.0),
                                        ("b", -999.0), ("b", 999999.0)])
def test_one_cell_far_outside_a_feature_is_named(name, value):
    """M46: one cell of -999 or 999999 in a numeric feature is named."""
    frame = _table()
    frame.loc[17, name] = value
    result = _run(frame)
    want = _assert_as_by_hand(result, frame, name)
    assert (value, 1) in {(v, k) for v, k, _ in want}


def test_the_target_of_a_regression_is_looked_at():
    frame = _table()
    frame.loc[23, "y"] = 999999.0
    result = _run(frame, task="regression")
    want = _assert_as_by_hand(result, frame, "y")
    assert [(v, k) for v, k, _ in want] == [(999999.0, 1)]


def test_a_value_counts_once_however_many_rows_hold_it():
    """M46: four cells of -999 are one value, named once, in 4 rows."""
    frame = _table()
    frame.loc[[3, 70, 140, 250], "a"] = -999.0
    result = _run(frame)
    _assert_as_by_hand(result, frame, "a")
    assert _named(result, "a") == [(-999.0, 4)]


@pytest.mark.parametrize("value,named", [(310.0, False), (311.0, True)])
def test_the_gap_must_be_more_than_thirty_times_the_spread(value, named):
    """Above the whole numbers 0 to 10 the spread is 10: a gap of 300 is not more than 30
    times it, a gap of 301 is."""
    frame = _table()
    column = np.array([float(k % 11) for k in range(len(frame))])
    column[100] = value
    frame = q.with_column(frame, "c", column)
    result = _run(frame)
    want = _assert_as_by_hand(result, frame, "c")
    assert bool(want) == named
    assert ((value, 1) in _named(result, "c")) == named


SHAPES = {
    "two_values": ([0.0, 1.0], []),
    "three_values": ([0.0, 1.0, 1000.0], [1000.0]),
    "three_values_near": ([0.0, 1.0, 31.0], []),
    "two_beyond": ([-999.0, -998.0] + [float(v) for v in range(20, 40)], [-999.0, -998.0]),
    "three_beyond": ([-999.0, -998.0, -997.0] + [float(v) for v in range(20, 40)],
                     [-999.0, -998.0, -997.0]),
    "four_beyond": ([-999.0, -998.0, -997.0, -996.0] + [float(v) for v in range(20, 40)], []),
    "both_ends": ([-999.0] + [float(v) for v in range(20, 40)] + [999999.0],
                  [-999.0, 999999.0]),
}


@pytest.mark.parametrize("case", list(SHAPES))
def test_the_three_outermost_values_at_each_end_are_looked_at(case):
    """M46 on `both_ends`: the three outermost values at each end, and the values beyond
    a value far outside."""
    values, expected = SHAPES[case]
    frame = _table()
    frame = q.with_column(frame, "c", [values[k % len(values)] for k in range(len(frame))])
    result = _run(frame)
    want = _assert_as_by_hand(result, frame, "c")
    assert sorted({v for v, _, _ in want}) == sorted(expected)
    assert sorted({v for v, _ in _named(result, "c")}) == sorted(expected)


def test_only_rows_that_are_split_are_looked_at():
    """A far value in a row with no target is not named, and such a row is not counted."""
    frame = _table()
    frame.loc[[5, 6], "y"] = np.nan
    frame.loc[5, "a"] = -999.0
    frame.loc[[6, 40], "b"] = 999999.0
    result = _run(frame)
    assert _named(result, "a") == []
    assert _named(result, "b") == [(999999.0, 1)]
    _assert_as_by_hand(result, frame, "b")


def test_columns_that_reach_no_model_are_not_looked_at():
    """An identifier, a leak, a text column and a classification target are not looked
    at; a numeric feature beside them is."""
    frame = _table()
    frame.loc[9, "a"] = -999.0
    identifier = np.arange(1, len(frame) + 1, dtype=np.int64)
    identifier[-1] = 1_000_000
    frame.insert(0, "code", identifier)
    leak = 3.7 * frame["y"].to_numpy(dtype=float) + 12.0
    leak[30] = 999999.0
    frame = q.with_column(frame, "leak", leak)
    words = ["red" if k % 2 else "blue" for k in range(len(frame))]
    words[50] = "999999"
    frame = q.with_column(frame, "colour", pd.array(words, dtype="str"))
    result = _run(frame, task="regression")
    assert q.column(result, "code")["reason"] == "id_like"
    assert q.is_leak(q.column(result, "leak"))
    assert q.column(result, "colour")["kind"] == "categorical"
    assert {r["column"] for r in s.warning_records(result, CODE)} == {"a"}

    labelled = h.clf_frame(n=300, seed=8611)
    labelled["y"] = labelled["y"] * 999999
    labelled.loc[12, "x1"] = -999.0
    result = _run(labelled)
    assert result.to_dict()["task"]["type"] == "classification"
    assert {r["column"] for r in s.warning_records(result, CODE)} == {"x1"}


def test_an_infinite_cell_is_no_extreme_value():
    frame = _table()
    frame.loc[8, "a"] = np.inf
    frame.loc[9, "b"] = -np.inf
    result = _run(frame)
    assert CODE not in s.codes(result)


def _shape(name, n):
    rng = np.random.default_rng([8641, n, len(name)])
    if name == "normal":
        return rng.normal(size=n)
    if name == "log_normal_1":
        return rng.lognormal(sigma=1.0, size=n)
    if name == "log_normal_1.5":
        return rng.lognormal(sigma=1.5, size=n)
    if name == "volume":
        return np.round(rng.lognormal(mean=12.0, sigma=1.2, size=n))
    if name == "exponential":
        return rng.exponential(size=n)
    if name == "count":
        return rng.poisson(3.0, size=n).astype(float)
    raise ValueError(name)


@pytest.mark.parametrize("n", [200, 2000])
@pytest.mark.parametrize("shape", ["normal", "log_normal_1", "log_normal_1.5", "volume",
                                   "exponential", "count"])
def test_a_long_tail_is_not_named(shape, n):
    """Columns with long tails, and no value standing for another, are quiet."""
    rng = np.random.default_rng([8642, n])
    frame = pd.DataFrame({"x": _shape(shape, n), "z": rng.normal(size=n),
                          "y": rng.normal(size=n)})
    result = _run(frame)
    assert q.far_outside(frame["x"].to_numpy(dtype=float)) == []
    assert CODE not in s.codes(result)


def test_a_clean_table_has_no_such_warning_under_any_scheme():
    for scheme in q.SCHEMES:
        frame, statements = q.scheme_table(scheme, seed=8651)
        result = q.screened(frame, statements)
        assert CODE not in s.codes(result), scheme


class _ValueStep:
    """A step that changes nothing and records the cells of `a` it is fitted on."""

    def __init__(self, seen):
        self.seen = seen

    def fit(self, X, y=None):
        self.seen.append({int(r): float(v) for r, v in zip(X.index, X["a"].tolist())})
        return self

    def transform(self, X):
        return X.copy()


def test_the_warning_changes_no_number():
    """The far value is in every fit that holds its row, and the split is that of the
    table without it."""
    clean = _table()
    clean_result = _run(clean)
    row = q.training_rows(clean_result, clean)[60]
    frame = clean.copy()
    frame.loc[row, "a"] = 999999.0
    seen = []
    _, families = s.rec()
    result = s.run(frame, families=families, steps=[lambda: _ValueStep(seen)], **s.SAID)
    assert _named(result, "a") == [(999999.0, 1)]
    assert h.holdout_rows(result) == h.holdout_rows(clean_result)
    holding = [fit for fit in seen if row in fit]
    assert holding, "no fit held the row"
    assert all(fit[row] == 999999.0 for fit in holding)
    assert q.is_feature(q.column(result, "a"))


def test_the_command_line_writes_the_warning(tmp_path):
    """M46: through the command line, `profile.json` holds the warning."""
    frame = _table()
    frame.loc[17, "a"] = -999.0
    csv_path = tmp_path / "table.csv"
    frame.to_csv(csv_path, index=False)
    out = tmp_path / "out"
    proc = s.run_cli(["profile", "--data", csv_path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1", "--no-time-order",
                      "--no-entities"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = h.load_json_strict(out / "profile.json")
    records = [w for w in doc["warnings"] if w["code"] == CODE and w["column"] == "a"]
    assert len(records) == 1, doc["warnings"]
    assert "holds the value -999.0 in 1 row" in records[0]["message"]
