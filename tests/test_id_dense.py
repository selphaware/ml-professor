"""T61: the second clause of the id-like rule for whole numbers (spec step 5, rule 8b;
4.3, 7, 11; D19, D31, D74 point 5).

A whole-number column with no other role is id-like when every value is distinct and
the values are consecutive or strictly ordered in file order (D19), or when more than
95% of its values are distinct and the distinct values fill at least half of their
range, whatever the order of the rows (D74). Roles are read with
`cp8_helpers.roles(frame)` on a table of 300 rows: the column `code`, a noise column
and a noise target.
"""
import numpy as np
import pandas as pd
import pytest

import cp7_helpers as s
import cp8_helpers as e
import cp9_helpers as q
import mlp_helpers as h

N = 300


def _rising(seed=0):
    steps = np.random.default_rng([seed, 1]).integers(1, 50, size=N)
    return 1000 + np.cumsum(steps).astype(np.int64)


def _permuted(values, seed=0):
    values = np.asarray(values)
    return values[np.random.default_rng([seed, 2]).permutation(len(values))]


def _some_rows_gone(start=100001, dtype=np.int64):
    picked = np.random.default_rng(3).choice(400, size=N, replace=False)
    return np.array([start + int(k) for k in picked], dtype=dtype)


def _evens_and(last):
    return _permuted(np.array(list(range(2, 599, 2)) + [last], dtype=np.int64))


def _distinct_of_300(k):
    base = list(range(1, k + 1))
    extra = [base[j % k] for j in range(N - k)]
    return _permuted(np.array(base + extra, dtype=np.int64))


def _one_repeated():
    base = list(range(5001, 5300))
    return np.array(base + [base[150]], dtype=np.int64)


def _empty_cells():
    values = _some_rows_gone().astype(float)
    values[np.random.default_rng(4).choice(N, size=10, replace=False)] = np.nan
    return values


CASES = {
    # the first clause (D19)
    "in_file_order": (lambda: np.arange(1, N + 1, dtype=np.int64), True),
    "consecutive_in_no_order": (lambda: _permuted(np.arange(1, N + 1, dtype=np.int64)), True),
    "rising_with_gaps": (lambda: _rising(), True),
    "falling_with_gaps": (
        lambda: (90000 - np.cumsum(np.random.default_rng(5).integers(1, 50, size=N)))
        .astype(np.int64), True),
    "a_measure_in_no_order": (lambda: _permuted(_rising()), False),
    "few_values": (lambda: np.random.default_rng(6).integers(0, 11, size=N).astype(np.int64),
                   False),
    "fractions": (lambda: _permuted((2.0 * np.arange(N) + 1.0) / 2.0), False),
    # the second clause (D74)
    "no_order_some_rows_gone": (lambda: _some_rows_gone(), True),
    "no_order_some_rows_gone_as_floats": (lambda: _some_rows_gone().astype(float), True),
    "no_order_beyond_2_to_the_53": (lambda: _some_rows_gone(start=2 ** 60), True),
    "no_order_empty_cells": (_empty_cells, True),
    "no_order_one_row_repeated": (lambda: _permuted(_one_repeated()), True),
    "file_order_one_row_repeated": (lambda: np.sort(_one_repeated()), True),
    "286_distinct_of_300": (lambda: _distinct_of_300(286), True),
    "285_distinct_of_300": (lambda: _distinct_of_300(285), False),
    "fills_just_over_half_its_range": (lambda: _evens_and(600), True),
    "fills_half_its_range": (lambda: _evens_and(601), True),
    "fills_just_under_half_its_range": (lambda: _evens_and(602), False),
}


def _table(code, seed=0):
    rng = np.random.default_rng([seed, 7])
    return pd.DataFrame({"code": code, "noise": rng.normal(size=len(code)),
                         "y": rng.normal(size=len(code))})


def _present(values):
    out = []
    for v in np.asarray(values).tolist():
        if isinstance(v, float) and np.isnan(v):
            continue
        out.append(int(v))
    return out


def _facts(values):
    present = _present(values)
    distinct = sorted(set(present))
    span = distinct[-1] - distinct[0] + 1
    rising = all(a < b for a, b in zip(present, present[1:]))
    falling = all(a > b for a, b in zip(present, present[1:]))
    return present, distinct, span, rising, falling


@pytest.mark.parametrize("case", list(CASES))
def test_the_id_like_rule_for_whole_numbers(case):
    """F27 (M50 where a row is repeated): a whole-number identifier is a feature when the
    rows are shuffled, lost or repeated."""
    make, id_like = CASES[case]
    record = e.roles(_table(make()))["code"]
    if id_like:
        assert record["role"] == "excluded" and record["reason"] == "id_like", (case, record)
    else:
        assert record["role"] == "feature" and record["kind"] == "numeric", (case, record)
        assert record["reason"] is None


def test_the_cases_are_what_their_names_say():
    """The shares each case's name states, counted from the column alone."""
    for case in ("in_file_order", "consecutive_in_no_order"):
        present, distinct, span, rising, _ = _facts(CASES[case][0]())
        assert len(distinct) == N and span == N
        assert rising == (case == "in_file_order")
    for case, up in (("rising_with_gaps", True), ("falling_with_gaps", False)):
        present, distinct, span, rising, falling = _facts(CASES[case][0]())
        assert len(distinct) == N and (rising if up else falling)
        steps = {abs(b - a) for a, b in zip(present, present[1:])}
        assert min(steps) >= 1 and max(steps) <= 49 and 2 * N < span
    present, distinct, span, rising, falling = _facts(CASES["a_measure_in_no_order"][0]())
    assert len(distinct) == N and not rising and not falling and 2 * len(distinct) < span
    assert len(set(_present(CASES["few_values"][0]()))) <= 11
    fractions = CASES["fractions"][0]()
    assert not any(float(v).is_integer() for v in fractions)
    for case in ("no_order_some_rows_gone", "no_order_some_rows_gone_as_floats",
                 "no_order_beyond_2_to_the_53", "no_order_empty_cells",
                 "no_order_one_row_repeated", "file_order_one_row_repeated"):
        present, distinct, span, rising, falling = _facts(CASES[case][0]())
        assert 100 * len(distinct) > 95 * len(present), case
        assert 2 * len(distinct) >= span, case
        assert span > len(distinct) or len(distinct) < len(present), case
        if case.startswith("no_order"):
            assert not rising and not falling, case
    assert len(_present(CASES["no_order_empty_cells"][0]())) == N - 10
    assert min(_present(CASES["no_order_beyond_2_to_the_53"][0]())) >= 2 ** 60
    assert len(set(_present(CASES["no_order_beyond_2_to_the_53"][0]()))) == N
    present, distinct, _, _, _ = _facts(CASES["file_order_one_row_repeated"][0]())
    assert present == sorted(present) and len(distinct) == N - 1
    for k in (286, 285):
        present, distinct, span, _, _ = _facts(CASES[f"{k}_distinct_of_300"][0]())
        assert len(present) == N and len(distinct) == k and span == k
    for case, width in (("fills_just_over_half_its_range", 599),
                        ("fills_half_its_range", 600),
                        ("fills_just_under_half_its_range", 601)):
        present, distinct, span, _, _ = _facts(CASES[case][0]())
        assert len(distinct) == N and span == width, case


def _with_dropped_targets(permute):
    rng = np.random.default_rng(8)
    n = 340
    frame = pd.DataFrame({"code": np.arange(100001, 100001 + n, dtype=np.int64),
                          "z1": rng.normal(size=n), "z2": rng.normal(size=n),
                          "y": rng.normal(size=n)})
    frame.loc[rng.choice(n, size=40, replace=False), "y"] = np.nan
    if permute:
        frame = frame.iloc[rng.permutation(n)].reset_index(drop=True)
    return frame


@pytest.mark.parametrize("case", ["file_order", "no_order"])
def test_the_rule_reads_the_rows_that_have_a_target(case):
    """F27 on `no_order`: rows with no target leave gaps, and the identifier stays one."""
    frame = _with_dropped_targets(case == "no_order")
    assert int(frame["y"].isna().sum()) == 40
    result = q.screened(frame, s.SAID)
    assert h.rows_of(result)["dropped_missing_target"] == 40
    code = q.column(result, "code")
    assert code["role"] == "excluded" and code["reason"] == "id_like", code
    for name in ("z1", "z2"):
        assert q.is_feature(q.column(result, name)), name


def test_a_shuffled_identifier_reaches_no_model():
    """F27: a shuffled identifier of a classification is handed to no step and no model."""
    rng = np.random.default_rng(9)
    frame = h.clf_frame(n=N, seed=9)
    frame.insert(0, "code", _some_rows_gone())
    frame = frame.iloc[rng.permutation(N)].reset_index(drop=True)
    log = h.Log()
    result = s.run(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)],
                   **s.SAID)
    assert q.column(result, "code")["reason"] == "id_like"
    assert log.events, "no step or model was handed a row"
    for event in log.events:
        assert "code" not in event["columns"], event["columns"]


def test_the_time_column_and_the_entity_column_keep_their_roles():
    """A dense whole-number time column and entity column keep their own roles."""
    rng = np.random.default_rng(10)
    frame = pd.DataFrame({
        "t": rng.choice(400, size=N, replace=False).astype(np.int64) + 1,
        "who": rng.permutation(N).astype(np.int64) + 5000,
        "x": rng.normal(size=N),
        "y": rng.normal(size=N),
    })
    roles = e.roles(frame, time_col="t", group_col="who")
    assert roles["t"]["role"] == "time"
    assert roles["who"]["role"] == "group"
    assert roles["y"]["role"] == "target"
    assert roles["x"]["role"] == "feature"
