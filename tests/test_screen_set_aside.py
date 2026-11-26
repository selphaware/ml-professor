"""T63: training rows whose target is wildly off are set aside from the whole leak screen
(spec step 8, 4.3, 7; D74 point 2).

A row is marked when its target lies more than 10 median deviations from the median
target of the training rows or, where the previous value is a baseline, more than 10
median steps from its previous value. The marked rows are set aside when they are at
most 1% of the training rows (at least one row allowed); when more are marked, none is
set aside and as D74 had it, every row is scored. Since the ninth pass (D77 point 5, D78
point 7) rule 2 then still leaves its rows unscored and is also read against each fold's
training median: tests/test_screen_marked.py pins that, and this file accepts either
(D59).
"""
import math

import numpy as np
import pytest

import cp7_helpers as s
import cp9_helpers as q
import mlp_helpers as h

WILD = 999999.0


def _roles(result):
    return {r["name"]: (r["role"], r["reason"], r["leak_rule"])
            for r in h.records(result.columns)}


def _middle(frame, statements, training, n_cells):
    """`n_cells` training rows from the middle of the training rows, seven apart; under
    time order from the middle in time, one row at each of timestamps seven apart."""
    time = statements.get("time_col")
    if not time:
        ordered = sorted(training)
        mid = len(ordered) // 2
        return [ordered[mid + 7 * k] for k in range(n_cells)]
    keys = frame[time].tolist()
    stamps = sorted({keys[r] for r in training})
    mid = len(stamps) // 2
    picked = []
    for k in range(n_cells):
        stamp = stamps[mid + 7 * k]
        picked.append(min(r for r in training if keys[r] == stamp))
    return picked


def _with_wild_targets(frame, statements, n_cells):
    """`frame` with `copy` (3.7 y + 12) and the target of `n_cells` middle training rows
    set to 999999: the copy keeps what the target was. Returns the table and the rows."""
    clean = q.screened(frame, statements)
    rows = _middle(frame, statements, q.training_rows(clean, frame), n_cells)
    out = q.with_column(frame, "copy", 3.7 * frame["y"].to_numpy(dtype=float) + 12.0)
    out.loc[rows, "y"] = WILD
    return out, rows


def _assert_copy_is_the_only_leak(result):
    assert q.leaks(result) == ["copy"], q.leaks(result)
    assert q.column(result, "copy")["leak_rule"] == "predicts_target"


def _assert_copy_is_kept_and_named(result):
    assert q.leaks(result) == [], q.leaks(result)
    assert q.is_feature(q.column(result, "copy")), q.column(result, "copy")
    assert len(q.told(result, "strong_single_column", "copy")) == 1, s.codes(result)


def _assert_copy_is_kept_and_named_or_excluded(result):
    """Where more rows are marked than may be set aside.

    D59, D77 point 5, D78 point 7: as D74 built it every row was then scored, and the copy
    was kept and named by the order reading. Since the ninth pass rule 2 still leaves its
    rows unscored and is also read against each fold's training median, and the copy is
    excluded, the only leak. Either is accepted here; tests/test_screen_marked.py pins the
    second.
    """
    if q.leaks(result):
        _assert_copy_is_the_only_leak(result)
    else:
        _assert_copy_is_kept_and_named(result)


@pytest.mark.parametrize("scheme", q.SCHEMES)
def test_one_wild_target_cell_does_not_hide_a_copy(scheme):
    """F25: one target cell of 999999 among the training rows keeps a copy a feature."""
    frame, statements = q.scheme_table(scheme, seed=8401)
    wild, _ = _with_wild_targets(frame, statements, 1)
    result = q.screened(wild, statements)
    _assert_copy_is_the_only_leak(result)
    assert q.is_feature(q.column(result, "n1"))


@pytest.mark.parametrize("scheme", q.SCHEMES)
def test_one_wild_target_cell_makes_no_column_a_leak(scheme):
    """M42: a row set aside leaves every fold's means as well as its sums."""
    frame, statements = q.scheme_table(scheme, seed=8411)
    clean = q.screened(frame, statements)
    rows = _middle(frame, statements, q.training_rows(clean, frame), 1)
    wild = frame.copy()
    wild.loc[rows, "y"] = WILD
    result = q.screened(wild, statements)
    assert q.leaks(result) == []
    assert not q.told(result, "strong_single_column")
    assert _roles(result) == _roles(clean)


def test_a_wild_target_cell_in_the_holdout_changes_nothing_in_the_screen():
    """The screen reads training rows only: a wild holdout target moves no role."""
    frame, statements = q.scheme_table("random", seed=8421)
    frame = q.with_column(frame, "copy", 3.7 * frame["y"].to_numpy(dtype=float) + 12.0)
    clean = q.screened(frame, statements)
    wild = frame.copy()
    wild.loc[h.holdout_rows(clean)[5], "y"] = WILD
    result = q.screened(wild, statements)
    _assert_copy_is_the_only_leak(result)
    assert _roles(result) == _roles(clean)


@pytest.mark.parametrize("n,cells,out", [(600, 4, True), (600, 5, False),
                                         (120, 1, True), (120, 2, False)])
def test_at_most_one_percent_of_the_training_rows_is_set_aside(n, cells, out):
    """F25 where the rows are set aside; where they are too many, M44 as D74 built it (all
    scored, the copy kept and named by the order reading) or the copy excluded as the
    ninth pass has it (D59)."""
    frame = h.reg_frame(n=n, seed=8431)
    wild, _ = _with_wild_targets(frame, s.SAID, cells)
    result = q.screened(wild, s.SAID)
    allowed = max(1, math.floor(0.01 * (n - n // 4)))
    assert (cells <= allowed) == out
    if out:
        _assert_copy_is_the_only_leak(result)
    else:
        _assert_copy_is_kept_and_named_or_excluded(result)


@pytest.mark.parametrize("cells,out", [(2, True), (3, False)])
def test_under_time_order_the_row_after_a_wild_cell_counts_too(cells, out):
    """Each wild cell marks its row and the row after it: two cells are four rows of the
    450 training rows, three are six."""
    frame = q.walk(n=600, seed=8441)
    statements = {"time_col": "t", "group_col": False}
    wild, rows = _with_wild_targets(frame, statements, cells)
    result = q.screened(wild, statements)
    for row in rows:
        assert 75 <= row < 450
    if out:
        _assert_copy_is_the_only_leak(result)
    else:
        _assert_copy_is_kept_and_named_or_excluded(result)


def test_a_long_tail_of_the_target_is_scored_as_it_is():
    """About 8% of the targets 500 times larger: no row is set aside, and an exact copy of
    the target is still excluded."""
    frame = h.reg_frame(n=600, seed=8451).rename(columns={"z1": "n1"})
    tail = np.random.default_rng(8451).random(600) < 0.08
    frame.loc[tail, "y"] = frame.loc[tail, "y"] * 500.0
    frame = q.with_column(frame, "copy", 3.7 * frame["y"].to_numpy(dtype=float) + 12.0)
    result = q.screened(frame, s.SAID)
    training = q.training_rows(result, frame)
    y = frame["y"].to_numpy(dtype=float)[training]
    centre = float(np.median(y))
    spread = float(np.median(np.abs(y - centre)))
    far = int(np.sum(np.abs(y - centre) > q.SCREEN_FAR * spread))
    assert far > math.floor(q.SCREEN_SET_ASIDE * len(training)), far
    _assert_copy_is_the_only_leak(result)
