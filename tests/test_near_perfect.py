"""T70: the near-perfect warning (spec step 12a, 2.5, 4.3, 5.3 rule 20, 11; D77 point 2).

A headline skill of 0.98 or more, the level at which one column alone is excluded as a
suspected leak, carries exactly one `near_perfect_headline` warning with a null column.
Its message holds the skill as a number and names each column of `headline.rests_on`
in quotes as `repr` writes it; with an empty list it is raised all the same. In both
tasks and under every scheme. It changes nothing.
"""
import numpy as np
import pandas as pd
import pytest

import cp10_helpers as t
import cp7_helpers as s


def _assert_named(result, names):
    skill = result.headline["skill"]
    assert skill is not None and skill >= t.NEAR_PERFECT, skill
    found = t.warning_records(result, t.NEAR)
    assert len(found) == 1, s.codes(result)
    assert found[0]["column"] is None
    message = str(found[0]["message"])
    assert s.holds_number(message, skill, within=0.0005 * abs(skill) + 0.0005), message
    for name in names:
        assert repr(name) in message, (name, message)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_two_columns_that_give_the_target_together_are_named(task):
    """F30, M57: two columns that give the target only together are named by the warning."""
    frame = t.pair_table(task, n=400, seed=9201)
    result = t.run(frame)
    assert t.leaks(result) == []
    assert set(t.listed(result)) >= {"pair_a", "pair_b"}, t.rests_on(result)
    _assert_named(result, ["pair_a", "pair_b"])


def test_under_time_order_the_warning_is_raised_too():
    """M57: a regression under time order, against the better of its baselines."""
    frame = t.pair_table("regression", n=400, seed=9202)
    frame.insert(0, "t", np.arange(len(frame), dtype=np.int64))
    result = t.run(frame, {"time_col": "t", "group_col": False})
    assert t.doc(result)["validation"]["scheme"] == "time_ordered"
    _assert_named(result, ["pair_a", "pair_b"])


def test_under_the_grouped_scheme_the_warning_is_raised_too():
    """A classification of whole entities carries the warning too."""
    frame = t.pair_table("classification", n=400, seed=9203)
    frame.insert(0, "who", t.entities(80, 5, seed=9203))
    result = t.run(frame, {"time_col": False, "group_col": "who"})
    assert t.doc(result)["validation"]["scheme"] == "grouped"
    _assert_named(result, ["pair_a", "pair_b"])


def test_no_warning_below_the_level():
    """A headline of about 0.7: no near-perfect warning."""
    rng = np.random.default_rng(9204)
    n = 400
    x1 = rng.normal(size=n)
    frame = pd.DataFrame({"x1": x1, "z1": rng.normal(size=n),
                          "y": x1 + 0.6 * rng.normal(size=n)})
    result = t.run(frame)
    assert 0.5 < result.headline["skill"] < 0.9, result.headline["skill"]
    assert t.NEAR not in s.codes(result)


def test_a_near_perfect_headline_that_rests_on_no_one_column_is_named_too():
    """Sixty columns of one weight each: no fall reaches the floor, and the warning is raised."""
    rng = np.random.default_rng(9205)
    n = 1200
    frame = pd.DataFrame({f"w{k:02d}": rng.normal(size=n) for k in range(60)})
    frame["y"] = frame.sum(axis=1)
    result = t.run(frame)
    assert t.rests_on(result) == []
    assert all(t.fall(result, f"w{k:02d}") < t.floor(result) for k in range(60))
    _assert_named(result, [])
