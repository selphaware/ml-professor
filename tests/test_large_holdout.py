"""T74: a time-ordered holdout far above a quarter is named (spec step 7, 2.5, 4.3, 5.3
rule 21, 5.4, 11; D77 point 6, D78 point 7).

The holdout is the latest whole timestamps that make a quarter, so a last date that holds
many rows can make it far more. When 2 x `rows.holdout` >= `rows.training` +
`rows.dropped_gap` + `rows.holdout`, one `holdout_far_above_quarter` warning with a null
column says so; its message holds `H of the N rows` and the fixed passage of 5.4.
"""
import pytest

import cp10_helpers as t
import cp7_helpers as s
import mlp_helpers as h

TIME = {"time_col": "date", "group_col": False}


def _counts(result):
    return dict(h.rows_of(result))


def _assert_named(result, held, split):
    rows = _counts(result)
    assert rows["holdout"] == held
    assert rows["training"] + rows["dropped_gap"] + rows["holdout"] == split
    found = t.warning_records(result, t.FAR)
    assert len(found) == 1, s.codes(result)
    assert found[0]["column"] is None
    message = str(found[0]["message"])
    assert f"{held} of the {split} rows" in message, message
    assert t.FAR_ABOVE_PASSAGE in message, message


@pytest.mark.parametrize("share", [0.6, 0.5])
def test_a_holdout_of_half_the_rows_or_more_is_named(share):
    """F35, M64: 1,000 rows, that share on the last date: named, `600 of the 1000 rows`."""
    result = t.run(t.last_date_table(n=1000, share=share, seed=9701), TIME)
    held = int(round(share * 1000))
    assert 2 * held >= 1000
    _assert_named(result, held, 1000)


def test_the_rows_split_count_the_gap_rows():
    """With gap=1 the holdout still holds 500 of the 1000 rows split: named."""
    result = t.run(t.last_date_table(n=1000, share=0.5, seed=9702), TIME, gap=1)
    rows = _counts(result)
    assert rows["dropped_gap"] == 1
    _assert_named(result, 500, 1000)


def test_below_half_there_is_no_warning():
    """45% on the last date: no warning."""
    result = t.run(t.last_date_table(n=1000, share=0.45, seed=9703), TIME)
    assert _counts(result)["holdout"] == 450
    assert t.FAR not in s.codes(result)


def test_an_ordinary_time_split_has_no_warning():
    """One row a timestamp: no warning."""
    result = t.run(h.time_frame(n=400, seed=9704), TIME)
    assert t.FAR not in s.codes(result)
