"""T37: the fast profile's cap leaving too few rows (spec 2.4, step 6, 11; D52, D53).

Red today, every test. Errors are compared by `code` and by exact class:
`pytest.raises` of a parent class alone would accept the wrong one.
"""
import numpy as np
import pandas as pd

import cp3_helpers as c
import cp5_helpers as p
import mlp_helpers as h

CODE = "fast_profile_too_few_rows"


def _timed(counts, seed=0):
    """One numeric timestamp `t` per entry of `counts`, that many rows each, in order."""
    gen = np.random.default_rng(seed)
    t = np.repeat(np.arange(len(counts), dtype=float), counts)
    n = len(t)
    x1, x2 = gen.normal(size=n), gen.normal(size=n)
    return pd.DataFrame({"t": t, "x1": x1, "x2": x2, "y": x1 + gen.normal(size=n)})


def _rare(n, k, seed=0):
    """`n` rows, two normal features, and exactly `k` rows of class 1 at random positions."""
    gen = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[gen.choice(n, size=k, replace=False)] = 1
    return pd.DataFrame({"x1": gen.normal(size=n), "x2": gen.normal(size=n), "y": y})


def _flat(frame):
    return p.flat_family(h.Log(), len(frame))


def test_error_class_and_code():
    from mlprofessor import errors

    assert issubclass(errors.FastProfileRowsError, errors.ProfileError)
    assert errors.FastProfileRowsError.code == CODE
    assert errors.FastProfileError.code == "fast_profile_too_few_timestamps"


def _assert_fast_rows_error(err):
    from mlprofessor import errors

    assert type(err) is errors.FastProfileRowsError, f"{type(err).__name__}: {err}"
    assert err.code == CODE
    assert "full" in str(err), str(err)


def test_a_small_latest_timestamp_behind_a_large_one():
    """Finding 9: the cap keeps 60 rows; the full profile would run."""
    frame = _timed([100] * 30 + [2500, 60], seed=71)
    err = p.raised(frame, time_col="t", families=_flat(frame))
    _assert_fast_rows_error(err)
    assert "2000" in str(err) or "2,000" in str(err), str(err)

    result = c.run(frame, families=_flat(frame), time_col="t", profile="full")
    rows = h.rows_of(result)
    assert (rows["training"], rows["holdout"], rows["dropped_row_cap"]) == (3000, 2560, 0)


def test_too_few_of_a_class_in_the_sample():
    frame = _rare(40_000, 20, seed=72)
    for seed in (0, 1, 2):
        _assert_fast_rows_error(p.raised(frame, families=_flat(frame), seed=seed))
    result = c.run(frame, families=_flat(frame), profile="full")
    assert h.rows_of(result)["holdout"] == 10_000


def test_the_full_profiles_error_when_full_would_stop_too():
    """One test, five tables: the fast profile never sends the caller to a full profile that stops too."""
    stops_too = [
        ("gap", _timed([100] * 7 + [2500, 60], seed=73), {"time_col": "t", "gap": 3}, "gap_too_large"),
        ("three_stamps", _timed([3000, 2500, 60], seed=74), {"time_col": "t"}, "too_few_rows"),
        ("eight_of_a_class", _rare(40_000, 8, seed=75), {}, "too_few_rows"),
    ]
    for name, frame, kw, code in stops_too:
        full = p.raised(frame, families=_flat(frame), profile="full", **kw)
        assert full.code == code, f"{name}, full: {type(full).__name__}: {full}"
        fast = p.raised(frame, families=_flat(frame), **kw)
        assert type(fast) is type(full), f"{name}: fast {type(fast).__name__}: {fast}"
        assert fast.code == code, name

    untouched = [
        ("six_of_a_class", _rare(1500, 6, seed=76), {}),
        ("nine_stamps_of_ten", _timed([10] * 9, seed=77), {"time_col": "t"}),
    ]
    for name, frame, kw in untouched:
        fast = p.raised(frame, families=_flat(frame), **kw)
        assert fast.code == "too_few_rows", f"{name}: {type(fast).__name__}: {fast}"
