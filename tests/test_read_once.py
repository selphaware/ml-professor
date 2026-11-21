"""T40: each object column is read once, over all loaded rows (spec step 1a,
"Each column is read once"; step 3; D57; the fourth review, finding 1). Red today.

A cell that decides what an object column is must be seen whether or not its
row is dropped for another column's empty cell. The first four tests fail on
M14.
"""
import numpy as np
import pandas as pd

import cp6_helpers as g
import mlp_helpers as h

N = 240
ODD_ROW = 37


def _rec():
    return [h.family("rec", h.Log())]


def _base(seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "t": np.arange(N, dtype=float),
        "x1": rng.normal(size=N),
        "x2": rng.normal(size=N),
    }), rng


def _whole_number_target(seed):
    frame, rng = _base(seed)
    frame["y"] = pd.Series([int(v) for v in rng.integers(0, 60, size=N)], dtype=object)
    return frame


def test_a_text_cell_of_an_object_target_in_a_row_with_no_time_value():
    """M14: a text cell of an object target, in a row dropped for its time value."""
    frame = _whole_number_target(501)
    frame["y"] = frame["y"].astype(object)
    frame.iloc[ODD_ROW, frame.columns.get_loc("y")] = "odd"
    usable_code = g.raised(frame, families=_rec(), time_col="t").code

    dropped = frame.copy()
    dropped.loc[ODD_ROW, "t"] = np.nan
    err = g.raised(dropped, families=_rec(), time_col="t")
    assert err.code == usable_code, f"{type(err).__name__}: {err}"
    assert err.code == "task_ambiguous", f"{type(err).__name__}: {err}"


def test_a_whole_number_beyond_64_bits_in_a_target_row_with_no_time_value():
    """M14: a whole number too large for 64 bits, in a row dropped for its time value."""
    frame = _whole_number_target(502)
    frame.iloc[ODD_ROW, frame.columns.get_loc("y")] = 10 ** 25
    frame.loc[ODD_ROW, "t"] = np.nan
    err = g.raised(frame, families=_rec(), time_col="t")
    assert err.code == "task_unsupported", f"{type(err).__name__}: {err}"


def _object_time_column(seed):
    frame, rng = _base(seed)
    frame["t"] = pd.Series([int(k) for k in range(N)], dtype=object)
    frame["y"] = 2.0 * frame["x1"] + rng.normal(size=N)
    return frame


def test_a_text_cell_of_an_object_time_column_in_a_row_with_no_target():
    """M14: a text cell of an object time column, in a row dropped for its target."""
    frame = _object_time_column(503)
    frame.iloc[ODD_ROW, frame.columns.get_loc("t")] = "noon"
    usable = g.raised(frame, families=_rec(), time_col="t")
    assert usable.code == "time_column_invalid", f"{type(usable).__name__}: {usable}"

    dropped = frame.copy()
    dropped.loc[ODD_ROW, "y"] = np.nan
    err = g.raised(dropped, families=_rec(), time_col="t")
    assert err.code == usable.code, f"{type(err).__name__}: {err}"


def test_a_time_value_beyond_64_bits_in_a_row_with_no_target():
    """M14: a time value too large for 64 bits, in a row dropped for its target."""
    frame = _object_time_column(504)
    frame.iloc[ODD_ROW, frame.columns.get_loc("t")] = 10 ** 25
    frame.loc[ODD_ROW, "y"] = np.nan
    err = g.raised(frame, families=_rec(), time_col="t")
    assert err.code == "time_column_invalid", f"{type(err).__name__}: {err}"


def test_exact_integer_keys_are_decided_on_the_loaded_column():
    frame, rng = _base(505)
    stamps = np.arange(N, dtype=np.int64) * 100
    stamps[7] = 2 ** 53 + 1
    frame["t"] = stamps
    frame["y"] = 2.0 * frame["x1"] + rng.normal(size=N)
    frame.loc[7, "y"] = np.nan
    assert frame["t"].dtype == np.int64
    result = g.run(frame, families=_rec(), time_col="t")
    v = result.to_dict()["validation"]
    assert v["time_form"] == "numeric"
    for key in ("train_end", "holdout_start", "holdout_end"):
        value = v[key]
        assert isinstance(value, int) and not isinstance(value, bool), (key, value)
    assert (v["train_end"], v["holdout_start"]) == (17900, 18000), v
    assert v["holdout_end"] == 23900
