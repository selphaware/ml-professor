"""T6: `stages.read_time` and `stages.split_time` (spec steps 3 and 7, D7, D18)."""
import numpy as np
import pandas as pd
import pytest


def _arr(a):
    return np.asarray(a, dtype=int)


def _rows_in(keys, stamps):
    return np.flatnonzero(np.isin(keys, stamps))


def expected_split(keys, gap=0):
    """The split of step 7, worked out from the keys alone."""
    keys = np.asarray(keys)
    stamps, counts = np.unique(keys, return_counts=True)
    n, held, k = len(keys), 0, 0
    while 4 * held < n:
        k += 1
        held += counts[-k]
    hold_ts, rest = stamps[-k:], stamps[:-k]
    train_ts = rest[:len(rest) - gap]
    gap_ts = rest[len(rest) - gap:] if gap else rest[:0]
    blocks = np.array_split(train_ts, 6)
    folds = []
    for j in range(5):
        before = np.concatenate(blocks[:j + 1])
        folds.append({
            "training": _rows_in(keys, before[:len(before) - gap]),
            "gap": _rows_in(keys, before[len(before) - gap:] if gap else before[:0]),
            "validation": _rows_in(keys, blocks[j + 1]),
        })
    return {"training": _rows_in(keys, train_ts), "gap": _rows_in(keys, gap_ts),
            "holdout": _rows_in(keys, hold_ts), "folds": folds}


def _same(actual, expected):
    for part in ["training", "gap", "holdout"]:
        assert _arr(actual[part]).tolist() == _arr(expected[part]).tolist(), part
    assert len(actual["folds"]) == 5
    for k, (fa, fe) in enumerate(zip(actual["folds"], expected["folds"])):
        for part in ["training", "gap", "validation"]:
            assert _arr(fa[part]).tolist() == _arr(fe[part]).tolist(), (k + 1, part)


def _unique_keys(n=200, seed=0):
    return np.random.default_rng(seed).permutation(np.arange(n, dtype=float))


def _panel_keys(n_stamps=50, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 6, size=n_stamps)
    keys = np.repeat(np.arange(n_stamps, dtype=float) * 10.0, counts)
    return rng.permutation(keys)


# ---------------------------------------------------------------- read_time

def test_read_time_datetime_column():
    from mlprofessor import stages

    when = pd.Series(pd.date_range("2024-01-01", periods=5, freq="D")).iloc[[3, 0, 4, 1, 2]]
    when = when.reset_index(drop=True)
    when.iloc[2] = pd.NaT
    keys, form = stages.read_time(when)
    assert form == "datetime"
    assert keys.dtype == np.dtype("datetime64[ns]")
    assert np.isnat(keys[2])
    ok = [0, 1, 3, 4]
    assert np.argsort(keys[ok]).tolist() == np.argsort(when.iloc[ok].to_numpy()).tolist()


def test_read_time_datetime_with_zone_is_utc():
    from mlprofessor import stages

    when = pd.Series(pd.date_range("2024-03-30 22:00", periods=6, freq="h", tz="Europe/London"))
    keys, form = stages.read_time(when)
    assert form == "datetime"
    expected = when.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
    assert (keys == expected).all()


def test_read_time_numeric_column():
    from mlprofessor import stages

    values = pd.Series([3.0, 1.0, np.nan, 2.5, np.inf, 7.0])
    keys, form = stages.read_time(values)
    assert form == "numeric"
    assert keys.dtype == np.float64
    assert np.isnan(keys[2]) and np.isnan(keys[4])
    assert keys[[0, 1, 3, 5]].tolist() == [3.0, 1.0, 2.5, 7.0]
    keys, form = stages.read_time(pd.Series([5, 3, 9]))
    assert form == "numeric" and keys.dtype == np.float64 and keys.tolist() == [5.0, 3.0, 9.0]


@pytest.mark.parametrize("values,expected", [
    (["2024-01-05", "2024-01-03", None, "2024-02-01"],
     ["2024-01-05", "2024-01-03", None, "2024-02-01"]),
    (["2024-01-05 10:30", "2024-01-05T09:15:00", "2024-01-05 09:15:00.250", "2023-12-31 23:59"],
     ["2024-01-05T10:30", "2024-01-05T09:15:00", "2024-01-05T09:15:00.250", "2023-12-31T23:59"]),
    (["2024-01-05T10:00:00Z", "2024-01-05T09:00:00Z"],
     ["2024-01-05T10:00:00", "2024-01-05T09:00:00"]),
    (["2024-01-05T10:00:00+01:00", "2024-01-05T08:30:00-01:00"],
     ["2024-01-05T09:00:00", "2024-01-05T09:30:00"]),
])
def test_read_time_iso_text(values, expected):
    from mlprofessor import stages

    keys, form = stages.read_time(pd.Series(values, dtype="str"))
    assert form == "iso8601"
    assert keys.dtype == np.dtype("datetime64[ns]")
    want = np.array([np.datetime64("NaT") if e is None else np.datetime64(e) for e in expected],
                    dtype="datetime64[ns]")
    for got, exp in zip(keys, want):
        if np.isnat(exp):
            assert np.isnat(got)
        else:
            assert got == exp


@pytest.mark.parametrize("values", [
    ["25/01/2024", "26/01/2024", "27/01/2024"],
    ["01/12/2023", "02/12/2023", "03/12/2023"],
    ["5-Jan-2024", "6-Jan-2024"],
    ["2024-01", "2024-02"],
    [True, False, True],
    ["2024-02-30", "2024-03-01"],
    ["2024-01-05T10:00:00+01:00", "2024-01-06T10:00:00"],
    [1, "a", 2.5],
], ids=["day_first", "month_first", "month_name", "year_month", "boolean",
        "impossible", "mixed_offsets", "mixed_objects"])
def test_f7_read_time_refuses_other_forms(values):
    """F7: a time column that could be misread stops the run."""
    from mlprofessor import errors, stages

    if values and isinstance(values[0], bool):
        series = pd.Series(values, dtype=bool)
    elif all(isinstance(v, str) for v in values):
        series = pd.Series(values, dtype="str")
    else:
        series = pd.Series(values, dtype=object)
    with pytest.raises(errors.TimeColumnError):
        stages.read_time(series)


# --------------------------------------------------------------- split_time

def test_f7_every_training_key_precedes_every_holdout_key():
    """F7: no training row is at or after the holdout."""
    from mlprofessor import stages

    keys = _unique_keys(200)
    s = stages.split_time(keys)
    assert keys[_arr(s["training"])].max() < keys[_arr(s["holdout"])].min()
    for fold in s["folds"]:
        assert keys[_arr(fold["training"])].max() < keys[_arr(fold["validation"])].min()


def test_no_timestamp_straddles_the_cut():
    from mlprofessor import stages

    keys = _panel_keys(50, seed=1)
    s = stages.split_time(keys)
    hold = set(keys[_arr(s["holdout"])].tolist())
    assert not hold & set(keys[_arr(s["training"])].tolist())
    assert len(s["training"]) + len(s["holdout"]) + len(s["gap"]) == len(keys)


@pytest.mark.parametrize("seed", [2, 3])
def test_holdout_is_smallest_set_of_latest_timestamps(seed):
    from mlprofessor import stages

    keys = _panel_keys(50, seed=seed)
    s = stages.split_time(keys)
    exp = expected_split(keys)
    assert _arr(s["holdout"]).tolist() == exp["holdout"].tolist()
    assert 4 * len(s["holdout"]) >= len(keys)


@pytest.mark.parametrize("keys_kind", ["unique", "panel"])
def test_f7_split_depends_on_keys_not_their_order(keys_kind):
    """F7: shuffling the rows gives the same rows in every part."""
    from mlprofessor import stages

    keys = _unique_keys(150, 4) if keys_kind == "unique" else _panel_keys(45, 4)
    perm = np.random.default_rng(9).permutation(len(keys))
    a = stages.split_time(keys, gap=1)
    b = stages.split_time(keys[perm], gap=1)
    for part in ["training", "gap", "holdout"]:
        assert sorted(perm[_arr(b[part])].tolist()) == _arr(a[part]).tolist(), part
    for fa, fb in zip(a["folds"], b["folds"]):
        for part in ["training", "gap", "validation"]:
            assert sorted(perm[_arr(fb[part])].tolist()) == _arr(fa[part]).tolist(), part


@pytest.mark.parametrize("keys_kind", ["unique", "panel"])
def test_six_blocks_expanding_folds(keys_kind):
    from mlprofessor import stages

    keys = _unique_keys(130, 5) if keys_kind == "unique" else _panel_keys(61, 5)
    s = stages.split_time(keys)
    _same(s, expected_split(keys))
    for fold in s["folds"]:
        assert len(fold["gap"]) == 0
        assert keys[_arr(fold["training"])].max() < keys[_arr(fold["validation"])].min()


@pytest.mark.parametrize("gap", [2, 3])
def test_f7_gap_removed_before_holdout_and_every_block(gap):
    """F7: the gap's timestamps are in no training part."""
    from mlprofessor import stages

    keys = _panel_keys(70, seed=6)
    s = stages.split_time(keys, gap=gap)
    exp = expected_split(keys, gap)
    _same(s, exp)
    stamps = np.unique(keys)
    hold_start = keys[_arr(s["holdout"])].min()
    gap_stamps = sorted(set(keys[_arr(s["gap"])].tolist()))
    assert gap_stamps == stamps[stamps < hold_start][-gap:].tolist()
    train = set(_arr(s["training"]).tolist())
    assert not set(_arr(s["gap"]).tolist()) & train
    for fold in s["folds"]:
        fold_gap = set(_arr(fold["gap"]).tolist())
        assert fold_gap, "each fold has gap rows"
        assert not fold_gap & set(_arr(fold["training"]).tolist())
        assert fold_gap <= train
        val_start = keys[_arr(fold["validation"])].min()
        assert sorted(set(keys[list(fold_gap)].tolist())) == stamps[stamps < val_start][-gap:].tolist()


def test_split_of_datetime_keys_from_read_time():
    from mlprofessor import stages

    when = pd.Series(pd.date_range("2021-06-01", periods=120, freq="D"))
    when = when.iloc[np.random.default_rng(7).permutation(120)].reset_index(drop=True)
    keys, _ = stages.read_time(when)
    _same(stages.split_time(keys, gap=1), expected_split(keys, 1))


def test_too_few_training_timestamps():
    from mlprofessor import errors, stages

    keys = np.repeat(np.arange(7, dtype=float), 20)
    with pytest.raises(errors.TooFewRowsError):
        stages.split_time(keys)


def test_gap_leaving_too_few_training_timestamps():
    from mlprofessor import errors, stages

    with pytest.raises(errors.GapError):
        stages.split_time(np.arange(20, dtype=float), gap=10)


def test_gap_leaving_no_rows_before_a_block():
    from mlprofessor import errors, stages

    # 16 timestamps: holdout 4, gap 2, 10 training in blocks 2,2,2,2,1,1;
    # fold 1 trains on block 1 less the 2 gap timestamps: nothing.
    with pytest.raises(errors.GapError):
        stages.split_time(np.arange(16, dtype=float), gap=2)
