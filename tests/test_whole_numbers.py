"""T33: whole numbers are compared exactly (spec steps 1a, 3, 4, 5; 2.4; 2.7; section 6; D50, D53).

Red today, every test. Typed columns only, and object columns only where they
hold whole numbers too large for 64 bits. The stamps `START + 100 k` merge as
floats, which each test asserts so that it says why it exists.
"""
import numpy as np
import pandas as pd
import pytest

import cp3_helpers as c
import cp5_helpers as p
import mlp_helpers as h

START = p.START
STAMPS = [START + 100 * k for k in range(600)]
UNSIGNED_SHIFT = 2 ** 63
TYPES = ["int64", "uint64", "Int64", "UInt64"]


def _stamps_for(dtype):
    shift = UNSIGNED_SHIFT if dtype.lower() == "uint64" else 0
    return [s + shift for s in STAMPS]


def _column(values, dtype):
    if dtype in ("Int64", "UInt64"):
        return pd.array(values, dtype=dtype)
    return np.array(values, dtype=dtype)


@pytest.mark.parametrize("dtype", TYPES)
def test_read_time_keeps_integers_above_2_53(dtype):
    from mlprofessor import stages

    values = _stamps_for(dtype)
    assert len({float(v) for v in values}) < 600
    keys, form = stages.read_time(pd.Series(_column(values, dtype)))
    assert form == "numeric"
    assert keys.dtype == np.dtype("uint64" if dtype.lower() == "uint64" else "int64"), keys.dtype
    assert [int(k) for k in keys] == values
    assert len(set(keys.tolist())) == 600

    split = stages.split_time(np.ascontiguousarray(keys[::-1]))
    assert list(split["holdout"]) == list(range(150))
    assert list(split["training"]) == list(range(150, 600))
    assert len(split["folds"]) == 5
    for fold in split["folds"]:
        assert len(fold["validation"]) == 75

    previous = stages.previous_value(np.arange(600.0), keys)
    assert np.isnan(previous[0])
    assert list(previous[1:]) == [float(v) for v in range(599)]


def test_the_boundary_is_a_magnitude_above_2_53():
    """D53: integer keys only where some magnitude is above 2**53; float keys otherwise."""
    from mlprofessor import stages

    float_cases = [
        (pd.Series([5, 3, 9], dtype="int64"), [5.0, 3.0, 9.0]),
        (pd.Series([2 ** 53, 0, -2 ** 53], dtype="int64"), [2.0 ** 53, 0.0, -2.0 ** 53]),
        (pd.Series(pd.array([7, None, 2 ** 53], dtype="Int64")), [7.0, None, 2.0 ** 53]),
        (pd.Series([1, 2], dtype="uint8"), [1.0, 2.0]),
        (pd.Series([2.0 ** 60, 1.0, 3.0], dtype="float64"), [2.0 ** 60, 1.0, 3.0]),
    ]
    for values, expected in float_cases:
        keys, form = stages.read_time(values)
        assert form == "numeric", values.tolist()
        assert keys.dtype == np.float64, (values.tolist(), keys.dtype)
        for key, want in zip(keys.tolist(), expected):
            if want is None:
                assert np.isnan(key)
            else:
                assert key == want

    for values in ([2 ** 53 + 1, 0, 5], [-2 ** 53 - 1, 0, 5]):
        keys, form = stages.read_time(pd.Series(values, dtype="int64"))
        assert form == "numeric"
        assert keys.dtype == np.int64, (values, keys.dtype)
        assert [int(k) for k in keys] == values


def _walk_in(dtype):
    frame = p.nanosecond_walk(600)
    stamps = [int(v) for v in frame["t"]]
    if dtype.lower() == "uint64":
        stamps = [s + UNSIGNED_SHIFT for s in stamps]
    frame["t"] = _column(stamps, dtype)
    return frame, stamps


@pytest.mark.parametrize("dtype", TYPES)
def test_nanosecond_stamps_are_distinct_timestamps(dtype):
    """F7c: integer time keys above 2**53 stay distinct, so the previous-value baseline stands."""
    frame, stamps = _walk_in(dtype)
    assert len({float(s) for s in stamps}) < 600
    result = c.run(frame, "level", time_col="t")
    doc = result.to_dict()
    validation = doc["validation"]
    assert validation["scheme"] == "time_ordered"
    assert validation["time_form"] == "numeric"
    assert validation["n_train_timestamps"] == 450
    assert validation["n_holdout_timestamps"] == 150
    assert validation["timestamps_unique"] is True
    ordered = sorted(stamps)
    for key, want in (("train_end", ordered[449]), ("holdout_start", ordered[450]),
                      ("holdout_end", ordered[599])):
        assert type(validation[key]) is int, (key, validation[key])
        assert validation[key] == want, key
    assert [b["name"] for b in doc["baselines"]] == ["training_mean", "previous_value"]
    assert result.headline["baseline"] == "previous_value"
    assert c.count_code(result, "suspected_leak") == 0
    assert h.columns_of(result)["level_prev"]["role"] == "feature"

    twin = p.nanosecond_walk(600)
    twin["t"] = pd.to_datetime(twin["t"])
    other = c.run(twin, "level", time_col="t")
    assert h.holdout_rows(other) == h.holdout_rows(result)
    assert other.headline["score"] == result.headline["score"]
    assert other.headline["baseline_score"] == result.headline["baseline_score"]
    assert other.to_dict()["baselines"] == doc["baselines"]


def test_missing_and_repeated_integer_stamps():
    frame = p.nanosecond_walk(600)
    stamps = pd.array(frame["t"].tolist(), dtype="Int64")
    for row in (10, 200, 400):
        stamps[row] = pd.NA
    frame["t"] = stamps
    result = c.run(frame, "level", time_col="t")
    validation = result.to_dict()["validation"]
    assert h.rows_of(result)["dropped_missing_time"] == 3
    assert validation["n_train_timestamps"] + validation["n_holdout_timestamps"] == 597
    assert validation["timestamps_unique"] is True

    frame = p.nanosecond_walk(600)
    frame["t"] = np.array([START + k // 2 for k in range(600)], dtype=np.int64)
    result = c.run(frame, "level", time_col="t")
    doc = result.to_dict()
    validation = doc["validation"]
    assert validation["n_train_timestamps"] + validation["n_holdout_timestamps"] == 300
    assert validation["timestamps_unique"] is False
    assert [b["name"] for b in doc["baselines"]] in (
        ["training_mean"], ["training_mean", "previous_value"])


def _exact_columns():
    gen = np.random.default_rng(93)
    n = 200
    pick = gen.integers(0, 50, size=n)
    wide = np.array([2 ** 63 + 7 * int(k) for k in pick], dtype=np.uint64)
    pair = np.array([2 ** 53 + int(b) for b in gen.integers(0, 2, size=n)], dtype=np.int64)
    serial = [2 ** 53 + 1 + k for k in range(n)]
    shuffled = [serial[int(i)] for i in gen.permutation(n)]
    rising = np.cumsum(gen.integers(1, 5, size=n)).astype(object) + 2 ** 62
    scattered = [2 ** 61 + 3 * int(k) for k in gen.choice(1000, size=n, replace=False)]
    ends = [np.iinfo(np.int64).min if k % 2 else np.iinfo(np.int64).max for k in range(n)]
    draws = gen.integers(0, 10, size=n)
    nullable = pd.array([None if d == 0 else 2 ** 53 + int(d) for d in draws], dtype="Int64")
    frame = pd.DataFrame({
        "x1": gen.normal(size=n),
        "wide": wide,
        "pair": pair,
        "same": np.full(n, 2 ** 60 + 1, dtype=np.int64),
        "serial": np.array(serial, dtype=np.int64),
        "shuffled_serial": np.array(shuffled, dtype=np.int64),
        "rising": np.array([int(v) for v in rising], dtype=np.int64),
        "scattered": np.array(scattered, dtype=np.int64),
        "ends": np.array(ends, dtype=np.int64),
        "nullable": nullable,
        "y": gen.integers(0, 2, size=n),
    })
    return frame


EXPECTED_ROLES = {
    "x1": ("feature", None),
    "wide": ("feature", None),
    "pair": ("feature", None),
    "same": ("excluded", "constant"),
    "serial": ("excluded", "id_like"),
    "shuffled_serial": ("excluded", "id_like"),
    "rising": ("excluded", "id_like"),
    "scattered": ("feature", None),
    "ends": ("feature", None),
    "nullable": ("feature", None),
}


def test_distinct_values_are_counted_exactly():
    """Spec step 5, rules 8 to 8b: distinct whole numbers are counted as integers, at any size."""
    from mlprofessor import stages

    frame = _exact_columns()
    assert len({float(v) for v in frame["wide"]}) == 1
    assert len({int(v) for v in frame["wide"]}) == 50
    assert len({float(v) for v in frame["pair"]}) == 1
    assert len({int(v) for v in frame["pair"]}) == 2

    roles = {r["name"]: r for r in h.records(stages.column_roles(frame, "y"))}
    result = c.run(frame)
    cols = h.columns_of(result)
    for name, (role, reason) in EXPECTED_ROLES.items():
        assert (roles[name]["role"], roles[name]["reason"]) == (role, reason), (name, roles[name])
        assert (cols[name]["role"], cols[name]["reason"]) == (role, reason), (name, cols[name])
    c.assert_strict_json(result)


SHAPES = {
    "consecutive_26_digits": [10 ** 25 + k for k in range(200)],
    "one_at_2_64": list(range(199)) + [2 ** 64],
    "both_signs_beyond_int64": [(-1) ** k * (2 ** 63 + k) for k in range(200)],
    "one_below_int64": [-2 ** 63 - 1] + list(range(199)),
}
BEYOND_CASES = [(shape, "frame") for shape in SHAPES] + [
    ("consecutive_26_digits", "csv"), ("one_at_2_64", "csv"), ("one_below_int64", "csv")]


def _data(frame, source, tmp_path, name):
    if source == "frame":
        return frame
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("shape,source", BEYOND_CASES, ids=[f"{s}-{k}" for s, k in BEYOND_CASES])
def test_whole_numbers_beyond_64_bits_in_every_role(shape, source, tmp_path):
    """Finding 1: whole numbers too large for 64 bits end in a result or a named error, in every role."""
    from mlprofessor import errors

    big = pd.Series(SHAPES[shape], dtype=object)
    gen = np.random.default_rng(94)
    x1, z1, noise = gen.normal(size=200), gen.normal(size=200), gen.normal(size=200)

    spare = pd.DataFrame({"x1": x1, "z1": z1, "big": big, "y": gen.integers(0, 2, size=200)})
    result = c.run(_data(spare, source, tmp_path, "spare"))
    record = h.columns_of(result)["big"]
    assert (record["role"], record["reason"]) == ("excluded", "unsupported_type"), record
    assert h.warnings_of(result).count(("unsupported_type", "big")) == 1
    c.assert_strict_json(result)

    target = pd.DataFrame({"x1": x1, "z1": z1, "y": big})
    for task in ("auto", "classification", "regression"):
        err = p.raised(_data(target, source, tmp_path, f"target_{task}"), task=task)
        assert isinstance(err, errors.TaskUnsupportedError), f"{task}: {type(err).__name__}: {err}"

    timed = pd.DataFrame({"t": big, "x1": x1, "z1": z1, "y": 2.0 * x1 + noise})
    err = p.raised(_data(timed, source, tmp_path, "time"), time_col="t")
    assert isinstance(err, errors.TimeColumnError), f"{type(err).__name__}: {err}"


def test_command_line_on_a_26_digit_column(tmp_path):
    frame = h.clf_frame(n=200, n_noise=1)
    frame.insert(0, "account", pd.Series([10 ** 25 + 3 * k for k in range(200)], dtype=object))
    path = tmp_path / "accounts.csv"
    frame.to_csv(path, index=False)
    common = ["--data", path, "--profile", "fast", "--n-jobs", "1"]

    proc = h.run_cli(["profile", *common, "--target", "y", "--out", tmp_path / "spare"])
    assert proc.returncode == 0, proc.stderr
    assert any("unsupported_type" in line and "account" in line
               for line in proc.stdout.splitlines()), proc.stdout

    proc = h.run_cli(["profile", *common, "--target", "account", "--out", tmp_path / "target"])
    assert proc.returncode == 3, proc.stderr
    lines = proc.stderr.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("error: task_unsupported: "), proc.stderr

    proc = h.run_cli(["profile", *common, "--target", "y", "--time-col", "account",
                      "--out", tmp_path / "time"])
    assert proc.returncode == 3, proc.stderr
    lines = proc.stderr.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("error: time_column_invalid: "), proc.stderr
