"""T34: number columns held as objects, as a database driver returns them (spec step 1a, 2.5, 2.6, 2.7, 5.1, 5.2, section 6; D50, D51).

Red today, every test. Each typed twin is built from the same text as its
object column (`D(text)` beside `float(text)`), never by rounding a float, so
that the two agree exactly.
"""
import json
import math
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

import cp3_helpers as c
import cp5_helpers as p
import mlp_helpers as h

MISSING = {"amount": [5, 77], "visits": [12, 140, 260], "member": [33, 199]}


def _twin_tables():
    gen = np.random.default_rng(95)
    n = 300
    amount_text = [f"{v:.2f}" for v in gen.normal(50.0, 20.0, size=n)]
    visits = [int(v) for v in gen.integers(0, 40, size=n)]
    member = [bool(v) for v in gen.random(n) < 0.5]
    noise = gen.normal(size=n)
    logit = (0.08 * (np.array([float(t) for t in amount_text]) - 50.0)
             + 0.1 * (np.array(visits) - 20.0) + 1.5 * (np.array(member) - 0.5))
    y = (gen.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    def gap(name, values):
        return [None if k in MISSING[name] else v for k, v in enumerate(values)]

    driver = pd.DataFrame({
        "amount": pd.Series(gap("amount", [D(t) for t in amount_text]), dtype=object),
        "visits": pd.Series(gap("visits", visits), dtype=object),
        "member": pd.Series(gap("member", member), dtype=object),
        "noise": noise,
        "y": y,
    })
    typed = pd.DataFrame({
        "amount": np.array([np.nan if v is None else float(v)
                            for v in gap("amount", amount_text)], dtype=np.float64),
        "visits": pd.array(gap("visits", visits), dtype="Int64"),
        "member": pd.array(gap("member", member), dtype="boolean"),
        "noise": noise,
        "y": y,
    })
    return driver, typed


def test_object_number_columns_give_the_result_of_their_typed_twins():
    """Finding 4: Decimal, int and bool objects are read as the typed columns they stand for."""
    driver, typed = _twin_tables()
    run_obj = c.run(driver)
    run_typed = c.run(typed)
    cols, twin_cols = h.columns_of(run_obj), h.columns_of(run_typed)
    for name, missing in MISSING.items():
        record = cols[name]
        assert (record["role"], record["kind"], record["reason"]) == ("feature", "numeric", None), record
        assert record["dtype"] == "object", record
        assert record["n_missing"] == len(missing), record
        assert record["n_imputed"] == twin_cols[name]["n_imputed"], (record, twin_cols[name])
    assert c.count_code(run_obj, "unsupported_type") == 0
    assert h.holdout_rows(run_obj) == h.holdout_rows(run_typed)
    assert h.plain(dict(run_obj.headline)) == h.plain(dict(run_typed.headline))
    assert h.models_of(run_obj) == h.models_of(run_typed)
    assert (run_obj.predictions["y_pred"].to_numpy(dtype=float).tolist()
            == run_typed.predictions["y_pred"].to_numpy(dtype=float).tolist())


def test_a_step_is_handed_floats():
    """Spec 2.6: a step receives an object number column as floats, NaN where a cell was empty."""
    driver, typed = _twin_tables()
    frames = []
    for table in (driver, typed):
        _, log = p.run_kept(table)
        fit = max(log.calls("step", "fit"), key=lambda ev: len(ev["rows"]))
        frames.append(fit["frame"])
    got, want = frames
    for name in ("amount", "visits", "member", "noise"):
        assert got[name].dtype == np.float64, (name, got[name].dtype)
        assert want[name].dtype == np.float64, (name, want[name].dtype)
    pd.testing.assert_frame_equal(got[["amount", "visits", "member", "noise"]],
                                  want[["amount", "visits", "member", "noise"]])
    assert set(got["member"].dropna().tolist()) == {0.0, 1.0}


NUMERIC, EXCLUDED = "numeric", "excluded"
MIXTURES = {
    "decimals": ([D("1.5"), D("2.25"), D("-3")], NUMERIC),
    "whole_decimals": ([D("1"), D("2E+1"), D("30")], NUMERIC),
    "ints_and_none": ([1, 2, None, 4], NUMERIC),
    "numpy_ints": ([np.int64(1), np.int32(2), 3], NUMERIC),
    "bools_and_none": ([True, False, None], NUMERIC),
    "numpy_bools": ([np.bool_(True), False], NUMERIC),
    "floats_and_none": ([1.5, None, 2.5, math.nan], NUMERIC),
    "ints_floats_decimals": ([1, 2.5, D("3.75"), np.float32(4.5)], NUMERIC),
    "a_decimal_infinity": ([D("1.5"), D("Infinity"), D("2.5"), D("-Infinity")], NUMERIC),
    "a_decimal_not_a_number": ([D("1.5"), D("NaN"), D("2.5"), D("sNaN")], NUMERIC),
    "a_bool_beside_numbers": ([True, 2, 3.5], EXCLUDED),
    "a_bool_beside_decimals": ([False, D("2.5")], EXCLUDED),
    "text_beside_numbers": ([1, "two", 3.0], EXCLUDED),
    "a_complex_number": ([1.5, 2 + 3j], EXCLUDED),
    "lists": ([[1, 2], [3]], EXCLUDED),
    "bytes": ([b"a", b"b"], EXCLUDED),
}


@pytest.mark.parametrize("case", list(MIXTURES))
def test_which_mixtures_are_read_as_numbers(case):
    from mlprofessor import stages

    values, read_as = MIXTURES[case]
    frame = h.clf_frame(n=240)
    column = pd.Series([None] * 240, dtype=object)
    for k in range(240):
        column.iat[k] = values[k % len(values)]
    frame.insert(0, "c", column)
    assert frame["c"].dtype == object

    roles = {r["name"]: r for r in h.records(stages.column_roles(frame, "y"))}
    result = c.run(frame)
    cols = h.columns_of(result)
    for record in (roles["c"], cols["c"]):
        assert record["dtype"] == "object", record
        if read_as == NUMERIC:
            assert (record["role"], record["kind"], record["reason"]) == ("feature", "numeric", None), record
        else:
            assert (record["role"], record["reason"]) == ("excluded", "unsupported_type"), record
    for key in ("role", "kind", "reason", "n_missing", "n_infinite"):
        assert roles["c"][key] == cols["c"][key], key
    warned = h.warnings_of(result).count(("unsupported_type", "c"))
    assert warned == (1 if read_as == EXCLUDED else 0)
    assert c.count_code(result, "unsupported_type") == warned
    if case == "a_decimal_infinity":
        assert (cols["c"]["n_infinite"], cols["c"]["n_missing"]) == (120, 0)
        assert ("infinite_values", "c") in h.warnings_of(result)
    if case == "a_decimal_not_a_number":
        assert (cols["c"]["n_missing"], cols["c"]["n_infinite"]) == (120, 0)
    c.assert_strict_json(result)


def test_every_unsupported_type_exclusion_is_warned_about():
    """D51: every column excluded as unsupported_type gets a warning naming it."""
    frame = h.clf_frame(n=240)
    frame["durations"] = pd.to_timedelta(np.arange(240) % 17, unit="min")
    frame["complexes"] = np.array([complex(k % 7, 1) for k in range(240)], dtype=np.complex128)
    frame["months"] = pd.Series(pd.period_range("2000-01", periods=240, freq="M"))
    frame["pairs"] = pd.Series([(k % 5, k % 3) for k in range(240)], dtype=object)
    frame["mixed"] = pd.Series([k if k % 2 else f"s{k % 9}" for k in range(240)], dtype=object)
    five = {"durations", "complexes", "months", "pairs", "mixed"}
    result = c.run(frame)
    excluded = {name for name, r in h.columns_of(result).items()
                if r["role"] == "excluded" and r["reason"] == "unsupported_type"}
    assert excluded == five
    warned = [r for r in h.records(result.warnings) if r["code"] == "unsupported_type"]
    assert sorted(r["column"] for r in warned) == sorted(five)
    for record in warned:
        assert isinstance(record["message"], str) and record["message"].strip(), record


TARGET_KINDS = [
    ((False, True), [False, True], True),
    ((0, 1), [0, 1], 1),
    ((D("1.50"), D("2.50")), [1.5, 2.5], 2.5),
    ((D("0"), D("1")), [0, 1], 1),
]


def test_an_object_target_is_the_typed_target():
    """One test, four kinds: bool, int and Decimal objects as a classification target."""
    from mlprofessor import stages

    for pair, classes, positive in TARGET_KINDS:
        frame = h.clf_frame(n=240)
        labels = [pair[int(v)] for v in frame["y"]]
        labels[3] = None
        labels[100] = None
        frame["y"] = pd.Series(labels, dtype=object)
        where = f"target {pair!r}"

        task = stages.decide_task(frame["y"])
        assert task["type"] == "classification", where
        assert task["decided_by"] == "rule", where
        assert task["n_distinct"] == 2, where
        assert list(h.plain(task["classes"])) == classes, (where, task["classes"])
        assert h.plain(task["positive_class"]) == positive, where
        assert stages.usable_rows(frame, "y")["dropped_missing_target"] == 2, where

        result = c.run(frame)
        doc = result.to_dict()
        assert doc["task"]["classes"] == classes, (where, doc["task"]["classes"])
        assert [type(v) for v in doc["task"]["classes"]] == [type(v) for v in classes], where
        assert doc["task"]["positive_class"] == positive, where
        assert type(doc["task"]["positive_class"]) is type(positive), where
        assert doc["task"]["target_dtype"] == "object", where
        assert h.rows_of(result)["dropped_missing_target"] == 2, where
        assert sorted(set(h.plain(result.predictions["y_true"].tolist()))) == classes, where
        json.dumps(h.plain(doc), allow_nan=False)


def test_a_decimal_target_is_a_regression_target():
    frame = h.reg_frame(n=240)
    text = [f"{v:.3f}" for v in frame["y"]]
    as_objects = frame.copy()
    as_objects["y"] = pd.Series([D(t) for t in text], dtype=object)
    as_floats = frame.copy()
    as_floats["y"] = np.array([float(t) for t in text], dtype=np.float64)
    run_obj, run_typed = c.run(as_objects), c.run(as_floats)
    task = run_obj.to_dict()["task"]
    assert (task["type"], task["decided_by"], task["target_dtype"]) == ("regression", "rule", "object")
    assert h.plain(dict(run_obj.headline)) == h.plain(dict(run_typed.headline))
    assert (run_obj.predictions["y_true"].to_numpy(dtype=float).tolist()
            == run_typed.predictions["y_true"].to_numpy(dtype=float).tolist())


WHOLE = {"python_int": int, "decimal": D}


@pytest.mark.parametrize("held", list(WHOLE))
def test_object_whole_numbers_are_exact(held):
    """F7c: whole numbers held as objects are compared exactly, as time keys and when counted."""
    from mlprofessor import stages

    make = WHOLE[held]
    stamps = [p.START + 100 * k for k in range(600)]
    keys, form = stages.read_time(pd.Series([make(s) for s in stamps], dtype=object))
    assert form == "numeric"
    assert keys.dtype == np.int64, keys.dtype
    assert [int(k) for k in keys] == stamps
    keys, form = stages.read_time(pd.Series([make(5), make(3), make(2 ** 53)], dtype=object))
    assert keys.dtype == np.float64, keys.dtype
    assert keys.tolist() == [5.0, 3.0, float(2 ** 53)]

    typed = p.nanosecond_walk(600)
    held_frame = typed.copy()
    held_frame["t"] = pd.Series([make(int(v)) for v in typed["t"]], dtype=object)
    run_held = c.run(held_frame, "level", time_col="t")
    run_typed = c.run(typed, "level", time_col="t")
    validation = run_held.to_dict()["validation"]
    assert (validation["n_train_timestamps"], validation["n_holdout_timestamps"]) == (450, 150)
    assert validation["timestamps_unique"] is True
    assert validation == run_typed.to_dict()["validation"]
    assert h.plain(dict(run_held.headline)) == h.plain(dict(run_typed.headline))
    assert run_held.headline["baseline"] == "previous_value"

    gen = np.random.default_rng(96)
    n = 200
    wide = [2 ** 62 + 7 * int(k) for k in gen.integers(0, 50, size=n)]
    assert len({float(v) for v in wide}) == 1
    frame = pd.DataFrame({
        "x1": gen.normal(size=n),
        "wide": pd.Series([make(v) for v in wide], dtype=object),
        "serial": pd.Series([make(2 ** 53 + 1 + k) for k in range(n)], dtype=object),
        "same": pd.Series([make(2 ** 60 + 1)] * n, dtype=object),
        "y": gen.integers(0, 2, size=n),
    })
    roles = {r["name"]: r for r in h.records(stages.column_roles(frame, "y"))}
    assert (roles["wide"]["role"], roles["wide"]["reason"]) == ("feature", None), roles["wide"]
    assert (roles["serial"]["role"], roles["serial"]["reason"]) == ("excluded", "id_like")
    assert (roles["same"]["role"], roles["same"]["reason"]) == ("excluded", "constant")


def test_an_object_time_column_is_the_typed_time_column():
    held = h.time_frame(n=240)
    held["date"] = pd.Series([D(k) for k in range(240)], dtype=object)
    typed = h.time_frame(n=240)
    typed["date"] = np.arange(240, dtype=np.int64)
    runs = [c.run(frame, time_col="date") for frame in (held, typed)]
    for result in runs:
        validation = result.to_dict()["validation"]
        assert validation["time_form"] == "numeric"
        assert h.holdout_rows(result) == list(range(180, 240))
        assert validation["train_end"] == 179
    assert h.plain(dict(runs[0].headline)) == h.plain(dict(runs[1].headline))
