"""CP5, section H (D52): the type matrix, run against one implementation.

    python -B type_matrix.py <folder that holds the package mlprofessor> <label> [out.tsv] [filter]

    <folder>  the repo root for the application; verify/reference for the reference
    <label>   a word for the summary line
    out.tsv   one line per case: kind|role|extreme, the outcome, seconds, a detail
              (`-` for none)
    filter    only the cases whose name holds this text

Every kind of pandas column, in each of three roles (a spare column beside two
features, the target, the declared time column), with ordinary values and with
extreme ones (spec.md section 7, "The type matrix"). `profile()` must end in a
result or in a named error (a subclass of ProfileError), never in a raw
exception. When it ends in a result, every column appears once in the
accounting, the rows add up, every number is finite, and what was written is
strict JSON.

It prints the count of each outcome, then the raw exceptions grouped by where
they were raised, then every result that broke the contract. The kinds and the
extremes are what the suite's tests/test_type_matrix.py is written from.
Nothing is written inside the repo.
"""
import decimal
import json
import math
import os
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

PARENT, LABEL = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
ONLY = sys.argv[4] if len(sys.argv) > 4 else None
sys.dont_write_bytecode = True
sys.path.insert(0, PARENT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mlprofessor  # noqa: E402
from mlprofessor import errors  # noqa: E402

assert Path(mlprofessor.__file__).resolve().parent.parent == Path(PARENT).resolve(), mlprofessor.__file__

N = 200
D = decimal.Decimal
CELLS = [3, 57, 101, 150]          # where a few extreme cells go
ROLES = ("feature", "target", "time")


class Ctx:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.x1 = self.rng.normal(size=N)
        self.z1 = self.rng.normal(size=N)
        self.noise = self.rng.normal(size=N)
        self.pick50 = self.rng.integers(0, 50, size=N)
        self.bins = np.where(self.x1 > 0.5, 2, np.where(self.x1 < -0.5, 0, 1))


def put(values, cells):
    """`values` with the given cells written at CELLS (cycled)."""
    out = list(values)
    for i, position in enumerate(CELLS):
        out[position] = cells[i % len(cells)]
    return out


# ------------------------------------------------------------------ kinds --
# A kind is a function (role, extreme, ctx) -> a column (anything a DataFrame
# takes) or None when the kind does not allow that extreme.

def int_kind(dtype, nullable=False):
    info = np.iinfo(np.dtype(dtype.lower()) if nullable else dtype)
    lo, hi = int(info.min), int(info.max)

    def wrap(values):
        if nullable:
            return pd.array(values, dtype=dtype)
        return np.array(values, dtype=dtype)

    def build(role, extreme, c):
        centre = 0 if lo < 0 else 100
        if role == "time":
            values = [centre - (100 if lo < 0 else 100) + k for k in range(N)]
        else:
            values = [int(min(hi, max(lo, round(centre + 20 * v)))) for v in c.x1]
        if extreme == "ordinary":
            pass
        elif extreme == "two":
            values = [int(v > 0) for v in c.x1]
        elif extreme == "max_min":
            values = put(values, [hi, lo])
        elif extreme == "above_2p53":
            if hi <= 2 ** 53:
                return None
            start = 2 ** 53 + 1
            values = [start + k for k in range(N)] if role == "time" \
                else [start + int(k) for k in c.pick50]
        elif extreme == "nan":
            if not nullable:
                return None
            values = put(values, [None])
        else:
            return None
        return wrap(values)
    return build


def float_kind(dtype, nullable=False, sparse=False):
    base = np.dtype(dtype.lower()) if nullable else np.dtype(dtype)
    info = np.finfo(base)
    big, small = (1e300, 1e-300) if base == np.float64 else (1e38, 1e-38)

    def wrap(values):
        if sparse:
            return pd.arrays.SparseArray(np.array(values, dtype=base), fill_value=0.0)
        if nullable:
            return pd.array(values, dtype=dtype)
        return np.array(values, dtype=base)

    def build(role, extreme, c):
        if role == "time":
            values = [0.5 * k for k in range(N)]
        else:
            values = [float(2.0 * v + 0.3 * e) for v, e in zip(c.x1, c.noise)]
        if sparse:
            values = [0.0 if abs(v) < 0.7 else v for v in values] if role != "time" else values
        if extreme == "ordinary":
            pass
        elif extreme == "two":
            values = [float(v > 0) for v in c.x1]
        elif extreme == "max_min":
            values = put(values, [float(info.max), -float(info.max), float(info.tiny),
                                  float(info.smallest_subnormal)])
        elif extreme == "above_2p53":
            if base != np.float64:
                return None
            start = 2.0 ** 53
            values = [start + 2.0 * k for k in range(N)] if role == "time" \
                else [start + 2.0 * int(k) for k in c.pick50]
        elif extreme == "e300":
            values = put(values, [big, -big, small, -small])
        elif extreme == "inf":
            values = put(values, [math.inf, -math.inf])
        elif extreme == "nan":
            values = put(values, [None if nullable else math.nan])
        elif extreme == "neg_zero":
            values = put(values, [-0.0, 0.0])
        else:
            return None
        return wrap(values)
    return build


def bool_kind(nullable=False, sparse=False):
    def build(role, extreme, c):
        values = [bool(k % 2) for k in range(N)] if role == "time" else [bool(v > 0) for v in c.x1]
        if extreme in ("ordinary", "two"):
            pass
        elif extreme == "nan" and nullable:
            values = put(values, [None])
        else:
            return None
        if sparse:
            return pd.arrays.SparseArray(np.array(values, dtype=bool), fill_value=False)
        return pd.array(values, dtype="boolean") if nullable else np.array(values, dtype=bool)
    return build


WORDS = ["lo", "mid", "hi"]


def number_text(role, extreme, c):
    """Numbers written as text, or None. The values a CSV would carry."""
    if role == "time":
        values = [f"{0.5 * k:.1f}" for k in range(N)]
    else:
        values = [f"{2.0 * v + 0.3 * e:.4f}" for v, e in zip(c.x1, c.noise)]
    if extreme == "numbers":
        return values
    if extreme == "max_min":
        return put(values, ["1.7976931348623157e308", "-1.7976931348623157e308", "5e-324",
                            "2.2250738585072014e-308"])
    if extreme == "above_2p53":
        start = 2 ** 53 + 1
        return [str(start + k) for k in range(N)] if role == "time" \
            else [str(start + int(k)) for k in c.pick50]
    if extreme == "e300":
        return put(values, ["1e300", "-1e300", "1e-300", "-1e-300"])
    if extreme == "inf":
        return put(values, ["inf", "-inf"])
    if extreme == "nan":
        return put(values, [None])
    if extreme == "neg_zero":
        return put(values, ["-0.0", "0.0"])
    if extreme == "beyond_64":
        start = 10 ** 25
        return [str(start + k) for k in range(N)] if role == "time" \
            else [str(start + int(k)) for k in c.pick50]
    if extreme == "beyond_64_few":
        whole = [str(k) for k in range(N)] if role == "time" \
            else [str(int(round(20 * v))) for v in c.x1]
        return put(whole, [str(10 ** 25), str(-10 ** 25), str(2 ** 64), str(10 ** 30)])
    return None


def text_kind(dtype):
    """dtype: 'object', 'str', 'category', 'category_ordered', 'bytes'."""
    def wrap(values):
        if dtype == "object":
            return pd.Series(values, dtype=object)
        if dtype == "str":
            return pd.Series(values, dtype="str")
        if dtype == "string_python":
            return pd.Series(values, dtype=pd.StringDtype("python"))
        if dtype == "bytes":
            return pd.Series([None if v is None else v.encode("utf-8") for v in values],
                             dtype=object)
        present = sorted({v for v in values if v is not None})
        return pd.Categorical(values, categories=present, ordered=dtype == "category_ordered")

    def build(role, extreme, c):
        if extreme == "ordinary":
            if role == "time":
                start = pd.Timestamp("2020-01-01")
                values = [(start + pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in range(N)]
            else:
                values = [WORDS[b] for b in c.bins]
        elif extreme == "two":
            values = ["yes" if v > 0 else "no" for v in c.x1]
        else:
            values = number_text(role, extreme, c)
            if values is None:
                return None
        return wrap(values)
    return build


def iso_text_kind(role, extreme, c):
    """Dates written as ISO 8601 text."""
    start = pd.Timestamp("2020-01-01")
    if role == "time":
        values = [(start + pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in range(N)]
    else:
        values = [(start + pd.Timedelta(days=int(round(40 + 15 * v)))).strftime("%Y-%m-%d")
                  for v in c.x1]
    if extreme == "ordinary":
        pass
    elif extreme == "two":
        values = ["2020-01-01" if v > 0 else "2020-01-02" for v in c.x1]
    elif extreme == "max_min":
        values = put(values, ["9999-12-31T23:59:59", "0001-01-01"])
    elif extreme == "nan":
        values = put(values, [None])
    else:
        return None
    return pd.Series(values, dtype="str")


def csv_kind(role, extreme, c):
    """Marker: the column is `number_text`, written to a CSV file and read by the tool."""
    if extreme == "ordinary":
        return number_text(role, "numbers", c)
    if extreme == "two":
        return ["1" if v > 0 else "0" for v in c.x1]
    return number_text(role, extreme, c)


def datetime_kind(unit="ns", tz=None):
    def build(role, extreme, c):
        start = pd.Timestamp("2020-01-01")
        if role == "time":
            stamps = [start + pd.Timedelta(days=k) for k in range(N)]
        else:
            stamps = [start + pd.Timedelta(days=int(round(40 + 15 * v))) for v in c.x1]
        if extreme == "ordinary":
            pass
        elif extreme == "two":
            stamps = [start if v > 0 else start + pd.Timedelta(days=1) for v in c.x1]
        elif extreme == "max_min":
            if unit == "ns":
                stamps = put(stamps, [pd.Timestamp.max, pd.Timestamp.min])
            else:
                stamps = put(stamps, [pd.Timestamp("9999-12-31 23:59:59"),
                                      pd.Timestamp("0001-01-01 00:00:00")])
        elif extreme == "nan":
            stamps = put(stamps, [pd.NaT])
        else:
            return None
        series = pd.Series(stamps)
        if unit != "ns":
            series = pd.Series(pd.array(np.array([np.datetime64(s.to_pydatetime(), unit)
                                                  if s is not pd.NaT else np.datetime64("NaT")
                                                  for s in stamps], dtype=f"datetime64[{unit}]")))
        if tz is not None:
            if extreme == "max_min":
                series = pd.Series(put(list(series), [pd.Timestamp("2262-04-10"),
                                                      pd.Timestamp("1677-09-23")]))
            series = series.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        return series
    return build


def timedelta_kind(role, extreme, c):
    if role == "time":
        values = [pd.Timedelta(hours=k) for k in range(N)]
    else:
        values = [pd.Timedelta(minutes=int(round(600 + 200 * v))) for v in c.x1]
    if extreme == "ordinary":
        pass
    elif extreme == "two":
        values = [pd.Timedelta(days=1) if v > 0 else pd.Timedelta(days=2) for v in c.x1]
    elif extreme == "max_min":
        values = put(values, [pd.Timedelta.max, pd.Timedelta.min])
    elif extreme == "nan":
        values = put(values, [pd.NaT])
    else:
        return None
    return pd.Series(values, dtype="timedelta64[ns]")


def period_kind(role, extreme, c):
    start = pd.Period("2020-01-01", freq="D")
    values = [start + k for k in range(N)] if role == "time" \
        else [start + int(round(40 + 15 * v)) for v in c.x1]
    if extreme == "ordinary":
        pass
    elif extreme == "two":
        values = [start if v > 0 else start + 1 for v in c.x1]
    elif extreme == "max_min":
        values = put(values, [pd.Period("9999-12-31", freq="D"), pd.Period("0001-01-01", freq="D")])
    elif extreme == "nan":
        values = put(values, [pd.NaT])
    else:
        return None
    return pd.Series(values, dtype="period[D]")


def interval_kind(role, extreme, c):
    values = [pd.Interval(k, k + 1) for k in range(N)] if role == "time" \
        else [pd.Interval(int(b), int(b) + 1) for b in c.bins]
    if extreme == "ordinary":
        pass
    elif extreme == "two":
        values = [pd.Interval(0, 1) if v > 0 else pd.Interval(1, 2) for v in c.x1]
    elif extreme == "nan":
        values = put(values, [None])
    else:
        return None
    return pd.Series(pd.arrays.IntervalArray(values))


def complex_kind(dtype):
    def build(role, extreme, c):
        values = [complex(k, 1) for k in range(N)] if role == "time" \
            else [complex(2 * v, e) for v, e in zip(c.x1, c.z1)]
        if extreme == "ordinary":
            pass
        elif extreme == "two":
            values = [1 + 1j if v > 0 else 2 + 2j for v in c.x1]
        elif extreme == "e300":
            values = put(values, [complex(1e300, 1e300), complex(-1e300, 1e-300)]) \
                if dtype == "complex128" else put(values, [complex(1e38, 1e38)])
        elif extreme == "inf":
            values = put(values, [complex(math.inf, 0), complex(0, -math.inf)])
        elif extreme == "nan":
            values = put(values, [complex(math.nan, math.nan)])
        else:
            return None
        return np.array(values, dtype=dtype)
    return build


def decimal_kind(role, extreme, c):
    if role == "time":
        values = [D(k) for k in range(N)]
    else:
        values = [D(f"{2.0 * v + 0.3 * e:.4f}") for v, e in zip(c.x1, c.noise)]
    if extreme == "ordinary":
        pass
    elif extreme == "two":
        values = [D("1.5") if v > 0 else D("2.5") for v in c.x1]
    elif extreme == "max_min":
        values = put(values, [D("1E+400"), D("-1E+400"), D("1E-400"), D("-1E-400")])
    elif extreme == "above_2p53":
        start = 2 ** 53 + 1
        values = [D(start + k) for k in range(N)] if role == "time" \
            else [D(start + int(k)) for k in c.pick50]
    elif extreme == "e300":
        values = put(values, [D("1E+300"), D("-1E+300"), D("1E-300"), D("-1E-300")])
    elif extreme == "inf":
        values = put(values, [D("Infinity"), D("-Infinity")])
    elif extreme == "nan":
        values = put(values, [None, D("NaN"), D("sNaN")])
    elif extreme == "neg_zero":
        values = put(values, [D("-0"), D("0"), D("-0.0")])
    elif extreme == "beyond_64":
        start = 10 ** 25
        values = [D(start + k) for k in range(N)] if role == "time" \
            else [D(start + int(k)) for k in c.pick50]
    else:
        return None
    return pd.Series(values, dtype=object)


def object_number_kind(what):
    """Object columns as a database driver returns them: int, bool, float, or a mixture."""
    def build(role, extreme, c):
        if what == "bool":
            values = [bool(k % 2) for k in range(N)] if role == "time" else [bool(v > 0) for v in c.x1]
        elif what == "float":
            values = [0.5 * k for k in range(N)] if role == "time" \
                else [float(2.0 * v + 0.3 * e) for v, e in zip(c.x1, c.noise)]
        elif what == "int":
            values = list(range(N)) if role == "time" else [int(round(20 * v)) for v in c.x1]
        elif what == "numbers":                 # int, float and Decimal together
            values = [(k, float(k) + 0.5, D(k) + D("0.25"))[k % 3] for k in range(N)] \
                if role == "time" else \
                [(int(round(20 * v)), float(2 * v), D(f"{v:.3f}"))[k % 3]
                 for k, v in enumerate(c.x1)]
        else:                                   # int, float, Decimal and bool together
            values = [(int(round(20 * v)), float(2 * v), D(f"{v:.3f}"), bool(v > 0))[k % 4]
                      for k, v in enumerate(c.x1)]
        if extreme == "ordinary":
            pass
        elif extreme == "two":
            values = {"bool": [bool(v > 0) for v in c.x1],
                      "float": [1.5 if v > 0 else 2.5 for v in c.x1],
                      "int": [1 if v > 0 else 0 for v in c.x1]}.get(what)
            if values is None:
                return None
        elif extreme == "nan":
            values = put(values, [None, math.nan])
        elif extreme == "max_min" and what == "int":
            values = put(values, [2 ** 63 - 1, -2 ** 63, 2 ** 64 - 1])
        elif extreme == "above_2p53" and what == "int":
            start = 2 ** 53 + 1
            values = [start + k for k in range(N)] if role == "time" \
                else [start + int(k) for k in c.pick50]
        elif extreme == "beyond_64" and what == "int":
            start = 10 ** 25
            values = [start + k for k in range(N)] if role == "time" \
                else [start + int(k) for k in c.pick50]
        elif extreme == "beyond_64_few" and what == "int":
            values = put(values, [10 ** 25, -10 ** 25, 2 ** 64, None])
        elif extreme == "inf" and what == "float":
            values = put(values, [math.inf, -math.inf])
        elif extreme == "e300" and what == "float":
            values = put(values, [1e300, -1e300, 1e-300, -1e-300])
        elif extreme == "neg_zero" and what == "float":
            values = put(values, [-0.0, 0.0])
        else:
            return None
        return pd.Series(values, dtype=object)
    return build


def mixed_kind(role, extreme, c):
    values = [k if k % 2 else f"s{k}" for k in range(N)]
    if extreme == "ordinary":
        pass
    elif extreme == "two":
        values = [1 if v > 0 else "one" for v in c.x1]
    elif extreme == "nan":
        values = put(values, [None, math.nan, pd.NA, pd.NaT])
    elif extreme == "e300":
        values = put(values, [1e300, D("1E+400"), 10 ** 30, complex(1e300, 1)])
    else:
        return None
    return pd.Series(values, dtype=object)


def container_kind(what):
    def build(role, extreme, c):
        make = {"list": lambda k: [k, k + 1], "tuple": lambda k: (k, k + 1),
                "dict": lambda k: {"a": k}, "set": lambda k: {k, k + 1}}[what]
        values = [make(k) for k in range(N)]
        if extreme == "ordinary":
            pass
        elif extreme == "two":
            values = [make(0) if v > 0 else make(5) for v in c.x1]
        elif extreme == "nan":
            values = put(values, [None])
        else:
            return None
        return pd.Series(values, dtype=object)
    return build


def all_missing_kind(what):
    def build(role, extreme, c):
        if extreme != "ordinary":
            return None
        if what == "float":
            return np.full(N, np.nan)
        if what == "none":
            return pd.Series([None] * N, dtype=object)
        if what == "NA":
            return pd.Series([pd.NA] * N, dtype=object)
        if what == "NaT":
            return pd.Series([pd.NaT] * N, dtype="datetime64[ns]")
        if what == "Int64":
            return pd.array([None] * N, dtype="Int64")
        if what == "str":
            return pd.Series([None] * N, dtype="str")
    return build


def sparse_int(role, extreme, c):
    if role == "time":
        values = list(range(N))
    else:
        values = [int(round(3 * v)) for v in c.x1]
    if extreme == "ordinary":
        pass
    elif extreme == "two":
        values = [int(v > 0) for v in c.x1]
    elif extreme == "max_min":
        values = put(values, [2 ** 63 - 1, -2 ** 63])
    elif extreme == "above_2p53":
        start = 2 ** 53 + 1
        values = [start + k for k in range(N)] if role == "time" \
            else [start + int(k) for k in c.pick50]
    else:
        return None
    return pd.arrays.SparseArray(np.array(values, dtype="int64"), fill_value=0)


KINDS = {
    "int8": int_kind("int8"), "int16": int_kind("int16"), "int32": int_kind("int32"),
    "int64": int_kind("int64"),
    "uint8": int_kind("uint8"), "uint16": int_kind("uint16"), "uint32": int_kind("uint32"),
    "uint64": int_kind("uint64"),
    "float32": float_kind("float32"), "float64": float_kind("float64"),
    "Int8": int_kind("Int8", nullable=True), "Int64": int_kind("Int64", nullable=True),
    "UInt64": int_kind("UInt64", nullable=True),
    "Float32": float_kind("Float32", nullable=True), "Float64": float_kind("Float64", nullable=True),
    "bool": bool_kind(), "boolean": bool_kind(nullable=True),
    "text_object": text_kind("object"), "text_str": text_kind("str"),
    "text_string_python": text_kind("string_python"),
    "category": text_kind("category"), "category_ordered": text_kind("category_ordered"),
    "bytes": text_kind("bytes"), "iso_text": iso_text_kind,
    "datetime_ns": datetime_kind("ns"), "datetime_s": datetime_kind("s"),
    "datetime_utc": datetime_kind("ns", tz="UTC"),
    "datetime_london": datetime_kind("ns", tz="Europe/London"),
    "timedelta": timedelta_kind, "period": period_kind, "interval": interval_kind,
    "complex128": complex_kind("complex128"), "complex64": complex_kind("complex64"),
    "decimal": decimal_kind,
    "object_int": object_number_kind("int"), "object_bool": object_number_kind("bool"),
    "object_float": object_number_kind("float"), "object_numbers": object_number_kind("numbers"),
    "object_bool_beside_numbers": object_number_kind("mixed"),
    "mixed": mixed_kind,
    "list": container_kind("list"), "tuple": container_kind("tuple"),
    "dict": container_kind("dict"), "set": container_kind("set"),
    "all_missing_float": all_missing_kind("float"), "all_missing_none": all_missing_kind("none"),
    "all_missing_NA": all_missing_kind("NA"), "all_missing_NaT": all_missing_kind("NaT"),
    "all_missing_Int64": all_missing_kind("Int64"), "all_missing_str": all_missing_kind("str"),
    "sparse_float": float_kind("float64", sparse=True), "sparse_int": sparse_int,
    "sparse_bool": bool_kind(sparse=True),
    "csv": csv_kind,
}
EXTREMES = ["ordinary", "two", "one_value", "max_min", "above_2p53", "e300", "inf", "nan",
            "neg_zero", "beyond_64", "beyond_64_few"]


def cases():
    for kind in KINDS:
        for role in ROLES:
            for extreme in EXTREMES:
                if extreme == "two" and role != "target":
                    continue
                yield kind, role, extreme


def one_value(column):
    """The column with its first value in every row, in the same type; None if that is missing."""
    if isinstance(column, list):                # the cells of a CSV column
        return None if column[0] is None else [column[0]] * N
    series = pd.Series(column)
    first = series.iloc[0]
    if isinstance(first, (list, dict, set, tuple)) or not pd.isna(first):
        return series.take([0] * N).reset_index(drop=True)
    return None


def build_frame(kind, role, extreme):
    """The table of one case, or None when the kind does not allow the extreme."""
    c = Ctx()
    if extreme == "one_value":
        column = KINDS[kind](role, "ordinary", c)
        column = None if column is None else one_value(column)
    else:
        column = KINDS[kind](role, extreme, c)
    if column is None:
        return None, None
    if role == "feature":
        frame = pd.DataFrame({"x1": c.x1, "z1": c.z1})
        frame["c"] = column
        frame["y"] = (c.x1 + 0.5 * c.noise > 0).astype(int)
        args = dict(target="y")
    elif role == "target":
        frame = pd.DataFrame({"x1": c.x1, "z1": c.z1})
        frame["y"] = column
        args = dict(target="y")
    else:
        frame = pd.DataFrame({"x1": c.x1, "z1": c.z1})
        frame.insert(0, "t", column)
        frame["y"] = 2.0 * c.x1 + c.noise
        args = dict(target="y", time_col="t")
    return frame, args


def not_finite(node, path="$"):
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            found += not_finite(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            found += not_finite(value, f"{path}[{i}]")
    elif isinstance(node, (float, np.floating)) and not math.isfinite(float(node)):
        found.append(f"{path}={node!r}")
    return found


def is_number(value):
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(float(value)))


def judge_result(result, names):
    """'' when the result keeps the contract of H, else what is wrong with it."""
    doc = result.to_dict()
    wrong = []
    got = [col["name"] for col in doc["columns"]]
    if got != names:
        wrong.append(f"columns {got} != {names}")
    rows = {r["item"]: r["count"] for r in doc["rows"]}
    if rows.get("loaded") != sum(v for k, v in rows.items() if k != "loaded"):
        wrong.append(f"rows do not add up {rows}")
    bad = not_finite(doc) + not_finite(dict(result.headline), "headline")
    if bad:
        wrong.append("not finite: " + ", ".join(bad[:4]))
    try:
        json.dumps(doc, allow_nan=False)
    except (TypeError, ValueError) as exc:
        wrong.append(f"to_dict() is not JSON: {exc}")
    head = doc["headline"]
    for key in ("score", "baseline_score"):
        if not is_number(head[key]):
            wrong.append(f"headline.{key}={head[key]!r}")
    if head["skill"] is None:
        if head["baseline_score"] != 0 or not any(w["code"] == "skill_undefined"
                                                  for w in doc["warnings"]):
            wrong.append("skill null without its reason")
    elif not is_number(head["skill"]):
        wrong.append(f"headline.skill={head['skill']!r}")
    if head["interval_method"] is None:
        if not head["interval_reason"]:
            wrong.append("no interval and no reason")
    elif not (is_number(head["interval_low"]) and is_number(head["interval_high"])):
        wrong.append("interval ends are not numbers")
    if head["p_value"] is None:
        if not head["p_value_reason"]:
            wrong.append("no p-value and no reason")
    elif not is_number(head["p_value"]):
        wrong.append(f"p_value={head['p_value']!r}")
    for b in doc["baselines"]:
        if not is_number(b["score"]):
            wrong.append(f"baseline {b['name']} score {b['score']!r}")
    for m in doc["models"]:
        if m["status"] == "fitted" and not is_number(m["cv_mean"]):
            wrong.append(f"{m['config']} fitted with cv_mean {m['cv_mean']!r}")
    y_pred = result.predictions["y_pred"].to_numpy(dtype=float)
    if not np.isfinite(y_pred).all():
        wrong.append("a y_pred that is not finite")
    if len(result.predictions) != rows.get("holdout"):
        wrong.append("predictions rows != holdout")
    for part in ("rows", "columns", "models", "warnings", "predictions"):
        if not isinstance(getattr(result, part), pd.DataFrame):
            wrong.append(f"result.{part} is not a DataFrame")
    return "; ".join(wrong)


def where(exc):
    """The innermost frame of the traceback that is inside the package."""
    inside = [f for f in traceback.extract_tb(exc.__traceback__) if "mlprofessor" in f.filename]
    last = (inside or traceback.extract_tb(exc.__traceback__))[-1]
    return f"{Path(last.filename).name}:{last.lineno} {last.name}"


def run_case(kind, role, extreme, folder):
    frame, args = build_frame(kind, role, extreme)
    if frame is None:
        return None
    data, names = frame, [str(n) for n in frame.columns]
    if kind == "csv":
        data = str(Path(folder) / f"{role}_{extreme}.csv")
        frame.to_csv(data, index=False)
    started = time.perf_counter()
    out = Path(folder) / f"out_{kind}_{role}_{extreme}"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = mlprofessor.profile(data, profile="fast", seed=0, n_jobs=1,
                                         out_dir=str(out), **args)
        problem = judge_result(result, names)
        if not problem:
            try:
                json.loads((out / "profile.json").read_text(encoding="utf-8"),
                           parse_constant=lambda token: 1 / 0)
            except Exception as exc:  # noqa: BLE001
                problem = f"profile.json is not strict JSON: {type(exc).__name__}"
        outcome = "result" if not problem else "BAD RESULT: " + problem
        detail = ""
        if not problem:
            doc = result.to_dict()
            col = {c_["name"]: c_ for c_ in doc["columns"]}.get("c")
            detail = f"{doc['task']['type']}"
            if col:
                detail += f" c={col['role']}/{col['reason'] or col['kind']}"
            if role == "time":
                v = doc["validation"]
                detail += f" form={v['time_form']} stamps={v['n_train_timestamps']}+{v['n_holdout_timestamps']} unique={v['timestamps_unique']}"
    except errors.ProfileError as exc:
        outcome, detail = f"named: {exc.code}", str(exc)[:110]
    except Exception as exc:  # noqa: BLE001
        outcome = f"RAW: {type(exc).__name__}"
        detail = f"{where(exc)} | {str(exc)[:120]}"
    return outcome, detail.replace("\n", " "), time.perf_counter() - started


def main():
    lines, counts, raw, bad = [], {}, [], []
    folder = tempfile.mkdtemp(prefix="cp5-matrix-")
    total_seconds = 0.0
    for kind, role, extreme in cases():
        case = f"{kind}|{role}|{extreme}"
        if ONLY and ONLY not in case:
            continue
        try:
            got = run_case(kind, role, extreme, folder)
        except Exception as exc:  # noqa: BLE001  (the prototype's own fault)
            got = (f"PROTOTYPE: {type(exc).__name__}", str(exc)[:150], 0.0)
        if got is None:
            continue
        outcome, detail, seconds = got
        total_seconds += seconds
        key = outcome.split(":")[0]
        counts[key] = counts.get(key, 0) + 1
        lines.append(f"{case}\t{outcome}\t{seconds:.2f}\t{detail}")
        if key == "RAW":
            raw.append((case, outcome, detail))
        if key in ("BAD RESULT", "PROTOTYPE"):
            bad.append((case, outcome, detail))
    if OUT:
        Path(OUT).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{LABEL}] {len(lines)} cases in {total_seconds:.0f} s: "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    causes = {}
    for case, outcome, detail in raw:
        causes.setdefault(outcome + " @ " + detail.split(" | ")[0], []).append(case)
    for cause, found in sorted(causes.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(found):3d}  {cause}")
        print("       " + ", ".join(found[:8]) + (f" (+{len(found) - 8} more)" if len(found) > 8 else ""))
    for case, outcome, detail in bad:
        print(f"  {case}: {outcome} {detail}")


if __name__ == "__main__":
    main()
