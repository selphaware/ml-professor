"""Reference implementation of spec.md, in the spec's order of operations.

Judge fixture: the gate measures its bands on this code and breaks it in fourteen
named ways; the test suite is proved against those and against thirteen smaller
faults (see _variant.py). One module on purpose, so each step of the
methodology can be read top to bottom in `_run`.
"""
from __future__ import annotations

import codecs
import csv
import decimal
import json
import math
import os
import platform
import re
import time
import warnings as _pywarnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import threadpoolctl
from pandas.api import types as pt
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import errors as E
from . import extensions
from ._variant import BROKEN

# ------------------------------------------------------------ pinned values --
FAST_ROW_CAP = 2000
N_FOLDS = 5
N_BLOCKS = 6
MIN_ROWS = 100              # fewer usable rows than this: too small to profile
MIN_CLASS_ROWS = 10         # fewer usable rows of a class than this: too small
LEAK_SKILL = 0.98           # single-column cross-validated skill on training rows
STRONG_SKILL = 0.90         # at least this and below LEAK_SKILL: kept, with a warning (D30)
LEAK_MISSING_HIGH = 0.95    # missing for more than this share of one class ...
LEAK_MISSING_LOW = 0.05     # ... and less than this share of the other
ID_TEXT_SHARE = 0.95        # text column: distinct values > 95% of the rows
FREE_TEXT_SHARE = 0.50      # text column: distinct values > 50% of non-missing
NUMERIC_SHARE = 0.90        # text column: >= 90% of non-missing parse as numbers
SEPARATED_SHARE = 0.90      # text column: >= 90% parse or carry thousands separators (D28)
DATE_SHARE = 0.95           # text column: >= 95% of non-missing look like dates
SMALL_HOLDOUT = 50
MIN_BLOCK_TIMESTAMPS = 30   # fewer distinct holdout timestamps: no interval
AUTOCORR = 0.5              # lag-1 autocorrelation of the per-timestamp mean target
MAX_LEVELS = 30             # one-hot levels kept per categorical column
RESAMPLES = {"full": 1000, "fast": 200}
MISSING_LEVEL = "(missing)"
MAX_SEED = 2 ** 32 - 1      # the largest seed scikit-learn takes (D45)
# The largest 32-bit float, 3.4028234663852886e+38. scikit-learn's trees and forests
# read their input as 32-bit floats and raise on anything beyond; a feature value of
# greater magnitude is read as an infinite one is: missing, counted, warned (D48).
TOO_LARGE = float(np.finfo(np.float32).max)
# Whole numbers up to this magnitude are exact as 64-bit floats; above it two
# distinct integers can become one float, so integer time keys are kept as
# integers there (D50, D53).
EXACT_AS_FLOAT = 2 ** 53
# D49, D53: the run stops when more than this share of the holdout rows, and at
# least this many of them, have a copy among the training rows.
COPY_SHARE = 0.01
COPY_ROWS = 10

LINEAR_C = [1.0, 0.001, 0.01, 0.1, 10.0, 100.0]
LINEAR_ALPHA = [1.0, 0.01, 0.1, 10.0, 100.0, 1000.0]
THIRD = 1.0 / 3.0
FOREST_LEAF = [1, 5, 20]
BOOSTING_GRID = [(31, 0.1, 100), (31, 0.05, 300), (7, 0.1, 100), (7, 0.05, 300),
                 (63, 0.1, 100), (63, 0.05, 300)]

_Y = r"\d{4}"
_M = r"(?:0?[1-9]|1[0-2])"
_D = r"(?:0?[1-9]|[12]\d|3[01])"
_T = r"(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
_MON = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
DATE_FORMS = [
    rf"{_Y}-{_M}-{_D}{_T}", rf"{_Y}/{_M}/{_D}{_T}",
    rf"{_D}/{_M}/{_Y}{_T}", rf"{_M}/{_D}/{_Y}{_T}",
    rf"{_D}-{_M}-{_Y}{_T}", rf"{_M}-{_D}-{_Y}{_T}",
    rf"{_D}\.{_M}\.{_Y}{_T}",
    rf"{_D}[- ]{_MON}[- ]{_Y}{_T}", rf"{_MON} {_D},? {_Y}{_T}",
]
DATE_RE = re.compile("(?:" + "|".join(DATE_FORMS) + ")", re.I)
ISO_TIME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?)?(?:Z|[+-]\d{2}:\d{2})?")
ISO_TZ_RE = re.compile(r".*(?:Z|[+-]\d{2}:\d{2})")
# ISO 8601 in every respect but one: more than nine fractional digits. Only used to
# say what is wrong with a refused value (D52).
ISO_LONG_FRACTION_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\.\d{10,}(?:Z|[+-]\d{2}:\d{2})?")
# D28: a number written with thousands separators. Never parsed: `1,200` is 1.2 in some locales.
SEPARATED_RE = re.compile(r"[+-]?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?")

RANDOM_CAVEAT = ("Random split: this number is optimistic if the rows are ordered in time "
                 "or one entity appears on many rows.")
NO_P_VALUE = ("not computed under the time-ordered scheme: shuffling the holdout targets "
              "assumes independent rows")
NO_P_VALUE_NO_SKILL = ("not computed: the baseline's squared error on the holdout is zero, so "
                       "skill has no value to compare the shuffles with")
LIMITATIONS = [
    "Predictability here means how well these model families did on this data under this "
    "validation scheme.",
    "Low skill is not proof that the target cannot be predicted.",
    "High skill is not proof of cause.",
    "A random split is optimistic when rows are ordered in time or one entity appears on "
    "many rows.",
    "Look-ahead inside the caller's own features cannot be detected.",
    "Classification skill and regression skill are on different scales and are not comparable.",
    "A CSV file is read with pandas' types, so text identifiers can lose leading zeros; "
    "pass a DataFrame to control the types.",
    "Intervals are approximate: they tend to be too narrow when the holdout is short and, "
    "under time order, when errors are serially correlated.",
    "Negative skill means the models did worse than the baseline on the holdout; read it as "
    "no evidence of predictability.",
]


# ------------------------------------------------------------------ result --

class ProfileResult:
    """What `profile` returns. Every part is also in `to_dict()`."""

    def __init__(self, doc, predictions):
        self._doc = doc
        self.headline = dict(doc["headline"])
        self.rows = pd.DataFrame(doc["rows"], columns=["item", "count"])
        self.columns = pd.DataFrame(doc["columns"], columns=COLUMN_KEYS)
        self.models = pd.DataFrame(doc["models"], columns=MODEL_KEYS)
        self.warnings = pd.DataFrame(doc["warnings"], columns=["code", "column", "message"])
        self.predictions = predictions

    def to_dict(self):
        return json.loads(json.dumps(self._doc, allow_nan=False))

    def to_json(self, path):
        text = json.dumps(self._doc, indent=2, allow_nan=False)
        Path(path).write_text(text + "\n", encoding="utf-8")


COLUMN_KEYS = ["name", "position", "dtype", "role", "kind", "reason", "leak_rule",
               "n_missing", "n_coerced", "n_infinite", "n_imputed"]
MODEL_KEYS = ["config", "family", "params", "status", "error", "fold_scores",
              "cv_mean", "cv_std", "chosen"]


def summary(result):
    """Short plain-text account for the command line."""
    d = result.to_dict()
    h, rows = d["headline"], {r["item"]: r["count"] for r in d["rows"]}
    skill = "not defined" if h["skill"] is None else f"{h['skill']:.6g}"
    lines = [
        f"task: {d['task']['type']} ({d['task']['decided_by']}); "
        f"scheme: {d['validation']['scheme']}",
        f"rows: {rows['loaded']} loaded, {rows['training']} training, {rows['holdout']} holdout",
        f"headline: skill {skill} ({h['metric']} {h['score']:.6g} against "
        f"{h['baseline']} {h['baseline_score']:.6g}), model {h['model']}",
    ]
    # The interval on the score and on skill, and the p-value: each as a number,
    # or the reason there is none (D39).
    if h["interval_method"] is None:
        lines.append(f"95% interval: none. {h['interval_reason']}")
    else:
        lines.append(f"95% interval on {h['metric']}: {h['interval_low']:.6g} to "
                     f"{h['interval_high']:.6g} ({h['interval_method']})")
        if h["skill_low"] is None:
            code = "skill_undefined" if h["skill"] is None else "no_skill_interval"
            lines.append(f"95% interval on skill: none (see the warning {code})")
        else:
            lines.append(f"95% interval on skill: {h['skill_low']:.6g} to {h['skill_high']:.6g}")
    if h["p_value"] is None:
        lines.append(f"permutation p-value: none. {h['p_value_reason']}")
    else:
        lines.append(f"permutation p-value: {h['p_value']:.6g}")
    for w in d["warnings"]:                 # one line each: the code, then the column
        where = "" if w["column"] is None else f" column {w['column']!r}"
        lines.append(f"warning [{w['code']}]{where}: {w['message']}")
    return "\n".join(lines)


# -------------------------------------------------------------- public API --

def profile(data, target, task="auto", time_col=None, gap=0, profile="full", seed=0,
            out_dir=None, n_jobs=-1, duplicates="error"):
    """Measure how predictable `target` is from the other columns of one table."""
    started = time.perf_counter()
    if task not in ("auto", "classification", "regression"):
        raise E.InputError(f"task must be auto, classification or regression, not {task!r}")
    if profile not in ("full", "fast"):
        raise E.InputError(f"profile must be full or fast, not {profile!r}")
    if not isinstance(duplicates, str) or duplicates not in ("error", "keep"):
        raise E.InputError(f"duplicates must be error or keep, not {duplicates!r}")
    if isinstance(gap, bool) or not isinstance(gap, (int, np.integer)) or gap < 0:
        raise E.InputError(f"gap must be a whole number of at least 0, not {gap!r}")
    _check_seed(seed)                       # before anything is loaded (D45)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) \
            or not (n_jobs == -1 or n_jobs >= 1):
        raise E.InputError(f"n_jobs must be -1 or a positive whole number, not {n_jobs!r}")
    if gap and time_col is None:
        raise E.InputError("gap needs time_col: it counts timestamps")
    if time_col is not None and time_col == target:
        raise E.InputError("time_col and target are the same column")
    if out_dir is not None:
        _check_out_dir(out_dir)             # before anything is loaded (D39)
    threads = (os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"), \
            threadpoolctl.threadpool_limits(limits=threads, user_api="openmp"):
        doc, predictions = _run(data, target, task, time_col, int(gap), profile, int(seed), threads,
                                duplicates)
    doc["meta"]["timing"] = {"seconds": round(time.perf_counter() - started, 3)}
    result = ProfileResult(doc, predictions)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.to_json(out / "profile.json")
        predictions.to_csv(out / "predictions.csv", index=False, lineterminator="\n")
    return result


# ------------------------------------------------------------------- steps --

def _warn(warnings, code, message, column=None):
    warnings.append({"code": code, "column": column, "message": message})


def _check_seed(seed):
    """D45: a whole number from 0 to 2**32 - 1. A larger seed is never folded down:
    two seeds would then give one run, silently."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) \
            or not 0 <= seed <= MAX_SEED:
        raise E.InputError(f"seed must be a whole number from 0 to {MAX_SEED} (2**32 - 1), "
                           f"not {seed!r}")


def _not_finite(what, value, cause):
    """D44: the named error for a number the result rests on that is not finite."""
    return E.NonFiniteScoreError(
        f"{what} is not a finite number ({value!r}), so it is not reported. Likely cause: "
        f"{cause}. Nothing was written")


TARGET_CAUSE = ("a target value too large for squared error: the square of a value of about "
                "1e154 or more is beyond the largest number a double can hold. Look for a "
                "sentinel such as 1.8e308 in the target, or rescale it")


def _check_out_dir(out_dir):
    """D39: an output folder that cannot be used is an input error. Nothing is created."""
    if not isinstance(out_dir, (str, os.PathLike)):
        raise E.InputError(f"out_dir must be a folder path or None, not {out_dir!r}")
    place = Path(out_dir)
    while not place.exists() and place.parent != place:
        place = place.parent                # the nearest ancestor that exists
    if place.exists() and not place.is_dir():
        raise E.InputError(f"out_dir {str(out_dir)!r} cannot be used: {str(place)!r} exists "
                           "and is not a folder")


def _check_utf8(path):
    """D39: a file that is not valid UTF-8 is an input error wherever the bad byte sits."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise E.InputError(f"{path} is not valid UTF-8: {exc}") from exc


def _from_header(lines):
    """The lines of a file from its header line on.

    pandas skips a leading line that is empty or holds only spaces and tabs, so
    the header is the first line that is neither (D39). A line of other white
    space, of two quotes or of commas is the header to pandas, and here.
    """
    started = False
    for line in lines:
        if started or line.strip(" \t\r\n"):
            started = True
            yield line


def _csv_head(path):
    """The fields of the header line and of the first data row (`None`, `[]` when absent)."""
    limit = csv.field_size_limit(2 ** 31 - 1)   # pandas reads a cell of any length
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = csv.reader(_from_header(fh))
            header = next(rows, None)
            # As pandas counts rows: a line that is empty or holds only spaces and
            # tabs is skipped, and a line of only commas is a row.
            first = next((row for row in rows
                          if len(row) > 1 or "".join(row).strip(" \t")), [])
    finally:
        csv.field_size_limit(limit)             # the caller's setting, as it was (D39)
    return header, first


def _load(data, warnings):
    """Step 1 (D22): the table, with every row of the source."""
    if isinstance(data, pd.DataFrame):
        df = data.copy(deep=False)
        df.columns = [str(c) for c in df.columns]
        names = list(df.columns)
        dups = sorted({n for n in names if names.count(n) > 1})
        kind, label = "dataframe", "<DataFrame>"
    elif isinstance(data, (str, os.PathLike)):
        path = Path(data)
        if not path.is_file():
            raise E.InputError(f"no such file: {data}")
        _check_utf8(path)
        header, first = _csv_head(path)
        if header is None:
            raise E.InputError(f"{data} has no header row")
        # A cell is blank only when it is empty. One that holds white space is a name
        # like any other, so two of them are duplicates (D42).
        dups = sorted({n for n in header if n != "" and header.count(n) > 1})
        if not dups:
            # pandas makes an index of the leading fields of a first data row longer
            # than the header and names the fields after them. The index it returns
            # does not show that: for row numbers it is a RangeIndex, like the default.
            # A later row longer than the first data row is pandas' own ParserError.
            # F6d: the check is gone, and pandas reads the file without a word.
            if len(first) > len(header) and BROKEN != "F6d":
                raise E.InputError(
                    f"{data}: the first data row has {len(first)} fields and the header has "
                    f"{len(header)}; a data row may not have more fields than the header")
            try:
                df = pd.read_csv(path, low_memory=False)
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
                raise E.InputError(f"{data} could not be read as CSV: {exc}") from exc
            if len(df.columns) != len(header):
                raise E.InputError(f"{data}: data rows do not have the header's number of fields")
            for i, name in enumerate(header):
                if name == "":
                    _warn(warnings, "blank_header",
                          f"column {i + 1} has no header and is kept as {df.columns[i]!r}",
                          str(df.columns[i]))
        kind, label = "csv", str(data)
    else:
        raise E.InputError("data must be a pandas DataFrame or a path to a CSV file")
    if dups:
        raise E.DuplicateColumnsError("duplicate column names: " + ", ".join(dups))
    return df.reset_index(drop=True), kind, label


def _cell_missing(value):
    """Whether one cell of an object column is empty.

    None, NaN, `pd.NA`, `pd.NaT`, and a decimal not-a-number of either kind:
    pandas itself raises on a signalling one, so it is asked about no decimal.
    """
    if isinstance(value, decimal.Decimal):
        return value.is_nan()
    gone = pd.isna(value)                   # an array for a list: a list is a value
    return isinstance(gone, (bool, np.bool_)) and bool(gone)


def _cell_kind(value):
    """What a cell of an object column is, for D51: `bool`, `int`, `real`, or None
    for anything else.

    `int` is a whole number held exactly: a Python or numpy integer, or a decimal
    number with no fractional part, which is how a database driver returns a
    wide whole-number column (D50: such values are never merged). `real` is a
    float, or a decimal number with a fractional part or an infinite one. A
    complex number is not a number here (D48).
    """
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, decimal.Decimal):
        return "int" if value.is_finite() and value == value.to_integral_value() else "real"
    if isinstance(value, (float, np.floating)):
        return "real"
    return None


def _whole(value):
    """A cell of kind `int` as a Python integer. A decimal number of more than 30
    digits is not spelled out: any value that large is beyond 64 bits already."""
    if isinstance(value, decimal.Decimal) and value.adjusted() > 30:
        return (-1 if value < 0 else 1) * 10 ** 31
    return int(value)


def _as_float(value):
    """A number as a 64-bit float. One beyond the largest double is infinite, with its sign."""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _object_numbers(series):
    """D51: one column of generic objects, as the typed column it stands for.

    Returns the column to use in its place, or None to leave it as it is.
    - only booleans: the nullable boolean type;
    - only whole numbers: the nullable 64-bit integer type, signed where every
      value fits and unsigned where every value fits that instead. Where neither
      holds them the column is left alone: it is too large for 64 bits (D50);
    - whole numbers, floats and decimal numbers in any mixture: 64-bit floats;
    - a boolean beside a number, or any other value: left alone.
    An empty cell stays empty throughout.
    """
    cells = series.to_numpy(dtype=object)
    missing = np.fromiter((_cell_missing(v) for v in cells), dtype=bool, count=len(cells))
    kinds = {_cell_kind(v) for v in cells[~missing]}

    def typed(values, dtype):
        return pd.Series(pd.array(values, dtype=dtype), index=series.index, name=series.name)

    if not kinds or None in kinds or (len(kinds) > 1 and "bool" in kinds):
        if missing.any() and any(isinstance(v, decimal.Decimal) for v in cells[missing]):
            # a decimal not-a-number is an empty cell, in a form pandas can ask about
            return pd.Series([None if gone else v for v, gone in zip(cells, missing)],
                             index=series.index, dtype=object, name=series.name)
        return None
    if kinds == {"bool"}:
        return typed([None if gone else bool(v) for v, gone in zip(cells, missing)], "boolean")
    if kinds == {"int"}:
        whole = [None if gone else _whole(v) for v, gone in zip(cells, missing)]
        low = min(v for v in whole if v is not None)
        high = max(v for v in whole if v is not None)
        if -2 ** 63 <= low and high <= 2 ** 63 - 1:
            return typed(whole, "Int64")
        if low >= 0 and high <= 2 ** 64 - 1:
            return typed(whole, "UInt64")
        return None
    numbers = np.array([np.nan if gone else _as_float(v) for v, gone in zip(cells, missing)],
                       dtype="float64")
    return pd.Series(numbers, index=series.index, name=series.name)


def _numbers_from_objects(df):
    """D51, before any step: every object column that holds only numbers, typed.

    Returns the frame to work on and the type of every column as it was loaded,
    which is what the output reports.
    """
    loaded = {name: str(df[name].dtype) for name in df.columns}
    out = df
    for name in df.columns:
        if df[name].dtype != object:
            continue
        typed = _object_numbers(df[name])
        if typed is not None:
            if out is df:
                out = df.copy(deep=False)
            out[name] = typed
    return out, loaded


def _beyond_64_bits(series):
    """Whether a column holds whole numbers too large for 64 bits (D50).

    After `_numbers_from_objects` an object column whose values are all whole
    numbers is exactly that: one that fits has been given an integer type.
    """
    if series.dtype != object:
        return False
    kinds = {_cell_kind(v) for v in series.to_numpy(dtype=object) if not _cell_missing(v)}
    return kinds == {"int"}


def _exact_integers(series):
    """An integer-typed column as exact 64-bit integers: (values, which are missing).

    Unsigned where the type is, signed otherwise; 0 stands in a missing cell.
    """
    missing = series.isna().to_numpy().copy()
    kind = "uint64" if pt.is_unsigned_integer_dtype(series.dtype) else "int64"
    return series.to_numpy(dtype=kind, na_value=0), missing


def _to_nanoseconds(series, name):
    """A datetime column as nanosecond stamps, or the time-column error when a
    date lies outside what nanoseconds since 1970 can hold."""
    try:
        return series.to_numpy("datetime64[ns]")
    except (OverflowError, ValueError) as exc:
        raise E.TimeColumnError(
            f"time column {name!r} holds a date outside the range a nanosecond timestamp can "
            "hold (1677-09-21 to 2262-04-11), so its order cannot be read") from exc


def _time_keys(series, name):
    """Step 3 (D18): sortable keys for the time column. Returns (keys, missing, form, show)."""
    dtype = series.dtype
    if pt.is_datetime64_any_dtype(dtype):
        s = series
        if getattr(dtype, "tz", None) is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        stamps = _to_nanoseconds(s, name)
        return stamps.astype("int64"), np.isnat(stamps), "datetime", _show_stamp
    if pt.is_numeric_dtype(dtype) and not pt.is_bool_dtype(dtype) \
            and not pt.is_complex_dtype(dtype):
        whole = pt.is_integer_dtype(dtype)
        if whole and BROKEN != "F7c":
            # D50, D53: above 2**53 two distinct whole numbers can be one float, so
            # there the keys stay the integers they are. F7c: always floats.
            exact, missing = _exact_integers(series)
            present = exact[~missing]
            if len(present) and (int(present.max()) > EXACT_AS_FLOAT
                                 or int(present.min()) < -EXACT_AS_FLOAT):
                return exact, missing, "numeric", int
        keys = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        return keys, ~np.isfinite(keys), "numeric", (lambda k: int(k) if whole else float(k))
    if _beyond_64_bits(series):
        raise E.TimeColumnError(
            f"time column {name!r} holds whole numbers too large for 64 bits, which cannot be "
            "ordered exactly here; pass the column as 64-bit integers, or as datetimes")
    present = series.dropna()
    if (pt.is_string_dtype(dtype) or dtype == object) and all(isinstance(v, str) for v in present):
        text = present.str.strip()
        ok = text.map(lambda v: ISO_TIME_RE.fullmatch(v) is not None)
        if not ok.all() and BROKEN == "F7b":
            # Parsed leniently, value by value: 01/12/2023 becomes 12 January and
            # 13/12/2023 becomes 13 December, so the rows are in the wrong order.
            with _pywarnings.catch_warnings():
                _pywarnings.simplefilter("ignore")
                parsed = pd.to_datetime(text, format="mixed", dayfirst=False)
            stamps = np.full(len(series), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
            stamps[series.notna().to_numpy()] = parsed.to_numpy("datetime64[ns]")
            return stamps.astype("int64"), np.isnat(stamps), "iso8601", _show_stamp
        if not ok.all():
            first = text[~ok].iloc[0]
            if ISO_LONG_FRACTION_RE.fullmatch(first):
                # ISO 8601 but for the length of its fraction: say that, and nothing
                # about the order of day and month (D52)
                raise E.TimeColumnError(
                    f"time column {name!r} holds a value with more than nine digits in its "
                    f"fraction of a second (for example {first!r}). At most nine are read, "
                    "which is a nanosecond: shorten the fraction, or parse the column "
                    "yourself and pass a DataFrame")
            raise E.TimeColumnError(
                f"time column {name!r} holds text that is not ISO 8601 with the year first "
                f"(for example {first!r}). Day-first and month-first dates are "
                "never guessed: parse the column yourself and pass a DataFrame")
        zoned = text.map(lambda v: ISO_TZ_RE.fullmatch(v) is not None)
        if zoned.any() and not zoned.all():
            raise E.TimeColumnError(f"time column {name!r} mixes values with and without a "
                                    "time zone offset")
        try:
            parsed = pd.to_datetime(text, format="ISO8601", utc=bool(zoned.all()))
        except (ValueError, TypeError) as exc:
            raise E.TimeColumnError(f"time column {name!r} holds an invalid date: {exc}") from exc
        if zoned.all():
            parsed = parsed.dt.tz_localize(None)
        stamps = np.full(len(series), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        stamps[series.notna().to_numpy()] = _to_nanoseconds(parsed, name)
        return stamps.astype("int64"), np.isnat(stamps), "iso8601", _show_stamp
    raise E.TimeColumnError(
        f"time column {name!r} has type {dtype}; accepted forms are a datetime column, a "
        "numeric column, or text in ISO 8601 form with the year first")


def _show_stamp(key):
    return pd.Timestamp(int(key)).isoformat()


def _is_numeric(dtype):
    """Integer or float type. Boolean is not numeric here, and nor is complex (D48):
    turning a complex number into a float would drop its imaginary part in silence."""
    return pt.is_numeric_dtype(dtype) and not pt.is_bool_dtype(dtype) \
        and not pt.is_complex_dtype(dtype)


def _target_values(series):
    """The target as every step reads it.

    A text value is taken without the white space round it, so `yes` and `yes `
    are one class (D46). A text value that holds only white space is a missing
    target, like an empty cell (D48). A value that is not text is as it is.
    """
    cells = series.to_numpy(dtype=object)
    if not any(isinstance(v, str) for v in cells):
        return series
    if pt.is_string_dtype(series.dtype) and not isinstance(series.dtype, pd.CategoricalDtype) \
            and series.dtype != object:
        stripped = series.str.strip()
        return stripped.mask(stripped == "")
    read = [v.strip() if isinstance(v, str) else v for v in cells]
    return pd.Series([None if isinstance(v, str) and v == "" else v for v in read],
                     index=series.index, dtype=object, name=series.name)


def _decide_task(y, name, requested):
    """Step 4 (D4, D21): classification or regression, and how that was decided."""
    if _beyond_64_bits(y):
        raise E.TaskUnsupportedError(
            f"target {name!r} holds whole numbers too large for 64 bits, which are not read "
            "as numbers here; pass them as text if they are labels (D50)")
    try:
        n_distinct = int(y.nunique())
    except TypeError as exc:
        # lists, dictionaries, sets: values that cannot be compared for equality (D53)
        kind = type(y.dropna().iloc[0]).__name__ if y.notna().any() else str(y.dtype)
        raise E.TaskUnsupportedError(
            f"target {name!r} holds values that cannot be compared for equality ({kind}), so "
            "its distinct values cannot be counted") from exc
    numeric = _is_numeric(y.dtype)
    if n_distinct < 2:
        raise E.TargetConstantError(f"target {name!r} has {n_distinct} distinct value(s)")
    if requested == "auto":
        if BROKEN == "F5":
            kind = "regression" if numeric else "classification"
        elif n_distinct == 2:
            kind = "classification"
        elif numeric and n_distinct > 20:
            kind = "regression"
        else:
            raise E.TaskAmbiguousError(
                f"target {name!r} has {n_distinct} distinct values"
                f"{'' if numeric else ' and is not numeric'}: the rule cannot decide. "
                "Declare task='classification' or task='regression'")
        return kind, "rule", n_distinct, numeric
    if requested == "classification" and n_distinct != 2:
        raise E.TaskUnsupportedError(f"classification needs exactly two classes; target "
                                     f"{name!r} has {n_distinct}")
    if requested == "regression" and not numeric:
        raise E.TaskUnsupportedError(f"regression needs a numeric target; {name!r} is {y.dtype}")
    return requested, "declared", n_distinct, numeric


def _no_target(series):
    """Step 2: which target cells are missing or, for a numeric target, infinite.

    `series` is the target as `_target_values` reads it, so a text cell of only
    white space is already missing here.
    """
    gone = series.isna().to_numpy().copy()
    if _is_numeric(series.dtype):
        gone |= ~np.isfinite(series.to_numpy(dtype="float64", na_value=np.nan))
    return gone


def _classes(values):
    """The two classes in sorted order; the second is the positive class."""
    distinct = list(pd.unique(values))
    try:
        return sorted(distinct)
    except TypeError:
        return sorted(distinct, key=str)


def _whole_number_id(x):
    """D19: every value distinct, and consecutive or strictly ordered in file order.

    `x` holds the present values in file order: floats, or the exact integers of
    an integer-typed column (D50). Neighbours are compared, never subtracted, and
    the range is taken in Python integers, so nothing here can overflow.
    """
    if len(x) < 2 or len(np.unique(x)) != len(x):
        return False
    if x.dtype.kind == "f" and not np.all(x == np.floor(x)):
        return False
    if np.all(x[1:] > x[:-1]) or np.all(x[1:] < x[:-1]):
        return True
    return int(x.max()) - int(x.min()) + 1 == len(x)


def _separated(text):
    """D28: whether a stripped text value is a number written with thousands separators."""
    return SEPARATED_RE.fullmatch(text) is not None


def _roles(df, usable, target, time_col, loaded=None):
    """Step 5 (D19, D20, D28, D34): roles that need no target, judged on the usable rows.

    Returns the column records and, for each feature, its prepared values over
    every loaded row: float64 for numeric, object for categorical. `df` is the
    table after `_numbers_from_objects`; `loaded` gives each column's type as it
    was loaded, which is the type reported (D51).
    """
    records, prepared = [], {}
    n_rows = int(usable.sum())
    for position, name in enumerate(df.columns):
        series = df[name]
        rec = dict(name=name, position=position,
                   dtype=loaded[name] if loaded else str(series.dtype), role="feature",
                   kind=None, reason=None, leak_rule=None, n_missing=int(series.isna().sum()),
                   n_coerced=0, n_infinite=0, n_imputed=0)
        records.append(rec)
        if name == target or name == time_col:
            rec["role"] = "target" if name == target else "time"
            continue
        dtype, numbers, levels = series.dtype, None, None
        if isinstance(dtype, pd.CategoricalDtype):      # treated as text, by string form
            series = pd.Series([v if pd.isna(v) else str(v) for v in series],
                               index=series.index, dtype=object)
            dtype = series.dtype
        present = series[usable].dropna()
        if len(present) == 0:
            rec.update(role="excluded", reason="all_missing")
        elif pt.is_datetime64_any_dtype(dtype):
            rec.update(role="excluded", reason="datetime", kind="datetime")
        elif pt.is_bool_dtype(dtype) or (pt.is_numeric_dtype(dtype)
                                         and not pt.is_complex_dtype(dtype)):
            numbers = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64",
                                                                      na_value=np.nan)
        elif (pt.is_string_dtype(dtype) or dtype == object) \
                and all(isinstance(v, str) for v in present):
            text = present.str.strip()
            parses = pd.to_numeric(text, errors="coerce").notna().to_numpy(dtype=bool)
            separated = text.map(_separated).to_numpy(dtype=bool)
            distinct = int(text.nunique())          # text is compared after stripping
            if text.map(lambda v: DATE_RE.fullmatch(v) is not None).mean() >= DATE_SHARE:
                rec.update(role="excluded", reason="datetime", kind="datetime")
            elif distinct > 1 and separated.any() \
                    and (parses | separated).mean() >= SEPARATED_SHARE:
                # D34: never parsed, and never used with those cells left out. One
                # such cell is enough: the cells lost would be the largest values.
                rec.update(role="excluded", reason="numeric_with_separators", kind="text")
            elif parses.mean() >= NUMERIC_SHARE:
                numbers = pd.to_numeric(series.str.strip(), errors="coerce") \
                    .to_numpy(dtype="float64", na_value=np.nan)
                rec["n_coerced"] = int((series.notna().to_numpy() & np.isnan(numbers)).sum())
            else:
                if distinct == 1:
                    rec.update(role="excluded", reason="constant", kind="categorical")
                elif distinct > ID_TEXT_SHARE * n_rows:
                    rec.update(role="excluded", reason="id_like", kind="text")
                elif distinct > FREE_TEXT_SHARE * len(present):
                    rec.update(role="excluded", reason="free_text", kind="text")
                else:
                    # A level is the cell's text without the white space round it (D42):
                    # here, so that the screen, the steps and the matrix all see it so.
                    levels = series.str.strip().astype(object) \
                        .where(series.notna(), MISSING_LEVEL).to_numpy()
                    rec["kind"] = "categorical"
        else:
            rec.update(role="excluded", reason="unsupported_type")
        if numbers is not None:
            rec["kind"] = "numeric"
            # Infinite, or finite and too large for the 32-bit floats the tree models
            # read: one rule, by value. Read as missing, and counted (D48).
            infinite = np.abs(numbers) > TOO_LARGE
            rec["n_infinite"] = int(infinite.sum())
            numbers = np.where(infinite, np.nan, numbers)
            seen = numbers[usable]
            seen = seen[~np.isnan(seen)]
            if pt.is_integer_dtype(dtype):
                # D50: the distinct values of a whole-number column are counted on
                # the integers themselves. As floats, integers above 2**53 merge:
                # fifty distinct ones can count as one, and the column as constant.
                exact, gone = _exact_integers(series)
                seen = exact[usable & ~gone]
            distinct = len(np.unique(seen))
            if distinct == 0:
                rec.update(role="excluded", reason="all_missing")
            elif distinct == 1:
                rec.update(role="excluded", reason="constant")
            elif _whole_number_id(seen):
                rec.update(role="excluded", reason="id_like")
            else:
                prepared[name] = numbers
        elif levels is not None:
            prepared[name] = levels
    if BROKEN == "F6b":                     # a constant column vanishes from the accounting
        records = [r for r in records if r["reason"] != "constant"]
    return records, prepared


def _split_time(keys, gap):
    """Step 7 under time order (D7). `keys` is sorted. Returns index arrays."""
    stamps, first, counts = np.unique(keys, return_index=True, return_counts=True)
    n, total = len(keys), len(stamps)
    h, held = total, 0
    while h > 0 and 4 * held < n:           # latest whole timestamps holding >= 25%
        h -= 1
        held += counts[h]
    if h < N_BLOCKS:
        raise E.TooFewRowsError(f"{n} rows hold {total} distinct timestamps, too few to form "
                                "the holdout and six training blocks")
    if h - gap < N_BLOCKS:
        raise E.GapError(f"gap={gap} leaves {max(h - gap, 0)} training timestamps; six blocks "
                         "need at least six")
    stamp_of_row = np.repeat(np.arange(total), counts)
    blocks = np.array_split(np.arange(h - gap), N_BLOCKS)
    folds = []
    for k in range(1, N_BLOCKS):
        start = blocks[k][0]
        if start - gap < 1:
            raise E.GapError(f"gap={gap} leaves no training rows before validation block {k + 1}")
        folds.append({
            "train": np.flatnonzero(stamp_of_row < start - gap),
            "valid": np.flatnonzero((stamp_of_row >= start) & (stamp_of_row <= blocks[k][-1])),
            "gap_rows": int(((stamp_of_row >= start - gap) & (stamp_of_row < start)).sum()),
        })
    return {
        "train": np.flatnonzero(stamp_of_row < h - gap),
        "gap": np.flatnonzero((stamp_of_row >= h - gap) & (stamp_of_row < h)),
        "holdout": np.flatnonzero(stamp_of_row >= h),
        "folds": folds, "stamp_of_row": stamp_of_row,
        "train_end": stamps[h - gap - 1], "holdout_start": stamps[h], "holdout_end": stamps[-1],
        "n_train_stamps": int(h - gap), "n_holdout_stamps": int(total - h),
        "unique": bool(counts.max() == 1),
    }


def _fast_profile_stop(all_keys, gap, found):
    """The error for a fast run whose capped rows cannot form the split (D20, D46).

    `all_keys` are the keys of every usable row, in time order: what the full
    profile would split. If that split can be formed, with the caller's gap, the
    fast profile's own error tells the caller to use `profile="full"`, and the
    advice is true. If it cannot, the error is the one the full profile would
    raise, so that nobody is sent to a profile that stops too.
    """
    if BROKEN == "M7":
        return E.TooFewRowsError(f"{found}")
    try:
        _split_time(all_keys, gap)
    except (E.TooFewRowsError, E.GapError) as exc:
        return type(exc)(f"{exc} (the fast profile kept fewer rows still: {found})")
    return E.FastProfileError(f"{found}. Every usable row together can form the split: "
                              "use profile='full'")


def _fast_rows_stop(found, y_usable, classification, all_keys, gap):
    """The error for a fast run whose capped rows are too few to profile (D52, D53).

    The rows the cap kept are fewer than `MIN_ROWS`, or hold fewer than
    `MIN_CLASS_ROWS` of a class. `y_usable` and `all_keys` are the target and the
    time keys of every usable row: what the full profile would work on. If the
    full profile gets past this step and can form its split, the error is the
    fast profile's own, with a code that names rows, and its advice is true.
    If it cannot, the error is the one the full profile would raise.
    """
    if classification and min(int(y_usable.sum()), int(len(y_usable) - y_usable.sum())) \
            < MIN_CLASS_ROWS:
        return E.TooFewRowsError(f"the rarer class has fewer than {MIN_CLASS_ROWS} usable rows "
                                 f"(the fast profile kept fewer rows still: {found})")
    if all_keys is not None:
        try:
            _split_time(all_keys, gap)
        except (E.TooFewRowsError, E.GapError) as exc:
            return type(exc)(f"{exc} (the fast profile kept fewer rows still: {found})")
    return E.FastProfileRowsError(
        f"{found}. Every usable row together is enough to profile: use profile='full'")


def _holdout_copies(columns, y, train, holdout):
    """D49: how many holdout rows have a copy among the training rows.

    A copy is a training row with the same value in every feature column and the
    same target. `columns` are the features left after the leak screen, as a
    fitted step is handed them: 64-bit floats for a numeric column, where two
    empty cells are equal and 0.0 equals -0.0; levels for a categorical column,
    without the white space round them, an empty cell being the level
    `(missing)`. `y` is the target as step 4 reads it.
    """
    used = np.concatenate([train, holdout])
    code = np.zeros(len(used), dtype="int64")
    for values in list(columns) + [y]:
        cell = values[used]
        if cell.dtype == object:
            cell = cell.astype(str)
        # numpy counts every not-a-number as one value, and 0.0 and -0.0 as one
        _, part = np.unique(cell, return_inverse=True)
        # one number per distinct pair so far; renumbered each time, so it stays small
        _, code = np.unique(code * (int(part.max()) + 1) + part, return_inverse=True)
    return int(np.isin(code[len(train):], code[:len(train)]).sum())


def _split_random(n, y, classification, seed):
    """Step 7 with no time column (D6)."""
    strat = y if classification else None
    n_holdout = -(-n // 4)
    train, holdout = train_test_split(np.arange(n), test_size=n_holdout, random_state=seed,
                                      shuffle=True, stratify=strat)
    train, holdout = np.sort(train), np.sort(holdout)
    if classification:
        cuts = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(train, y[train])
    else:
        cuts = KFold(N_FOLDS, shuffle=True, random_state=seed).split(train)
    folds = [{"train": train[a], "valid": train[b], "gap_rows": 0} for a, b in cuts]
    return {"train": train, "gap": np.array([], dtype=int), "holdout": holdout, "folds": folds}


def _collisions(kinds, levels_of):
    """D37: names the preprocessed matrix would hold twice -> the columns that give them.

    A numeric column gives its own name; a categorical column gives
    `<column>=<level>` for each level in `levels_of[column]`.
    """
    givers = {}
    for column, kind in kinds.items():
        names = [column] if kind == "numeric" else [f"{column}={lv}" for lv in levels_of[column]]
        for name in names:
            givers.setdefault(name, []).append(column)
    return {name: columns for name, columns in givers.items() if len(columns) > 1}


def _collision_error(clashes):
    """The named error for `_collisions`, naming every column involved."""
    involved = []
    for columns in clashes.values():
        involved += [c for c in columns if c not in involved]
    shown = "; ".join(f"{name!r} from columns " + " and ".join(repr(c) for c in columns)
                      for name, columns in list(clashes.items())[:5])
    return E.FeatureNameCollisionError(
        "two columns of the preprocessed matrix would have the same name: " + shown
        + ". A numeric column keeps its own name and a level of a categorical column is named "
          "<column>=<level>. Rename one of: " + ", ".join(repr(c) for c in involved))


class _Prep:
    """Step 8: imputation, scaling and encoding, fitted on the frame it is given."""

    def __init__(self, kinds):
        self.kinds = dict(kinds)
        self.numeric = [c for c, k in kinds.items() if k == "numeric"]
        self.categorical = [c for c, k in kinds.items() if k == "categorical"]

    def _parts(self, frame):
        num = frame[self.numeric].to_numpy(dtype="float64", na_value=np.nan) if self.numeric \
            else np.empty((len(frame), 0))
        # An infinite cell and a too-large one are missing cells (D48). Every value
        # left is within the 32-bit range, so the mean and the deviation learned
        # below are always finite: there is no "deviation that is not finite" case.
        num = np.where(np.abs(num) > TOO_LARGE, np.nan, num)
        cats = [frame[c].astype(object).where(frame[c].notna(), MISSING_LEVEL).to_numpy()
                for c in self.categorical]
        return num, cats

    def fit(self, frame, stats_frame=None, level_values=None):
        """Learn every statistic from `frame`.

        The other two arguments exist only for the broken variants M1 and M2:
        rows to learn the numeric statistics from, and values to learn each
        categorical column's levels from, in place of the rows of `frame`.
        """
        num, cats = self._parts(frame)
        if stats_frame is not None:
            num = self._parts(stats_frame)[0]
        if level_values is not None:
            cats = level_values
        with _pywarnings.catch_warnings():
            _pywarnings.simplefilter("ignore", RuntimeWarning)
            self.median = np.nanmedian(num, axis=0) if num.shape[1] else np.empty(0)
        self.median = np.where(np.isnan(self.median), 0.0, self.median)
        filled = np.where(np.isnan(num), self.median, num)
        self.mean = filled.mean(axis=0)
        self.scale = filled.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        self.levels = []
        for values in cats:
            names, counts = np.unique(values.astype(str), return_counts=True)
            order = np.lexsort((names, -counts))[:MAX_LEVELS]
            self.levels.append(sorted(names[order]))
        clashes = _collisions(self.kinds, dict(zip(self.categorical, self.levels)))
        if clashes:                                 # never two columns under one name (D37)
            raise _collision_error(clashes)
        return self

    def transform(self, frame):
        num, cats = self._parts(frame)
        parts = [(np.where(np.isnan(num), self.median, num) - self.mean) / self.scale]
        for values, levels in zip(cats, self.levels):
            text = values.astype(str)
            parts.append(np.column_stack([(text == lv).astype("float64") for lv in levels]))
        return np.hstack(parts)

    def names(self):
        out = list(self.numeric)
        for name, levels in zip(self.categorical, self.levels):
            out += [f"{name}={lv}" for lv in levels]
        return out


def _families(profile, classification):
    """D8, D16: families in tie-break order, each with its grid (default first)."""
    custom = extensions.active()["families"]
    if custom is not None:
        return [dict(f, custom=True) for f in custom]

    def linear(task, params, seed, n_jobs):
        if classification:
            return LogisticRegression(C=params["C"], max_iter=1000)
        return Ridge(alpha=params["alpha"])

    def tree(task, params, seed, n_jobs):
        cls = DecisionTreeClassifier if classification else DecisionTreeRegressor
        return cls(random_state=seed)

    def forest(task, params, seed, n_jobs):
        cls = RandomForestClassifier if classification else RandomForestRegressor
        return cls(n_estimators=100, random_state=seed, n_jobs=n_jobs, **params)

    def boosting(task, params, seed, n_jobs):
        cls = HistGradientBoostingClassifier if classification \
            else HistGradientBoostingRegressor
        return cls(early_stopping=False, random_state=seed, **params)

    linear_grid = [{"C": c} for c in LINEAR_C] if classification \
        else [{"alpha": a} for a in LINEAR_ALPHA]
    if profile == "fast":
        return [{"name": "linear", "grid": linear_grid[:1], "make": linear},
                {"name": "tree", "grid": [{}], "make": tree}]
    first, second = ("sqrt", THIRD) if classification else (THIRD, "sqrt")
    forest_grid = [{"min_samples_leaf": leaf, "max_features": mf}
                   for leaf in FOREST_LEAF for mf in (first, second)]
    boosting_grid = [{"max_leaf_nodes": a, "learning_rate": b, "max_iter": c}
                     for a, b, c in BOOSTING_GRID]
    return [{"name": "linear", "grid": linear_grid, "make": linear},
            {"name": "forest", "grid": forest_grid, "make": forest},
            {"name": "boosting", "grid": boosting_grid, "make": boosting}]


def _fit_predict(family, model, x_fit, y_fit, x_new, classification, frames=None):
    """Fit on one set of rows, predict another. Forests predict on one thread."""
    if family.get("custom"):
        fit_frame, new_frame, y_series = frames
        model.fit(fit_frame, y_series)
        x_new = new_frame
    else:
        model.fit(x_fit, y_fit)
        if hasattr(model, "estimators_") and hasattr(model, "n_jobs"):
            model.set_params(n_jobs=1)      # summing trees across threads is not order-stable
    if classification:
        return np.asarray(model.predict_proba(x_new))[:, 1].astype("float64")
    return np.asarray(model.predict(x_new)).astype("float64")


def _auc(y, score, ties_share=True):
    """ROC AUC by ranks, ties averaged. None when only one class is present.

    `ties_share=False` exists only for the broken variant M12: tied scores get
    ranks in the order they come, so a tie counts for whichever row came first.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    if ties_share:
        ranks = _ranks(score)
    else:
        ranks = np.empty(len(score))
        ranks[np.argsort(score, kind="mergesort")] = np.arange(1, len(score) + 1)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _ranks(score):
    order = np.argsort(score, kind="mergesort")
    ordered = score[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    counts = np.diff(np.r_[starts, len(score)])
    ranks = np.empty(len(score))
    ranks[order] = np.repeat(starts + 1 + (counts - 1) / 2, counts)
    return ranks


def _mse(y, pred):
    """Mean squared error. It can overflow to infinity; the caller decides what that means."""
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.mean((y - pred) ** 2))


def _mean(values):
    """The mean, with numpy's overflow warning kept off standard error: a result
    that is not finite is dealt with by the caller, under its own name (D44)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return values.mean()


def _mean_and_spread(scores):
    """The mean and the sample standard deviation of finite fold scores.

    Computed as numpy computes them. Where that overflows although the answer can
    be held (fold scores above about 1e154, whose squares cannot), the scores are
    divided by the largest first, so that a number that exists is reported (D44).
    """
    a = np.asarray(scores, dtype="float64")
    ddof = 0 if BROKEN == "M11" else 1      # M11: the population deviation
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.mean(a))
        spread = float(np.std(a, ddof=ddof)) if len(a) > 1 else None
        scale = float(np.max(np.abs(a)))
        if scale > 0 and not math.isfinite(mean):
            mean = float(np.mean(a / scale) * scale)
        if scale > 0 and spread is not None and not math.isfinite(spread):
            spread = float(np.std(a / scale, ddof=ddof) * scale)
    return mean, spread


def _require_finite(doc):
    """D44, the backstop: no number of the output may be infinite or not-a-number.

    The rules of `_run` stop the run, or fail a configuration, where such a number
    first appears. This looks at every number about to be written, so that none
    can leave as a null, whatever arithmetic produced it.
    """
    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                label = value.get("config", value.get("name")) if isinstance(value, dict) else None
                walk(value, f"{path}[{label if label is not None else i}]")
        elif isinstance(node, (float, np.floating)) and not math.isfinite(node):
            raise _not_finite(f"`{path}`", node, "values too large for the arithmetic behind "
                              "this number: " + TARGET_CAUSE)
    for part in ("validation", "headline", "baselines", "cv", "models"):
        walk(doc[part], part)


def _previous_of(prev, rows, where):
    """The previous value of each of `rows`. Every one exists: step 7 leaves at least
    `gap + 1` timestamps before the first validation block, or stops with
    `gap_too_large`. One that is missing is a defect here, never a reason to fall
    back to the training mean."""
    before = prev[rows]
    if np.isnan(before).any():
        raise RuntimeError(f"a row of {where} has no previous value: the split's own rule "
                           "excludes this, so the split or the baseline is wrong")
    return before


def _screen_column(values, numeric, y, folds, classification, seed, prev, also_fit=None):
    """D11: one column's cross-validated skill over the selection folds.

    None when it cannot be computed: the baseline's error over the pooled
    validation rows is zero (D36). A baseline's error, or the column's own, that
    is not finite stops the run (D44). `also_fit` exists only for the broken
    variant M3: rows added to every fold's training rows for the tree.
    """
    cls = DecisionTreeClassifier if classification else DecisionTreeRegressor
    truth, guess, mean_guess, rows = [], [], [], []
    for fold in folds:
        if not fold["scored"]:
            continue
        a, b = fold["train"], fold["valid"]
        fold_mean = _mean(y[a])
        if also_fit is not None:
            a = np.concatenate([a, also_fit])
        if numeric:
            xa, xb = values[a].reshape(-1, 1), values[b].reshape(-1, 1)
        else:
            levels = {lv: i for i, lv in enumerate(sorted(set(values[a].astype(str))))}
            xa = np.array([levels[v] for v in values[a].astype(str)], dtype="float64").reshape(-1, 1)
            xb = np.array([levels.get(v, -1) for v in values[b].astype(str)],
                          dtype="float64").reshape(-1, 1)
        model = cls(max_depth=6, min_samples_leaf=5, random_state=seed).fit(xa, y[a])
        guess.append(model.predict_proba(xb)[:, 1] if classification else model.predict(xb))
        truth.append(y[b])
        mean_guess.append(np.full(len(b), fold_mean))
        rows.append(b)
    truth, guess = np.concatenate(truth), np.concatenate(guess)
    if classification:
        auc = _auc(truth, guess)
        return None if auc is None else 2 * auc - 1
    bases = [("training_mean", _mse(truth, np.concatenate(mean_guess)))]
    if prev is not None:
        before = _previous_of(prev, np.concatenate(rows), "a validation block")
        bases.append(("previous_value", _mse(truth, before)))
    for name, error in bases:
        if not math.isfinite(error):
            raise _not_finite(f"In the leak screen, the squared error of the baseline {name} "
                              "over the validation rows of the selection folds", error,
                              TARGET_CAUSE)
    own = _mse(truth, guess)
    if not math.isfinite(own):
        raise _not_finite("In the leak screen, the squared error of a single-column tree over "
                          "the validation rows of the selection folds", own, TARGET_CAUSE)
    base = min(error for _, error in bases)
    return 1 - own / base if base > 0 else None


def _cube_root_up(n):
    """Smallest whole L with L**3 >= n."""
    length = max(1, round(n ** (1 / 3)))
    while length ** 3 < n:
        length += 1
    while length > 1 and (length - 1) ** 3 >= n:
        length -= 1
    return length


def _uncertainty(y, pred, base_pred, classification, time_groups, n_resamples, seed, score,
                 skill_defined=True):
    """Step 12 (D9): interval on the holdout score and skill, and the p-value."""
    out = dict(interval_method=None, interval_level=0.95, interval_low=None, interval_high=None,
               skill_low=None, skill_high=None, n_resamples=n_resamples, n_resamples_used=0,
               n_resamples_skill_used=0, block_length=None, interval_reason=None, p_value=None,
               n_permutations=None, p_value_reason=None)

    def measure(rows):
        """(score, skill) of one resample. None when the score cannot be computed (one
        class); a skill of None when only the skill cannot (a zero baseline error: D36)."""
        if classification:
            # M12: in a resample, tied predictions do not share their rank
            auc = _auc(y[rows], pred[rows], ties_share=BROKEN != "M12")
            return None if auc is None else (auc, 2 * auc - 1)
        mse, base = _mse(y[rows], pred[rows]), _mse(y[rows], base_pred[rows])
        return mse, (1 - mse / base if base > 0 else None)

    rng = np.random.default_rng([seed, 5])
    n = len(y)
    if time_groups is None:
        draws = (rng.integers(0, n, n) for _ in range(n_resamples))
        out["interval_method"] = "bootstrap_rows"
    elif len(time_groups) < MIN_BLOCK_TIMESTAMPS:
        draws = ()
        out["interval_reason"] = (f"the holdout has {len(time_groups)} distinct timestamps; a "
                                  f"block bootstrap needs at least {MIN_BLOCK_TIMESTAMPS}")
    else:
        total = len(time_groups)
        length = _cube_root_up(total)
        n_blocks = -(-total // length)
        out.update(interval_method="bootstrap_blocks", block_length=length)

        def blocks():
            for _ in range(n_resamples):
                if BROKEN == "M4":              # timestamps one by one: no blocks at all
                    picked = rng.integers(0, total, total)
                else:
                    starts = rng.integers(0, total - length + 1, n_blocks)
                    picked = np.concatenate([np.arange(s, s + length) for s in starts])[:total]
                yield np.concatenate([time_groups[t] for t in picked])
        draws = blocks()
    results = [m for m in (measure(rows) for rows in draws) if m is not None]
    scores = np.array([m[0] for m in results], dtype="float64")
    # A resample with a zero baseline error is left out of the skill interval only (D40).
    skills = np.array([m[1] for m in results if m[1] is not None], dtype="float64")
    out["n_resamples_used"], out["n_resamples_skill_used"] = len(scores), len(skills)
    # M8: the 5th and 95th percentiles, under a level that still says 0.95
    ends = [5.0, 95.0] if BROKEN == "M8" else [2.5, 97.5]
    if out["interval_method"] and len(scores) * 2 >= n_resamples:
        out["interval_low"], out["interval_high"] = (float(v) for v in
                                                     np.percentile(scores, ends))
        if len(skills) * 2 >= n_resamples:
            out["skill_low"], out["skill_high"] = (float(v) for v in
                                                   np.percentile(skills, ends))
    elif out["interval_method"]:
        out.update(interval_method=None, block_length=None,
                   interval_reason="the score could be computed in fewer than half of the "
                                   "resamples: the others held a single class")
    if time_groups is not None:
        out["p_value_reason"] = NO_P_VALUE
        return out
    if not skill_defined:                   # nothing to compare the shuffles with (D36)
        out["p_value_reason"] = NO_P_VALUE_NO_SKILL
        return out
    rng = np.random.default_rng([seed, 6])
    if classification:
        ranks, n_pos = _ranks(pred), int(y.sum())
        floor = n_pos * (n_pos + 1) / 2
        scale = n_pos * (n - n_pos)
        shuffled = ((ranks[rng.permutation(y) == 1].sum() - floor) / scale
                    for _ in range(n_resamples))
        # at least the observed skill; M5 counts only those strictly above it
        hits = sum(s > score if BROKEN == "M5" else s >= score for s in shuffled)
    else:
        shuffled = (_mse(rng.permutation(y), pred) for _ in range(n_resamples))
        hits = sum(s < score if BROKEN == "M5" else s <= score for s in shuffled)
    out.update(p_value=(1 + int(hits)) / (1 + n_resamples), n_permutations=n_resamples)
    return out


def _plain(value):
    """A class value as it is written: a boolean, a number or text (D53).

    Anything that is none of the three (bytes, a date, a period, an interval, a
    complex number, a tuple) is written as its text form.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return str(value)


# --------------------------------------------------------------------- run --

def _run(data, target, requested, time_col, gap, profile, seed, n_jobs, duplicates="error"):
    warnings = []
    # 1. load; count the rows
    df, source_kind, source_label = _load(data, warnings)
    if BROKEN == "F6":
        df = df.dropna().reset_index(drop=True)     # rows vanish before anyone counts them
    for name in (target, time_col):
        if name is not None and name not in df.columns:
            raise E.ColumnNotFoundError(f"no column named {name!r}")
    n_loaded = len(df)
    try:
        duplicate_rows = int(df.duplicated().sum()) if n_loaded else 0
    except (TypeError, ArithmeticError):            # cells that cannot be hashed or compared
        duplicate_rows = 0
    if duplicate_rows:
        _warn(warnings, "duplicate_rows",
              f"{duplicate_rows} row(s) are exact copies of an earlier row; they are kept")
    # An object column that holds only numbers is read as numbers from here on, in
    # every role (D51). The output goes on reporting the type it was loaded with.
    df, loaded = _numbers_from_objects(df)

    # 2. drop rows with a missing target, then rows with a missing time value. A text
    # target is read without the white space round it from here on, and a cell that
    # holds nothing else is a missing target (D46, D48).
    target_dtype = loaded[target]
    y_all = _target_values(df[target])
    no_target = _no_target(y_all)
    time_form, show = None, None
    no_time = np.zeros(n_loaded, dtype=bool)
    if time_col is not None:
        # 3. read the time column in its accepted form
        keys_all, missing_time, time_form, show = _time_keys(df[time_col], time_col)
        no_time = missing_time & ~no_target
    usable = ~no_target & ~no_time
    if usable.sum() < MIN_ROWS:
        raise E.TooFewRowsError(f"{int(usable.sum())} usable rows; at least {MIN_ROWS} are needed")

    # 4. decide the task
    task, decided_by, n_distinct, numeric_target = _decide_task(y_all[usable], target, requested)
    classification = task == "classification"
    if classification:
        classes = _classes(y_all[usable])
        # a missing cell of a nullable type compares as missing, not as False
        y_work_all = (y_all == classes[1]).fillna(False).to_numpy(dtype=bool).astype("int64")
    else:
        classes = None
        # The target is of integer or float type here (step 4), so every usable value
        # is a number as it stands: nothing is coerced, and nothing can be.
        y_work_all = y_all.to_numpy(dtype="float64", na_value=np.nan)
        whole = y_work_all[usable]
        if decided_by == "rule" and np.all(whole == np.floor(whole)):
            _warn(warnings, "whole_number_regression_target",
                  f"target {target!r} holds only whole numbers ({n_distinct} distinct) and was "
                  "read as a continuous target; declare the task if it is a code or an identifier",
                  target)

    # 5. column roles that need no target
    records, prepared = _roles(df, usable, target, time_col, loaded)
    by_name = {r["name"]: r for r in records}
    for rec in records:
        if rec["reason"] == "unsupported_type":
            # never excluded in silence (D51): the caller may have meant it as a feature
            if _beyond_64_bits(df[rec["name"]]):
                why = "holds whole numbers too large for 64 bits (D50)"
            else:
                held = df[rec["name"]].dropna()
                example = f"; for example a value of type {type(held.iloc[0]).__name__}" \
                    if len(held) and rec["dtype"] == "object" else ""
                why = f"has a type that is not read (stored as {rec['dtype']}{example})"
            _warn(warnings, "unsupported_type",
                  f"column {rec['name']!r} {why} and is excluded. Convert it to numbers, "
                  "text or dates if it should be a feature", rec["name"])
        if rec["n_coerced"]:
            _warn(warnings, "values_coerced",
                  f"{rec['n_coerced']} text value(s) in numeric column {rec['name']!r} could not "
                  "be read as numbers and are treated as missing", rec["name"])
        if rec["reason"] == "numeric_with_separators":
            _warn(warnings, "numeric_with_separators",
                  f"column {rec['name']!r} holds numbers written with thousands separators "
                  "(such as 1,200) and is excluded. Such a value is never read as a number, "
                  "because the same text means 1.2 in some locales, and a column used without "
                  "those cells would be missing its largest values: parse the column and pass "
                  "a DataFrame", rec["name"])
        if rec["n_infinite"]:
            _warn(warnings, "infinite_values",
                  f"{rec['n_infinite']} value(s) in {rec['name']!r} are infinite, or too large "
                  f"for the tree models (a magnitude above {TOO_LARGE!r}, the largest 32-bit "
                  "float), and are treated as missing", rec["name"])
        if rec["reason"] == "datetime" and time_col is None:
            _warn(warnings, "date_column_undeclared",
                  f"column {rec['name']!r} holds dates but no time_col was declared, so the "
                  "rows were split at random; " + RANDOM_CAVEAT, rec["name"])
    # Two features may not give the matrix one name (D37). Judged here, on every
    # level of the usable rows, so the answer does not depend on the cap, the
    # seed or the rows a fit happens to see.
    clashes = _collisions(
        {name: by_name[name]["kind"] for name in prepared},
        {name: sorted(set(values[usable].astype(str))) for name, values in prepared.items()
         if by_name[name]["kind"] == "categorical"})
    if clashes:
        raise _collision_error(clashes)

    # 6. the fast profile's row cap; rows are put in time order here
    rows = np.flatnonzero(usable)
    if time_col is not None:
        rows = rows[np.argsort(keys_all[rows], kind="stable")]
    capped = 0
    usable_rows = rows                      # every usable row: what the full profile splits
    if profile == "fast" and len(rows) > FAST_ROW_CAP:
        if time_col is not None:
            stamps, counts = np.unique(keys_all[rows], return_counts=True)
            kept, t = 0, len(stamps)
            while t > 0 and kept + counts[t - 1] <= FAST_ROW_CAP:
                t -= 1
                kept += counts[t]
            if kept == 0:
                raise _fast_profile_stop(
                    keys_all[usable_rows], gap,
                    f"the most recent timestamp alone holds {counts[-1]} rows, more than the "
                    f"fast profile's cap of {FAST_ROW_CAP}")
            capped, rows = len(rows) - kept, rows[len(rows) - kept:]
        else:
            keep = np.sort(np.random.default_rng([seed, 1]).choice(len(rows), FAST_ROW_CAP,
                                                                   replace=False))
            if BROKEN == "M10":             # the head of the file, not a sample from the seed
                keep = np.arange(FAST_ROW_CAP)
            capped, rows = len(rows) - FAST_ROW_CAP, rows[keep]
    y = y_work_all[rows]
    too_few = kept_too_few = None
    if len(rows) < MIN_ROWS:
        too_few = f"{len(rows)} usable rows; at least {MIN_ROWS} are needed"
        kept_too_few = f"kept {len(rows)} rows, and at least {MIN_ROWS} are needed"
    elif classification and min(int(y.sum()), int(len(y) - y.sum())) < MIN_CLASS_ROWS:
        too_few = f"the rarer class has fewer than {MIN_CLASS_ROWS} usable rows"
        kept_too_few = f"kept fewer than {MIN_CLASS_ROWS} rows of the rarer class"
    if too_few and capped:
        # The cap left too few rows. Say "use the full profile" only when the full
        # profile would get past this point and form its split (D52, D53).
        raise _fast_rows_stop(
            f"the fast profile's cap of {FAST_ROW_CAP} rows {kept_too_few}",
            y_work_all[usable_rows], classification,
            keys_all[usable_rows] if time_col is not None else None, gap)
    if too_few:
        raise E.TooFewRowsError(too_few)

    # 7. split off the holdout, applying the gap
    time_ordered = time_col is not None
    if time_ordered and BROKEN != "F7":
        keys = keys_all[rows]
        try:
            split = _split_time(keys, gap)
        except E.TooFewRowsError as exc:
            if not capped:
                raise
            # The cap left too few timestamps. Say "use the full profile" only when
            # the full profile would get past this step (D46).
            raise _fast_profile_stop(keys_all[usable_rows], gap,
                                     f"under the fast profile's cap of {FAST_ROW_CAP}, {exc}") \
                from exc
    else:
        split = _split_random(len(rows), y, classification, seed)   # F7: shuffled regardless
        if time_ordered:
            keys = keys_all[rows]
            _, counts = np.unique(keys, return_counts=True)
            split.update(train_end=keys[split["train"]].max(),
                         holdout_start=keys[split["holdout"]].min(),
                         holdout_end=keys[split["holdout"]].max(),
                         n_train_stamps=len(np.unique(keys[split["train"]])),
                         n_holdout_stamps=len(np.unique(keys[split["holdout"]])),
                         unique=bool(counts.max() == 1))
    train, holdout = split["train"], split["holdout"]
    if classification:
        if len(np.unique(y[holdout])) < 2 or len(np.unique(y[train])) < 2:
            raise E.OneClassError("the holdout or the training rows hold a single class, so "
                                  "ROC AUC cannot be computed")
        for fold in split["folds"]:
            fold["scored"] = (len(np.unique(y[fold["train"]])) == 2
                              and len(np.unique(y[fold["valid"]])) == 2)
    else:
        for fold in split["folds"]:
            fold["scored"] = True
    unscored = [i + 1 for i, f in enumerate(split["folds"]) if not f["scored"]]
    if len(unscored) == len(split["folds"]):
        raise E.OneClassError("every fold holds a single class in its training or validation rows")
    if unscored:
        _warn(warnings, "fold_not_scored",
              f"fold(s) {', '.join(map(str, unscored))} hold a single class and were not scored")
    used = np.concatenate([train, holdout])
    for name, values in prepared.items():
        # counted over the rows that reach a model. M13: over every loaded row
        cell = values if BROKEN == "M13" else values[rows[used]]
        by_name[name]["n_imputed"] = int(np.isnan(cell).sum()) if cell.dtype != object \
            else int((cell == MISSING_LEVEL).sum())

    # the target `gap + 1` timestamps earlier, where timestamps are unique (D10)
    prev = None
    if time_ordered and not classification and split["unique"]:
        order = np.argsort(keys, kind="stable")
        prev = np.full(len(y), np.nan)
        prev[order[gap + 1:]] = y[order[:len(y) - gap - 1]]

    # The baselines of step 11 need no model, only the target and the split, so they
    # are scored on the holdout here, before the leak screen and before any fit. One
    # whose score is not finite stops the run at once: when the target itself is the
    # cause, the caller gets one clear error, and never a list of configurations that
    # failed one after another (D44). Both are checked, the unused one too: its score
    # is a number in the output.
    truth = y[holdout]
    if classification:
        baselines, base_pred = [{"name": "chance", "score": 0.5, "used": True}], None
    else:
        base_pred = np.full(len(holdout), _mean(y[train]))
        baselines = [{"name": "training_mean", "score": _mse(truth, base_pred), "used": True}]
        if prev is not None:
            before = _previous_of(prev, holdout, "the holdout")
            previous = _mse(truth, before)
            baselines.append({"name": "previous_value", "score": previous, "used": False})
            if previous < baselines[0]["score"] and BROKEN != "F8":
                baselines[0]["used"], baselines[1]["used"] = False, True
                base_pred = before
        for b in baselines:
            if not math.isfinite(b["score"]):
                raise _not_finite(f"The squared error of the baseline {b['name']} on the "
                                  "holdout", b["score"], TARGET_CAUSE)

    # 8. from here on every fitted step sees training rows only: the leak screen first
    if BROKEN != "F2":
        # M3: the screen looks at rows it must not see
        judged = np.arange(len(rows)) if BROKEN == "M3" else train
        for name in list(prepared):
            values, numeric = prepared[name][rows], by_name[name]["kind"] == "numeric"
            rule = None
            if classification:
                gone = np.isnan(values[judged]) if numeric else values[judged] == MISSING_LEVEL
                share = [gone[y[judged] == c].mean() for c in (0, 1)]
                if max(share) > LEAK_MISSING_HIGH and min(share) < LEAK_MISSING_LOW:
                    rule = "missing_by_class"
            alone = None
            if rule is None:
                # None: the skill cannot be computed, so the column is not flagged (D36).
                # M9: the screen forgets the previous-value baseline, so the lag of a
                # persistent series is measured against the training mean alone.
                alone = _screen_column(values, numeric, y, split["folds"], classification,
                                       seed, None if BROKEN == "M9" else prev,
                                       holdout if BROKEN == "M3" else None)
                if alone is not None and alone >= LEAK_SKILL:
                    rule = "predicts_target"
            if rule is None and alone is not None and alone >= STRONG_SKILL:  # kept, told (D30)
                _warn(warnings, "strong_single_column",
                      f"column {name!r} on its own predicts the target with skill {alone:.2f} "
                      "on the training rows. It is kept as a feature: check that it is known "
                      "before the outcome", name)
            if rule:
                by_name[name].update(role="excluded", reason="suspected_leak", leak_rule=rule,
                                     n_imputed=0)
                del prepared[name]
                why = ("it is filled in for one class and empty for the other"
                       if rule == "missing_by_class"
                       else "on its own it predicts the target almost perfectly")
                _warn(warnings, "suspected_leak",
                      f"column {name!r} is excluded as a suspected leak: {why}", name)
    if not prepared:
        excluded = "; ".join(f"{r['name']} ({r['reason']})" for r in records
                             if r["role"] == "excluded")
        raise E.NoFeaturesError("no usable feature column is left. Excluded, with the reason "
                                "for each: " + (excluded or "none"))

    # Copies of a holdout row among the training rows (D49, D53). Random scheme only:
    # under time order rows at different times are different observations, and rows
    # at one timestamp are on one side of the split. Judged on the features that are
    # left, before any model is fitted, so a run that stops has cost nothing.
    n_copies = copy_share = None
    if time_col is None:
        n_copies = _holdout_copies([values[rows] for values in prepared.values()], y,
                                   train, holdout)
        copy_share = n_copies / len(holdout)
        if n_copies >= COPY_ROWS and copy_share > COPY_SHARE:
            found = (f"{n_copies} of the {len(holdout)} holdout rows ({100 * copy_share:.1f}%) "
                     "have a copy among the training rows: the same value in every feature "
                     "column and the same target")
            if duplicates == "error" and BROKEN != "F9":    # F9: the rule is never applied
                dates = [r["name"] for r in records if r["reason"] == "datetime"]
                raise E.HoldoutCopiesError(
                    f"{found}. A model can score on such a row by recalling its copy, so the "
                    "headline would overstate how predictable the target is. Remove the "
                    "repeated rows, or pass duplicates='keep' (--duplicates keep) if they are "
                    "separate observations that happen to agree; the output then records the "
                    "count" + (
                        ". Note: the column(s) " + ", ".join(repr(d) for d in dates)
                        + " hold dates and were excluded because no time column was declared, "
                        "so rows that differ only in their date count as copies. If the rows "
                        "are ordered in time, declare time_col: under time order this rule "
                        "does not apply" if dates else ""))
            if duplicates == "keep":
                _warn(warnings, "holdout_copies_kept",
                      f"{found}, and were kept (duplicates='keep'). A model can score on such "
                      "a row by recalling its copy, so the headline may overstate how "
                      "predictable the target is")
    kinds = {name: by_name[name]["kind"] for name in prepared}
    index = pd.Index(rows, name="row")
    step_makers = extensions.active()["steps"] or []

    def feature_frame(part):
        return pd.DataFrame({name: values[rows[part]] for name, values in prepared.items()},
                            index=index[part])

    def matrices(fit_rows, new_rows, final=False):
        """Preprocessing fitted on `fit_rows` only, applied to both sets of rows."""
        fit_frame, new_frame = feature_frame(fit_rows), feature_frame(new_rows)
        y_fit = pd.Series(y[fit_rows], index=fit_frame.index, name=target)
        for make in step_makers:                    # steps supplied through the seam
            step = make()
            step.fit(fit_frame, y_fit)
            fit_frame, new_frame = step.transform(fit_frame), step.transform(new_frame)
        # M1: the final fit's statistics also see the holdout. M2: levels from every row.
        seen_too = pd.concat([fit_frame, new_frame]) if BROKEN == "M1" and final else None
        every_level = [prepared[name] for name, kind in kinds.items() if kind == "categorical"] \
            if BROKEN == "M2" else None
        prep = _Prep(kinds).fit(fit_frame, seen_too, every_level)
        x_fit, x_new = prep.transform(fit_frame), prep.transform(new_frame)
        names = prep.names()
        frames = (pd.DataFrame(x_fit, index=fit_frame.index, columns=names),
                  pd.DataFrame(x_new, index=new_frame.index, columns=names), y_fit)
        return x_fit, x_new, frames

    def finite(guess):
        """A prediction that is not finite fails its configuration, as raising would
        (D44). M6: it is scored and written as it is."""
        if BROKEN != "M6" and not np.isfinite(guess).all():
            raise ValueError("the model predicted a value that is not finite")
        return guess

    # 9. select by five folds inside the training rows, preprocessing refitted in each
    families = _families(profile, classification)
    fold_data = [matrices(f["train"], f["valid"]) if f["scored"] else None
                 for f in split["folds"]]
    models, makers = [], []
    for family in families:
        for position, params in enumerate(family["grid"]):
            record = dict(config=f"{family['name']}-{position}", family=family["name"],
                          params=dict(params), status="fitted", error=None, fold_scores=[],
                          cv_mean=None, cv_std=None, chosen=False)
            try:
                for k, (fold, mats) in enumerate(zip(split["folds"], fold_data), 1):
                    if mats is None:
                        record["fold_scores"].append(None)
                        continue
                    model = family["make"](task, params, seed, n_jobs)
                    guess = finite(_fit_predict(family, model, mats[0], y[fold["train"]],
                                                mats[1], classification, mats[2]))
                    fold_truth = y[fold["valid"]]
                    fold_score = _auc(fold_truth, guess) if classification \
                        else _mse(fold_truth, guess)
                    if not math.isfinite(fold_score):
                        # failed, like one that raised: never a null where a score belongs
                        raise ValueError(f"its score on fold {k} is not finite ({fold_score!r})")
                    record["fold_scores"].append(fold_score)
            except Exception as exc:        # a family that raises is recorded, never dropped
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                              fold_scores=[None] * len(split["folds"]))
                _warn(warnings, "model_failed",
                      f"configuration {record['config']} failed: {record['error']}")
            scored = [s for s in record["fold_scores"] if s is not None]
            if record["status"] == "fitted" and scored:
                record["cv_mean"], record["cv_std"] = _mean_and_spread(scored)
                for key in ("cv_mean", "cv_std"):
                    if record[key] is not None and not math.isfinite(record[key]):
                        raise _not_finite(f"`{key}` of configuration {record['config']}, from "
                                          "fold scores that are each finite", record[key],
                                          "squared errors too large to average: " + TARGET_CAUSE)
            models.append(record)
            makers.append((family, params))
    # best mean first; a tie goes to the earlier family, then the earlier grid position
    ranked = sorted((i for i, m in enumerate(models) if m["cv_mean"] is not None),
                    key=lambda i: (-models[i]["cv_mean"] if classification
                                   else models[i]["cv_mean"], i))

    def all_failed():
        return E.AllModelsFailedError("every configuration failed: " +
                                      "; ".join(f"{m['config']}: {m['error']}" for m in models))
    if not ranked:
        raise all_failed()

    # 10. refit the chosen configuration on all training rows, and predict the holdout.
    # One that raises here is marked failed and keeps its fold scores; the next best
    # by cv_mean takes its place (D38). So is one whose prediction is not finite (D44):
    # that depends on the model and the features alone, never on the holdout targets.
    # The holdout plays no part in the choice.
    fit_rows = np.concatenate([train, holdout]) if BROKEN == "F1" else train
    x_fit, x_holdout, frames = matrices(fit_rows, holdout, final=True)
    chosen = pred = None
    for i in ranked:
        family, params = makers[i]
        try:
            model = family["make"](task, params, seed, n_jobs)
            pred = finite(_fit_predict(family, model, x_fit, y[fit_rows], x_holdout,
                                       classification, frames))
        except Exception as exc:            # recorded, never dropped, and never the headline
            models[i].update(status="failed", error=f"{type(exc).__name__}: {exc}")
            _warn(warnings, "model_failed",
                  f"configuration {models[i]['config']} failed after it was chosen: "
                  f"{models[i]['error']}")
            continue
        chosen = models[i]
        break
    if chosen is None:
        raise all_failed()
    chosen["chosen"] = True

    # 11. score it once on the holdout. The baselines were scored on the holdout above.
    if classification:
        metric, score = "roc_auc", _auc(truth, pred)
    else:
        metric, score = "mse", _mse(truth, pred)
    if not math.isfinite(score):
        # The predictions are finite, so this is the size of an error, not a model
        # that misbehaved. The run stops: the next configuration is not tried, because
        # one is passed over only for raising, never for its score (D38, D44).
        raise _not_finite(f"The score of the chosen configuration {chosen['config']} on the "
                          "holdout", score,
                          "a prediction or a target value so far from the other that its "
                          "squared error is beyond the largest number a double can hold")
    baseline = next(b for b in baselines if b["used"])
    if BROKEN == "F3":
        score = chosen["cv_mean"]                   # the best cross-validation score
    if BROKEN == "F4" and classification:
        majority = int(y[train].mean() >= 0.5)
        metric, score = "accuracy", float(((pred >= 0.5).astype(int) == truth).mean())
        baseline = {"name": "majority_class", "score": float((truth == majority).mean())}
        baselines = [dict(baseline, used=True)]
        skill = (score - baseline["score"]) / (1 - baseline["score"]) if baseline["score"] < 1 \
            else 0.0
    elif classification:
        skill = 2 * score - 1
    else:
        # With a baseline error of exactly zero skill has no value: null, never an
        # error, an infinity or a zero in its place (D36).
        skill = 1 - score / baseline["score"] if baseline["score"] > 0 else None
    if skill is not None and not math.isfinite(skill):
        # null is for a baseline error of exactly zero and for nothing else (D44)
        raise _not_finite("`headline.skill`, from a score and a baseline score that are each "
                          "finite", skill, "a baseline error so small beside the model's "
                          "error that their quotient is beyond the largest number a double "
                          "can hold")

    # 12. the interval and the p-value, from the holdout predictions alone
    groups = None
    if time_ordered:
        _, inverse = np.unique(keys[holdout], return_inverse=True)
        groups = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    uncertainty = _uncertainty(truth, pred, base_pred, classification, groups, RESAMPLES[profile],
                               seed, score, skill is not None)
    if skill is None:
        _warn(warnings, "skill_undefined",
              f"skill is not reported: the baseline {baseline['name']} has a squared error of "
              "exactly zero on the holdout, so 1 - error / baseline error has no value. The "
              "score and the baselines are reported as measured")
    elif uncertainty["interval_method"] and uncertainty["skill_low"] is None:
        # skill has a value and its interval has none: a code of its own (D42)
        _warn(warnings, "no_skill_interval",
              "the interval on skill is not reported: the baseline's squared error was exactly "
              f"zero in {uncertainty['n_resamples_used'] - uncertainty['n_resamples_skill_used']}"
              f" of the {uncertainty['n_resamples_used']} resamples, so skill could be computed "
              "in fewer than half of them")
    if uncertainty["interval_method"] == "bootstrap_blocks":
        _warn(warnings, "interval_approximate",
              "the interval comes from a block bootstrap over the holdout timestamps and is "
              "approximate: it tends to be too narrow when the holdout is short or the errors "
              "are serially correlated")
    if uncertainty["interval_reason"]:
        _warn(warnings, "no_interval", "no interval: " + uncertainty["interval_reason"])
    rarer = int(min(truth.sum(), len(truth) - truth.sum())) if classification else len(truth)
    if rarer < SMALL_HOLDOUT:
        _warn(warnings, "small_holdout",
              f"the holdout has {rarer} row(s)" + (" of the rarer class" if classification else "")
              + f", fewer than {SMALL_HOLDOUT}: the headline is imprecise")
    if time_ordered and not classification and not split["unique"]:
        stamps_train = keys[train]
        _, inverse = np.unique(stamps_train, return_inverse=True)
        means = np.bincount(inverse, weights=y[train]) / np.bincount(inverse)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            varies = len(means) > 2 and means[1:].std() > 0 and means[:-1].std() > 0
            lag1 = float(np.corrcoef(means[1:], means[:-1])[0, 1]) if varies else None
        # Means that do not vary have no autocorrelation, and there is no warning. One
        # that cannot be computed because a value overflows is not passed over (D44).
        if lag1 is not None and not math.isfinite(lag1):
            raise _not_finite("The lag-1 autocorrelation of the per-timestamp mean of the "
                              "target over the training timestamps", lag1, TARGET_CAUSE)
        if lag1 is not None:
            if lag1 > AUTOCORR:
                _warn(warnings, "autocorrelated_panel_target",
                      f"the per-timestamp mean of the target has lag-1 autocorrelation "
                      f"{lag1:.2f}; against the training mean, skill here may only reflect "
                      "that one period resembles the last", target)

    # 13. assemble the result
    source_rows = rows[holdout]
    order = np.argsort(source_rows, kind="stable")
    predictions = pd.DataFrame({
        "row": source_rows[order],
        "y_true": y_all.to_numpy()[source_rows[order]],
        "y_pred": pred[order],
    })
    n_gap = int(len(split["gap"]))
    row_items = [("loaded", n_loaded), ("dropped_missing_target", int(no_target.sum())),
                 ("dropped_missing_time", int(no_time.sum())), ("dropped_gap", n_gap),
                 ("dropped_row_cap", int(capped)), ("training", int(len(train))),
                 ("holdout", int(len(holdout)))]
    validation = {
        "scheme": "time_ordered" if time_ordered else "random",
        "holdout_share": len(holdout) / (len(train) + len(holdout) + n_gap),
        "n_folds": N_FOLDS, "stratified": bool(classification and not time_ordered),
        "gap": gap, "time_col": time_col, "time_form": time_form,
        "train_end": show(split["train_end"]) if time_ordered else None,
        "holdout_start": show(split["holdout_start"]) if time_ordered else None,
        "holdout_end": show(split["holdout_end"]) if time_ordered else None,
        "n_train_timestamps": split["n_train_stamps"] if time_ordered else None,
        "n_holdout_timestamps": split["n_holdout_stamps"] if time_ordered else None,
        "timestamps_unique": split["unique"] if time_ordered else None,
        "folds": [{"fold": i + 1, "training_rows": int(len(f["train"])),
                   "validation_rows": int(len(f["valid"])), "gap_rows": int(f["gap_rows"]),
                   "scored": bool(f["scored"])} for i, f in enumerate(split["folds"])],
        "caveat": None if time_ordered else RANDOM_CAVEAT,
        # holdout rows with a copy among the training rows; null under time order (D49)
        "n_holdout_copies": n_copies, "holdout_copy_share": copy_share,
    }
    headline = {"metric": metric, "score": score, "model": chosen["config"],
                "baseline": baseline["name"], "baseline_score": baseline["score"],
                "skill": skill}
    headline.update(uncertainty)
    doc = {
        "meta": {"tool": "mlprofessor", "schema_version": 1, "profile": profile, "seed": seed,
                 "n_jobs": n_jobs,
                 "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                              "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                              "threadpoolctl": threadpoolctl.__version__}},
        "input": {"source": source_label, "kind": source_kind, "target": target,
                  "time_col": time_col, "task": requested, "gap": gap, "n_rows": n_loaded,
                  "n_columns": int(df.shape[1]), "duplicate_rows": duplicate_rows,
                  "duplicates": duplicates},
        "rows": [{"item": k, "count": v} for k, v in row_items],
        "columns": records,
        "task": {"type": task, "decided_by": decided_by, "n_distinct": n_distinct,
                 "target_dtype": target_dtype,
                 "classes": [_plain(c) for c in classes] if classes else None,
                 "positive_class": _plain(classes[1]) if classes else None},
        "validation": validation,
        "headline": headline,
        "baselines": baselines,
        "cv": {"metric": "roc_auc" if classification else "mse",
               "greater_is_better": bool(classification), "n_folds": N_FOLDS,
               "n_scored_folds": N_FOLDS - len(unscored), "n_configurations": len(models),
               "chosen": chosen["config"], "chosen_mean": chosen["cv_mean"],
               "chosen_std": chosen["cv_std"]},
        # F6c: a family none of whose configurations was chosen is left out of the table
        "models": models if BROKEN != "F6c"
        else [m for m in models if m["family"] == chosen["family"]],
        "warnings": warnings,
        "limitations": list(LIMITATIONS),
    }
    _require_finite(doc)
    return doc, predictions
