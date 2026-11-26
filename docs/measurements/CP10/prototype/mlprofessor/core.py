"""Reference implementation of spec.md, in the spec's order of operations.

Judge fixture: the gate measures its bands on this code and breaks it in thirty-seven
named ways; the test suite is proved against those and against fifty-one smaller
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
from concurrent.futures import ThreadPoolExecutor
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
from . import _proto as P

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
MIN_GROUPS = N_FOLDS + 1    # one entity for the holdout and one for each fold, at the least
MIN_BOOT_GROUPS = 30        # fewer holdout entities than this: no interval (grouped scheme)
# D74, D76: the whole-number identifier's second clause. More than this share of the
# values distinct, and the distinct values filling at least that share of their range.
ID_DISTINCT_SHARE, ID_RANGE_SHARE = 0.95, 0.5
# D74: the leak screen. A training row whose target lies more than SCREEN_FAR median
# deviations from the median target, or SCREEN_FAR median steps from its previous
# value, is set aside from the screen, when such rows are at most SCREEN_SET_ASIDE of
# the training rows. For rule 2 the SCREEN_UNSCORED of the pooled validation rows that
# the column misses by most, and the same share that the baseline misses by most, are
# not scored.
SCREEN_FAR = 10
SCREEN_SET_ASIDE = 0.01
SCREEN_UNSCORED = 0.01
# D76: the second reading of the warning tier. With the training rows in a column's
# order, the target moves the other way between at most this share of neighbouring
# rows; and at least this many pairs are needed for the share to have a value.
ORDER_AGAINST = 0.10
ORDER_PAIRS = 20
# D74: one row carries more than this share of a squared error: a warning.
LARGEST_ROW_SHARE = 0.5
# D74, D76: a value far outside all the other values of its column. Among the
# EXTREME_OUTERMOST distinct values at each end, the gap to the next value inward is
# more than EXTREME_GAP times the spread of the EXTREME_SPREAD distinct values after it.
EXTREME_GAP = 30
EXTREME_SPREAD = 10
EXTREME_OUTERMOST = 3
# D74: under the grouped scheme, a fold that trains on fewer rows than this and on
# fewer rows than it validates on: no choice between configurations is made.
SELECTION_FLOOR = 50

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

# D62: when each answer to the copies stop fits, `keep` first. A fixed text.
COPIES_ADVICE = ("Pass duplicates='keep' (--duplicates keep) if the repeated rows are different "
                 "cases that look alike, as in a table of few-valued columns. Pass "
                 "duplicates='group' (--duplicates group) if they are one case recorded more "
                 "than once: no combination of feature values is then on both sides of the "
                 "split. Or remove the repeated rows.")
# D66: what is said of a unit that holds a quarter or more of the rows it is drawn
# among. Fixed texts.
LARGE_KEPT = ("It is kept among the training rows and the holdout is made of smaller units: "
              "the headline does not test it.")
LARGE_IS_HOLDOUT = ("The units smaller than a quarter do not together hold a quarter, so this "
                    "one unit is the holdout: the headline rests on it alone.")
# D77: what is said of a time-ordered holdout far above a quarter. A fixed text.
LARGE_HOLDOUT = ("The models were fitted on the rows before it, and the headline is measured "
                 "on it: fewer rows were learned from than the split usually leaves.")
# D66: a whole number reads as a calendar date written YYYYMMDD when its year is in
# this range, both ends included, and its month and day exist.
DATE_NUMBER_YEARS = (1900, 2100)
# D69: a column of such dates may hold values that are no dates (a sentinel such as
# 99991231 or 0): for each distinct value that is no date, at least this many
# distinct values that are.
DATES_PER_OTHER_VALUE = 9
# D72: or at least this many distinct dates beside at most that many distinct
# values that are no dates: a short column of dates with a sentinel or two.
FEW_DATES, FEW_OTHER_VALUES = 5, 2
# D69: what `duplicates="group"` measures. A fixed text.
GROUP_MEASURES = ("With duplicates='group' the headline measures whether a combination of "
                  "feature values that the model has not seen can be predicted. On a table of "
                  "few-valued columns that can give negative skill where duplicates='keep' "
                  "finds signal.")
# D74 (1): what the panel warning says of the baseline. A fixed text.
PANEL_BASELINE = ("With no entity column the previous value of a row is the mean target of "
                  "the timestamp before, over every series at once: if the rows are several "
                  "series, declare group_col, and each series is measured against its own "
                  "previous value.")
RANDOM_CAVEAT = ("Random split: this number is optimistic if the rows are ordered in time "
                 "or one entity appears on many rows.")
GROUPED_CAVEAT = ("Random split of whole entities: this number is optimistic if the rows are "
                  "ordered in time.")
NO_P_VALUE = ("not computed under the time-ordered scheme: shuffling the holdout targets "
              "assumes independent rows")
NO_P_VALUE_GROUPED = ("not computed under the grouped scheme: shuffling the holdout targets "
                      "assumes independent rows, and the rows of one entity are not")
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
        # No float that is not finite leaves in any part of the document, whoever
        # built it: the check is here, where every result is made (D44, D57).
        _require_finite(doc)
        self._doc = doc
        self.headline = dict(doc["headline"])
        if BROKEN == "F19":
            self.headline = {k: round(v, 3) if isinstance(v, float) else v
                             for k, v in self.headline.items()}
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
               "n_missing", "n_coerced", "n_infinite", "n_imputed",
               "shuffled_score", "shuffled_skill", "skill_fall"]
MODEL_KEYS = ["config", "family", "params", "status", "error", "fold_scores",
              "cv_mean", "cv_std", "chosen", "largest_row_share", "largest_row"]


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
    rests = h.get("rests_on") or []
    floor = h.get("rests_on_floor")
    lines.append("rests on" + ("" if floor is None else f" (columns whose fall is at least "
                                                         f"{floor:.4g})") + ": "
                 + (", ".join(f"{e['column']!r} (fall {e['skill_fall']:.6g})" for e in rests)
                    if rests else "none"))
    for w in d["warnings"]:                 # one line each: the code, then the column
        where = "" if w["column"] is None else f" column {w['column']!r}"
        lines.append(f"warning [{w['code']}]{where}: {w['message']}")
    return "\n".join(lines)


# -------------------------------------------------------------- public API --

def profile(data, target, task="auto", time_col=None, gap=0, profile="full", seed=0,
            out_dir=None, n_jobs=-1, duplicates="error", group_col=None):
    """Measure how predictable `target` is from the other columns of one table."""
    started = time.perf_counter()
    if task not in ("auto", "classification", "regression"):
        raise E.InputError(f"task must be auto, classification or regression, not {task!r}")
    if profile not in ("full", "fast"):
        raise E.InputError(f"profile must be full or fast, not {profile!r}")
    if not isinstance(duplicates, str) or duplicates not in ("error", "keep", "group"):
        raise E.InputError(f"duplicates must be error, keep or group, not {duplicates!r}")
    if isinstance(gap, bool) or not isinstance(gap, (int, np.integer)) or gap < 0:
        raise E.InputError(f"gap must be a whole number of at least 0, not {gap!r}")
    _check_seed(seed)                       # before anything is loaded (D45)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) \
            or not (n_jobs == -1 or n_jobs >= 1):
        raise E.InputError(f"n_jobs must be -1 or a positive whole number, not {n_jobs!r}")
    # D63: every call says both things. `False` is the caller's statement that the
    # rows have no time order, and for `group_col` that no rows belong together.
    # `None` is "not said": it is an argument like any other here, and the run
    # stops for it at step 5, where the table can be named. `True` says nothing.
    if time_col is True:
        raise E.InputError("time_col must be a column name, None, or False for a table whose "
                           "rows have no time order; not True")
    if group_col is not None and group_col is not False and not isinstance(group_col, str):
        raise E.InputError("group_col must be a column name, None, or False for a table with "
                           f"no entities; not {group_col!r}")
    if gap and (time_col is None or time_col is False):
        raise E.InputError("gap needs time_col: it counts timestamps")
    if time_col is not None and time_col is not False and time_col == target:
        raise E.InputError("time_col and target are the same column")
    if isinstance(group_col, str) and group_col == target:
        raise E.InputError("group_col and target are the same column")
    if isinstance(group_col, str) and time_col is not None and time_col is not False \
            and group_col == time_col:
        raise E.InputError("group_col and time_col are the same column")
    if out_dir is not None:
        _check_out_dir(out_dir)             # before anything is loaded (D39)
    threads = (os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"), \
            threadpoolctl.threadpool_limits(limits=threads, user_api="openmp"):
        doc, predictions = _run(data, target, task, time_col, int(gap), profile, int(seed), threads,
                                duplicates, group_col)
    doc["meta"]["timing"] = {"seconds": round(time.perf_counter() - started, 3)}
    if BROKEN == "F21":
        doc["meta"]["finished"] = time.time()
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

    Returns the frame to work on, the type of every column as it was loaded,
    which is what the output reports, and the names of the columns that hold
    whole numbers too large for 64 bits (D50). Each column is read here once,
    over all of its loaded rows, and every later step goes by this reading: an
    odd cell counts wherever its row ends up (D57).
    """
    loaded = {name: str(df[name].dtype) for name in df.columns}
    out, beyond = df, set()
    for name in df.columns:
        if df[name].dtype != object:
            continue
        typed = _object_numbers(df[name])
        if typed is not None:
            if out is df:
                out = df.copy(deep=False)
            out[name] = typed
        if _beyond_64_bits(out[name]):
            beyond.add(name)
    return out, loaded, beyond


def _beyond_64_bits(series):
    """Whether a column holds whole numbers too large for 64 bits (D50).

    After `_object_numbers` an object column whose values are all whole numbers
    is exactly that: one that fits has been given an integer type. It is asked
    of the whole column as loaded, never of some of its rows.
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
    beyond = E.TimeColumnError(
        f"time column {name!r} holds a date outside the range a nanosecond timestamp can "
        "hold (1677-09-21 to 2262-04-11), so its order cannot be read")
    # Asked before the conversion, never left to it: on a column of a coarser unit that
    # holds such a date beside ordinary ones, pandas 3.1.1 does not always raise. On
    # 1,500 rows with one cell of 9999-12-31 the conversion ended the process with an
    # access violation (found by the perturbation battery, 2026-11-22).
    if bool(((series < pd.Timestamp.min) | (series > pd.Timestamp.max)).any()):
        raise beyond
    try:
        return series.to_numpy("datetime64[ns]")
    except (OverflowError, ValueError) as exc:
        raise beyond from exc


def _time_keys(series, name, judged=None):
    """Step 3 (D18): sortable keys for the time column. Returns (keys, missing, form, show).

    `judged` exists only for the broken variant M14: the rows whose values decide
    whether whole-number keys stay integers, in place of every loaded row.
    """
    dtype = series.dtype
    if pt.is_datetime64_any_dtype(dtype):
        s = series
        if getattr(dtype, "tz", None) is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        stamps = _to_nanoseconds(s, name)
        return stamps.astype("int64"), np.isnat(stamps), "datetime", _show_stamp
    if pt.is_numeric_dtype(dtype) and not pt.is_bool_dtype(dtype) \
            and not pt.is_complex_dtype(dtype):
        _no_other_value_among_dates(series, name)
        whole = pt.is_integer_dtype(dtype)
        if whole and BROKEN != "F7c":
            # D50, D53: above 2**53 two distinct whole numbers can be one float, so
            # there the keys stay the integers they are. F7c: always floats.
            exact, missing = _exact_integers(series)
            present = exact[~missing] if judged is None else exact[~missing & judged]
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
        if BROKEN != "M28":
            _no_other_value_among_dates(series, name)
        if BROKEN != "M28" and len(text) and text.map(_date_digits).all() \
                and _calendar_dates(text.astype("int64").to_numpy()):
            # D66: dates written as eight digits are ordered as the numbers they are
            keys = pd.to_numeric(series.str.strip(), errors="coerce") \
                .to_numpy(dtype="float64", na_value=np.nan)
            return keys, np.isnan(keys), "numeric", int
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


def _decide_task(y, name, requested, beyond=None):
    """Step 4 (D4, D21): classification or regression, and how that was decided.

    `beyond`: whether the target column holds whole numbers too large for 64 bits,
    as step 1a read it over all loaded rows. None: `y` is the whole column.
    """
    if _beyond_64_bits(y) if beyond is None else beyond:
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
        elif not numeric:
            # Three classes or more. Declaring the task would not help, so the
            # message does not suggest it (D57).
            raise E.TaskAmbiguousError(
                f"target {name!r} has {n_distinct} distinct values and is not numeric. Three "
                "or more classes are not supported: a classification needs exactly two. Keep "
                "the rows of two classes, or merge the classes into two, and run again")
        else:
            raise E.TaskAmbiguousError(
                f"target {name!r} is numeric with {n_distinct} distinct values: more than two "
                "classes, and three or more classes are not supported; and too few values to "
                "be read as a quantity without being told so. If it is a quantity, pass "
                "task='regression'")
        return kind, "rule", n_distinct, numeric
    if requested == "classification" and n_distinct != 2:
        raise E.TaskUnsupportedError(f"classification needs exactly two classes; target "
                                     f"{name!r} has {n_distinct}")
    if requested == "regression" and not numeric:
        raise E.TaskUnsupportedError(f"regression needs a numeric target; {name!r} is {y.dtype}")
    return requested, "declared", n_distinct, numeric


def _no_target(series):
    """Step 2: which target cells are missing or infinite.

    `series` is the target as `_target_values` reads it, so a text cell of only
    white space is already missing here. An infinite number is no target: in a
    numeric column, and as a float among the values of an object column.
    """
    gone = series.isna().to_numpy().copy()
    if _is_numeric(series.dtype):
        gone |= ~np.isfinite(series.to_numpy(dtype="float64", na_value=np.nan))
    elif series.dtype == object:
        gone |= _infinite_floats(series)
    return gone


def _infinite_floats(series):
    """Which cells of an object column hold an infinite float. In a target that is
    not read as numbers such a cell is no class: it is a missing target (D57)."""
    return np.fromiter((isinstance(v, (float, np.floating)) and math.isinf(v)
                        for v in series.to_numpy(dtype=object)), dtype=bool, count=len(series))


def _classes(values):
    """The two classes in sorted order; the second is the positive class."""
    distinct = list(pd.unique(values))
    try:
        return sorted(distinct)
    except TypeError:
        return sorted(distinct, key=str)


def _whole_number_id(x):
    """Rule 8b (D19, D74): a whole-number column is an identifier when

    - every value is distinct, and the values are consecutive or strictly ordered
      in file order (D19); or
    - more than 95% of the values are distinct, and the distinct values fill at
      least half of the range from the smallest to the largest, whatever the
      order of the rows (D74): so an identifier stays one when the rows are
      shuffled, some are missing, or a few are repeated.

    `x` holds the present values in file order: floats, or the exact integers of
    an integer-typed column (D50). Neighbours are compared, never subtracted, and
    the range is taken in Python integers, so nothing here can overflow.
    F27: D19's clause alone, as it was. M50: the second clause asks for every
    value to be distinct, so one repeated row makes the identifier a feature.
    """
    if len(x) < 2:
        return False
    if x.dtype.kind == "f" and not np.all(x == np.floor(x)):
        return False
    distinct = np.unique(x)
    span = int(distinct.max()) - int(distinct.min()) + 1
    if len(distinct) == len(x):
        if np.all(x[1:] > x[:-1]) or np.all(x[1:] < x[:-1]) or span == len(x):
            return True
    if BROKEN == "F27" or (BROKEN == "M50" and len(distinct) != len(x)):
        return False
    return len(distinct) > ID_DISTINCT_SHARE * len(x) and len(distinct) >= ID_RANGE_SHARE * span


def _reads_as_date(values):
    """Which of `values`, whole numbers in a numpy array, read as a calendar date
    written YYYYMMDD: a year from 1900 to 2100, a month from 1 to 12 and a day that
    month has, 29 February only in a leap year (D66)."""
    values = np.asarray(values)
    low, high = DATE_NUMBER_YEARS[0] * 10000 + 101, DATE_NUMBER_YEARS[1] * 10000 + 1231
    ok = (values >= low) & (values <= high)
    held = np.where(ok, values, low).astype("int64")
    year, month, day = held // 10000, held // 100 % 100, held % 100
    ok = ok & (month >= 1) & (month <= 12) & (day >= 1)
    month = np.where(ok, month, 1)
    length = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])[month]
    leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    length = np.where((month == 2) & leap, 29, length)
    return ok & (day <= length)


def _calendar_dates(values):
    """Whether every one of `values`, whole numbers in a numpy array, reads as a
    calendar date written YYYYMMDD, and there is at least one (D66)."""
    values = np.asarray(values)
    return len(values) > 0 and bool(_reads_as_date(values).all())


def _date_digits(text):
    """Whether a stripped text value is eight digits, the form of a date number as text."""
    return len(text) == 8 and text.isascii() and text.isdigit()


def _date_values(series, rows):
    """D66, D69: the distinct values of a column among `rows`, as rule 2a reads them.

    Returns (how many distinct values read as a calendar date written YYYYMMDD, the
    distinct values that do not as [(value, cells that hold it), ...], the most
    frequent first, how many cells hold a date), or None when the rule does not
    look at the column: a boolean,
    a complex number, mixed objects. In a float column a value with a fractional
    part and an infinite cell are values, and no dates; in a text column a value
    is its text without the white space round it, and reads as a date when it is
    eight digits that form one.
    """
    dtype = series.dtype
    low, high = DATE_NUMBER_YEARS[0] * 10000 + 101, DATE_NUMBER_YEARS[1] * 10000 + 1231
    if pt.is_bool_dtype(dtype) or pt.is_complex_dtype(dtype):
        return None
    if pt.is_integer_dtype(dtype):
        exact, gone = _exact_integers(series)
        distinct, cells = np.unique(exact[rows & ~gone], return_counts=True)
        dates = _reads_as_date(distinct)
        shown = [int(v) for v in distinct[~dates]]
    elif pt.is_float_dtype(dtype):
        values = series.to_numpy(dtype="float64", na_value=np.nan)[rows]
        distinct, cells = np.unique(values[~np.isnan(values)], return_counts=True)
        with np.errstate(invalid="ignore"):
            whole = np.isfinite(distinct) & (distinct == np.floor(distinct))
        dates = whole & (distinct >= low) & (distinct <= high)
        dates[dates] = _reads_as_date(distinct[dates].astype("int64"))
        shown = [int(v) if w and abs(v) < 2 ** 53 else float(v)
                 for v, w in zip(distinct[~dates].tolist(), whole[~dates].tolist())]
    elif pt.is_string_dtype(dtype) or dtype == object:
        present = series[rows].dropna()
        if not all(isinstance(v, str) for v in present):
            return None
        counted = pd.Series([v.strip() for v in present], dtype=object).value_counts(sort=False)
        distinct, cells = counted.index.to_numpy(dtype=object), counted.to_numpy()
        dates = np.array([_date_digits(v) for v in distinct], dtype=bool)
        if dates.any():
            dates[dates] = _reads_as_date(np.array([int(v) for v in distinct[dates]],
                                                   dtype="int64"))
        shown = list(distinct[~dates])
    else:
        return None
    others = sorted(zip(shown, (int(c) for c in cells[~dates])), key=lambda pair: -pair[1])
    return int(dates.sum()), others, int(cells[dates].sum())


def _date_numbers(series, rows):
    """D66, D69: whether a column holds dates written as whole numbers, YYYYMMDD.
    Returns (whether it does, the distinct values that are no dates with their cells).

    Judged on the cells of `rows` that are not empty, by their distinct values: at
    least two read as calendar dates, and for each one that does not, at least
    `DATES_PER_OTHER_VALUE` do. So a sentinel counts once, however many cells
    hold it, and a column of a few codes of which some read as dates is no date
    column. F14b: every value must read as a date, as built in the sixth pass:
    one sentinel cell, and the column is a number again.
    """
    found = _date_values(series, rows)
    if found is None:
        return False, []
    dates, others, date_cells = found
    if BROKEN == "F14b":
        return dates >= 2 and not others, others
    if BROKEN == "M31":
        # M31: the cells are counted, not the distinct values: 95% of them must be
        # dates, so a sentinel in a third of the rows makes the column a number
        held = sum(cells for _, cells in others)
        return dates >= 2 and 20 * held <= held + date_cells, others
    enough = dates >= DATES_PER_OTHER_VALUE * len(others) \
        or (len(others) <= FEW_OTHER_VALUES and dates >= FEW_DATES)
    return dates >= 2 and enough, others


def _no_other_value_among_dates(series, name):
    """D69: a time column of dates written YYYYMMDD that also holds a value that is
    no date stops the run. Such a value has no place in time: it is never ordered
    as a number, and its rows are never dropped without the caller's say. Judged
    on every non-missing value of the column."""
    dates, others = _date_numbers(series, np.ones(len(series), dtype=bool))
    if not (dates and others) or BROKEN == "M32":   # M32: it is ordered as a number
        return
    total = sum(cells for _, cells in others)
    shown = ", ".join(f"{value!r} in {cells} row{'' if cells == 1 else 's'}"
                      for value, cells in others[:5])
    more = "" if len(others) <= 5 else f", and {len(others) - 5} other value(s)"
    raise E.TimeColumnError(
        f"time column {name!r} holds dates written as whole numbers (YYYYMMDD), and {total} "
        f"row{'' if total == 1 else 's'} with a value that is no date: {shown}{more}. Such a "
        "value has no place in time, so it is not ordered as a number, and its rows are not "
        "dropped without your say. Replace it with an empty cell, and those rows are dropped "
        "and counted; or with the date it stands for. Then run again")


def _separated(text):
    """D28: whether a stripped text value is a number written with thousands separators."""
    return SEPARATED_RE.fullmatch(text) is not None


def _roles(df, usable, target, time_col, loaded=None, group_col=None):
    """Step 5 (D19, D20, D28, D34): roles that need no target, judged on the usable rows.

    Returns the column records and, for each feature, its prepared values over
    every loaded row: float64 for numeric, object for categorical. `df` is the
    table after `_numbers_from_objects`; `loaded` gives each column's type as it
    was loaded, which is the type reported (D51).
    """
    records, prepared = [], {}
    n_rows = int(usable.sum())
    named = {target: "target"}
    if isinstance(time_col, str):
        named[time_col] = "time"
    if isinstance(group_col, str):
        named[group_col] = "group"
    for position, name in enumerate(df.columns):
        series = df[name]
        rec = dict(name=name, position=position,
                   dtype=loaded[name] if loaded else str(series.dtype), role="feature",
                   kind=None, reason=None, leak_rule=None, n_missing=int(series.isna().sum()),
                   n_coerced=0, n_infinite=0, n_imputed=0,
                   shuffled_score=None, shuffled_skill=None, skill_fall=None)
        records.append(rec)
        if name in named:
            rec["role"] = named[name]
            # The infinite cells of a numeric target or time column are counted like
            # any column's (5.2): each is a missing target or a missing time value.
            # M20: they are left out of the count.
            if rec["role"] != "group" and _is_numeric(series.dtype) and BROKEN != "M20":
                held = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64",
                                                                       na_value=np.nan)
                rec["n_infinite"] = int(np.isinf(held).sum())
            elif rec["role"] == "target" and series.dtype == object and BROKEN != "M20":
                rec["n_infinite"] = int(_infinite_floats(series).sum())
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
        elif BROKEN not in ("F14", "M29") and _date_numbers(series, usable)[0]:
            # D66: dates written as whole numbers are dates, before any rule that would
            # read them as a measure or as an identifier. F14: they are a feature.
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
            elif BROKEN == "M29" and _date_numbers(series, usable)[0]:
                # M29: the id-like rule is asked first, so one row a day in date order
                # is an identifier and nothing says that the table holds dates
                rec.update(role="excluded", reason="datetime", kind="datetime")
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


def _fast_rows_stop(found, y_usable, classification, all_keys, gap, groups_usable=None,
                    seed=0):
    """The error for a fast run whose capped rows are too few to profile (D52, D53).

    The rows the cap kept are fewer than `MIN_ROWS`, or hold fewer than
    `MIN_CLASS_ROWS` of a class. `y_usable` and `all_keys` are the target and the
    time keys of every usable row: what the full profile would work on. If the
    full profile gets past this step and can form its split, the error is the
    fast profile's own, with a code that names rows, and its advice is true.
    If it cannot, the error is the one the full profile would raise.
    `groups_usable`: with an entity column and no time column, the entity of
    every usable row, so that the split asked about is the one of whole entities.
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
    elif groups_usable is not None:
        try:
            _split_grouped(groups_usable, y_usable, classification, seed)
        except E.TooFewGroupsError as exc:
            return type(exc)(f"{exc} (the fast profile kept fewer rows still: {found})")
    return E.FastProfileRowsError(
        f"{found}. Every usable row together is enough to profile: use profile='full'")


def _row_keys(columns, y, used):
    """One whole number per row of `used`: equal for two rows that are copies of one
    another (D49), the same value in every one of `columns` and the same `y`. With
    `y` None the target plays no part: equal for two rows that agree in every one
    of `columns`, which is what `duplicates="group"` keeps together (D62).

    `columns` are feature columns as a fitted step is handed them: 64-bit floats
    for a numeric column, where two empty cells are equal and 0.0 equals -0.0;
    levels for a categorical column, without the white space round them, an
    empty cell being the level `(missing)`. `y` is the target as step 4 reads it.
    """
    code = np.zeros(len(used), dtype="int64")
    for values in list(columns) + ([] if y is None else [y]):
        cell = values[used]
        if cell.dtype == object:
            cell = cell.astype(str)
        # numpy counts every not-a-number as one value, and 0.0 and -0.0 as one
        _, part = np.unique(cell, return_inverse=True)
        # one number per distinct pair so far; renumbered each time, so it stays small
        _, code = np.unique(code * (int(part.max()) + 1) + part, return_inverse=True)
    return code


def _holdout_copies(columns, y, train, holdout):
    """D49: how many holdout rows have a copy among the training rows, on the
    features left after the leak screen. A copy has the same target too: D62 changed
    what `duplicates="group"` keeps together, and not what a copy is. M30: it did."""
    code = _row_keys(columns, None if BROKEN == "M30" else y, np.concatenate([train, holdout]))
    return int(np.isin(code[len(train):], code[:len(train)]).sum())


def _clusters(keys):
    """Rows in groups: two rows are in one group when they have the same value in one
    of the arrays of `keys`, directly or through other rows. Returns one whole
    number per row, 0..groups-1."""
    n = len(keys[0])
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for key in keys:
        order = np.argsort(key, kind="stable")
        same = np.flatnonzero(key[order][1:] == key[order][:-1])
        for a, b in zip(order[same].tolist(), order[same + 1].tolist()):
            a, b = find(a), find(b)
            if a != b:
                parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1]


def _group_codes(series, name):
    """D57: the entity column as one whole number per loaded row, -1 for an empty cell.

    Two cells are one entity when their values are equal. Text is compared
    without the white space round it, and a text cell that holds nothing else is
    an empty cell. Whole numbers are compared exactly at any size; `0.0` and
    `-0.0` are one value. Values that cannot be compared for equality (lists,
    dictionaries, sets) are the named error.
    """
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        series = pd.Series([v if pd.isna(v) else str(v) for v in series], index=series.index,
                           dtype=object)
    elif isinstance(dtype, pd.SparseDtype):
        series = series.sparse.to_dense()
    dtype = series.dtype
    try:
        if pt.is_string_dtype(dtype) or dtype == object:
            cells = np.empty(len(series), dtype=object)
            for i, value in enumerate(series.to_numpy(dtype=object)):
                if _cell_missing(value):
                    cells[i] = None
                elif isinstance(value, str):
                    cells[i] = value.strip() or None
                else:
                    cells[i] = value
            codes, _ = pd.factorize(cells, use_na_sentinel=True)
        elif pt.is_float_dtype(dtype):
            values = series.to_numpy(dtype="float64", na_value=np.nan)
            codes, _ = pd.factorize(values + 0.0, use_na_sentinel=True)
        else:
            codes, _ = pd.factorize(series, use_na_sentinel=True)
    except TypeError as exc:
        raise E.GroupColumnError(
            f"entity column {name!r} holds values that cannot be compared for equality "
            f"({exc}), so its rows cannot be put into entities") from exc
    return np.asarray(codes, dtype="int64")


def _previous_by_group(y, keys, groups, gap):
    """D57: for each row, the target of the latest row of its own entity that lies
    at least `gap + 1` distinct timestamps earlier; NaN where the entity has none.

    Timestamps are those of every row given, whatever its entity, so that a gap
    means what it means for the split. Within one entity they are unique.
    """
    _, stamp = np.unique(keys, return_inverse=True)
    order = np.lexsort((stamp, groups))
    g, s = groups[order].astype("int64"), stamp[order].astype("int64")
    width = int(stamp.max()) + gap + 2
    here = g * width + s
    pos = np.searchsorted(here, g * width + s - gap - 1, side="right") - 1
    found = (s - gap - 1 >= 0) & (pos >= 0)
    found[found] = g[pos[found]] == g[found]
    prev = np.full(len(y), np.nan)
    prev[order[found]] = y[order[pos[found]]]
    return prev


def _lag1_of_stamp_means(keys, y):
    """The lag-1 autocorrelation of the per-timestamp mean of `y`, timestamps in order.
    None when those means do not vary; not finite when a value overflows."""
    _, inverse = np.unique(keys, return_inverse=True)
    means = np.bincount(inverse, weights=y) / np.bincount(inverse)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        varies = len(means) > 2 and means[1:].std() > 0 and means[:-1].std() > 0
        return float(np.corrcoef(means[1:], means[:-1])[0, 1]) if varies else None


def _cap_groups(groups, seed):
    """Step 6 with entities (D57): which of the rows the fast profile keeps.

    Whole entities, taken in an order drawn from the seed: an entity is kept
    when all of its rows still fit under the cap. So no more than `FAST_ROW_CAP`
    rows are kept, and no entity left out would have fitted beside those kept.
    M17: a sample of rows, whatever their entity.
    """
    if BROKEN == "M17":
        return np.sort(np.random.default_rng([seed, 1]).choice(len(groups), FAST_ROW_CAP,
                                                               replace=False))
    groups = pd.factorize(np.asarray(groups))[0]        # numbered as they first appear
    sizes = np.bincount(groups)
    kept, total = np.zeros(len(sizes), dtype=bool), 0
    if BROKEN not in ("F15", "M35"):
        # D72: an entity that holds a quarter or more of the rows is always kept; the
        # caller has seen to it that those fit. F15, M35: it is left to the draw.
        for g in np.flatnonzero(4 * sizes >= len(groups)).tolist():
            kept[g], total = True, total + int(sizes[g])
    for g in np.random.default_rng([seed, 1]).permutation(len(sizes)).tolist():
        if kept[g]:
            continue
        if total + sizes[g] <= FAST_ROW_CAP:
            kept[g], total = True, total + int(sizes[g])
    return np.flatnonzero(kept[groups])


def _split_grouped(groups, y, classification, seed):
    """Step 7 for whole entities (D57). `groups` holds one whole number per row, 0..G-1.

    No entity has rows on both sides of the holdout, and within the training
    rows each entity is in the validation part of exactly one fold. For
    classification an entity counts with the class most of its rows hold (the
    positive class on a tie), and the rule below is applied to the entities of
    each class apart.

    Holdout (D64, D66), with the entities in an order drawn from the seed: an
    entity joins it while the holdout, with it, still holds less than a quarter
    of the rows, and any other is passed over; the smallest entity passed over,
    the first among equals, completes it; then every entity the holdout can
    spare, the largest first, goes back to the training rows. So the holdout
    holds at least a quarter, needs every entity it has, and never takes a large
    entity where smaller ones could fill it. F13: the entities in that order
    until they hold a quarter, as built in the fifth pass.
    Folds: the training entities in an order drawn from the seed, each to the
    fold that so far holds the fewest rows.

    `too_large` in the result lists every entity that holds a quarter or more of
    the rows it is drawn among, as (rows, rows drawn among, whether it is the
    holdout). Such an entity is the holdout exactly when the entities smaller
    than a quarter do not together hold a quarter; otherwise it is never held out.

    The entities are numbered here as they first appear among the rows, so the
    split depends on which rows share an entity and not on what it is called.
    """
    groups = pd.factorize(np.asarray(groups))[0]
    sizes = np.bincount(groups)
    n_groups = len(sizes)
    if n_groups < MIN_GROUPS:
        raise E.TooFewGroupsError(
            f"the rows hold {n_groups} entit{'y' if n_groups == 1 else 'ies'}; at least "
            f"{MIN_GROUPS} are needed, one for the holdout and one for each of the "
            f"{N_FOLDS} folds")
    rng = np.random.default_rng([seed, 7])
    if classification:
        positive = np.bincount(groups, weights=y, minlength=n_groups)
        stratum = (2 * positive >= sizes).astype("int64")
    else:
        stratum = np.zeros(n_groups, dtype="int64")
    held = np.zeros(n_groups, dtype=bool)
    fold_of = np.full(n_groups, -1)
    orders, too_large = {}, []
    for s in (0, 1):
        members = np.flatnonzero(stratum == s)
        members = members[rng.permutation(len(members))]
        total, taken = int(sizes[members].sum()), 0
        if BROKEN == "F13":                 # as built in the fifth pass: until a quarter is held
            for g in members.tolist():
                if 4 * taken >= total:
                    break
                held[g], taken = True, taken + int(sizes[g])
            orders[s] = members[~held[members]]
            continue
        passed = []
        for g in members.tolist():          # 1. while the holdout stays below a quarter
            if 4 * (taken + int(sizes[g])) < total:
                held[g], taken = True, taken + int(sizes[g])
            else:
                passed.append(g)
        if passed:
            # 2. completed by the smallest unit passed over, the first among equals
            last = min(passed, key=lambda g: int(sizes[g]))
            held[last], taken = True, taken + int(sizes[last])
            # 3. what the holdout can spare goes back, the largest first
            spare = [g for g in members.tolist() if held[g] and g != last]
            if BROKEN == "M26":             # M26: step 3 is missing
                spare = []
            for g in sorted(spare, key=lambda g: -int(sizes[g])):
                if 4 * (taken - int(sizes[g])) >= total:
                    held[g], taken = False, taken - int(sizes[g])
        # D66: every unit that holds a quarter or more of the rows it is drawn among
        # is either the whole holdout or never in it. Either way the caller is told.
        too_large += [(int(sizes[g]), total, bool(held[g])) for g in members.tolist()
                      if 4 * int(sizes[g]) >= total]
        orders[s] = members[~held[members]]
    if int((~held).sum()) < N_FOLDS:
        raise E.TooFewGroupsError(
            f"with a quarter of the rows held out as whole entities, {int((~held).sum())} "
            f"entit{'y is' if int((~held).sum()) == 1 else 'ies are'} left for training; "
            f"the {N_FOLDS} folds need at least {N_FOLDS}")
    if BROKEN == "M15":                     # the folds are cut through the entities
        train, holdout = np.flatnonzero(~held[groups]), np.flatnonzero(held[groups])
        cuts = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(train, y[train]) \
            if classification else KFold(N_FOLDS, shuffle=True, random_state=seed).split(train)
        return {"train": train, "gap": np.array([], dtype=int), "holdout": holdout,
                "folds": [{"train": train[a], "valid": train[b], "gap_rows": 0}
                          for a, b in cuts], "too_large": too_large}
    load = np.zeros(N_FOLDS, dtype="int64")
    for s in (0, 1):
        own = np.zeros(N_FOLDS, dtype="int64")
        for g in orders[s][rng.permutation(len(orders[s]))].tolist():
            k = min(range(N_FOLDS), key=lambda f: (own[f], load[f], f))
            fold_of[g], own[k], load[k] = k, own[k] + sizes[g], load[k] + sizes[g]
    row_fold = fold_of[groups]
    train, holdout = np.flatnonzero(row_fold >= 0), np.flatnonzero(held[groups])
    folds = [{"train": np.flatnonzero((row_fold >= 0) & (row_fold != k)),
              "valid": np.flatnonzero(row_fold == k), "gap_rows": 0} for k in range(N_FOLDS)]
    return {"train": train, "gap": np.array([], dtype=int), "holdout": holdout, "folds": folds,
            "too_large": too_large}


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
    # What the model hands back is read here, inside the caller's `try`: one number
    # for each row. A classifier gives two columns of probabilities, and the second
    # is the positive class; a regressor gives the rows' values in any shape that
    # holds exactly that many. Anything else raises, and fails the configuration.
    if classification:
        proba = np.asarray(model.predict_proba(x_new), dtype="float64")
        if proba.ndim != 2 or proba.shape != (len(x_new), 2):
            raise ValueError(f"predict_proba returned shape {proba.shape}; expected "
                             f"({len(x_new)}, 2)")
        return proba[:, 0 if BROKEN == "F16" else 1].copy()
    values = np.asarray(model.predict(x_new), dtype="float64").reshape(-1)
    if len(values) != len(x_new):
        raise ValueError(f"predict returned {len(values)} values for {len(x_new)} rows")
    return values


def _predict(model, x_new, classification):
    """What a fitted model predicts for `x_new`, read as `_fit_predict` reads it."""
    if classification:
        proba = np.asarray(model.predict_proba(x_new), dtype="float64")
        if proba.ndim != 2 or proba.shape != (len(x_new), 2):
            raise ValueError(f"predict_proba returned shape {proba.shape}; expected "
                             f"({len(x_new)}, 2)")
        return proba[:, 0 if BROKEN == "F16" else 1].copy()
    values = np.asarray(model.predict(x_new), dtype="float64").reshape(-1)
    if len(values) != len(x_new):
        raise ValueError(f"predict returned {len(values)} values for {len(x_new)} rows")
    return values


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
    # Every part, so that no float can leave as a null from any of them (D57).
    # M18: five parts only, as it was.
    parts = ("validation", "headline", "baselines", "cv", "models") if BROKEN == "M18" else doc
    for part in parts:
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


def _screen_baselines(y, folds, prev, by_entity):
    """The screen's baselines over the pooled validation rows of the scored folds.

    Returns (rows, [(name, predictions)]): `rows` are positions among the rows
    split, fold after fold. The training mean is each fold's own; the previous
    value is each row's, and with an entity column a row whose entity has no
    earlier row is given its fold's training mean in its place (D57).
    """
    rows, means = [], []
    for fold in folds:
        if not fold["scored"] or not len(fold["train"]) or not len(fold["valid"]):
            continue
        rows.append(fold["valid"])
        means.append(np.full(len(fold["valid"]), _mean(y[fold["train"]])))
    rows, mean_pred = np.concatenate(rows), np.concatenate(means)
    base_preds = [("training_mean", mean_pred)]
    if prev is not None and by_entity:
        before = prev[rows]
        base_preds.append(("previous_value", np.where(np.isnan(before), mean_pred, before)))
    elif prev is not None:
        base_preds.append(("previous_value", _previous_of(prev, rows, "a validation block")))
    return rows, base_preds


def _screen_folds(y, train, folds, prev, by_entity, classification):
    """Step 8 (D74): the folds the screen is measured on, in a regression.

    A training row whose target is wildly off is set aside from the whole screen:
    it is in no fold's training rows and in no fold's validation rows, so in no
    mean, no tree, no line and no sum. Two readings of "wildly off", one for each
    baseline, both robust, so that the row cannot hide behind what it does to a
    mean:

      - its target lies more than `SCREEN_FAR` median deviations from the median
        target of the training rows;
      - its target lies more than `SCREEN_FAR` median steps from its previous
        value, where the previous value is a baseline (among the training rows
        that have one).

    The rows of both readings together are set aside when they are at most
    `SCREEN_SET_ASIDE` of the training rows, and never fewer than one row is
    allowed. More than that is the shape of the target, not a stray row: nothing is
    set aside, and the screen scores every row, as it did before D74.

    Returns (folds, rows set aside, whether every row is to be scored). F25:
    nothing is ever set aside and every row is scored. M42: the rows are left out
    of the validation rows only, and stay among each fold's training rows.
    """
    none = np.array([], dtype=int)
    if classification:
        return folds, none, False, None
    if BROKEN == "F25":
        return folds, none, True, None
    pooled, base_preds = _screen_baselines(y, folds, prev, by_entity)
    for _, pred in base_preds:
        if not math.isfinite(_mse(y[pooled], pred)):
            return folds, none, False, None     # the run stops where a column is measured (D44)
    allowed = max(1, int(SCREEN_SET_ASIDE * len(train)))
    gone = np.zeros(len(y), dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        readings = [(train, np.abs(y[train] - np.median(y[train])))]
        if prev is not None:
            has = train[~np.isnan(prev[train])]
            if len(has):
                readings.append((has, np.abs(y[has] - prev[has])))
        for rows, miss in readings:
            scale = float(np.median(miss))
            if scale > 0 and math.isfinite(scale):
                gone[rows[miss > SCREEN_FAR * scale]] = True
    if P.DEBUG:
        P.LOG["marked"] = (int(gone.sum()), int(allowed), int(len(train)))
    if gone.sum() > allowed:
        if P.D == "always":         # rule 2 leaves its rows unscored all the same
            return folds, none, False, None
        if P.D == "unmarked":       # ... but never a row that is marked
            return folds, none, False, gone
        if P.D == "median":         # ... and against a baseline the rows left in cannot move
            return folds, none, False, "median"
        return folds, none, True, None
    if not gone.any():
        return folds, none, False, None
    kept = [dict(fold, train=fold["train"] if BROKEN == "M42" else fold["train"][~gone[fold["train"]]],
                 valid=fold["valid"][~gone[fold["valid"]]]) for fold in folds]
    return kept, np.flatnonzero(gone), False, None


def _theil_line(x, y):
    """A straight line that a few rows cannot move (D74): the median of the slopes
    between paired rows, and the median of what is left.

    The rows are sorted by the column's value, then by the target. With n rows and
    m = n // 2, the i-th row is paired with row i + (n - m), for i = 0 .. m - 1; a
    pair whose two values of the column are equal has no slope. The slope is the
    median of the pairs' slopes, and the level the median of target - slope x
    value over all the rows. Returns (level, slope), or None when no pair has a
    slope or the arithmetic gives no finite line.
    """
    order = np.lexsort((y, x))
    xs, ys = x[order], y[order]
    n = len(xs)
    m = n // 2
    if m == 0:
        return None
    dx, dy = xs[n - m:] - xs[:m], ys[n - m:] - ys[:m]
    ok = dx > 0
    if not ok.any():
        return None
    slope = float(np.median(dy[ok] / dx[ok]))
    level = float(np.median(ys - slope * xs))
    if not (math.isfinite(slope) and math.isfinite(level)):
        return None
    return level, slope


def _least_squares_line(x, y):
    """M41 only: the least-squares line, which one cell far off can turn."""
    x0, y0 = float(x.mean()), float(y.mean())
    dx = x - x0
    if not float(dx @ dx) > 0:
        return None
    slope = float(dx @ (y - y0)) / float(dx @ dx)
    if not math.isfinite(slope):
        return None
    return y0 - slope * x0, slope


def _line_predict(x_fit, y_fit, x_new, fold_mean):
    """Step 8 (D74): the line through one numeric column, fitted on the rows of a
    fold's training part where the column is present. Where the column is missing
    in a validation row, and where no line can be fitted, it predicts the fold's
    training mean."""
    out = np.full(len(x_new), float(fold_mean))
    have = ~np.isnan(x_fit)
    x, y = x_fit[have], y_fit[have]
    if len(x) < 2:
        return out
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        line = _least_squares_line(x, y) if BROKEN == "M41" else _theil_line(x, y)
        if line is None:
            return out
        level, slope = line
        seen = ~np.isnan(x_new)
        out[seen] = level + slope * x_new[seen]
    return out


def _scored_rows(own_errors, base_errors, eligible=None):
    """Step 8 (D74): which pooled validation rows count for rule 2. The
    `SCREEN_UNSCORED` of the rows that the column misses by most are not scored,
    and the same share of the rows that the baseline misses by most: k = that
    share of the rows, rounded up, and at least one, from each. Among equal errors
    the earlier row goes. With too few rows for that (2k >= n) every row counts.
    `eligible` (D77): when rows are marked and could not be set aside, only a row
    that is not marked may be left unscored."""
    n = len(own_errors)
    keep = np.ones(n, dtype=bool)
    k = max(1, math.ceil(SCREEN_UNSCORED * n))
    if 2 * k < n:
        for errors in (own_errors, base_errors):
            order = np.argsort(-errors, kind="stable")
            if eligible is not None:
                order = order[eligible[order]]
            keep[order[:k]] = False
    return keep


def _level_predict(keys_fit, y_fit, keys_new, numeric):
    """D77: a column read by its levels. The mean target of each level among the rows
    of a fit, for each new row; not-a-number for a level those rows do not hold. A
    missing cell of a numeric column is no level."""
    out = np.full(len(keys_new), np.nan)
    have = ~np.isnan(keys_fit) if numeric else np.ones(len(keys_fit), dtype=bool)
    if not have.any():
        return out
    levels, inverse = np.unique(keys_fit[have], return_inverse=True)
    with np.errstate(over="ignore", invalid="ignore"):
        means = np.bincount(inverse, weights=y_fit[have], minlength=len(levels)) \
            / np.bincount(inverse, minlength=len(levels))
    at = np.searchsorted(levels, keys_new)
    at = np.minimum(at, len(levels) - 1)
    seen = levels[at] == keys_new
    out[seen] = means[at[seen]]
    return out


def _screen_column(values, numeric, y, folds, classification, seed, prev, also_fit=None,
                   by_entity=False, every_row=False, level_mode=0, marked=None, log=None):
    """D11, D74, D77: one column's cross-validated skill over the folds of the screen.

    `level_mode`: 0, the column is not read by its levels; 1, it is, for the log
    only; 2, it is. `marked`: the rows that may not be left unscored (D77).
    """
    cls = DecisionTreeClassifier if classification else DecisionTreeRegressor
    use_line = numeric and not classification and BROKEN != "F24"
    direct = classification and BROKEN != "M47"
    truth, guess, other, mean_guess, rows, level, median_guess = [], [], [], [], [], [], []
    robust = isinstance(marked, str)
    if robust:
        marked = None
    for fold in folds:
        if not fold["scored"] or not len(fold["train"]) or not len(fold["valid"]):
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
        if robust:
            median_guess.append(np.full(len(b), float(np.median(y[fold["train"]]))))
        if use_line:
            other.append(_line_predict(values[a], y[a], values[b], fold_mean))
        elif direct and numeric:
            stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1,
                                           random_state=seed).fit(xa, y[a])
            other.append(stump.predict_proba(xb)[:, 1])
        elif direct and P.B_UNSEEN == "base":
            codes_a, codes_b = xa[:, 0].astype(int), xb[:, 0].astype(int)
            share = np.bincount(codes_a, weights=y[a], minlength=len(levels)) \
                / np.maximum(np.bincount(codes_a, minlength=len(levels)), 1)
            other.append(np.where(codes_b >= 0, share[np.maximum(codes_b, 0)], fold_mean))
        elif direct:
            level.append(_level_predict(xa[:, 0], y[a].astype("float64"), xb[:, 0], False))
        if level_mode and not (classification and not numeric):
            level.append(_level_predict(xa[:, 0], y[a].astype("float64"), xb[:, 0], numeric))
    truth, guess = np.concatenate(truth), np.concatenate(guess)
    if classification:
        auc = _auc(truth, guess)
        if auc is None:
            return None, None
        found = {"tree": 2 * auc - 1}
        if other:
            second = _auc(truth, np.concatenate(other))
            if second is not None:
                found["direct"] = 2 * second - 1
        if level:
            by_level = np.concatenate(level)
            seen = ~np.isnan(by_level)
            if P.B_UNSEEN == "base":
                third = _auc(truth, np.where(seen, by_level, np.concatenate(mean_guess)))
            else:
                # D78: rows whose level was not seen are not scored; with too few rows
                # scored the reading by levels has no value
                raw = _auc(truth[seen], by_level[seen]) if seen.any() else None
                rarer = int(min(truth[seen].sum(), seen.sum() - truth[seen].sum()))
                enough = int(seen.sum()) >= P.B_MIN
                for one in (0, 1):
                    scored, held = int((truth[seen] == one).sum()), int((truth == one).sum())
                    enough = enough and (scored >= P.B_MIN_CLASS or scored == held)
                third = raw if enough else None
                if log is not None:
                    log["levels_raw"] = (None if raw is None else 2 * raw - 1, int(seen.sum()),
                                         int(len(seen)), rarer)
            if third is not None:
                found["levels"] = 2 * third - 1
        if log is not None:
            log.update({name: (value, value) for name, value in found.items()})
        if level_mode < 2 and numeric:
            found.pop("levels", None)
        skill = max(found.values())
        return skill, skill
    mean_pred = np.concatenate(mean_guess)
    base_preds = [("training_mean", mean_pred)]
    if prev is not None and by_entity:
        before = prev[np.concatenate(rows)]
        base_preds.append(("previous_value", np.where(np.isnan(before), mean_pred, before)))
    elif prev is not None:
        base_preds.append(("previous_value",
                           _previous_of(prev, np.concatenate(rows), "a validation block")))
    for name, pred in base_preds:
        error = _mse(truth, pred)
        if not math.isfinite(error):
            raise _not_finite(f"In the leak screen, the squared error of the baseline {name} "
                              "over the validation rows of the selection folds", error,
                              TARGET_CAUSE)
    own = _mse(truth, guess)
    if not math.isfinite(own):
        raise _not_finite("In the leak screen, the squared error of a single-column tree over "
                          "the validation rows of the selection folds", own, TARGET_CAUSE)
    readers = [("tree", lambda base: guess)]
    only = {}
    if other:
        line = np.concatenate(other)
        if np.isfinite(line).all() and math.isfinite(_mse(truth, line)):
            readers.append(("line", lambda base: line))
    if level:
        by_level = np.concatenate(level)
        seen = ~np.isnan(by_level)

        def from_levels(base):
            return np.where(seen, by_level, base if P.B_UNSEEN == "base" else mean_pred)
        if np.isfinite(by_level[seen]).all() \
                and all(math.isfinite(_mse(truth, from_levels(pred))) for _, pred in base_preds):
            readers.append(("levels", from_levels))
            if P.B_UNSEEN == "skip":
                only[from_levels] = seen
    eligible = None if marked is None else ~marked[np.concatenate(rows)]
    robust_preds = [("training_median", np.concatenate(median_guess))] if robust else []
    with np.errstate(over="ignore", invalid="ignore"):

        def skill_of(reader, unscored):
            """The lower of the skills against each baseline; None when one has no error."""
            lowest = None
            for _, base_pred in base_preds + (robust_preds if unscored else []):
                base_error = (truth - base_pred) ** 2
                own_errors = (truth - reader(base_pred)) ** 2
                may = eligible
                if reader in only:
                    if not only[reader].any():
                        return None
                    own_errors, base_error = own_errors[only[reader]], base_error[only[reader]]
                    may = None if eligible is None else eligible[only[reader]]
                keep = _scored_rows(own_errors, base_error, may) if unscored \
                    else np.ones(len(own_errors), dtype=bool)
                base_sum = float(base_error[keep].sum())
                if not base_sum > 0:
                    return None             # a baseline with no error cannot be beaten (D36)
                value = 1 - float(own_errors[keep].sum()) / base_sum
                lowest = value if lowest is None else min(lowest, value)
            return lowest

        each = {name: (skill_of(reader, not every_row), skill_of(reader, False))
                for name, reader in readers}
        if log is not None:
            log.update(each)
            log["seen"] = float(seen.mean()) if level else None
            if level and "levels" in each:
                log["levels_raw"] = (each["levels"][1], int(seen.sum()), int(len(seen)),
                                     each["levels"][0])
        if level_mode < 2:
            each.pop("levels", None)
        if "levels" in each and only and (int(seen.sum()) < P.B_MIN or None in each["levels"]):
            # D78: too few rows scored (or no error to beat among them): no reading by levels
            each.pop("levels")
            if log is not None:
                log.pop("levels", None)

        def best(which):
            found = [pair[which] for pair in each.values()]
            return None if any(f is None for f in found) else max(found)

        told = best(1)
        decided = told if every_row else best(0)
    return decided, told


def _against_order(x, y):
    """D76: the rows are put in the order of the column `x`, ties by the target (rows
    where either is missing are left out); among the neighbouring pairs whose
    values of `x` differ, the share in which the target moves the other way: the
    smaller of the share in which it falls and the share in which it rises. So 0
    means that the target rises, or falls, with the column without exception.
    None with fewer than `ORDER_PAIRS` such pairs."""
    have = ~(np.isnan(x) | np.isnan(y))
    order = np.lexsort((y[have], x[have]))
    xs, ys = x[have][order], y[have][order]
    moved = xs[1:] > xs[:-1]
    if int(moved.sum()) < ORDER_PAIRS:
        return None
    up = float((ys[1:][moved] > ys[:-1][moved]).mean())
    down = float((ys[1:][moved] < ys[:-1][moved]).mean())
    return min(up, down)


def _order_share(x, y, form, seed=0):
    """D77: the share against the order, under each form measured for the plan.

    old     rows of equal column value in the target's order (D76 as built)
    random  rows of equal column value in an order drawn from the seed
    mean    one point for each distinct value: the mean target of its rows
    median  one point for each distinct value: the median target of its rows
    pairs   for each two neighbouring distinct values, every row of the one with
            every row of the other; the shares are averaged over the neighbours
    """
    if form == "old":
        return _against_order(x, y)
    have = ~(np.isnan(x) | np.isnan(y))
    x, y = x[have], y[have]
    if not len(x):
        return None
    if form == "random":
        order = np.lexsort((np.random.default_rng([seed, 9]).random(len(x)), x))
        xs, ys = x[order], y[order]
        moved = xs[1:] > xs[:-1]
        if int(moved.sum()) < ORDER_PAIRS:
            return None
        return min(float((ys[1:][moved] > ys[:-1][moved]).mean()),
                   float((ys[1:][moved] < ys[:-1][moved]).mean()))
    levels, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(levels) - 1 < ORDER_PAIRS:
        return None
    if form in ("mean", "median"):
        if form == "mean":
            with np.errstate(over="ignore", invalid="ignore"):
                point = np.bincount(inverse, weights=y) / counts
        else:
            ys = y[np.lexsort((y, inverse))]
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            point = (ys[starts + (counts - 1) // 2] + ys[starts + counts // 2]) / 2
        return min(float((point[1:] > point[:-1]).mean()), float((point[1:] < point[:-1]).mean()))
    # pairs: every row of a value with every row of the next value
    _, rank = np.unique(y, return_inverse=True)
    width = int(rank.max()) + 2
    key = np.sort(inverse.astype("int64") * width + rank)
    starts = np.r_[0, np.cumsum(counts)]
    inner = inverse < len(levels) - 1
    this, nxt, r = inverse[inner], inverse[inner] + 1, rank[inner]
    below = np.searchsorted(key, nxt * width + r, side="left") - starts[nxt]
    above = starts[nxt + 1] - np.searchsorted(key, nxt * width + r, side="right")
    weight = 1.0 / (counts[this] * counts[nxt])
    up = np.bincount(this, weights=above * weight, minlength=len(levels) - 1)
    down = np.bincount(this, weights=below * weight, minlength=len(levels) - 1)
    return min(float(up.mean()), float(down.mean()))


def _far_outside(values):
    """D74, D76: the distinct values of one numeric column that lie far outside all
    its other values. Returns [(value, rows that hold it, nearest other value), ...].

    The distinct finite values are put in order. From each end, the
    `EXTREME_OUTERMOST` outermost are looked at: a value is far outside when the
    gap between it and the next value inward is more than `EXTREME_GAP` times the
    spread of the `EXTREME_SPREAD` distinct values after that one (as many as the
    column has, when fewer; at least one). So are the values beyond it. Judged on
    distinct values, so a value counts once however many rows hold it, and a
    column needs three distinct values for one of them to be far from the others.
    M46: the gap is compared with the whole range of the other values.
    """
    values = values[np.isfinite(values)]
    distinct, counts = np.unique(values, return_counts=True)
    n, found = len(distinct), []
    if n < 3:
        return found
    for d, held in ((distinct, counts), (distinct[::-1], counts[::-1])):
        edge = -1
        for j in range(min(EXTREME_OUTERMOST, n - 2)):
            m = n - (j + 2) if BROKEN == "M46" else min(EXTREME_SPREAD, n - (j + 2))
            gap = abs(float(d[j + 1]) - float(d[j]))
            spread = abs(float(d[j + 1 + m]) - float(d[j + 1]))
            if spread > 0 and gap > EXTREME_GAP * spread:
                edge = j
        found += [(float(d[j]), int(held[j]), float(d[edge + 1])) for j in range(edge + 1)]
    return found


def _largest_row(errors):
    """D74: the share of a sum of squared errors that its largest row carries, and
    where that row is (the first of several). (None, None) when every error is
    zero: there is nothing to have a share of. The errors are divided by the
    largest before they are added, so the sum cannot overflow while each error is
    finite; an error that is not finite stops the run (D44)."""
    at = int(np.argmax(errors))
    top = float(errors[at])
    if not math.isfinite(top):
        raise _not_finite("The largest squared error of one row", top, TARGET_CAUSE)
    if not top > 0:
        return None, None
    return 1.0 / float((errors / top).sum()), at


_CORES = [1]            # F22 only: the cores of the run in hand


def _cube_root_up(n):
    """Smallest whole L with L**3 >= n."""
    length = max(1, round(n ** (1 / 3)))
    while length ** 3 < n:
        length += 1
    while length > 1 and (length - 1) ** 3 >= n:
        length -= 1
    return length


def _uncertainty(y, pred, base_pred, classification, time_groups, n_resamples, seed, score,
                 skill_defined=True, entity_groups=None):
    """Step 12 (D9): interval on the holdout score and skill, and the p-value.

    `entity_groups`, under the grouped scheme: the holdout rows of each entity.
    A resample then draws whole entities, and there is no p-value (D57).
    """
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

    rng = np.random.default_rng([seed, 5, _CORES[0]] if BROKEN == "F22" else [seed, 5])
    n = len(y)
    if entity_groups is not None and len(entity_groups[1]) < MIN_BOOT_GROUPS:
        draws = ()
        out["interval_reason"] = (f"the holdout has {len(entity_groups[1])} entities; a "
                                  f"bootstrap of whole entities needs at least {MIN_BOOT_GROUPS}")
    elif entity_groups is not None:
        # `entity_groups`: the holdout rows put entity by entity, and each entity's size
        by_entity, sizes = entity_groups
        held, starts = len(sizes), np.r_[0, np.cumsum(sizes)[:-1]]
        out["interval_method"] = "bootstrap_groups"

        def entities():
            for _ in range(n_resamples):
                if BROKEN == "M21":             # rows one by one, whatever their entity
                    yield rng.integers(0, n, n)
                    continue
                drawn = rng.integers(0, held, held)
                size = sizes[drawn]
                at = np.repeat(starts[drawn] - np.r_[0, np.cumsum(size)[:-1]], size)
                yield by_entity[at + np.arange(int(size.sum()))]
        draws = entities()
    elif time_groups is None:
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
    if entity_groups is not None:
        out["p_value_reason"] = NO_P_VALUE_GROUPED
        return out
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

def _repeated_rows(df):
    """How many rows exactly repeat an earlier row (`input.duplicate_rows`).

    A cell that cannot be hashed (a list, a dictionary, a set) is compared by its
    text (D72): pandas raises on such a column, and until D72 the count was then
    0. M39: it still is.
    """
    try:
        return int(df.duplicated().sum())
    except (TypeError, ArithmeticError):
        if BROKEN == "M39":
            return 0

    def text_of(value):
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    held = df.copy()
    for name in held.columns:
        if held[name].dtype == object:
            held[name] = held[name].map(text_of)
    try:
        return int(held.duplicated().sum())
    except (TypeError, ArithmeticError):
        return int(held.astype(str).duplicated().sum())


def _missing_cells(series):
    """Which cells of a column are empty, without asking pandas about a decimal."""
    if series.dtype == object:
        return np.fromiter((_cell_missing(v) for v in series.to_numpy(dtype=object)),
                           dtype=bool, count=len(series))
    return series.isna().to_numpy().copy()


def _run(data, target, requested, time_col, gap, profile, seed, n_jobs, duplicates="error",
         group_col=None):
    warnings = []
    _CORES[0] = n_jobs
    # `time_col=False` is the caller's statement that the rows have no time order,
    # and `group_col=False` that there are no entities (D63). Either way the column
    # is not used: what the caller passed is reported in `input`.
    stated_time, time_col = time_col, (None if time_col is False else time_col)
    group_name = group_col if isinstance(group_col, str) else None
    # 1. load; count the rows
    df, source_kind, source_label = _load(data, warnings)
    if BROKEN == "F6":
        df = df.dropna().reset_index(drop=True)     # rows vanish before anyone counts them
    for name in (target, time_col, group_name):
        if name is not None and name not in df.columns:
            raise E.ColumnNotFoundError(f"no column named {name!r}")
    n_loaded = len(df)
    duplicate_rows = _repeated_rows(df) if n_loaded else 0
    if duplicate_rows:
        _warn(warnings, "duplicate_rows",
              f"{duplicate_rows} row(s) are exact copies of an earlier row; they are kept")
    # An object column that holds only numbers is read as numbers from here on, in
    # every role (D51). The output goes on reporting the type it was loaded with.
    # Each column is read once, here, over all of its loaded rows (D57).
    as_loaded = df
    df, loaded, beyond = _numbers_from_objects(df)

    # 2. drop rows with a missing target, then rows with a missing time value. A text
    # target is read without the white space round it from here on, and a cell that
    # holds nothing else is a missing target (D46, D48).
    target_dtype = loaded[target]
    y_all = _target_values(df[target])
    no_target = _no_target(y_all)
    beyond_target = target in beyond
    time_form, show = None, None
    no_time = np.zeros(n_loaded, dtype=bool)
    if time_col is not None:
        # 3. read the time column in its accepted form
        if BROKEN == "M14":
            # M14: the column is read a second time, over the rows that will be used
            # only, so an odd cell in a row with no target is never seen
            held = ~no_target & ~_missing_cells(as_loaded[time_col])
            again = as_loaded[time_col]
            if again.dtype == object:
                again = again.where(pd.Series(held, index=again.index), None)
                typed = _object_numbers(again)
                again = again if typed is None else typed
            keys_all, missing_time, time_form, show = _time_keys(again, time_col, judged=held)
        else:
            keys_all, missing_time, time_form, show = _time_keys(df[time_col], time_col)
        no_time = missing_time & ~no_target
    usable = ~no_target & ~no_time
    if usable.sum() < MIN_ROWS:
        raise E.TooFewRowsError(f"{int(usable.sum())} usable rows; at least {MIN_ROWS} are needed")
    if BROKEN == "M14" and as_loaded[target].dtype == object:
        again = as_loaded[target].where(pd.Series(usable, index=as_loaded.index), None)
        typed = _object_numbers(again)
        again = again if typed is None else typed
        y_all, beyond_target = _target_values(again), _beyond_64_bits(again)

    # The entity column (D57): one whole number per row. An empty cell in a row that
    # will be used is the named error: such a row belongs to no entity the tool knows.
    groups_all = None
    if group_name is not None:
        groups_all = _group_codes(df[group_name], group_name)
        empty = int((groups_all[usable] < 0).sum())
        if empty:
            raise E.GroupColumnError(
                f"entity column {group_name!r} is empty in {empty} of the {int(usable.sum())} "
                "rows that have a target. A row with no entity cannot be kept on one side of "
                "the split with the rest of its entity: fill the cells in, or remove the rows")

    # 4. decide the task
    task, decided_by, n_distinct, numeric_target = _decide_task(y_all[usable], target, requested,
                                                                beyond_target)
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
    records, prepared = _roles(df, usable, target, time_col, loaded, group_name)
    by_name = {r["name"]: r for r in records}
    undeclared = []
    for rec in records:
        if rec["reason"] == "unsupported_type":
            # never excluded in silence (D51): the caller may have meant it as a feature
            if rec["name"] in beyond:
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
        if rec["n_infinite"] and rec["role"] not in ("target", "time"):
            _warn(warnings, "infinite_values",
                  f"{rec['n_infinite']} value(s) in {rec['name']!r} are infinite, or too large "
                  f"for the tree models (a magnitude above {TOO_LARGE!r}, the largest 32-bit "
                  "float), and are treated as missing", rec["name"])
        if rec["reason"] == "datetime" and time_col is None:
            undeclared.append(rec["name"])
            # its text ends with the caveat of the scheme, which step 7 decides (D69)
            if BROKEN != "F18":
                _warn(warnings, "date_column_undeclared", "", rec["name"])
    # D58: a column of dates and no word from the caller about time order. The tool
    # does not go ahead under a random split on its own say: on the VIX level that
    # gave skill 0.939, against 0.023 with the time column declared. F7d: it does.
    # D63: every call says two things, and the tool assumes neither. F11: it assumes
    # both. F7d: it assumes time order away where a column holds dates. M19: it
    # assumes there are no entities under a time column. M24: with both unsaid it
    # asks for the one and leaves the other for the next call.
    entities = ("Nothing was said about entities: rows that belong together, as the rows of "
                "one customer or one series do. Pass group_col (--group-col) with the name of "
                "the column that says which entity a row belongs to, or group_col=False "
                "(--no-entities) if no rows belong together")
    if stated_time is None and BROKEN != "F11" and not (BROKEN == "F7d" and undeclared):
        seen = ("Column(s) " + ", ".join(repr(name) for name in undeclared) + " hold dates. "
                if undeclared else "")
        raise E.TimeOrderUndeclaredError(
            "Nothing was said about time order. " + seen + "Pass time_col (--time-col) with "
            "the name of the time column if the rows are ordered in time: the holdout is then "
            "the latest rows. Pass time_col=False (--no-time-order) if their order in time "
            "does not matter: the rows are then split at random. "
            + (entities + ". " if group_col is None and BROKEN != "M24" else "")
            + "Nothing was written")
    if group_col is None and BROKEN != "F11" and not (BROKEN == "M19" and time_col is not None):
        seen = ""
        if time_col is not None:
            stamps = len(np.unique(keys_all[usable]))
            if stamps < int(usable.sum()):
                seen = (f" The timestamps repeat: {int(usable.sum())} rows on {stamps} "
                        "timestamps, which is what several series in one table look like.")
        raise E.EntitiesUndeclaredError(entities + "." + seen + " Nothing was written")
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
    if profile == "fast" and len(rows) > FAST_ROW_CAP and time_col is None \
            and groups_all is not None and BROKEN != "F15":
        # D70, D72: the cap keeps whole entities. One of more rows than the cap could
        # only be left out; and one that holds a quarter or more of the usable rows is
        # never left to the draw: left out, the run would describe another table. So
        # such entities are always kept (`_cap_groups`), and when they cannot be, the
        # run stops and sends the caller to the full profile. For entity columns
        # only: under `duplicates="group"` the sets are formed among the rows the
        # cap keeps. F15: left to the draw, and only the row count says so.
        whole = np.unique(groups_all[rows], return_inverse=True)[1]
        sizes = np.bincount(whole)
        large = sizes[4 * sizes >= len(rows)]
        if BROKEN == "M35":                 # M35: only an entity above the cap stops the run
            large = large[:0]
        if int(sizes.max()) > FAST_ROW_CAP or int(large.sum()) > FAST_ROW_CAP:
            y_rows = y_work_all[rows]
            if BROKEN != "M34":             # M34: the full profile's own error is masked
                if classification and min(int(y_rows.sum()),
                                          int(len(y_rows) - y_rows.sum())) < MIN_CLASS_ROWS:
                    raise E.TooFewRowsError(
                        f"the rarer class has fewer than {MIN_CLASS_ROWS} usable rows")
                _split_grouped(whole, y_rows, classification, seed)     # its own error
            if int(sizes.max()) > FAST_ROW_CAP:
                found = (f"one entity of the column {group_name!r} holds {int(sizes.max())} of "
                         f"the {len(rows)} usable rows "
                         f"({100 * int(sizes.max()) / len(rows):.1f}%), more than the fast "
                         f"profile's cap of {FAST_ROW_CAP} rows")
            else:
                found = (f"{len(large)} entities of the column {group_name!r} each hold a "
                         f"quarter or more of the {len(rows)} usable rows and together "
                         f"{int(large.sum())}, more than the fast profile's cap of "
                         f"{FAST_ROW_CAP} rows")
            raise E.FastProfileGroupError(
                f"{found}. The cap keeps whole entities and never leaves out one that holds a "
                "quarter or more of the rows: the result would describe another table. The "
                "full profile can split these rows: use profile='full'")
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
        elif groups_all is not None:
            # whole entities, where they fit (D57)
            keep = _cap_groups(np.unique(groups_all[rows], return_inverse=True)[1], seed)
            capped, rows = len(rows) - len(keep), rows[keep]
        else:
            keep = np.sort(np.random.default_rng([seed, 1]).choice(len(rows), FAST_ROW_CAP,
                                                                   replace=False))
            if BROKEN == "M10":             # the head of the file, not a sample from the seed
                keep = np.arange(FAST_ROW_CAP)
            capped, rows = len(rows) - FAST_ROW_CAP, rows[keep]
    y = y_work_all[rows]
    groups = None if groups_all is None else np.unique(groups_all[rows], return_inverse=True)[1]
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
            keys_all[usable_rows] if time_col is not None else None, gap,
            groups_all[usable_rows] if groups_all is not None else None, seed)
    if too_few:
        raise E.TooFewRowsError(too_few)

    # 7. split off the holdout, applying the gap
    time_ordered = time_col is not None
    grouped, units, n_same_features = False, None, None
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
        def units_of(part):
            """The rows `part` (positions in the loaded table) as whole units: the
            declared entities, joined under `duplicates="group"` with the sets of rows
            that agree in every feature column (D57, D62). None when every row stands
            alone. Also the number of rows that agree with another, or None if not
            asked."""
            found = None
            if groups_all is not None:
                found = np.unique(groups_all[part], return_inverse=True)[1]
            same_features = None
            if duplicates == "group":
                # D62: rows that agree in every feature column of step 5. The target
                # plays no part. F12: it does, as built in the fifth pass.
                copy_of = _row_keys(list(prepared.values()),
                                    y_work_all if BROKEN == "F12" else None, part)
                same_features = int((np.bincount(copy_of)[copy_of] > 1).sum())
                if same_features and BROKEN != "M22":     # M22: found, counted, and not used
                    found = copy_of if found is None else _clusters([copy_of, found])
                if BROKEN == "M25":
                    # M25: the sets are right, and the count still reads the target
                    counted = _row_keys(list(prepared.values()), y_work_all, part)
                    same_features = int((np.bincount(counted)[counted] > 1).sum())
            return found, same_features

        if not time_ordered:
            units, n_same_features = units_of(rows)
            if n_same_features and BROKEN != "M22":
                _warn(warnings, "same_features_grouped",
                      f"{n_same_features} of the {len(rows)} rows agree with another row in "
                      "every feature column. With duplicates='group' the rows that agree are "
                      "kept together: no combination of feature values is on both sides of "
                      "the split or in two folds"
                      + ("" if BROKEN == "M37" else ". " + GROUP_MEASURES))
        if units is not None and BROKEN != "F10":
            try:
                split = _split_grouped(units, y, classification, seed)
            except E.TooFewGroupsError as exc:
                if not capped:
                    raise
                # The cap left too few entities. Say "use the full profile" only when
                # the full profile would form its split (D52, D57).
                try:
                    _split_grouped(units_of(usable_rows)[0], y_work_all[usable_rows],
                                   classification, seed)
                except E.TooFewGroupsError as full:
                    raise full from None        # the full profile would stop too: its error
                message = (f"the fast profile's cap of {FAST_ROW_CAP} rows kept too few "
                           f"entities: {exc}. Every usable row together holds enough: "
                           "use profile='full'")
                raise E.FastProfileRowsError(message) from exc
        else:
            split = _split_random(len(rows), y, classification, seed)   # F7: shuffled regardless
        grouped = units is not None             # F10: said, and not done
        for size, total, is_holdout in split.get("too_large", []):
            found = (f"One unit of {size} rows holds {100 * size / total:.1f}% of the {total} "
                     "rows" + (" of the units of its class. " if classification else ". "))
            if is_holdout:
                _warn(warnings, "large_group_is_holdout", found + LARGE_IS_HOLDOUT)
            elif BROKEN != "M27":           # M27: only the unit that is the holdout is named
                _warn(warnings, "large_group_kept_in_training", found + LARGE_KEPT)
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
    if time_ordered and BROKEN != "F7" and P.E > 0:
        held_share = len(holdout) / (len(train) + len(holdout) + len(split["gap"]))
        if held_share >= P.E:
            first = int((keys[holdout] == keys[holdout].min()).sum())
            _warn(warnings, "holdout_far_above_quarter",
                  f"The holdout holds {len(holdout)} of the "
                  f"{len(train) + len(holdout) + len(split['gap'])} rows "
                  f"({100 * held_share:.1f}%), far more than a quarter: it is made of whole "
                  f"timestamps, and its earliest timestamp alone holds {first} rows. "
                  + LARGE_HOLDOUT)
    for w in warnings:
        if w["code"] == "date_column_undeclared":
            w["message"] = (f"column {w['column']!r} holds dates and time_col=False was passed, "
                            "so the rows were not split by time. "
                            + (GROUPED_CAVEAT if grouped and BROKEN != "M36"
                               else RANDOM_CAVEAT))
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
    if unscored and BROKEN == "M38":             # M38: one record for the run
        _warn(warnings, "fold_not_scored",
              f"fold(s) {', '.join(map(str, unscored))} hold a single class and were not scored")
    elif unscored:
        for k in unscored:                          # one for each fold not scored (D72)
            _warn(warnings, "fold_not_scored",
                  f"fold {k} holds a single class in its training or its validation rows "
                  "and was not scored")
    used = np.concatenate([train, holdout])
    for name, values in prepared.items():
        # counted over the rows that reach a model. M13: over every loaded row
        cell = values if BROKEN == "M13" else values[rows[used]]
        by_name[name]["n_imputed"] = int(np.isnan(cell).sum()) if cell.dtype != object \
            else int((cell == MISSING_LEVEL).sum())

    # The previous value (D10, D57, D74). With no entity column: the mean target of
    # the timestamp `gap + 1` distinct timestamps earlier. With one: each entity's
    # own, from its latest row at least `gap + 1` timestamps earlier; a row whose
    # entity has no such row has none.
    prev, by_entity, panel_warning = None, False, None
    if time_ordered and not classification:
        if groups is not None:
            _, stamp = np.unique(keys, return_inverse=True)
            cell = groups.astype("int64") * (int(stamp.max()) + 1) + stamp
            if len(np.unique(cell)) != len(cell):
                twice = int(len(cell) - len(np.unique(cell)))
                raise E.GroupColumnError(
                    f"entity column {group_name!r}: {twice} row(s) share their entity and their "
                    "timestamp with another row, so an entity is not one series and has no "
                    "one previous value. Declare a column that names one series (join the "
                    "columns that together do), or pass group_col=False")
            prev, by_entity = _previous_by_group(y, keys, groups, gap), True
        else:
            if split["unique"] or BROKEN != "F23":
                # D10, D74: the mean target of the timestamp `gap + 1` distinct
                # timestamps before. With unique timestamps that is the target of
                # the row before. F23: only with unique timestamps, as it was, so
                # one repeated row leaves the training mean as the only baseline.
                # M40: the target of the last row of that timestamp, not the mean.
                _, stamp = np.unique(keys, return_inverse=True)
                with np.errstate(over="ignore", invalid="ignore"):
                    of_stamp = np.bincount(stamp, weights=y) / np.bincount(stamp)
                if BROKEN == "M40":
                    of_stamp = np.zeros(int(stamp.max()) + 1)
                    of_stamp[stamp] = y
                prev = np.full(len(y), np.nan)
                has = stamp - gap - 1 >= 0
                prev[has] = of_stamp[stamp[has] - gap - 1]
        if groups is None and not split["unique"]:
            lag1 = _lag1_of_stamp_means(keys[train], y[train])
            # Means that do not vary have no autocorrelation. One that cannot be
            # computed because a value overflows is not passed over (D44).
            if lag1 is not None and not math.isfinite(lag1):
                raise _not_finite("The lag-1 autocorrelation of the per-timestamp mean of the "
                                  "target over the training timestamps", lag1, TARGET_CAUSE)
            # M23: the cut-off the first harness session abandoned, 0.8
            if lag1 is not None and lag1 > (0.8 if BROKEN == "M23" else AUTOCORR):
                # The caller has said that there are no entities (D63); the tool's doubt
                # about that statement is a warning, as it was (D10, D58). Its text
                # names the baseline the run used (D74), so it is written at the end.
                panel_warning = lag1

    # The baselines of step 11 need no model, only the target and the split, so they
    # are scored on the holdout here, before the leak screen and before any fit. One
    # whose score is not finite stops the run at once: when the target itself is the
    # cause, the caller gets one clear error, and never a list of configurations that
    # failed one after another (D44). Both are checked, the unused one too: its score
    # is a number in the output.
    truth = y[holdout]
    n_without_previous = None
    if classification:
        baselines, base_pred = [{"name": "chance", "score": 0.5, "used": True}], None
    else:
        base_pred = np.full(len(holdout), _mean(y[train]))
        baselines = [{"name": "training_mean", "score": _mse(truth, base_pred), "used": True}]
        # F8b: with entities declared, the headline is still scored against the training
        # mean alone
        if prev is not None and not (by_entity and BROKEN == "F8b"):
            if by_entity:
                # a row whose entity has no earlier row gets what is known of it: the
                # training mean (D57)
                before = prev[holdout]
                n_without_previous = int(np.isnan(before).sum())
                before = np.where(np.isnan(before), base_pred, before)
            else:
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
        if n_without_previous:
            _warn(warnings, "no_previous_value",
                  f"{n_without_previous} of the {len(holdout)} holdout rows belong to an entity "
                  "with no earlier row, so they have no previous value; for them the "
                  "previous-value baseline predicts the training mean")

    # 8. from here on every fitted step sees training rows only: the leak screen first
    screen_started = time.perf_counter()
    if BROKEN != "F2":
        # M3: the screen looks at rows it must not see
        judged = np.arange(len(rows)) if BROKEN == "M3" else train
        # M9, and M16 for a panel: the screen forgets the previous-value baseline
        screen_prev = None if BROKEN == "M9" or (BROKEN == "M16" and by_entity) else prev
        # D74: rows whose target is wildly off are set aside from the whole screen
        screen_folds, _, every_row, marked = _screen_folds(y, train, split["folds"],
                                                           screen_prev, by_entity,
                                                           classification)
        if P.DEBUG:
            P.LOG["screen"], P.LOG["order"], P.LOG["levels"] = {}, {}, {}
        by_its_levels = {}
        for name in list(prepared):
            values, numeric = prepared[name][rows], by_name[name]["kind"] == "numeric"
            rule = None
            if classification:
                gone = np.isnan(values[judged]) if numeric else values[judged] == MISSING_LEVEL
                share = [gone[y[judged] == c].mean() for c in (0, 1)]
                if max(share) > LEAK_MISSING_HIGH and min(share) < LEAK_MISSING_LOW:
                    rule = "missing_by_class"
            alone = told = None
            if rule is None:
                # None: the skill cannot be computed, so the column is not flagged (D36).
                # Two readings of one measurement: `alone` decides rule 2, `told` rule 3.
                # D77: a text column, and a numeric column of few distinct values, is
                # also read by its levels
                few = True
                if numeric:
                    held = values[train]
                    held = held[~np.isnan(held)]
                    distinct = len(np.unique(held))
                    few = distinct > 0 and len(held) >= P.B_ROWS * distinct \
                        and (P.B_MAX == 0 or distinct <= P.B_MAX)
                    if P.DEBUG:
                        P.LOG["levels"][name] = (int(distinct), int(len(held)))
                level_mode = 0 if P.B != "levels" else 2 if few else (1 if P.DEBUG else 0)
                seen_log = {} if P.DEBUG else None
                alone, told = _screen_column(values, numeric, y, screen_folds, classification,
                                             seed, screen_prev,
                                             holdout if BROKEN == "M3" else None, by_entity,
                                             every_row, level_mode, marked, seen_log)
                if P.DEBUG:
                    P.LOG["screen"][name] = dict(seen_log, alone=alone, told=told,
                                                 level_mode=level_mode)
                by_its_levels[name] = level_mode == 2 or (classification and not numeric
                                                          and BROKEN != "M47")
                if alone is not None and alone >= LEAK_SKILL:
                    rule = "predicts_target"
            if rule is None and told is not None and told >= STRONG_SKILL:  # kept, told (D30)
                _warn(warnings, "strong_single_column",
                      f"column {name!r} on its own predicts the target with skill {told:.2f} "
                      "on the training rows. It is kept as a feature: check that it is known "
                      "before the outcome", name)
            elif rule is None and told is not None and numeric and not classification:
                # D76: the second reading of the warning tier. With the training rows
                # in the column's order the target moves the other way between at most
                # a tenth of neighbouring rows, and it does not follow the previous
                # value as closely. A warning only. Like the first reading it is not
                # taken of a column whose skill cannot be computed (D36). M43: no
                # second condition. M44: the column is excluded for it.
                share = _order_share(values[train], y[train], P.C, seed)
                before = None if screen_prev is None else _order_share(screen_prev[train],
                                                                       y[train], P.C, seed)
                if P.DEBUG:
                    P.LOG["order"][name] = {
                        form: (_order_share(values[train], y[train], form, seed),
                               None if screen_prev is None
                               else _order_share(screen_prev[train], y[train], form, seed))
                        for form in P.C_FORMS}
                if share is not None and share <= ORDER_AGAINST and (
                        BROKEN == "M43" or before is None or before > ORDER_AGAINST):
                    if BROKEN == "M44":
                        rule = "predicts_target"
                    else:
                        _warn(warnings, "strong_single_column",
                              f"put in the order of column {name!r}, the target moves the "
                              f"other way between {100 * share:.1f}% of neighbouring training "
                              "rows: the column follows the target almost without exception. "
                              "It is kept as a feature: check that it is known before the "
                              "outcome", name)
            if rule:
                by_name[name].update(role="excluded", reason="suspected_leak", leak_rule=rule,
                                     n_imputed=0)
                del prepared[name]
                why = ("it is filled in for one class and empty for the other"
                       if rule == "missing_by_class"
                       else "on its own it predicts the target almost perfectly")
                _warn(warnings, "suspected_leak",
                      f"column {name!r} is excluded as a suspected leak: {why}"
                      + (". If it names the entity each row belongs to, declare it as "
                         "group_col (--group-col)"
                         if rule == "predicts_target" and by_its_levels.get(name) else ""),
                      name)
    if P.DEBUG:
        P.LOG["screen_seconds"] = time.perf_counter() - screen_started
    # D74, D76: a numeric feature, or a regression target, that holds a value far
    # outside all its other values, among the rows that are split. A warning only: no
    # number is changed and nothing is excluded for it.
    looked = [(name, values[rows]) for name, values in prepared.items()
              if by_name[name]["kind"] == "numeric"]
    if not classification:
        looked.append((target, y))
    for name, held in looked:
        for value, n_held, nearest in _far_outside(np.asarray(held, dtype="float64")):
            _warn(warnings, "extreme_value",
                  f"column {name!r} holds the value {value!r} in {n_held} "
                  f"row{'' if n_held == 1 else 's'}, far outside all its other values (the "
                  f"nearest is {nearest!r}). It is used as it is: no number of this run "
                  "leaves it out. If it stands for a missing value, replace it with an empty "
                  "cell", name)
    if not prepared:
        excluded = "; ".join(f"{r['name']} ({r['reason']})" for r in records
                             if r["role"] == "excluded")
        raise E.NoFeaturesError("no usable feature column is left. Excluded, with the reason "
                                "for each: " + (excluded or "none"))

    # Copies of a holdout row among the training rows (D49, D53). Not under time
    # order: there rows at different times are different observations, and rows at
    # one timestamp are on one side of the split. Judged on the features that are
    # left, before any model is fitted, so a run that stops has cost nothing.
    n_copies = copy_share = None
    leaks = [r["name"] for r in records if r["reason"] == "suspected_leak"]
    if time_col is None and duplicates == "group" and leaks and BROKEN != "F12b":
        # D69: the units were formed on the features of step 5. With a column excluded
        # by the screen, rows that differed only in it agree in every feature that is
        # left, whatever their targets, and may be on both sides: a held-out
        # combination then has rows among the training rows after all. F12b: only
        # the rows that are copies, target and all, are counted, as built.
        code = _row_keys([values[rows] for values in prepared.values()], None,
                         np.concatenate([train, holdout]))
        both = np.isin(code[len(train):], code[:len(train)])
        # M33: the stop keeps the threshold of the copies rule, 1% and ten rows
        enough = both.any() if BROKEN != "M33" else \
            (both.sum() >= COPY_ROWS and both.mean() > COPY_SHARE)
        if enough:
            n_both, combos = int(both.sum()), len(np.unique(code[len(train):][both]))
            # every message of this error names a date column that was excluded because
            # no time column was declared (spec 11): rows that differ only there agree too
            dates = [r["name"] for r in records if r["reason"] == "datetime"]
            raise E.HoldoutCopiesError(
                f"{n_both} of the {len(holdout)} holdout rows "
                f"({100 * n_both / len(holdout):.1f}%) agree with a training row in every "
                f"feature column that is left: {combos} combination(s) of feature values are "
                "on both sides of the split. duplicates='group' kept together the rows that "
                "agreed in every feature column before the split, and the leak screen then "
                "excluded " + ", ".join(repr(n) for n in leaks) + ". Remove the excluded "
                "column(s) and run again. " + COPIES_ADVICE + (
                    " Note: the column(s) " + ", ".join(repr(d) for d in dates)
                    + " hold dates and were excluded because no time column was declared, "
                    "so rows that differ only in their date agree as well. If the rows are "
                    "ordered in time, declare time_col: under time order this rule does not "
                    "apply." if dates else ""))
    if time_col is None:
        n_copies = _holdout_copies([values[rows] for values in prepared.values()], y,
                                   train, holdout)
        copy_share = n_copies / len(holdout)
        if n_copies >= COPY_ROWS and copy_share > COPY_SHARE:
            found = (f"{n_copies} of the {len(holdout)} holdout rows ({100 * copy_share:.1f}%) "
                     "have a copy among the training rows: the same value in every feature "
                     "column and the same target")
            if duplicates != "keep" and BROKEN != "F9":     # F9: the rule is never applied
                dates = [r["name"] for r in records if r["reason"] == "datetime"]
                after = ""
                if duplicates == "group":
                    # the sets were formed before the leak screen, on the features of step 5
                    leaks = [r["name"] for r in records if r["reason"] == "suspected_leak"]
                    after = (" duplicates='group' kept together the rows that agreed in every "
                             "feature column before the split; these rows became copies when "
                             "the leak screen excluded "
                             + (", ".join(repr(n) for n in leaks) or "a column")
                             + ". Remove the excluded column(s) and run again.")
                raise E.HoldoutCopiesError(
                    f"{found}. A model can score on such a row by recalling its copy, so the "
                    "headline would overstate how predictable the target is. "
                    + COPIES_ADVICE + after + (
                        " Note: the column(s) " + ", ".join(repr(d) for d in dates)
                        + " hold dates and were excluded because no time column was declared, "
                        "so rows that differ only in their date count as copies. If the rows "
                        "are ordered in time, declare time_col: under time order this rule "
                        "does not apply." if dates else ""))
            if duplicates == "keep":
                _warn(warnings, "holdout_copies_kept",
                      f"{found}, and were kept (duplicates='keep'). A model can score on such "
                      "a row by recalling its copy, so the headline may overstate how "
                      "predictable the target is")
    kinds = {name: by_name[name]["kind"] for name in prepared}
    index = pd.Index(rows, name="row")
    step_makers = extensions.active()["steps"] or []
    final_fit = {}

    def feature_frame(part):
        return pd.DataFrame({name: values[rows[part]] for name, values in prepared.items()},
                            index=index[part])

    def matrices(fit_rows, new_rows, final=False):
        """Preprocessing fitted on `fit_rows` only, applied to both sets of rows.

        The rows of a fit are handed over in the order they have here: under time
        order that is time order, ties in file order (D74 point 4); otherwise file
        order. M49: file order under time order too, so a model that reads its
        rows by position (a forest) sees them in another order.
        """
        if BROKEN == "M49" and time_ordered:
            fit_rows = fit_rows[np.argsort(rows[fit_rows], kind="stable")]
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
        if final:
            final_fit.update(prep=prep, names=names)
        frames = (pd.DataFrame(x_fit, index=fit_frame.index, columns=names),
                  pd.DataFrame(x_new, index=new_frame.index, columns=names), y_fit)
        return x_fit, x_new, frames, y[fit_rows]

    def finite(guess, n_rows):
        """One finite number per row, or the configuration fails, as it would by
        raising (D44). `_fit_predict` has read the shape. M6: a value that is not
        finite is scored and written as it is."""
        if guess.shape != (n_rows,):
            raise ValueError(f"the model returned predictions of shape {guess.shape} for "
                             f"{n_rows} rows")
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
                          cv_mean=None, cv_std=None, chosen=False,
                          largest_row_share=None, largest_row=None)
            failed = None
            cv_errors, cv_rows = [], []
            for k, (fold, mats) in enumerate(zip(split["folds"], fold_data), 1):
                if mats is None:
                    record["fold_scores"].append(None)
                    continue
                # Only what the model does is inside the `try`: it is made, fitted, asked
                # to predict, and its answer read. A configuration that raises there is
                # recorded, never dropped.
                try:
                    model = family["make"](task, params, seed, n_jobs)
                    guess = finite(_fit_predict(family, model, mats[0], mats[3],
                                                mats[1], classification, mats[2]),
                                   len(fold["valid"]))
                except Exception as exc:
                    failed = f"{type(exc).__name__}: {exc}"
                    break
                # The tool's own scoring is outside it: a fault there is the tool's, and
                # stops the run with its own exception (D57).
                fold_truth = y[fold["valid"]]
                fold_score = _auc(fold_truth, guess) if classification \
                    else _mse(fold_truth, guess)
                if not math.isfinite(fold_score):
                    # failed, like one that raised: never a null where a score belongs
                    failed = f"ValueError: its score on fold {k} is not finite ({fold_score!r})"
                    break
                record["fold_scores"].append(fold_score)
                if not classification:
                    with np.errstate(over="ignore", invalid="ignore"):
                        cv_errors.append((fold_truth - guess) ** 2)
                    cv_rows.append(fold["valid"])
            if failed is None and cv_errors:
                # D74: the largest row's share of this configuration's squared error
                # over the validation rows of the scored folds
                share, at = _largest_row(np.concatenate(cv_errors))
                if share is not None:
                    record["largest_row_share"] = share
                    record["largest_row"] = int(rows[np.concatenate(cv_rows)][at])
            if failed is not None:
                record.update(status="failed", error=failed,
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
                    key=lambda i: (-models[i]["cv_mean"] if classification or BROKEN == "F17"
                                   else models[i]["cv_mean"], i))
    # D74 (6): under the grouped scheme, when some fold has fewer than 50 training rows
    # and fewer training rows than validation rows, the other folds validate on those
    # same few rows, and no choice is made on so few: every fold stays scored and
    # reported, the configurations are ranked in configuration order, and a warning
    # says so. F28: the choice is made all the same. M48: any fold of fewer than 50
    # training rows is enough.
    if grouped and BROKEN != "F28":
        starved = [len(f["train"]) for f in split["folds"] if len(f["train"]) < SELECTION_FLOOR
                   and (BROKEN == "M48" or len(f["train"]) < len(f["valid"]))]
        if starved:
            ranked = sorted(ranked)
            _warn(warnings, "selection_not_made",
                  f"one fold trains on {min(starved)} rows, fewer than {SELECTION_FLOOR} and "
                  "fewer than it validates on: one unit, or the units of one fold, hold nearly "
                  "all of the training rows, and the other folds validate on those same few "
                  "rows. No choice between configurations is made on so few: they are "
                  "taken in configuration order")

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
    x_fit, x_holdout, frames, y_fit = matrices(fit_rows, holdout, final=True)
    chosen = pred = chosen_model = chosen_family = None
    for i in ranked:
        family, params = makers[i]
        try:
            model = family["make"](task, params, seed, n_jobs)
            pred = finite(_fit_predict(family, model, x_fit, y_fit, x_holdout,
                                       classification, frames), len(holdout))
        except Exception as exc:            # recorded, never dropped, and never the headline
            models[i].update(status="failed", error=f"{type(exc).__name__}: {exc}")
            _warn(warnings, "model_failed",
                  f"configuration {models[i]['config']} failed after it was chosen: "
                  f"{models[i]['error']}")
            continue
        chosen, chosen_model, chosen_family = models[i], model, family
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

    # D74 (3): the share of a squared error that its single largest row carries: on
    # the holdout and over the selection folds, for the model and for the headline
    # baseline. A statement about the result: nothing is left out of any number for
    # it. One warning for each statement above one half. F26: the share is taken of
    # the absolute errors. M45: over the selection folds only the chosen
    # configuration keeps its share.
    largest = {"largest_row_share": None, "largest_row": None,
               "baseline_largest_row_share": None, "baseline_largest_row": None}
    cv_largest = dict(largest)
    if not classification and BROKEN != "F4":
        if BROKEN == "M45":
            for m in models:
                if m is not chosen:
                    m["largest_row_share"] = m["largest_row"] = None
        with np.errstate(over="ignore", invalid="ignore"):
            for key, guess in (("", pred), ("baseline_", base_pred)):
                errors = np.abs(truth - guess) if BROKEN == "F26" else (truth - guess) ** 2
                share, at = _largest_row(errors)
                if share is not None:
                    largest[key + "largest_row_share"] = share
                    largest[key + "largest_row"] = int(rows[holdout][at])
            cv_largest["largest_row_share"] = chosen["largest_row_share"]
            cv_largest["largest_row"] = chosen["largest_row"]
            fold_rows, fold_bases = _screen_baselines(y, split["folds"], prev, by_entity)
            share, at = _largest_row((y[fold_rows] - dict(fold_bases)[baseline["name"]]) ** 2)
            if share is not None:
                cv_largest["baseline_largest_row_share"] = share
                cv_largest["baseline_largest_row"] = int(rows[fold_rows][at])
        for key, what in (("", f"the chosen model {chosen['config']}"),
                          ("baseline_", f"the baseline {baseline['name']}")):
            share = largest[key + "largest_row_share"]
            if share is not None and share > LARGEST_ROW_SHARE:
                _warn(warnings, "largest_row_dominates",
                      f"one holdout row, row {largest[key + 'largest_row']}, carries "
                      f"{100 * share:.1f}% of the squared error of {what} on the holdout. The "
                      "score and the skill are reported as measured, with that row in them: "
                      "they say more about that one row than about the rest. Look at the row")
        ruled = [m for m in models if m["largest_row_share"] is not None
                 and m["largest_row_share"] > LARGEST_ROW_SHARE]
        if ruled:
            named = sorted({m["largest_row"] for m in ruled})
            _warn(warnings, "largest_row_dominates",
                  f"over the selection folds one row carries more than half of the squared "
                  f"error of {len(ruled)} of the {len(models)} configurations "
                  f"(row{'s' if len(named) > 1 else ''} {', '.join(map(str, named))}): the "
                  "choice of model, and the model fitted on the training rows, may rest on "
                  "it. Look at the row")
        share = cv_largest["baseline_largest_row_share"]
        if share is not None and share > LARGEST_ROW_SHARE:
            _warn(warnings, "largest_row_dominates",
                  f"over the selection folds one row, row {cv_largest['baseline_largest_row']}, "
                  f"carries {100 * share:.1f}% of the squared error of the baseline "
                  f"{baseline['name']}. Look at the row")

    # 12. the interval and the p-value, from the holdout predictions alone
    time_groups = entity_groups = None
    if time_ordered:
        _, inverse = np.unique(keys[holdout], return_inverse=True)
        time_groups = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    elif grouped:
        # the holdout rows of each entity, as one array and where each entity starts
        _, inverse = np.unique(units[holdout], return_inverse=True)
        entity_groups = (np.argsort(inverse, kind="stable"), np.bincount(inverse))
    uncertainty = _uncertainty(truth, pred, base_pred, classification, time_groups,
                               RESAMPLES[profile], seed, score, skill is not None, entity_groups)
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
    if panel_warning is not None:
        # M51: it names the training mean, whatever baseline the run used
        used = "training_mean" if BROKEN == "M51" else baseline["name"]
        _warn(warnings, "autocorrelated_panel_target",
              "the timestamps repeat and group_col=False was passed. The per-timestamp mean "
              f"of the target has lag-1 autocorrelation {panel_warning:.2f}, so the table may "
              f"hold several series. Skill is measured against the baseline {used}. "
              + PANEL_BASELINE, target)

    # D77, D78: what the headline rests on. For each feature, the score of the chosen
    # configuration on the holdout with that feature's part of the holdout matrix moved
    # to other holdout rows, the mean over `A_K` shuffles; nothing is refitted. Columns
    # that are exact copies of one another are moved together and share one fall.
    rests_on, floor = [], None
    if P.A:
        started_a = time.perf_counter()
        prep, names = final_fit["prep"], final_fit["names"]
        blocks, at = {}, len(prep.numeric)
        for j, name in enumerate(prep.numeric):
            blocks[name] = [j]
        for name, levels in zip(prep.categorical, prep.levels):
            blocks[name] = list(range(at, at + len(levels)))
            at += len(levels)
        custom = bool(chosen_family.get("custom"))
        # D78 (3): two features are copies when the parts of the final fit's matrix they
        # give are the same in every training row and every holdout row: a numeric
        # feature's column agreeing to within COPY_TOLERANCE, a categorical feature's
        # block of level columns holding the same columns in any order
        if P.A_COPIES == "values":
            reach = rows[np.concatenate([train, holdout])]
            copy_sets = {}
            for name, values in prepared.items():
                held = values[reach]
                if kinds[name] == "numeric":
                    held = np.where(np.isnan(held) | (np.abs(held) > TOO_LARGE), np.nan,
                                    held) + 0.0
                    key = ("numeric", held.tobytes())
                else:
                    key = ("categorical", "\x00".join(held.astype(str)))
                copy_sets.setdefault(key, []).append(name)
            copy_sets = list(copy_sets.values())
        else:
            whole = np.vstack([x_fit, x_holdout])
            head = min(16, len(whole))

            def block(name):
                part = whole[:, blocks[name]]
                if kinds[name] == "numeric":
                    return part
                return part[:, np.lexsort(part[::-1])] if part.shape[1] else part

            seen_blocks = {name: block(name) for name in prepared}

            def alike(a, b):
                x, z = seen_blocks[a], seen_blocks[b]
                if kinds[a] != kinds[b] or x.shape != z.shape:
                    return False
                if kinds[a] == "numeric":
                    for sign in ((1.0, -1.0) if P.COPY_SIGN else (1.0,)):
                        if np.all(np.abs(x[:head] - sign * z[:head]) <= P.COPY_TOLERANCE)                                 and np.all(np.abs(x - sign * z) <= P.COPY_TOLERANCE):
                            return True
                    return False
                return bool(np.array_equal(x[:head], z[:head]) and np.array_equal(x, z))

            copy_sets, placed = [], set()
            for name in prepared:
                if name in placed:
                    continue
                members = [name] + [other for other in prepared
                                    if other != name and other not in placed
                                    and P.A_COPIES != "none" and alike(name, other)]
                placed.update(members)
                copy_sets.append(members)

        def shuffled(members):
            rng = np.random.default_rng([seed, 8, by_name[members[0]]["position"]])
            cols = [c for name in members for c in blocks[name]]
            got = []
            for _ in range(P.A_K):
                order = rng.permutation(len(holdout))
                moved = x_holdout.copy()
                moved[:, cols] = x_holdout[order][:, cols]
                new = pd.DataFrame(moved, index=frames[1].index, columns=names) if custom \
                    else moved
                try:
                    guess = finite(_predict(chosen_model, new, classification), len(holdout))
                except Exception as exc:            # the model cannot predict this matrix
                    return f"{type(exc).__name__}: {exc}"
                got.append(_auc(truth, guess) if classification else _mse(truth, guess))
            return got

        if not custom and hasattr(chosen_model, "estimators_") and n_jobs > 1 and len(copy_sets) > 1:
            with ThreadPoolExecutor(max_workers=min(n_jobs, len(copy_sets))) as pool:
                every = list(pool.map(shuffled, copy_sets))
        else:
            every = [shuffled(members) for members in copy_sets]
        for members, got in zip(copy_sets, every):
            mean_score = None
            if not isinstance(got, str):
                with np.errstate(over="ignore", invalid="ignore"):
                    mean_score = float(np.mean(got))
                if not math.isfinite(mean_score):
                    got, mean_score = f"its score is not finite ({mean_score!r})", None
            for name in members:
                rec = by_name[name]
                if mean_score is None:
                    _warn(warnings, "shuffled_score_not_computed",
                          f"what the headline rests on is not stated for column {name!r}: the "
                          f"chosen configuration {chosen['config']} could not score the holdout "
                          f"with its values shuffled ({got})", name)
                    continue
                rec["shuffled_score"] = mean_score
                if BROKEN == "F4":
                    continue
                if classification:
                    without = 2 * mean_score - 1
                else:
                    without = 1 - mean_score / baseline["score"] if baseline["score"] > 0 else None
                if skill is not None and without is not None:
                    rec["shuffled_skill"], rec["skill_fall"] = without, skill - without
                if P.DEBUG:
                    P.LOG.setdefault("rests", {})[name] = [float(g) for g in got]
        if P.DEBUG:
            P.LOG["copies"] = [members for members in copy_sets if len(members) > 1]
        # D80 (1): the floor of the list. The half-width of the headline skill's
        # interval, or a stand-in where there is none; never below A_MIN.
        half = None
        if uncertainty["skill_low"] is not None and uncertainty["skill_high"] is not None:
            half = (uncertainty["skill_high"] - uncertainty["skill_low"]) / 2
        rows_half = None
        if skill is not None and (half is None or P.DEBUG):
            # the half-width of a bootstrap of the holdout rows, as if they were independent
            rng_rows = np.random.default_rng([seed, 10])
            found_rows = []
            for _ in range(RESAMPLES[profile]):
                pick = rng_rows.integers(0, len(holdout), len(holdout))
                if classification:
                    a_rows = _auc(truth[pick], pred[pick])
                    if a_rows is not None:
                        found_rows.append(2 * a_rows - 1)
                else:
                    b_rows = _mse(truth[pick], base_pred[pick])
                    if b_rows > 0:
                        found_rows.append(1 - _mse(truth[pick], pred[pick]) / b_rows)
            if len(found_rows) * 2 >= RESAMPLES[profile]:
                lo_rows, hi_rows = np.percentile(found_rows, [2.5, 97.5])
                rows_half = float(hi_rows - lo_rows) / 2
        if half is not None and P.A_HALF:
            floor = max(P.A_MIN, half)
        elif P.A_STANDIN == "rows" and rows_half is not None and P.A_HALF:
            floor = max(P.A_MIN, rows_half)
        elif P.A_STANDIN not in ("rows", "none") and P.A_HALF:
            floor = max(P.A_MIN, float(P.A_STANDIN))
        else:
            floor = P.A_MIN
        if P.DEBUG:
            P.LOG["floor"] = {"half": half, "rows_half": rows_half, "floor": floor}
        order = sorted((r for r in records if r["skill_fall"] is not None),
                       key=lambda r: (-r["skill_fall"], r["position"]))
        named = [r for r in order if r["skill_fall"] >= floor][:P.A_TOP]
        rests_on = [{"column": r["name"], "skill_fall": r["skill_fall"]} for r in named]
        if skill is not None and skill >= P.A_WARN and BROKEN != "F4":
            if named:
                rests = ("It rests most on " + "; ".join(
                    f"{r['name']!r} (skill {r['shuffled_skill']:.3f} with its holdout values "
                    f"shuffled, a fall of {r['skill_fall']:.3f})" for r in named)
                    + ". Check that each of these columns is known before the outcome")
            else:
                rests = (f"No column's fall reaches {P.A_MIN}: shuffling any one column's "
                         "holdout values leaves the skill almost where it is. Check that the "
                         "columns are known before the outcome")
            _warn(warnings, "near_perfect_headline",
                  f"The headline skill is {skill:.4f}, at or above {P.A_WARN}: the level at "
                  "which one column alone is excluded as a suspected leak. " + rests)
        if P.DEBUG:
            P.LOG["a_seconds"] = time.perf_counter() - started_a

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
    # the entities, or the sets of copies, on each side: counted from the rows
    sided = units if units is not None else groups
    validation = {
        "scheme": "time_ordered" if time_ordered else "grouped" if grouped else "random",
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
        "caveat": None if time_ordered else GROUPED_CAVEAT if grouped else RANDOM_CAVEAT,
        # holdout rows with a copy among the training rows; null under time order (D49)
        "n_holdout_copies": n_copies, "holdout_copy_share": copy_share,
        # rows with a copy among the rows split, counted before the split:
        # `duplicates="group"` only (D57, D60)
        "n_rows_same_features": n_same_features,
        "group_col": group_name,
        "n_groups_training": None if sided is None else int(len(np.unique(sided[train]))),
        "n_groups_holdout": None if sided is None else int(len(np.unique(sided[holdout]))),
        # holdout rows whose entity has no earlier row: time order with entities (D57)
        "n_holdout_without_previous": n_without_previous,
    }
    headline = {"metric": metric, "score": score, "model": chosen["config"],
                "baseline": baseline["name"], "baseline_score": baseline["score"],
                "skill": skill}
    headline.update(uncertainty)
    headline.update(largest)
    headline["rests_on"] = rests_on
    headline["rests_on_floor"] = floor if P.A else None
    doc = {
        "meta": {"tool": "mlprofessor", "schema_version": 1, "profile": profile, "seed": seed,
                 "n_jobs": n_jobs,
                 "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                              "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                              "threadpoolctl": threadpoolctl.__version__}},
        "input": {"source": source_label, "kind": source_kind, "target": target,
                  "time_col": stated_time, "task": requested, "gap": gap, "n_rows": n_loaded,
                  "n_columns": int(df.shape[1]), "duplicate_rows": duplicate_rows,
                  "duplicates": duplicates, "group_col": group_col},
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
               "chosen_std": chosen["cv_std"], **cv_largest},
        # F6c: a family none of whose configurations was chosen is left out of the table
        "models": models if BROKEN != "F6c"
        else [m for m in models if m["family"] == chosen["family"]],
        "warnings": warnings,
        "limitations": [] if BROKEN == "F20" else list(LIMITATIONS),
    }
    _require_finite(doc)
    return doc, predictions
