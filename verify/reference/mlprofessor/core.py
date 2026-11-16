"""Reference implementation of spec.md, in the spec's order of operations.

Judge fixture: the gate measures its bands on this code and breaks it in eight
named ways (see _variant.py). One module on purpose, so each step of the
methodology can be read top to bottom in `_run`.
"""
from __future__ import annotations

import csv
import json
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
LEAK_MISSING_HIGH = 0.95    # missing for more than this share of one class ...
LEAK_MISSING_LOW = 0.05     # ... and less than this share of the other
ID_TEXT_SHARE = 0.95        # text column: distinct values > 95% of the rows
FREE_TEXT_SHARE = 0.50      # text column: distinct values > 50% of non-missing
NUMERIC_SHARE = 0.90        # text column: >= 90% of non-missing parse as numbers
DATE_SHARE = 0.95           # text column: >= 95% of non-missing look like dates
SMALL_HOLDOUT = 50
MIN_BLOCK_TIMESTAMPS = 30   # fewer distinct holdout timestamps: no interval
AUTOCORR = 0.5              # lag-1 autocorrelation of the per-timestamp mean target
MAX_LEVELS = 30             # one-hot levels kept per categorical column
RESAMPLES = {"full": 1000, "fast": 200}
MISSING_LEVEL = "(missing)"

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

RANDOM_CAVEAT = ("Random split: this number is optimistic if the rows are ordered in time "
                 "or one entity appears on many rows.")
NO_P_VALUE = ("not computed under the time-ordered scheme: shuffling the holdout targets "
              "assumes independent rows")
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
    lines = [
        f"task: {d['task']['type']} ({d['task']['decided_by']}); "
        f"scheme: {d['validation']['scheme']}",
        f"rows: {rows['loaded']} loaded, {rows['training']} training, {rows['holdout']} holdout",
        f"headline: skill {h['skill']:.4f} ({h['metric']} {h['score']:.6g} against "
        f"{h['baseline']} {h['baseline_score']:.6g}), model {h['model']}",
    ]
    if h["interval_low"] is not None:
        lines.append(f"95% interval on skill: {h['skill_low']:.4f} to {h['skill_high']:.4f} "
                     f"({h['interval_method']})")
    if h["p_value"] is not None:
        lines.append(f"permutation p-value: {h['p_value']:.4g}")
    for w in d["warnings"]:
        lines.append(f"warning [{w['code']}]: {w['message']}")
    return "\n".join(lines)


# -------------------------------------------------------------- public API --

def profile(data, target, task="auto", time_col=None, gap=0, profile="full", seed=0,
            out_dir=None, n_jobs=-1):
    """Measure how predictable `target` is from the other columns of one table."""
    started = time.perf_counter()
    if task not in ("auto", "classification", "regression"):
        raise E.InputError(f"task must be auto, classification or regression, not {task!r}")
    if profile not in ("full", "fast"):
        raise E.InputError(f"profile must be full or fast, not {profile!r}")
    for name, value, low in (("gap", gap, 0), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
            raise E.InputError(f"{name} must be a whole number of at least {low}, not {value!r}")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) \
            or not (n_jobs == -1 or n_jobs >= 1):
        raise E.InputError(f"n_jobs must be -1 or a positive whole number, not {n_jobs!r}")
    if gap and time_col is None:
        raise E.InputError("gap needs time_col: it counts timestamps")
    if time_col is not None and time_col == target:
        raise E.InputError("time_col and target are the same column")
    threads = (os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"), \
            threadpoolctl.threadpool_limits(limits=threads, user_api="openmp"):
        doc, predictions = _run(data, target, task, time_col, int(gap), profile, int(seed), threads)
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
        with open(path, newline="", encoding="utf-8-sig") as fh:
            header = next((row for row in csv.reader(fh) if row), None)
        if header is None:
            raise E.InputError(f"{data} has no header row")
        dups = sorted({n for n in header if n.strip() and header.count(n) > 1})
        if not dups:
            try:
                df = pd.read_csv(path, low_memory=False)
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
                raise E.InputError(f"{data} could not be read as CSV: {exc}") from exc
            if not isinstance(df.index, pd.RangeIndex) or len(df.columns) != len(header):
                raise E.InputError(f"{data}: data rows do not have the header's number of fields")
            for i, name in enumerate(header):
                if not name.strip():
                    _warn(warnings, "blank_header",
                          f"column {i + 1} has no header and is kept as {df.columns[i]!r}",
                          str(df.columns[i]))
        kind, label = "csv", str(data)
    else:
        raise E.InputError("data must be a pandas DataFrame or a path to a CSV file")
    if dups:
        raise E.DuplicateColumnsError("duplicate column names: " + ", ".join(dups))
    return df.reset_index(drop=True), kind, label


def _time_keys(series, name):
    """Step 3 (D18): sortable keys for the time column. Returns (keys, missing, form, show)."""
    dtype = series.dtype
    if pt.is_datetime64_any_dtype(dtype):
        s = series
        if getattr(dtype, "tz", None) is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        stamps = s.to_numpy("datetime64[ns]")
        return stamps.astype("int64"), np.isnat(stamps), "datetime", _show_stamp
    if pt.is_numeric_dtype(dtype) and not pt.is_bool_dtype(dtype):
        keys = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        whole = pt.is_integer_dtype(dtype)
        return keys, ~np.isfinite(keys), "numeric", (lambda k: int(k) if whole else float(k))
    present = series.dropna()
    if (pt.is_string_dtype(dtype) or dtype == object) and all(isinstance(v, str) for v in present):
        text = present.str.strip()
        ok = text.map(lambda v: ISO_TIME_RE.fullmatch(v) is not None)
        if not ok.all():
            raise E.TimeColumnError(
                f"time column {name!r} holds text that is not ISO 8601 with the year first "
                f"(for example {text[~ok].iloc[0]!r}). Day-first and month-first dates are "
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
        stamps = np.full(len(series), np.datetime64("NaT"), dtype="datetime64[ns]")
        stamps[series.notna().to_numpy()] = parsed.to_numpy("datetime64[ns]")
        return stamps.astype("int64"), np.isnat(stamps), "iso8601", _show_stamp
    raise E.TimeColumnError(
        f"time column {name!r} has type {dtype}; accepted forms are a datetime column, a "
        "numeric column, or text in ISO 8601 form with the year first")


def _show_stamp(key):
    return pd.Timestamp(int(key)).isoformat()


def _decide_task(y, name, requested):
    """Step 4 (D4, D21): classification or regression, and how that was decided."""
    n_distinct = int(y.nunique())
    numeric = pt.is_numeric_dtype(y.dtype) and not pt.is_bool_dtype(y.dtype)
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
    """Step 2: which target cells are missing or, for a numeric target, infinite."""
    gone = series.isna().to_numpy().copy()
    if pt.is_numeric_dtype(series.dtype) and not pt.is_bool_dtype(series.dtype):
        gone |= ~np.isfinite(pd.to_numeric(series, errors="coerce")
                             .to_numpy(dtype="float64", na_value=np.nan))
    return gone


def _classes(values):
    """The two classes in sorted order; the second is the positive class."""
    distinct = list(pd.unique(values))
    try:
        return sorted(distinct)
    except TypeError:
        return sorted(distinct, key=str)


def _whole_number_id(values):
    """D19: every value distinct, and consecutive or strictly ordered in file order."""
    x = values[~np.isnan(values)]
    if len(x) < 2 or not np.all(x == np.floor(x)) or len(np.unique(x)) != len(x):
        return False
    step = np.diff(x)
    return bool(np.all(step > 0) or np.all(step < 0) or x.max() - x.min() + 1 == len(x))


def _roles(df, usable, target, time_col):
    """Step 5 (D19, D20): roles that need no target, judged on the usable rows.

    Returns the column records and, for each feature, its prepared values over
    every loaded row: float64 for numeric, object for categorical.
    """
    records, prepared = [], {}
    n_rows = int(usable.sum())
    for position, name in enumerate(df.columns):
        series = df[name]
        rec = dict(name=name, position=position, dtype=str(series.dtype), role="feature",
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
            if text.map(lambda v: DATE_RE.fullmatch(v) is not None).mean() >= DATE_SHARE:
                rec.update(role="excluded", reason="datetime", kind="datetime")
            elif pd.to_numeric(text, errors="coerce").notna().mean() >= NUMERIC_SHARE:
                numbers = pd.to_numeric(series.str.strip(), errors="coerce") \
                    .to_numpy(dtype="float64", na_value=np.nan)
                rec["n_coerced"] = int((series.notna().to_numpy() & np.isnan(numbers)).sum())
            else:
                distinct = int(present.nunique())
                if distinct == 1:
                    rec.update(role="excluded", reason="constant", kind="categorical")
                elif distinct > ID_TEXT_SHARE * n_rows:
                    rec.update(role="excluded", reason="id_like", kind="text")
                elif distinct > FREE_TEXT_SHARE * len(present):
                    rec.update(role="excluded", reason="free_text", kind="text")
                else:
                    levels = series.astype(object).where(series.notna(), MISSING_LEVEL).to_numpy()
                    rec["kind"] = "categorical"
        else:
            rec.update(role="excluded", reason="unsupported_type")
        if numbers is not None:
            rec["kind"] = "numeric"
            infinite = np.isinf(numbers)
            rec["n_infinite"] = int(infinite.sum())
            numbers = np.where(infinite, np.nan, numbers)
            seen = numbers[usable]
            distinct = len(np.unique(seen[~np.isnan(seen)]))
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
    return records, prepared


def _split_time(keys, gap, capped):
    """Step 7 under time order (D7). `keys` is sorted. Returns index arrays."""
    stamps, first, counts = np.unique(keys, return_index=True, return_counts=True)
    n, total = len(keys), len(stamps)
    h, held = total, 0
    while h > 0 and 4 * held < n:           # latest whole timestamps holding >= 25%
        h -= 1
        held += counts[h]
    if h < N_BLOCKS:
        if capped:
            raise E.FastProfileError(
                f"the {n} most recent rows hold {total} distinct timestamps, too few to form "
                "the holdout and six training blocks. Use profile='full'")
        raise E.TooFewRowsError(f"{total} distinct timestamps are too few to form the holdout "
                                "and six training blocks")
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


class _Prep:
    """Step 8: imputation, scaling and encoding, fitted on the frame it is given."""

    def __init__(self, kinds):
        self.numeric = [c for c, k in kinds.items() if k == "numeric"]
        self.categorical = [c for c, k in kinds.items() if k == "categorical"]

    def _parts(self, frame):
        num = frame[self.numeric].to_numpy(dtype="float64", na_value=np.nan) if self.numeric \
            else np.empty((len(frame), 0))
        cats = [frame[c].astype(object).where(frame[c].notna(), MISSING_LEVEL).to_numpy()
                for c in self.categorical]
        return num, cats

    def fit(self, frame):
        num, cats = self._parts(frame)
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


def _auc(y, score):
    """ROC AUC by ranks, ties averaged. None when only one class is present."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return float((_ranks(score)[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _ranks(score):
    order = np.argsort(score, kind="mergesort")
    ordered = score[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    counts = np.diff(np.r_[starts, len(score)])
    ranks = np.empty(len(score))
    ranks[order] = np.repeat(starts + 1 + (counts - 1) / 2, counts)
    return ranks


def _mse(y, pred):
    return float(np.mean((y - pred) ** 2))


def _screen_column(values, numeric, y, folds, classification, seed, prev):
    """D11: one column's cross-validated skill over the selection folds."""
    cls = DecisionTreeClassifier if classification else DecisionTreeRegressor
    truth, guess, mean_guess, rows = [], [], [], []
    for fold in folds:
        if not fold["scored"]:
            continue
        a, b = fold["train"], fold["valid"]
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
        mean_guess.append(np.full(len(b), y[a].mean()))
        rows.append(b)
    truth, guess = np.concatenate(truth), np.concatenate(guess)
    if classification:
        auc = _auc(truth, guess)
        return 0.0 if auc is None else 2 * auc - 1
    base = _mse(truth, np.concatenate(mean_guess))
    if prev is not None:
        before = prev[np.concatenate(rows)]
        known = ~np.isnan(before)
        if known.any():
            base = min(base, _mse(truth[known], before[known]))
    return 1 - _mse(truth, guess) / base if base > 0 else 0.0


def _cube_root_up(n):
    """Smallest whole L with L**3 >= n."""
    length = max(1, round(n ** (1 / 3)))
    while length ** 3 < n:
        length += 1
    while length > 1 and (length - 1) ** 3 >= n:
        length -= 1
    return length


def _uncertainty(y, pred, base_pred, classification, time_groups, n_resamples, seed, score):
    """Step 12 (D9): interval on the holdout score and skill, and the p-value."""
    out = dict(interval_method=None, interval_level=0.95, interval_low=None, interval_high=None,
               skill_low=None, skill_high=None, n_resamples=n_resamples, n_resamples_used=0,
               block_length=None, interval_reason=None, p_value=None, n_permutations=None,
               p_value_reason=None)

    def measure(rows):
        if classification:
            auc = _auc(y[rows], pred[rows])
            return None if auc is None else (auc, 2 * auc - 1)
        mse, base = _mse(y[rows], pred[rows]), _mse(y[rows], base_pred[rows])
        return (mse, 1 - mse / base) if base > 0 else None

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
                starts = rng.integers(0, total - length + 1, n_blocks)
                picked = np.concatenate([np.arange(s, s + length) for s in starts])[:total]
                yield np.concatenate([time_groups[t] for t in picked])
        draws = blocks()
    results = [m for m in (measure(rows) for rows in draws) if m is not None]
    out["n_resamples_used"] = len(results)
    if out["interval_method"] and len(results) * 2 >= n_resamples:
        scores, skills = np.array(results).T
        out["interval_low"], out["interval_high"] = (float(v) for v in
                                                     np.percentile(scores, [2.5, 97.5]))
        out["skill_low"], out["skill_high"] = (float(v) for v in
                                               np.percentile(skills, [2.5, 97.5]))
    elif out["interval_method"]:
        out["interval_method"] = None
        out["interval_reason"] = "more than half of the resamples held a single class"
    if time_groups is not None:
        out["p_value_reason"] = NO_P_VALUE
        return out
    rng = np.random.default_rng([seed, 6])
    if classification:
        ranks, n_pos = _ranks(pred), int(y.sum())
        floor = n_pos * (n_pos + 1) / 2
        scale = n_pos * (n - n_pos)
        hits = sum((ranks[rng.permutation(y) == 1].sum() - floor) / scale >= score
                   for _ in range(n_resamples))
    else:
        hits = sum(_mse(rng.permutation(y), pred) <= score for _ in range(n_resamples))
    out.update(p_value=(1 + int(hits)) / (1 + n_resamples), n_permutations=n_resamples)
    return out


def _plain(value):
    """A JSON-ready copy of a target value."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return str(value)


# --------------------------------------------------------------------- run --

def _run(data, target, requested, time_col, gap, profile, seed, n_jobs):
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
    except TypeError:                               # unhashable cells: not comparable
        duplicate_rows = 0
    if duplicate_rows:
        _warn(warnings, "duplicate_rows",
              f"{duplicate_rows} row(s) are exact copies of an earlier row; they are kept")

    # 2. drop rows with a missing target, then rows with a missing time value
    y_all = df[target]
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
        y_work_all = (y_all == classes[1]).to_numpy().astype("int64")
    else:
        classes = None
        y_work_all = pd.to_numeric(y_all, errors="coerce").to_numpy(dtype="float64",
                                                                    na_value=np.nan)
        whole = y_work_all[usable]
        if decided_by == "rule" and np.all(whole == np.floor(whole)):
            _warn(warnings, "whole_number_regression_target",
                  f"target {target!r} holds only whole numbers ({n_distinct} distinct) and was "
                  "read as a continuous target; declare the task if it is a code or an identifier",
                  target)

    # 5. column roles that need no target
    records, prepared = _roles(df, usable, target, time_col)
    by_name = {r["name"]: r for r in records}
    for rec in records:
        if rec["n_coerced"]:
            _warn(warnings, "values_coerced",
                  f"{rec['n_coerced']} text value(s) in numeric column {rec['name']!r} could not "
                  "be read as numbers and are treated as missing", rec["name"])
        if rec["n_infinite"]:
            _warn(warnings, "infinite_values",
                  f"{rec['n_infinite']} infinite value(s) in {rec['name']!r} are treated as "
                  "missing", rec["name"])
        if rec["reason"] == "datetime" and time_col is None:
            _warn(warnings, "date_column_undeclared",
                  f"column {rec['name']!r} holds dates but no time_col was declared, so the "
                  "rows were split at random; " + RANDOM_CAVEAT, rec["name"])

    # 6. the fast profile's row cap; rows are put in time order here
    rows = np.flatnonzero(usable)
    if time_col is not None:
        rows = rows[np.argsort(keys_all[rows], kind="stable")]
    capped = 0
    if profile == "fast" and len(rows) > FAST_ROW_CAP:
        if time_col is not None:
            stamps, counts = np.unique(keys_all[rows], return_counts=True)
            kept, t = 0, len(stamps)
            while t > 0 and kept + counts[t - 1] <= FAST_ROW_CAP:
                t -= 1
                kept += counts[t]
            if kept == 0:
                raise E.FastProfileError(
                    f"the most recent timestamp alone holds {counts[-1]} rows, more than the "
                    f"fast profile's cap of {FAST_ROW_CAP}. Use profile='full'")
            capped, rows = len(rows) - kept, rows[len(rows) - kept:]
        else:
            keep = np.sort(np.random.default_rng([seed, 1]).choice(len(rows), FAST_ROW_CAP,
                                                                   replace=False))
            capped, rows = len(rows) - FAST_ROW_CAP, rows[keep]
    if len(rows) < MIN_ROWS:
        raise E.TooFewRowsError(f"{len(rows)} usable rows; at least {MIN_ROWS} are needed")
    y = y_work_all[rows]
    if classification and min(int(y.sum()), int(len(y) - y.sum())) < MIN_CLASS_ROWS:
        raise E.TooFewRowsError(f"the rarer class has fewer than {MIN_CLASS_ROWS} usable rows")

    # 7. split off the holdout, applying the gap
    time_ordered = time_col is not None
    if time_ordered and BROKEN != "F7":
        keys = keys_all[rows]
        split = _split_time(keys, gap, capped > 0)
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
        cell = values[rows[used]]
        by_name[name]["n_imputed"] = int(np.isnan(cell).sum()) if cell.dtype != object \
            else int((cell == MISSING_LEVEL).sum())

    # the target `gap + 1` timestamps earlier, where timestamps are unique (D10)
    prev = None
    if time_ordered and not classification and split["unique"]:
        order = np.argsort(keys, kind="stable")
        prev = np.full(len(y), np.nan)
        prev[order[gap + 1:]] = y[order[:len(y) - gap - 1]]

    # 8. from here on every fitted step sees training rows only: the leak screen first
    if BROKEN != "F2":
        for name in list(prepared):
            values, numeric = prepared[name][rows], by_name[name]["kind"] == "numeric"
            rule = None
            if classification:
                gone = np.isnan(values[train]) if numeric else values[train] == MISSING_LEVEL
                share = [gone[y[train] == c].mean() for c in (0, 1)]
                if max(share) > LEAK_MISSING_HIGH and min(share) < LEAK_MISSING_LOW:
                    rule = "missing_by_class"
            if rule is None and _screen_column(values, numeric, y, split["folds"], classification,
                                               seed, prev) >= LEAK_SKILL:
                rule = "predicts_target"
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
        raise E.NoFeaturesError("no usable feature column is left")
    kinds = {name: by_name[name]["kind"] for name in prepared}
    index = pd.Index(rows, name="row")
    step_makers = extensions.active()["steps"] or []

    def feature_frame(part):
        return pd.DataFrame({name: values[rows[part]] for name, values in prepared.items()},
                            index=index[part])

    def matrices(fit_rows, new_rows, family):
        """Preprocessing fitted on `fit_rows` only, applied to both sets of rows."""
        fit_frame, new_frame = feature_frame(fit_rows), feature_frame(new_rows)
        y_fit = pd.Series(y[fit_rows], index=fit_frame.index, name=target)
        for make in step_makers:                    # steps supplied through the seam
            step = make()
            step.fit(fit_frame, y_fit)
            fit_frame, new_frame = step.transform(fit_frame), step.transform(new_frame)
        prep = _Prep(kinds).fit(fit_frame)
        x_fit, x_new = prep.transform(fit_frame), prep.transform(new_frame)
        names = prep.names()
        frames = (pd.DataFrame(x_fit, index=fit_frame.index, columns=names),
                  pd.DataFrame(x_new, index=new_frame.index, columns=names), y_fit)
        return x_fit, x_new, frames

    # 9. select by five folds inside the training rows, preprocessing refitted in each
    families = _families(profile, classification)
    fold_data = [matrices(f["train"], f["valid"], None) if f["scored"] else None
                 for f in split["folds"]]
    models, best = [], None
    for family in families:
        for position, params in enumerate(family["grid"]):
            record = dict(config=f"{family['name']}-{position}", family=family["name"],
                          params=dict(params), status="fitted", error=None, fold_scores=[],
                          cv_mean=None, cv_std=None, chosen=False)
            try:
                for fold, mats in zip(split["folds"], fold_data):
                    if mats is None:
                        record["fold_scores"].append(None)
                        continue
                    model = family["make"](task, params, seed, n_jobs)
                    guess = _fit_predict(family, model, mats[0], y[fold["train"]], mats[1],
                                         classification, mats[2])
                    truth = y[fold["valid"]]
                    record["fold_scores"].append(_auc(truth, guess) if classification
                                                 else _mse(truth, guess))
            except Exception as exc:        # a family that raises is recorded, never dropped
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                              fold_scores=[None] * len(split["folds"]))
                _warn(warnings, "model_failed",
                      f"configuration {record['config']} failed: {record['error']}")
            scored = [s for s in record["fold_scores"] if s is not None]
            if record["status"] == "fitted" and scored:
                record["cv_mean"] = float(np.mean(scored))
                record["cv_std"] = float(np.std(scored, ddof=1)) if len(scored) > 1 else None
                better = best is None or (record["cv_mean"] > best[0]["cv_mean"] if classification
                                          else record["cv_mean"] < best[0]["cv_mean"])
                if better:
                    best = (record, family, params)
            models.append(record)
    if best is None:
        raise E.AllModelsFailedError("every configuration failed: " +
                                     "; ".join(f"{m['config']}: {m['error']}" for m in models))
    chosen, family, params = best
    chosen["chosen"] = True

    # 10. refit the chosen configuration on all training rows
    fit_rows = np.concatenate([train, holdout]) if BROKEN == "F1" else train
    x_fit, x_holdout, frames = matrices(fit_rows, holdout, family)
    model = family["make"](task, params, seed, n_jobs)
    pred = _fit_predict(family, model, x_fit, y[fit_rows], x_holdout, classification, frames)

    # 11. score it once on the holdout; score the baselines on the holdout
    truth = y[holdout]
    if classification:
        metric, score = "roc_auc", _auc(truth, pred)
        baselines = [{"name": "chance", "score": 0.5, "used": True}]
        base_pred = None
    else:
        metric, score = "mse", _mse(truth, pred)
        base_pred = np.full(len(holdout), y[train].mean())
        baselines = [{"name": "training_mean", "score": _mse(truth, base_pred), "used": True}]
        if prev is not None:
            previous = _mse(truth, prev[holdout])
            baselines.append({"name": "previous_value", "score": previous, "used": False})
            if previous < baselines[0]["score"] and BROKEN != "F8":
                baselines[0]["used"], baselines[1]["used"] = False, True
                base_pred = prev[holdout]
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
        skill = 1 - score / baseline["score"] if baseline["score"] > 0 else 0.0

    # 12. the interval and the p-value, from the holdout predictions alone
    groups = None
    if time_ordered:
        _, inverse = np.unique(keys[holdout], return_inverse=True)
        groups = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    uncertainty = _uncertainty(truth, pred, base_pred, classification, groups, RESAMPLES[profile],
                               seed, score)
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
        if len(means) > 2 and means[1:].std() > 0 and means[:-1].std() > 0:
            lag1 = float(np.corrcoef(means[1:], means[:-1])[0, 1])
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
                  "n_columns": int(df.shape[1]), "duplicate_rows": duplicate_rows},
        "rows": [{"item": k, "count": v} for k, v in row_items],
        "columns": records,
        "task": {"type": task, "decided_by": decided_by, "n_distinct": n_distinct,
                 "target_dtype": str(y_all.dtype),
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
        "models": models,
        "warnings": warnings,
        "limitations": list(LIMITATIONS),
    }
    return doc, predictions
