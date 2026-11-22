"""The steps of the methodology as functions (spec.md, 2.7).

The second half of the seam for tests: each step can be called on its own,
before the whole run exists. `profile()` uses the same code.
"""
import numpy as np
import pandas as pd

from . import core
from . import errors as E


def load_table(data):
    """Step 1. Returns (frame, info): the table with source positions as its index."""
    warnings = []
    frame, kind, source = core._load(data, warnings)
    return frame, {"kind": kind, "source": source, "warnings": warnings}


def _usable(frame, target, time_col, group_col=None):
    """(no target, no time, the frame with its object number columns typed, loaded types)."""
    for name in (target, time_col, group_col):
        if name is not None and name not in frame.columns:
            raise E.ColumnNotFoundError(f"no column named {name!r}")
    # an object column that holds only numbers is read as numbers, in every step (D51)
    frame, loaded, _ = core._numbers_from_objects(frame)
    # a text target cell that holds only white space is a missing target (D48)
    no_target = core._no_target(core._target_values(frame[target]))
    no_time = np.zeros(len(frame), dtype=bool)
    if time_col is not None:
        no_time = core._no_target(frame[time_col]) & ~no_target
    return no_target, no_time, frame, loaded


def usable_rows(frame, target, time_col=None):
    """Step 2. The positions of the usable rows, and what was dropped, by reason."""
    no_target, no_time, _, _ = _usable(frame, target, time_col)
    return {"usable": np.flatnonzero(~no_target & ~no_time),
            "dropped_missing_target": int(no_target.sum()),
            "dropped_missing_time": int(no_time.sum())}


def column_roles(frame, target, time_col=None, group_col=None):
    """Step 5. One record per column: the roles that need no target."""
    no_target, no_time, frame, loaded = _usable(frame, target, time_col, group_col)
    records, _ = core._roles(frame.reset_index(drop=True), ~no_target & ~no_time, target,
                             time_col, loaded, group_col)
    return pd.DataFrame(records, columns=core.COLUMN_KEYS)


def decide_task(values, task="auto"):
    """Step 4. Classification or regression, how it was decided, and the evidence."""
    if task not in ("auto", "classification", "regression"):
        raise E.InputError(f"task must be auto, classification or regression, not {task!r}")
    # text is read without the white space round it (D46); only white space is missing (D48);
    # objects that are all numbers are read as numbers (D51)
    series = _typed(values)
    series = core._target_values(series)
    series = series[~core._no_target(series)]
    kind, decided_by, n_distinct, numeric = core._decide_task(series, "target", task)
    classes = [core._plain(c) for c in core._classes(series)] if kind == "classification" else None
    warnings = []
    if kind == "regression" and decided_by == "rule":
        numbers = series.to_numpy(dtype="float64", na_value=np.nan)
        if np.all(numbers == np.floor(numbers)):
            warnings.append("whole_number_regression_target")
    return {"type": kind, "decided_by": decided_by, "n_distinct": n_distinct, "classes": classes,
            "positive_class": classes[1] if classes else None, "warnings": warnings}


def _typed(values):
    """`values` as a column, an object column that holds only numbers typed (D51)."""
    series = pd.Series(values).reset_index(drop=True)
    if series.dtype == object:
        typed = core._object_numbers(series)
        series = series if typed is None else typed
    return series


def read_time(values):
    """Step 3. Returns (keys, form): sortable keys, NaT or NaN where the value is missing.

    The keys of a numeric column are 64-bit floats, with one exception (D50, D53):
    an integer-typed column that holds a magnitude above 2**53 gives exact 64-bit
    integers, since as floats two of its values could become one. Text of eight
    digits that all read as calendar dates, YYYYMMDD, is ordered as those numbers
    and has the form `numeric` (D66).
    """
    keys, missing, form, _ = core._time_keys(_typed(values), "time")
    if form == "numeric":
        if keys.dtype.kind in "iu":             # exact whole numbers
            if not missing.any():
                return keys, form
            exact = keys.astype(object)
            exact[missing] = None
            return exact, form
        return np.where(missing, np.nan, keys), form
    return keys.astype("datetime64[ns]"), form


def split_random(n_rows, classes=None, seed=0, groups=None):
    """Step 7, random scheme. Positions 0..n_rows-1; `classes` makes it stratified.

    The seed is in the range of `profile()` (D45). Text labels are compared without
    the white space round them, as the target is read everywhere (D46). `groups`,
    when given, holds one value per row, and the rows of one value stay together:
    on one side of the holdout, and in the validation part of one fold (D57).
    """
    core._check_seed(seed)
    y = None
    if classes is not None:
        labels = core._target_values(pd.Series(np.asarray(classes, dtype=object)))
        y = (labels == core._classes(labels)[1]).to_numpy().astype("int64")
    if groups is not None:
        codes = core._group_codes(pd.Series(groups).reset_index(drop=True), "groups")
        if len(codes) != int(n_rows) or (codes < 0).any():
            raise E.GroupColumnError("groups must hold one value for each of the rows, "
                                     "none of them empty")
        split = core._split_grouped(np.unique(codes, return_inverse=True)[1], y,
                                    classes is not None, int(seed))
    else:
        split = core._split_random(int(n_rows), y, classes is not None, int(seed))
    return {"training": split["train"], "holdout": split["holdout"],
            "folds": [{"training": np.sort(f["train"]), "validation": np.sort(f["valid"])}
                      for f in split["folds"]]}


def split_time(keys, gap=0):
    """Step 7, time-ordered scheme. Positions into `keys`, which may be in any order."""
    keys = np.asarray(keys)
    order = np.argsort(keys, kind="stable")
    split = core._split_time(keys[order], int(gap))
    back = lambda part: np.sort(order[part])  # noqa: E731
    folds = []
    for fold in split["folds"]:
        stamps = split["stamp_of_row"]
        first = stamps[fold["valid"]].min()
        gap_part = np.flatnonzero((stamps >= first - gap) & (stamps < first))
        folds.append({"training": back(fold["train"]), "gap": back(gap_part),
                      "validation": back(fold["valid"])})
    return {"training": back(split["train"]), "gap": back(split["gap"]),
            "holdout": back(split["holdout"]), "folds": folds}


def roc_auc(y, score):
    """ROC AUC of `score` for the 0/1 labels `y`; ties share their average rank."""
    value = core._auc(np.asarray(y).astype("int64"), np.asarray(score, dtype="float64"))
    if value is None:
        raise E.OneClassError("ROC AUC needs both classes")
    return value


def mse(y, pred):
    """Mean squared error."""
    return core._mse(np.asarray(y, dtype="float64"), np.asarray(pred, dtype="float64"))


def skill(metric, score, baseline_score):
    """Skill over the baseline: 0 is no better than it, 1 is perfect.

    None for `"mse"` when the baseline's error is zero: skill has no value there (D36).
    """
    if metric == "roc_auc":
        return (score - baseline_score) / (1 - baseline_score)
    if metric == "mse":
        return None if baseline_score == 0 else 1 - score / baseline_score
    raise ValueError(f"metric must be roc_auc or mse, not {metric!r}")


def previous_value(y, keys, gap=0, groups=None):
    """For each row, the target `gap + 1` timestamps earlier; NaN where there is none.

    With `groups`, one value per row: the target of the latest row of the same
    group that lies at least `gap + 1` distinct timestamps earlier (D57).
    """
    y, keys = np.asarray(y, dtype="float64"), np.asarray(keys)
    if groups is not None:
        codes = core._group_codes(pd.Series(groups).reset_index(drop=True), "groups")
        _, stamp = np.unique(keys, return_inverse=True)
        cell = codes.astype("int64") * (int(stamp.max()) + 1) + stamp
        if (codes < 0).any() or len(np.unique(cell)) != len(cell):
            raise ValueError("previous_value needs a group for every row, and unique "
                             "timestamps within each group")
        return core._previous_by_group(y, keys, codes, int(gap))
    if len(np.unique(keys)) != len(keys):
        raise ValueError("previous_value needs unique timestamps")
    order = np.argsort(keys, kind="stable")
    out = np.full(len(y), np.nan)
    out[order[gap + 1:]] = y[order[:len(y) - gap - 1]]
    return out


class Preprocessor:
    """Step 8's built-in preprocessing for one fit: fit on some rows, apply to any.

    `fit` raises `FeatureNameCollisionError` when two of the columns it would
    give the matrix have the same name (D37). An infinite cell and one too large
    for the tree models are missing cells to it (D48), so every statistic it
    learns is finite.
    """

    def __init__(self, kinds):
        self._prep = core._Prep(dict(kinds))

    def fit(self, frame):
        self._prep.fit(frame)
        return self

    def transform(self, frame):
        return pd.DataFrame(self._prep.transform(frame), index=frame.index,
                            columns=self._prep.names())
