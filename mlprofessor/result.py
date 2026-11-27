"""`ProfileResult` (spec 2.3): the whole of `profile.json` and the holdout predictions."""
import copy
import json
import math
import os

import numpy as np
import pandas as pd

from mlprofessor.errors import NonFiniteScoreError

ROWS_ITEMS = [
    "loaded", "dropped_missing_target", "dropped_missing_time", "dropped_gap",
    "dropped_row_cap", "training", "holdout",
]
COLUMN_KEYS = [
    "name", "position", "dtype", "role", "kind", "reason", "leak_rule",
    "n_missing", "n_coerced", "n_infinite", "n_imputed",
    "shuffled_score", "shuffled_skill", "skill_fall",
]
MODEL_KEYS = [
    "config", "family", "params", "status", "error", "fold_scores",
    "cv_mean", "cv_std", "chosen", "largest_row_share", "largest_row",
]
WARNING_KEYS = ["code", "column", "message"]
PREDICTION_KEYS = ["row", "y_true", "y_pred"]

RANDOM_CAVEAT = ("Random split: this number is optimistic if the rows are ordered in "
                 "time or one entity appears on many rows.")
GROUPED_CAVEAT = ("Random split of whole entities: this number is optimistic if the rows "
                  "are ordered in time.")
LIMITATIONS = [
    "Predictability here means how well these model families did on this data under "
    "this validation scheme.",
    "Low skill is not proof that the target cannot be predicted.",
    "High skill is not proof of cause.",
    "A random split is optimistic when rows are ordered in time or one entity appears "
    "on many rows.",
    "Look-ahead inside the caller's own features cannot be detected.",
    "Classification skill and regression skill are on different scales and are not "
    "comparable.",
    "A CSV file is read with pandas' types, so text identifiers can lose leading zeros; "
    "pass a DataFrame to control the types.",
    "Intervals are approximate: they tend to be too narrow when the holdout is short "
    "and, under time order, when errors are serially correlated.",
    "Negative skill means the models did worse than the baseline on the holdout; read "
    "it as no evidence of predictability.",
]


def _non_finite_numbers(value, path):
    """`(path, value)` of every float of `value` that is infinite or not-a-number."""
    if isinstance(value, dict):
        return [found for key, item in value.items()
                for found in _non_finite_numbers(item, f"{path}.{key}")]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [found for i, item in enumerate(value)
                for found in _non_finite_numbers(item, f"{path}[{i}]")]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return [(path, value)]
    return []


def check_finite(document):
    """Raise `NonFiniteScoreError` naming each number of the document that is not finite.

    Spec 3, "Numbers that are not finite", rule 6: every part of the document,
    lists and nested records included, is read before anything is returned or
    written, so that no such number leaves as a null.
    """
    found = [hit for part, value in document.items()
             for hit in _non_finite_numbers(value, str(part))]
    if found:
        named = ", ".join(f"{path} ({value!r})" for path, value in found)
        raise NonFiniteScoreError(
            f"Numbers of the output are not finite: {named}. Likely cause, for a score: a "
            "target value too large for squared error, whose arithmetic overflows.")


def plain(value):
    """A JSON-ready copy: numpy scalars as Python values, NA and NaT as None."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [plain(item) for item in value.tolist()]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise NonFiniteScoreError(
            f"A number that is not finite ({value!r}) reached the output; it is never "
            "written as null.")
    return value


def _frame(records, keys):
    """A DataFrame of `records` with the columns `keys`, object cells kept as they are."""
    data = {key: pd.Series([record[key] for record in records], dtype=object)
            for key in keys}
    frame = pd.DataFrame(data, columns=keys)
    for key in keys:
        values = [record[key] for record in records]
        if values and all(isinstance(v, (bool, np.bool_)) for v in values):
            frame[key] = frame[key].astype(bool)
        elif values and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                            for v in values):
            frame[key] = frame[key].astype("int64")
        elif values and all(isinstance(v, (float, int, np.number)) and v is not None
                            and not isinstance(v, bool) for v in values):
            frame[key] = frame[key].astype(float)
    return frame


class ProfileResult:
    """The outcome of `profile()`: the headline, the tables and the output."""

    def __init__(self, document, predictions):
        check_finite(document)
        self._document = plain(document)
        self._predictions = predictions.reset_index(drop=True).copy()

    @property
    def headline(self):
        return copy.deepcopy(self._document["headline"])

    @property
    def rows(self):
        return _frame(self._document["rows"], ["item", "count"])

    @property
    def columns(self):
        return _frame(self._document["columns"], COLUMN_KEYS)

    @property
    def models(self):
        return _frame(self._document["models"], MODEL_KEYS)

    @property
    def warnings(self):
        return _frame(self._document["warnings"], WARNING_KEYS)

    @property
    def predictions(self):
        return self._predictions.copy()

    def to_dict(self):
        """The whole of `profile.json`, as a new dict each call."""
        return copy.deepcopy(self._document)

    def to_json(self, path):
        """Write `profile.json` to `path`."""
        text = json.dumps(self._document, indent=2, allow_nan=False, ensure_ascii=False)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")

    def write_predictions(self, path):
        """Write `predictions.csv` to `path`: `row,y_true,y_pred`, in increasing `row`."""
        holdout_predictions = self._predictions[PREDICTION_KEYS].sort_values(
            "row", kind="stable")
        holdout_predictions.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")

    def write(self, out_dir):
        """Create `out_dir` if missing and write `profile.json` and `predictions.csv`."""
        folder = os.fspath(out_dir)
        os.makedirs(folder, exist_ok=True)
        self.to_json(os.path.join(folder, "profile.json"))
        self.write_predictions(os.path.join(folder, "predictions.csv"))

    def __repr__(self):
        headline = self._document.get("headline", {})
        return (f"ProfileResult(metric={headline.get('metric')!r}, "
                f"skill={headline.get('skill')!r}, model={headline.get('model')!r})")
