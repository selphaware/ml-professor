"""Preprocessing of step 8, fitted on the rows of one fit only.

Numeric columns: missing (and infinite) cells take the median of the fit rows
(0 if the column is all missing there), then are standardised with the mean and
standard deviation of the fit rows (a zero deviation is read as 1).
Categorical columns: a missing cell is the level `(missing)`; one 0/1 column per
kept level, at most the 30 most frequent in the fit rows (ties by level text),
in sorted order; any other level is all zeros.
"""
import numpy as np
import pandas as pd

from .errors import FeatureNameCollisionError

MISSING_LEVEL = "(missing)"
MAX_LEVELS = 30


def numeric_values(column):
    """Float copy of a column with NaN for missing and infinite cells."""
    values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def level_values(column):
    """Text levels of a column, a missing cell read as `(missing)`."""
    missing = column.isna().to_numpy()
    levels = column.astype(object).to_numpy(copy=True)
    levels[missing] = MISSING_LEVEL
    return np.array([str(level) for level in levels], dtype=object)


def kept_levels(levels):
    """The at most 30 most frequent levels (ties by text), in sorted order."""
    counts = pd.Series(levels, dtype=object).value_counts(sort=False)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return sorted(level for level, _ in ranked[:MAX_LEVELS])


def check_feature_names(kinds, levels_by_column):
    """Raise when two matrix columns would share a name.

    `levels_by_column` maps each categorical column to the levels that name
    matrix columns.
    """
    owners = {}
    for col, kind in kinds.items():
        if kind == "numeric":
            owners.setdefault(col, []).append(col)
        else:
            for level in levels_by_column[col]:
                owners.setdefault(f"{col}={level}", []).append(col)
    clashes = {name: cols for name, cols in owners.items() if len(cols) > 1}
    if clashes:
        involved = sorted({col for cols in clashes.values() for col in cols},
                          key=list(kinds).index)
        names = ", ".join(repr(name) for name in clashes)
        raise FeatureNameCollisionError(
            f"Feature names must be distinct; {names} would be given by more than one "
            f"column. Columns involved: {', '.join(repr(c) for c in involved)}. "
            "Rename one of them.")


class Preprocessor:
    """Median imputation and standardisation, and one-hot levels, for one fit."""

    def __init__(self, kinds):
        self.kinds = dict(kinds)
        self.numeric_cols = [c for c, k in self.kinds.items() if k == "numeric"]
        self.categorical_cols = [c for c, k in self.kinds.items() if k == "categorical"]
        unknown = set(self.kinds.values()) - {"numeric", "categorical"}
        if unknown:
            raise ValueError(f"unknown column kinds: {sorted(unknown)}")
        self.medians = {}
        self.means = {}
        self.scales = {}
        self.levels = {}

    def fit(self, frame):
        """Learn every statistic from the rows of `frame` alone; return self."""
        for col in self.numeric_cols:
            fit_values = numeric_values(frame[col])
            observed = fit_values[~np.isnan(fit_values)]
            median = float(np.median(observed)) if observed.size else 0.0
            imputed = np.where(np.isnan(fit_values), median, fit_values)
            mean = float(imputed.mean()) if imputed.size else 0.0
            deviation = float(imputed.std()) if imputed.size else 0.0
            self.medians[col] = median
            self.means[col] = mean
            self.scales[col] = deviation if deviation > 0 and np.isfinite(deviation) else 1.0
        for col in self.categorical_cols:
            self.levels[col] = kept_levels(level_values(frame[col]))
        check_feature_names(self.kinds, self.levels)
        return self

    def transform(self, frame):
        """The preprocessed matrix for the rows of `frame`, same index."""
        blocks = {}
        for col in self.numeric_cols:
            values = numeric_values(frame[col])
            values = np.where(np.isnan(values), self.medians[col], values)
            blocks[col] = (values - self.means[col]) / self.scales[col]
        for col in self.categorical_cols:
            levels = level_values(frame[col])
            for level in self.levels[col]:
                blocks[f"{col}={level}"] = (levels == level).astype(float)
        return pd.DataFrame(blocks, index=frame.index, columns=list(blocks), dtype=float)
