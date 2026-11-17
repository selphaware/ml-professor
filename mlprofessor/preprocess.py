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
