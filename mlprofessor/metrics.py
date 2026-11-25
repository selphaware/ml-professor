"""Holdout metrics, skill over a baseline, and the previous-value baseline."""
import math

import numpy as np
import pandas as pd

from mlprofessor.errors import OneClassError

METRICS = ("roc_auc", "mse")


def average_ranks(values):
    """Ranks 1..n of `values`; tied values share their average rank."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    n = len(values)
    # Start of each run of equal values in sorted order.
    starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    ends = np.r_[starts[1:], n]
    # Positions start..end-1 hold ranks start+1..end; their mean is (start+1+end)/2.
    run_rank = (starts + 1 + ends) / 2.0
    sorted_ranks = np.repeat(run_rank, ends - starts)
    ranks = np.empty(n, dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def roc_auc(y, score):
    """ROC AUC of `score` for labels 0 and 1 (1 positive), by average ranks."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    if y.shape != score.shape or y.ndim != 1:
        raise ValueError("y and score must be one-dimensional and the same length")
    positive = y == 1
    n_pos = int(positive.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise OneClassError(
            f"ROC AUC needs both classes; found {n_pos} positive and {n_neg} "
            "negative rows. Provide rows of both classes.")
    ranks = average_ranks(score)
    u_stat = ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


def mse(y, pred):
    """Mean squared error."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if y.shape != pred.shape:
        raise ValueError("y and pred must be the same length")
    with np.errstate(over="ignore", invalid="ignore"):
        # Infinite when a squared error overflows; the caller decides what that means.
        return float(np.mean((y - pred) ** 2))


def squared_errors(y, pred):
    """The squared error of each row; infinite where one overflows (the caller decides)."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        return (y - pred) ** 2


def largest_row_share(errors):
    """`(share, position)` of the largest of finite squared `errors` (step 11).

    The share is the largest error over the sum of them, each divided by the
    largest first so that no sum overflows; the position is that of the first
    of several largest. `(None, None)` when every error is zero.
    """
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return None, None
    position = int(np.argmax(errors))
    top = float(errors[position])
    if top == 0:
        return None, None
    share = 1.0 / math.fsum((errors / top).tolist())
    return min(share, 1.0), position


def skill(metric, score, baseline_score):
    """Skill over the baseline: 0 at the baseline, 1 at perfect, may be negative.

    For `mse`, None when the baseline's error is exactly zero: skill has no value.
    """
    if metric == "roc_auc":
        return float((score - baseline_score) / (1.0 - baseline_score))
    if metric == "mse":
        if baseline_score == 0:
            return None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return float(1.0 - np.float64(score) / np.float64(baseline_score))
    raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")


def previous_value(y, keys, gap=0, groups=None):
    """For each row, the mean target of the rows `gap + 1` distinct timestamps earlier.

    NaN for a row at one of the first `gap + 1` timestamps. With one row a
    timestamp that is the target of the row `gap + 1` timestamps earlier.

    With `groups`, the target of the latest row of the same group at least
    `gap + 1` distinct timestamps earlier, the timestamps being those of all
    the rows given; NaN where the group has no such row.
    """
    y = np.asarray(y, dtype=float)
    keys = np.asarray(keys)
    if y.shape != keys.shape or y.ndim != 1:
        raise ValueError("y and keys must be one-dimensional and the same length")
    if int(gap) < 0:
        raise ValueError("gap must be 0 or more")
    lag = int(gap) + 1
    if len(keys) and np.any(_missing(keys)):
        raise ValueError("keys must not hold missing values")
    if groups is not None:
        return _previous_in_groups(y, keys, lag, groups)
    out = np.full(len(y), np.nan)
    if not len(y):
        return out
    # The mean target of each distinct timestamp, then each row is given the mean of
    # the timestamp `lag` places before its own. With one row a timestamp that mean is
    # the row's own target, exactly.
    _, stamp_rank = np.unique(keys, return_inverse=True)
    stamp_rank = stamp_rank.reshape(-1).astype(np.int64)
    n_stamps = int(stamp_rank.max()) + 1
    counts = np.bincount(stamp_rank, minlength=n_stamps)
    with np.errstate(over="ignore", invalid="ignore"):
        sums = np.bincount(stamp_rank, weights=y, minlength=n_stamps)
        means = sums / counts
    has_before = stamp_rank >= lag
    out[has_before] = means[stamp_rank[has_before] - lag]
    return out


def _group_codes(groups, n_rows):
    """One code per row for `groups`, read as an entity column is; ValueError if one is empty."""
    from mlprofessor.columns import read_entities

    codes = np.asarray(read_entities(groups), dtype=np.int64)
    if len(codes) != n_rows:
        raise ValueError(f"groups holds {len(codes)} values for {n_rows} rows")
    if (codes < 0).any():
        raise ValueError(f"{int((codes < 0).sum())} value(s) of groups are empty")
    return codes


def repeated_in_groups(keys, codes):
    """A mask of the rows that share their group code and their timestamp with another row."""
    _, stamp_rank = np.unique(np.asarray(keys), return_inverse=True)
    pairs = pd.DataFrame({"g": np.asarray(codes), "t": stamp_rank.reshape(-1)})
    return pairs.duplicated(keep=False).to_numpy(dtype=bool)


def _previous_in_groups(y, keys, lag, groups):
    codes = _group_codes(groups, len(y))
    out = np.full(len(y), np.nan)
    if not len(y):
        return out
    if repeated_in_groups(keys, codes).any():
        raise ValueError("previous_value needs unique timestamps within each group; "
                         "two rows of one group share a timestamp")
    _, stamp_rank = np.unique(keys, return_inverse=True)
    stamp_rank = stamp_rank.reshape(-1).astype(np.int64)
    # One sortable number per row: its group, then its timestamp shifted by the lag,
    # so the latest row of a group at or below `rank - lag` is found by one search.
    width = int(stamp_rank.max()) + lag + 1
    placed = codes * width + stamp_rank + lag
    order = np.argsort(placed, kind="mergesort")
    sorted_placed = placed[order]
    wanted = codes * width + stamp_rank
    found = np.searchsorted(sorted_placed, wanted, side="right") - 1
    has = found >= 0
    has[has] = codes[order[found[has]]] == codes[has]
    out[has] = y[order[found[has]]]
    return out


def _missing(keys):
    """True where a key is NaN or NaT."""
    if np.issubdtype(keys.dtype, np.datetime64):
        return np.isnat(keys)
    if np.issubdtype(keys.dtype, np.number):
        return np.isnan(keys.astype(float))
    return np.zeros(len(keys), dtype=bool)


def lag1_autocorrelation(values):
    """Lag-1 autocorrelation of a series in order (the usual estimator).

    NaN when it has no value (fewer than two values, or none that vary) and
    when it cannot be computed because a number on the way is not finite.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if len(values) < 2:
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        centred = values - values.mean()
        spread = float((centred ** 2).sum())
        if spread == 0.0 or not np.isfinite(spread):
            return float("nan")
        return float((centred[:-1] * centred[1:]).sum() / spread)


EXTREME_GAP = 30
EXTREME_SPREAD = 10
EXTREME_OUTERMOST = 3


def far_outside(values):
    """The values far outside all the others (step 11): `[(value, rows, nearest), ...]`.

    Over the distinct finite values, sorted: from each end, of the three
    outermost, a value is far outside when its gap to the next value inward
    is more than 30 times the range of the ten distinct values after that one
    (as many as there are, at least one); the values beyond it are too.
    `rows` counts the cells that hold the value, `nearest` is the nearest
    value that is not far outside. The low end first.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)] + 0.0
    distinct, counts = np.unique(values, return_counts=True)
    n = len(distinct)
    named = []
    if n < 3:
        return named
    for order in (np.arange(n), np.arange(n)[::-1]):
        ordered = distinct[order]
        last = 0
        with np.errstate(over="ignore", invalid="ignore"):
            for j in range(1, min(EXTREME_OUTERMOST, n - 2) + 1):
                gap = abs(float(ordered[j]) - float(ordered[j - 1]))
                m = min(EXTREME_SPREAD, n - j - 1)
                spread = abs(float(ordered[j + m]) - float(ordered[j]))
                if gap > EXTREME_GAP * spread:
                    last = j
        for j in range(last):
            named.append((float(ordered[j]), int(counts[order[j]]), float(ordered[last])))
    return named
