"""Step 7: split off the holdout and form the selection folds (D6, D7)."""
import math

import numpy as np
import pandas as pd

from mlprofessor import errors

N_FOLDS = 5
HOLDOUT_SHARE_DENOMINATOR = 4


def _class_groups(n_rows, classes):
    """Positions 0..n_rows-1 grouped by class label, one array per class; one group if unstratified."""
    if classes is None:
        return [np.arange(n_rows)]
    labels = pd.Series(np.asarray(classes, dtype=object)).reset_index(drop=True)
    if len(labels) != n_rows:
        raise ValueError(f"classes has {len(labels)} labels for {n_rows} rows")
    codes, _ = pd.factorize(labels, use_na_sentinel=False)
    return [np.flatnonzero(codes == code) for code in range(codes.max() + 1)]


def _holdout_quotas(group_sizes, holdout_size):
    """Holdout rows per class: a quarter of each class, rounded so the quotas sum to the holdout."""
    exact = np.asarray(group_sizes, dtype=float) / HOLDOUT_SHARE_DENOMINATOR
    quotas = np.floor(exact).astype(int)
    extra = holdout_size - int(quotas.sum())
    # The classes with the largest remainders take one row more; ties go to the earlier class.
    order = np.argsort(-(exact - quotas), kind="stable")
    quotas[order[:extra]] += 1
    return quotas


def _deal_into_folds(shuffled_groups):
    """Deal rows to the folds in turn, class after class, so fold sizes differ by at most one."""
    fold_of_position = {}
    turn = 0
    for group in shuffled_groups:
        for position in group:
            fold_of_position[int(position)] = turn % N_FOLDS
            turn += 1
    return fold_of_position


def split_random(n_rows, classes=None, seed=0):
    """The random scheme: a holdout of ceil(n/4) rows, then five folds of the training rows.

    With `classes` the holdout and the folds are stratified on them. Every draw
    comes from `seed`. All position arrays are in increasing order.
    """
    rng = np.random.default_rng(seed)
    groups = _class_groups(n_rows, classes)
    holdout_size = math.ceil(n_rows / HOLDOUT_SHARE_DENOMINATOR)
    quotas = _holdout_quotas([len(g) for g in groups], holdout_size)

    holdout_parts, training_groups = [], []
    for group, quota in zip(groups, quotas):
        shuffled = rng.permutation(group)
        holdout_parts.append(shuffled[:quota])
        training_groups.append(shuffled[quota:])
    holdout_rows = np.sort(np.concatenate(holdout_parts)).astype(np.intp)
    training_rows = np.sort(np.concatenate(training_groups)).astype(np.intp)

    shuffled_training = [rng.permutation(group) for group in training_groups]
    fold_of_position = _deal_into_folds(shuffled_training)
    fold_labels = np.array([fold_of_position[int(p)] for p in training_rows], dtype=int)

    folds = []
    for fold in range(N_FOLDS):
        in_fold = fold_labels == fold
        folds.append({
            "training": training_rows[~in_fold],
            "validation": training_rows[in_fold],
        })
    return {"training": training_rows, "holdout": holdout_rows, "folds": folds}


# ---------------------------------------------------------------- time order

N_TIME_BLOCKS = N_FOLDS + 1

# Step 3: the one text form a time column may take. Year first, always; a zone
# only after a time of day. Nothing day-first or month-first ever reaches a parser.
_ISO_8601 = (
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{3})?)?(?P<zone>Z|[+-]\d{2}:\d{2})?)?"
)


def _time_error(message):
    return errors.TimeColumnError(f"time column: {message}")


def _datetime_keys(values):
    """A datetime-typed column, in UTC when it carries a zone, as datetime64[ns]."""
    if getattr(values.dt, "tz", None) is not None:
        values = values.dt.tz_convert("UTC").dt.tz_localize(None)
    try:
        return values.astype("datetime64[ns]").to_numpy(dtype="datetime64[ns]", copy=True)
    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime) as exc:
        raise _time_error(f"a timestamp is out of range ({exc})") from exc


def _numeric_keys(values):
    """A numeric column as float64, missing and infinite values as NaN."""
    keys = values.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    keys[~np.isfinite(keys)] = np.nan
    return keys


def _iso_keys(values):
    """Text in the ISO 8601 form of step 3, as instants in datetime64[ns]."""
    present = values.notna().to_numpy()
    text = values[present].astype(str)
    matched = text.str.fullmatch(_ISO_8601).to_numpy(dtype=bool)
    if not matched.all():
        first = text[~matched].iloc[0]
        raise _time_error(
            f"value {first!r} is not ISO 8601 with the year first "
            "(YYYY-MM-DD, optionally with a time and a zone)"
        )
    zoned = text.str.extract(f"^{_ISO_8601}$")["zone"].notna().to_numpy(dtype=bool)
    if zoned.any() and not zoned.all():
        raise _time_error("some values carry a zone and some do not")
    keys = np.full(len(values), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    if len(text) == 0:
        return keys
    try:
        parsed = pd.to_datetime(text, format="ISO8601", utc=True, errors="raise")
    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime) as exc:
        first_line = (str(exc) or type(exc).__name__).splitlines()[0].split(" You might")[0]
        raise _time_error(f"a value is not a real date and time: {first_line}") from exc
    keys[present] = _datetime_keys(parsed)
    return keys


def read_time(values):
    """Step 3: read a time column in one of its three accepted forms.

    Returns `(keys, form)`: keys sort in time order (datetime64[ns] with NaT,
    or float64 with NaN); form is `datetime`, `numeric` or `iso8601`.
    """
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    values = values.reset_index(drop=True)
    dtype = values.dtype
    if pd.api.types.is_bool_dtype(dtype):
        raise _time_error("a boolean column is not a time")
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return _datetime_keys(values), "datetime"
    if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype):
        return _numeric_keys(values), "numeric"
    if pd.api.types.is_string_dtype(dtype):
        present = values[values.notna()]
        if dtype == object and not present.map(lambda v: isinstance(v, str)).all():
            raise _time_error("the column mixes values of different types")
        return _iso_keys(values), "iso8601"
    raise _time_error(f"a column of type {dtype} is not an accepted time form")


def _holdout_stamp_count(counts, n_rows):
    """The fewest latest timestamps whose rows number at least a quarter of all rows."""
    held_from_latest = np.cumsum(counts[::-1])
    return int(np.argmax(HOLDOUT_SHARE_DENOMINATOR * held_from_latest >= n_rows)) + 1


def split_time(keys, gap=0):
    """The time-ordered scheme: a holdout of the latest whole timestamps, a gap
    before it, and five expanding folds over six contiguous blocks of the
    training timestamps, each with `gap` timestamps taken off before its
    validation block. All position arrays are in increasing order.
    """
    gap = int(gap)
    if gap < 0:
        raise ValueError(f"gap must be zero or more, not {gap}")
    keys = np.asarray(keys)
    missing = np.isnat(keys) if np.issubdtype(keys.dtype, np.datetime64) else np.isnan(keys)
    if missing.any():
        raise ValueError("split_time takes keys with no missing value")
    n_rows = len(keys)
    stamps, stamp_of_row, counts = np.unique(keys, return_inverse=True, return_counts=True)
    stamp_of_row = stamp_of_row.reshape(-1)
    n_stamps = len(stamps)

    n_holdout_stamps = _holdout_stamp_count(counts, n_rows) if n_rows else 0
    first_holdout = n_stamps - n_holdout_stamps
    if first_holdout < N_TIME_BLOCKS:
        raise errors.TooFewRowsError(
            f"{n_stamps} distinct timestamps leave {first_holdout} before the holdout; "
            f"the six blocks need at least {N_TIME_BLOCKS}"
        )
    n_training_stamps = first_holdout - gap
    if n_training_stamps < N_TIME_BLOCKS:
        raise errors.GapError(
            f"a gap of {gap} timestamps leaves {max(n_training_stamps, 0)} training "
            f"timestamps; the six blocks need at least {N_TIME_BLOCKS}"
        )

    def rows_between(start, stop):
        return np.flatnonzero((stamp_of_row >= start) & (stamp_of_row < stop)).astype(np.intp)

    blocks = np.array_split(np.arange(n_training_stamps), N_TIME_BLOCKS)
    folds = []
    for fold in range(N_FOLDS):
        validation_stamps = blocks[fold + 1]
        block_start = int(validation_stamps[0])
        train_stop = block_start - gap
        if train_stop <= 0:
            raise errors.GapError(
                f"a gap of {gap} timestamps leaves no training timestamp "
                f"before validation block {fold + 2}"
            )
        folds.append({
            "training": rows_between(0, train_stop),
            "gap": rows_between(train_stop, block_start),
            "validation": rows_between(block_start, int(validation_stamps[-1]) + 1),
        })
    return {
        "training": rows_between(0, n_training_stamps),
        "gap": rows_between(n_training_stamps, first_holdout),
        "holdout": rows_between(first_holdout, n_stamps),
        "folds": folds,
    }
