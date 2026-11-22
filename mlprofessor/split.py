"""Step 7: split off the holdout and form the selection folds (D6, D7)."""
import math
import re
from numbers import Integral

import numpy as np
import pandas as pd

from mlprofessor import errors
from mlprofessor.load import beyond_64_bits, read_numbers
from mlprofessor.columns import read_entities, stripped_text, text_of_dates
from mlprofessor.task import _sorted_classes, read_target_values

MAX_SEED = 2**32 - 1
N_FOLDS = 5
HOLDOUT_SHARE_DENOMINATOR = 4


def _class_groups(n_rows, classes):
    """Positions 0..n_rows-1 grouped by class label, one array per class; one group if unstratified."""
    if classes is None:
        return [np.arange(n_rows)]
    # Text labels are compared without the white space round them (D46).
    labels = read_target_values(pd.Series(np.asarray(classes, dtype=object))).reset_index(drop=True)
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


MIN_UNITS = 6
MIN_FOLD_UNITS = N_FOLDS


def units_of(groups, n_rows):
    """Each row's unit, numbered 0.. in the order the units first appear (step 7).

    The numbering depends only on which rows share a value, never on the values.
    """
    codes = read_entities(groups)
    if len(codes) != n_rows:
        raise errors.GroupColumnError(
            f"groups holds {len(codes)} values for {n_rows} rows; one value per row is needed.")
    if (codes < 0).any():
        raise errors.GroupColumnError(
            f"{int((codes < 0).sum())} of the rows split have an empty entity cell; every row "
            "split must name its entity. Fill the cells in or remove the rows.")
    _, first, inverse = np.unique(codes, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    return rank[inverse.reshape(-1)]


def _unit_classes(unit, n_units, classes, n_rows):
    """Each unit's class: 1 when most of its rows hold the positive class (ties too), else 0."""
    labels = read_target_values(pd.Series(np.asarray(classes, dtype=object))).reset_index(drop=True)
    if len(labels) != n_rows:
        raise ValueError(f"classes has {len(labels)} labels for {n_rows} rows")
    distinct = _sorted_classes(pd.unique(labels.to_numpy()))
    positive = distinct[-1] if len(distinct) < 2 else distinct[1]
    is_positive = np.array([value == positive for value in labels.to_numpy()], dtype=float)
    positive_rows = np.bincount(unit, weights=is_positive, minlength=n_units)
    sizes = np.bincount(unit, minlength=n_units)
    return (2 * positive_rows >= sizes).astype(np.int64)


def _holdout_units(ordered_units, sizes):
    """The holdout of one pool of units, taken in `ordered_units` order (step 7, property 3).

    A unit joins while the holdout, with it, still holds less than a quarter of the
    pool's rows; the smallest unit passed over (the first among equals) completes it;
    then every unit it can spare (one without which it still holds a quarter) goes back.
    """
    total = int(sizes[ordered_units].sum())
    joined, passed_over, held_rows = [], [], 0
    for u in ordered_units:
        u = int(u)
        if HOLDOUT_SHARE_DENOMINATOR * (held_rows + int(sizes[u])) < total:
            joined.append(u)
            held_rows += int(sizes[u])
        else:
            passed_over.append(u)
    if not passed_over:
        return joined
    # `min` keeps the first in the order among units of equal size.
    completing = min(passed_over, key=lambda u: int(sizes[u]))
    holdout = joined + [completing]
    held_rows += int(sizes[completing])
    # Spare the largest first, then in the drawn order; one pass suffices, since giving
    # a unit back never lets another be spared that could not be before.
    kept = []
    for position in sorted(range(len(holdout)), key=lambda p: (-int(sizes[holdout[p]]), p)):
        u = holdout[position]
        if HOLDOUT_SHARE_DENOMINATOR * (held_rows - int(sizes[u])) >= total:
            held_rows -= int(sizes[u])
        else:
            kept.append(u)
    return kept


def large_units(groups, n_rows, classes, holdout):
    """Each unit of a quarter or more of the rows it is drawn among (step 7, D66).

    The rows it is drawn among are all the rows split, or for classification the rows
    of the units of its class. One record per such unit, in the order units first
    appear: its rows, those rows it is drawn among, and whether it is in `holdout`.
    """
    unit = units_of(groups, n_rows)
    n_units = int(unit.max()) + 1 if n_rows else 0
    sizes = np.bincount(unit, minlength=n_units)
    unit_class = (np.zeros(n_units, dtype=np.int64) if classes is None
                  else _unit_classes(unit, n_units, classes, n_rows))
    pool_rows = np.bincount(unit_class, weights=sizes, minlength=2).astype(np.int64)
    held = np.zeros(n_units, dtype=bool)
    held[unit[np.asarray(holdout, dtype=np.intp)]] = True
    found = []
    for u in range(n_units):
        among = int(pool_rows[unit_class[u]])
        if HOLDOUT_SHARE_DENOMINATOR * int(sizes[u]) >= among:
            found.append({"rows": int(sizes[u]), "among": among, "held": bool(held[u])})
    return found


def _split_whole_units(unit, n_rows, classes, rng):
    """The grouped scheme of step 7: the holdout and five folds of whole units."""
    n_units = int(unit.max()) + 1 if n_rows else 0
    if n_units < MIN_UNITS:
        raise errors.TooFewGroupsError(
            f"The rows split hold {n_units} entities; a split of whole entities needs at "
            f"least {MIN_UNITS}.")
    sizes = np.bincount(unit, minlength=n_units)
    unit_class = (np.zeros(n_units, dtype=np.int64) if classes is None
                  else _unit_classes(unit, n_units, classes, n_rows))

    held = np.zeros(n_units, dtype=bool)
    for label in (0, 1):
        pool = np.flatnonzero(unit_class == label)
        if len(pool) == 0:
            continue
        for u in _holdout_units(rng.permutation(pool), sizes):
            held[u] = True
    training_units = np.flatnonzero(~held)
    if len(training_units) < MIN_FOLD_UNITS:
        raise errors.TooFewGroupsError(
            f"The rows split hold {n_units} entities; holding a quarter of the rows out as "
            f"whole entities leaves {len(training_units)} for the folds, and the five folds "
            f"need at least {MIN_FOLD_UNITS}.")

    # Each training unit, in an order drawn from the seed, to the fold that so far holds
    # the fewest rows of its class (then the fewest rows, then the first such fold).
    fold_of_unit = np.full(n_units, -1, dtype=np.int64)
    class_rows = np.zeros((2, N_FOLDS), dtype=np.int64)
    fold_rows = np.zeros(N_FOLDS, dtype=np.int64)
    for u in rng.permutation(training_units):
        label = int(unit_class[u])
        fold = min(range(N_FOLDS), key=lambda f: (class_rows[label, f], fold_rows[f], f))
        fold_of_unit[u] = fold
        class_rows[label, fold] += sizes[u]
        fold_rows[fold] += sizes[u]

    in_holdout = held[unit]
    holdout_rows = np.flatnonzero(in_holdout).astype(np.intp)
    training_rows = np.flatnonzero(~in_holdout).astype(np.intp)
    fold_labels = fold_of_unit[unit[training_rows]]
    folds = [{"training": training_rows[fold_labels != fold],
              "validation": training_rows[fold_labels == fold]} for fold in range(N_FOLDS)]
    return {"training": training_rows, "holdout": holdout_rows, "folds": folds}


def split_random(n_rows, classes=None, seed=0, groups=None):
    """The random scheme: a holdout of ceil(n/4) rows, then five folds of the training rows.

    With `classes` the holdout and the folds are stratified on them. With
    `groups` (one value per row) the rows of one value stay together: the
    grouped scheme of step 7. Every draw comes from `seed`. All position
    arrays are in increasing order.
    """
    if (isinstance(seed, bool) or not isinstance(seed, Integral)
            or not 0 <= seed <= MAX_SEED):
        raise errors.InputError(
            f"seed must be a whole number from 0 to {MAX_SEED}; got {seed!r}.")
    rng = np.random.default_rng(int(seed))
    if groups is not None:
        return _split_whole_units(units_of(groups, n_rows), n_rows, classes, rng)
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
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?P<zone>Z|[+-]\d{2}:\d{2})?)?"
)
# The same form with a fraction of any length of ten digits or more.
_LONG_FRACTION = _ISO_8601.replace(r"\.\d{1,9}", r"\.\d{10,}")


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


# A 64-bit float holds every whole number up to this magnitude, and only some above it.
EXACT_FLOAT_LIMIT = 2 ** 53


def _integer_keys(values, rows):
    """Exact int64 (or uint64) keys of `rows` for an integer-typed column with a
    magnitude above 2**53 among its non-missing values (step 3; D50, D53), when
    none of those rows is missing; None otherwise."""
    if not pd.api.types.is_integer_dtype(values.dtype):
        return None
    present = values.notna().to_numpy(dtype=bool)
    if not present.any() or not present[rows].all():
        return None
    unsigned = pd.api.types.is_unsigned_integer_dtype(values.dtype)
    keys = values[present].to_numpy(dtype=np.uint64 if unsigned else np.int64, copy=True)
    if unsigned:
        above = keys > EXACT_FLOAT_LIMIT
    else:
        above = (keys > EXACT_FLOAT_LIMIT) | (keys < -EXACT_FLOAT_LIMIT)
    if not bool(above.any()):
        return None
    return values.iloc[rows].to_numpy(dtype=keys.dtype, copy=True)


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
        if re.fullmatch(_LONG_FRACTION.replace("(?P<zone>", "(?:"), first):
            raise _time_error(
                f"value {first!r} has a fraction of a second with more than nine "
                "digits, which a nanosecond timestamp cannot hold; use at most nine"
            )
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


def _dates_as_text(values):
    """True when every non-missing value of the column is text of eight digits that
    reads as a calendar date YYYYMMDD (step 3; D66)."""
    present = values[values.notna()]
    if len(present) == 0 or not all(isinstance(v, str) for v in present.to_numpy()):
        return False
    return text_of_dates(stripped_text(present))


def _date_text_keys(values):
    """Text of eight-digit dates as the float64 numbers they read as, NaN where missing."""
    text = stripped_text(values)
    keys = np.full(len(values), np.nan, dtype=np.float64)
    present = text.notna().to_numpy(dtype=bool)
    keys[present] = [float(int(value)) for value in text[present]]
    return keys


def read_time(values, rows=None):
    """Step 3: read a time column in one of its three accepted forms.

    Returns `(keys, form)`: keys sort in time order (datetime64[ns] with NaT,
    or float64 with NaN; exact int64 or uint64 for whole numbers of which one
    has a magnitude above 2**53); form is `datetime`, `numeric` or `iso8601`.
    Text of eight digits that all read as dates YYYYMMDD is `numeric` (D66).
    With `rows`, the column is read (step 1a, and the 2**53 rule) over all its
    values, and the keys returned are those of the rows given, in their order.
    """
    values = read_numbers(values).reset_index(drop=True)
    rows = (np.arange(len(values), dtype=np.int64) if rows is None
            else np.asarray(rows, dtype=np.int64))
    dtype = values.dtype
    if pd.api.types.is_bool_dtype(dtype):
        raise _time_error("a boolean column is not a time")
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return _datetime_keys(values.iloc[rows]), "datetime"
    if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype):
        exact = _integer_keys(values, rows)
        if exact is not None:
            return exact, "numeric"
        return _numeric_keys(values.iloc[rows]), "numeric"
    if beyond_64_bits(values):
        raise _time_error(
            "the column holds whole numbers too large for 64 bits, which are not read as "
            "a time")
    if pd.api.types.is_string_dtype(dtype) and _dates_as_text(values):
        return _date_text_keys(values.iloc[rows]), "numeric"
    values = values.iloc[rows].reset_index(drop=True)
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
