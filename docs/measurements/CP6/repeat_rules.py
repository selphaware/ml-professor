"""Holdout rows that repeat a training row, counted by three rules (CP6 prototype).

    import repeat_rules
    found = repeat_rules.count_repeats(frame, "churned", "classification", seed=0)
    found["d49"]          # {"count": ..., "share": ..., "stop": ...}
    found["first_rule"]   # the same three
    found["calibrated"]   # the same three, and "chance": what chance alone gives

Needs pandas and numpy and nothing else: no part of the tool, of its tests or of
the gate is imported. This is the measurement prototype of CP6 made to run on its
own, so that the three rules can be tried on real tables. It is not the tool.
README.md beside this file has the call, an example and the constants.

    python repeat_rules.py        runs the example of the README

The three rules, each asked of a random split into training and holdout rows:

d49          a holdout row has a copy among the training rows: the same value in
             every feature column and the same target (decision D49).
first_rule   a copy, or a training row that agrees with it in more than half of
             the columns compared, two rare values among them (the rule first
             proposed for D57).
calibrated   a training row agrees with it by more than chance explains in this
             table. See `calibrated` below.

Each rule stops the run when more than 1% of the holdout rows, and at least 10
of them, are counted.
"""
import decimal
import itertools
import re

import numpy as np
import pandas as pd
from pandas.api import types as pt

# ------------------------------------------------------------------ constants

# The stop, the same for the three rules (D49): more than this share of the
# holdout rows, and at least this many of them.
STOP_SHARE = 0.01
STOP_ROWS = 10

# The six constants of the calibrated rule. README.md says what each does.
FLOOR = 20          # rows that must bear a thing out before the table is believed on it
RARE = 5            # a value held by this many rows or fewer is judged by its kind
GATE = 0.05         # chance matches expected, below which a match is improbable
WIDE = 0.5          # the same, for rows judged on an exact copy alone
MARGIN = 4.0        # spreads of the chance count taken off before a row counts as a repeat
MAX_DIFFER = 9      # the most facts in which a near-copy may differ

# The first rule's constants.
RARE_HOLDERS = 5    # a value is rare when no more than this many rows hold it
RARE_AGREEMENTS = 2  # two rows are repeats when they share this many rare values ...
#                     ... and agree in more than half of the columns compared

# How a column is read, as the tool's step 5 reads it.
MISSING_LEVEL = "(missing)"
TOO_LARGE = float(np.finfo(np.float32).max)
ID_TEXT_SHARE = 0.95        # text column: distinct values > 95% of the rows
FREE_TEXT_SHARE = 0.50      # text column: distinct values > 50% of non-missing
NUMERIC_SHARE = 0.90        # text column: >= 90% of non-missing parse as numbers
SEPARATED_SHARE = 0.90      # text column: >= 90% parse or carry thousands separators
DATE_SHARE = 0.95           # text column: >= 95% of non-missing look like dates

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
SEPARATED_RE = re.compile(r"[+-]?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?")


# ------------------------------------------------------------ reading a table

def _cell_missing(value):
    """Whether one cell of an object column is empty."""
    if isinstance(value, decimal.Decimal):
        return value.is_nan()
    gone = pd.isna(value)                   # an array for a list: a list is a value
    return isinstance(gone, (bool, np.bool_)) and bool(gone)


def _cell_kind(value):
    """`bool`, `int` or `real` for a number in an object column; None for anything else."""
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, decimal.Decimal):
        return "int" if value.is_finite() and value == value.to_integral_value() else "real"
    if isinstance(value, (float, np.floating)):
        return "real"
    return None


def _as_float(value):
    try:
        return float(value)
    except OverflowError:
        return float("inf") if value > 0 else float("-inf")


def _object_numbers(series):
    """An object column that holds only numbers, as the typed column it stands for.

    Only booleans: the nullable boolean type. Only whole numbers: a nullable
    64-bit integer type, or None when they do not fit in 64 bits. Whole numbers,
    floats and decimal numbers in any mixture: 64-bit floats. Anything else: None.
    """
    cells = series.to_numpy(dtype=object)
    missing = np.fromiter((_cell_missing(v) for v in cells), dtype=bool, count=len(cells))
    kinds = {_cell_kind(v) for v in cells[~missing]}
    if not kinds or None in kinds or (len(kinds) > 1 and "bool" in kinds):
        return None

    def typed(values, dtype):
        return pd.Series(pd.array(values, dtype=dtype), index=series.index, name=series.name)

    if kinds == {"bool"}:
        return typed([None if gone else bool(v) for v, gone in zip(cells, missing)], "boolean")
    if kinds == {"int"}:
        whole = [None if gone else int(v) for v, gone in zip(cells, missing)]
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


def _is_numeric(dtype):
    """Integer or float type. Boolean is not numeric here, and nor is complex."""
    return pt.is_numeric_dtype(dtype) and not pt.is_bool_dtype(dtype) \
        and not pt.is_complex_dtype(dtype)


def _whole_number_id(x):
    """Every value distinct, and consecutive or strictly ordered in file order."""
    if len(x) < 2 or len(np.unique(x)) != len(x):
        return False
    if x.dtype.kind == "f" and not np.all(x == np.floor(x)):
        return False
    if np.all(x[1:] > x[:-1]) or np.all(x[1:] < x[:-1]):
        return True
    return int(x.max()) - int(x.min()) + 1 == len(x)


def _read_column(series, usable, n_rows):
    """One column that is not the target: `(values, None)` when it is compared,
    `(None, reason)` when it is left out. The values are 64-bit floats for a
    numeric column and text levels for a categorical one, over every row."""
    if series.dtype == object:
        typed = _object_numbers(series)
        if typed is not None:
            series = typed
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):          # treated as text, by string form
        series = pd.Series([v if pd.isna(v) else str(v) for v in series],
                           index=series.index, dtype=object)
        dtype = series.dtype
    present = series[usable].dropna()
    numbers = None
    if len(present) == 0:
        return None, "all_missing"
    if pt.is_datetime64_any_dtype(dtype):
        return None, "datetime"
    if pt.is_bool_dtype(dtype) or (pt.is_numeric_dtype(dtype) and not pt.is_complex_dtype(dtype)):
        numbers = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    elif (pt.is_string_dtype(dtype) or dtype == object) \
            and all(isinstance(v, str) for v in present):
        text = present.str.strip()
        parses = pd.to_numeric(text, errors="coerce").notna().to_numpy(dtype=bool)
        separated = text.map(lambda v: SEPARATED_RE.fullmatch(v) is not None).to_numpy(dtype=bool)
        distinct = int(text.nunique())                  # text is compared after stripping
        if text.map(lambda v: DATE_RE.fullmatch(v) is not None).mean() >= DATE_SHARE:
            return None, "datetime"
        if distinct > 1 and separated.any() and (parses | separated).mean() >= SEPARATED_SHARE:
            return None, "numeric_with_separators"
        if parses.mean() >= NUMERIC_SHARE:
            numbers = pd.to_numeric(series.str.strip(), errors="coerce") \
                .to_numpy(dtype="float64", na_value=np.nan)
        elif distinct == 1:
            return None, "constant"
        elif distinct > ID_TEXT_SHARE * n_rows:
            return None, "id_like"
        elif distinct > FREE_TEXT_SHARE * len(present):
            return None, "free_text"
        else:
            levels = series.str.strip().astype(object).where(series.notna(), MISSING_LEVEL)
            return levels.to_numpy(dtype=object), None
    else:
        return None, "unsupported_type"
    # Infinite, or too large for the 32-bit floats the tree models read: missing.
    numbers = np.where(np.abs(numbers) > TOO_LARGE, np.nan, numbers)
    seen = numbers[usable]
    seen = seen[~np.isnan(seen)]
    if pt.is_integer_dtype(dtype):                      # distinct values, on the integers
        missing = series.isna().to_numpy().copy()
        kind = "uint64" if pt.is_unsigned_integer_dtype(dtype) else "int64"
        seen = series.to_numpy(dtype=kind, na_value=0)[usable & ~missing]
    distinct = len(np.unique(seen))
    if distinct == 0:
        return None, "all_missing"
    if distinct == 1:
        return None, "constant"
    if _whole_number_id(seen):
        return None, "id_like"
    return numbers, None


def _target_values(series):
    """The target as read: text without the white space round it; a text cell of
    only white space is a missing target."""
    cells = series.to_numpy(dtype=object)
    if not any(isinstance(v, str) for v in cells):
        return series
    read = [v.strip() if isinstance(v, str) else v for v in cells]
    return pd.Series([None if isinstance(v, str) and v == "" else v for v in read],
                     index=series.index, dtype=object, name=series.name)


def _read_target(series, name, task):
    """`(y, usable, classes)`: the target of every row as a float (the second class
    is 1.0, the first 0.0), which rows have a target, and the two classes."""
    if series.dtype == object:
        typed = _object_numbers(series)
        if typed is not None:
            series = typed
    if task == "regression":
        if not _is_numeric(series.dtype):
            raise ValueError(f"regression needs a numeric target; {name!r} is {series.dtype}")
        y = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        return y, np.isfinite(y), None
    values = _target_values(series)
    gone = values.isna().to_numpy().copy()
    if _is_numeric(values.dtype):
        gone |= ~np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))
    cells = values.to_numpy(dtype=object)
    distinct = list(pd.unique(pd.Series(cells[~gone], dtype=object)))
    try:
        classes = sorted(distinct)
    except TypeError:
        classes = sorted(distinct, key=str)
    if len(classes) != 2:
        raise ValueError(f"classification needs exactly two classes; target {name!r} has "
                         f"{len(classes)}")
    y = np.full(len(cells), np.nan)
    y[~gone] = [1.0 if v == classes[1] else 0.0 for v in cells[~gone]]
    return y, ~gone, classes


# ------------------------------------------------------------------ the split

def _approximate_mode(class_counts, n_draws, rng):
    """How many of `n_draws` rows each class gives: in proportion, the odd rows by lot."""
    continuous = class_counts / class_counts.sum() * n_draws
    floored = np.floor(continuous)
    need_to_add = int(n_draws - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        for value in np.sort(np.unique(remainder))[::-1]:
            (inds,) = np.where(remainder == value)
            add_now = min(len(inds), need_to_add)
            inds = rng.choice(inds, size=add_now, replace=False)
            floored[inds] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break
    return floored.astype(int)


def random_holdout(n, classes, seed):
    """A quarter of rows 0..n-1, rounded up, at random from `seed`: the holdout.

    `classes`: one value per row to keep in proportion (classification), or None.
    Written to give the rows that scikit-learn's `train_test_split(numpy.arange(n),
    test_size=ceil(n / 4), random_state=seed, shuffle=True, stratify=classes)`
    gives, which is how the tool splits (README.md says how far that was checked).
    """
    n_holdout = -(-n // 4)
    n_train = n - n_holdout
    rng = np.random.RandomState(seed)
    if classes is None:
        return np.sort(rng.permutation(n)[:n_holdout])
    _, y_indices = np.unique(classes, return_inverse=True)
    class_counts = np.bincount(y_indices)
    if class_counts.min() < 2 or n_holdout < len(class_counts) or n_train < len(class_counts):
        raise ValueError("too few rows of a class to split them between training and holdout")
    class_indices = np.split(np.argsort(y_indices, kind="mergesort"),
                             np.cumsum(class_counts)[:-1])
    n_i = _approximate_mode(class_counts, n_train, rng)
    t_i = _approximate_mode(class_counts - n_i, n_holdout, rng)
    holdout = []
    for i in range(len(class_counts)):
        permutation = rng.permutation(class_counts[i])
        of_class = class_indices[i].take(permutation, mode="clip")
        holdout.extend(of_class[n_i[i]:n_i[i] + t_i[i]])
    return np.sort(np.array(holdout, dtype="int64"))


# ------------------------------------------------------- the first two rules

def _codes(values):
    """One whole number per distinct value. Every not-a-number is one value, and
    0.0 and -0.0 are one value; text is compared as text."""
    cell = values
    if cell.dtype == object:
        cell = cell.astype(str)
    return np.unique(cell, return_inverse=True)[1].astype("int64").ravel()


def exact_copies(columns, y, train, holdout):
    """D49: which holdout rows have a copy among the training rows, the same value
    in every one of `columns` and the same `y`. One boolean per holdout row."""
    used = np.concatenate([train, holdout])
    code = np.zeros(len(used), dtype="int64")
    for values in list(columns) + [y]:
        part = _codes(values[used])
        # one number per distinct combination so far; renumbered, so it stays small
        code = np.unique(code * (int(part.max()) + 1) + part,
                         return_inverse=True)[1].astype("int64").ravel()
    return np.isin(code[len(train):], code[:len(train)])


def rare_value_pairs(columns, y, classification, used):
    """The first rule: the pairs of rows of `used` that are repeats of one another.

    Two rows are repeats when they agree in more than half of the columns
    compared, and at least `RARE_AGREEMENTS` of the values they share are rare:
    present (not an empty cell), and held by no more than `RARE_HOLDERS` of the
    rows of `used`. The columns compared are the feature columns and, for
    regression, the target; for classification the two rows must also have the
    same class. Returns two arrays of positions into `used`, `first < second`.
    """
    compared = list(columns) + ([] if classification else [y])
    n, codes, pairs = len(used), [], []
    for values in compared:
        cell = values[used]
        present = ~np.isnan(cell) if cell.dtype != object else cell != MISSING_LEVEL
        if cell.dtype == object:
            cell = cell.astype(str)
        _, part, holders = np.unique(cell, return_inverse=True, return_counts=True)
        part = part.astype("int64").ravel()
        codes.append(part)
        rare = np.flatnonzero(present & (holders[part] >= 2) & (holders[part] <= RARE_HOLDERS))
        if len(rare) < 2:
            continue
        rare = rare[np.argsort(part[rare], kind="stable")]      # rows of one value together
        value = part[rare]
        starts = np.flatnonzero(np.r_[True, value[1:] != value[:-1]])
        sizes = np.diff(np.r_[starts, len(value)])
        for size in range(2, RARE_HOLDERS + 1):
            first = starts[sizes == size]
            if not len(first):
                continue
            rows = rare[first[:, None] + np.arange(size)[None, :]]
            for a in range(size):
                for b in range(a + 1, size):
                    pairs.append(rows[:, a].astype("int64") * n + rows[:, b])
    empty = np.array([], dtype="int64")
    if not pairs:
        return empty, empty
    pair, shared = np.unique(np.concatenate(pairs), return_counts=True)
    pair = pair[shared >= RARE_AGREEMENTS]
    first, second = pair // n, pair % n
    if classification:
        same = y[used][first] == y[used][second]
        first, second = first[same], second[same]
    agree = np.zeros(len(first), dtype="int64")
    for part in codes:
        agree += part[first] == part[second]
    most = 2 * agree > len(compared)
    return first[most], second[most]


def first_rule(columns, y, classification, train, holdout):
    """Which holdout rows repeat a training row by the first rule: a copy, or a
    row that `rare_value_pairs` pairs with a training row."""
    used = np.concatenate([train, holdout])
    repeat = exact_copies(columns, y, train, holdout).copy()
    first, second = rare_value_pairs(columns, y, classification, used)
    across = (first < len(train)) & (second >= len(train))      # used: training rows first
    repeat[second[across] - len(train)] = True
    return repeat


# ------------------------------------------------------- the calibrated rule

def facts_of(codes, target):
    """The columns that stand for a fact: every column but those another column settles.

    One column settles another when each of its values goes with a single value
    of the other: `pclass` settles `class`, a price settles the same price in
    another currency, a postcode settles its latitude. Agreeing on the first is
    then agreeing on both, and counts once. That can only be learned where
    values repeat and the rows that share them differ elsewhere: when every row
    is in the table twice, each column "settles" all the others. So a column
    counts as settling only when at least `FLOOR` of its rows follow a row of the
    same value, and it settles fewer than half of the other columns. Of two
    columns that settle each other the earlier is kept. The target always stands.
    """
    p = len(codes)
    table = np.stack(codes, axis=1).astype(np.int32)
    settles = np.zeros((p, p), dtype=bool)
    for x in range(p):
        order = np.argsort(codes[x], kind="stable")
        same = codes[x][order][1:] == codes[x][order][:-1]
        if same.sum() < FLOOR:
            continue                            # too few repeated values to learn from
        rows = table[order]
        settles[x] = ~((rows[1:] != rows[:-1]) & same[:, None]).any(axis=0)
        settles[x, x] = False
    settles &= (2 * settles.sum(axis=1) < p - 1)[:, None]
    keep = []
    for y in range(p):
        by = np.flatnonzero(settles[:, y])
        if y != target and any(not settles[y, x] or x < y for x in by):
            continue
        keep.append(y)
    return keep


def _cross_pairs(key, m, queries):
    """`training row * n + holdout row` for every training row and every row of
    `queries` (a mask over the rows; holdout rows only) that share a key."""
    n = len(key)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    group = np.cumsum(np.r_[True, sorted_key[1:] != sorted_key[:-1]]) - 1
    is_train = order < m
    is_query = queries[order]
    n_train = np.bincount(group, weights=is_train).astype("int64")
    n_query = np.bincount(group, weights=is_query).astype("int64")
    if not (n_train * n_query).any():
        return np.array([], dtype="int64")
    train_rows = order[is_train]                        # grouped, since `order` is
    train_start = np.r_[0, np.cumsum(n_train)[:-1]]
    query_rows, query_group = order[is_query], group[is_query]
    per_query = n_train[query_group]
    total = int(per_query.sum())
    if total > 60_000_000:
        raise MemoryError(f"the search for near-copies met {total} pairs of rows that agree in "
                          "the same columns: too many to list. See README.md, 'When it gives up'")
    offset = np.arange(total, dtype="int64") \
        - np.repeat(np.r_[0, np.cumsum(per_query)[:-1]], per_query)
    t = train_rows[np.repeat(train_start[query_group], per_query) + offset]
    return t.astype("int64") * n + np.repeat(query_rows, per_query)


def calibrated(columns, y, classification, train, holdout):
    """The calibrated rule: holdout rows that agree with a training row by more
    than chance explains in this table.

    1. Facts. The columns compared are the feature columns and the target. A
       column that another column settles is not a fact of its own (`facts_of`).

    2. The chance that one independent row agrees with holdout row h in a fact:
       the share of the other rows that hold h's value. For a value held by
       `RARE` rows or fewer, h among them, Good and Turing's estimate from the
       count of counts instead, (c + 1) x (N[c+1] + 1) / (n x N[c]), where N[c]
       is the number of values held by exactly c rows. A value's own count says
       little when it is small: where every row is in the table twice, each
       value is "held by one other row" because of the copy itself.

    3. The chance that it agrees with h in every fact. Facts are taken from the
       coarsest (fewest distinct values) to the finest. While at least `FLOOR`
       other rows share h's values in all the facts taken so far, the share of
       rows that do is counted, jointly, so that dependent coarse facts are
       never multiplied. Past that point each further fact multiplies in its
       chance from step 2, or its share inside the last counted cell if larger.

    4. Near-copies. U(h) is the largest number of facts (never the class) in
       which a training row may differ from h while a match stays improbable:
       fewer than `GATE` training rows are expected to agree with h, by chance,
       in all facts but at most U of them. U(h) is at most `MAX_DIFFER`. A row
       for which even an exact copy is not that improbable is still judged, on
       an exact copy alone, when fewer than `WIDE` are expected.

    5. The count. At each level d, from 0 to `MAX_DIFFER`, h is counted when
       some training row has its class and differs from it in at most d facts,
       and in at most U(h). The rows counted at a level, less the number chance
       gives at that level and `MARGIN` spreads of it, to the nearest row, are
       the repeats at that level; the count is the largest of them. The rows
       judged on an exact copy alone are counted apart, in the same way, and
       added.

    `columns` and `y` hold one value per row; `train` and `holdout` are positions
    into them. Returns the count and what it was made from; `facts` and `settled`
    are positions into `columns`, with `len(columns)` standing for the target.
    """
    used = np.concatenate([train, holdout])
    n, m, n_hold = len(used), len(train), len(holdout)
    codes = [_codes(values[used]) for values in columns] + [_codes(y[used])]
    target = len(codes) - 1
    keep = facts_of(codes, target)
    settled = [i for i in range(len(codes)) if i not in keep]
    n_levels = {i: int(codes[i].max()) + 1 for i in keep}
    keep.sort(key=lambda i: (classification and i != target, n_levels[i], i))
    facts = [codes[i] for i in keep]            # coarse to fine; the class first
    k = len(facts)
    class_at = keep.index(target) if classification else None
    hold = np.arange(m, n)

    # 2. the chance of agreeing with h in each fact
    share = np.zeros((n_hold, k))
    held_by = np.zeros((n_hold, k), dtype="int64")
    for j, f in enumerate(facts):
        holders = np.bincount(f)
        c = holders[f[hold]]
        held_by[:, j] = c
        share[:, j] = (c - 1) / (n - 1)
        few = c <= RARE
        if few.any():
            n_with = np.bincount(holders, minlength=RARE + 2)
            share[few, j] = np.minimum(1.0, (c[few] + 1) * (n_with[c[few] + 1] + 1)
                                       / (n * n_with[c[few]]))

    # 3. the counted cell, and the chance of agreeing with h in every fact
    depth = np.zeros(n, dtype="int64")
    in_cell = np.full(n, n - 1, dtype="int64")
    level_keys = [np.zeros(n, dtype="int64")]
    key, alive = np.zeros(n, dtype="int64"), np.ones(n, dtype=bool)
    for j, f in enumerate(facts):
        key = np.unique(key * (int(f.max()) + 1) + f,
                        return_inverse=True)[1].astype("int64").ravel()
        follow = np.bincount(key)[key] - 1
        alive &= follow >= FLOOR
        if not alive.any():
            break
        depth[alive], in_cell[alive] = j + 1, follow[alive]
        level_keys.append(key)
    for level in np.unique(depth[hold]):        # inside the last counted cell, if larger
        mask = depth[hold] == level
        key = level_keys[level]
        for j in range(level, k):
            common = mask & (held_by[:, j] > RARE)
            if not common.any():
                continue
            both = np.unique(key * (int(facts[j].max()) + 1) + facts[j],
                             return_inverse=True)[1].ravel()
            inside = (np.bincount(both)[both[hold[common]]] - 1) / in_cell[hold[common]]
            share[common, j] = np.maximum(share[common, j], inside)
    past = np.arange(k)[None, :] >= depth[hold][:, None]
    log_pi = np.log(in_cell[hold] / (n - 1)) + np.where(past, np.log(share), 0.0).sum(axis=1)

    # 4. how many facts a near-copy of h may differ in
    x = 1.0 / share
    if classification:
        x[:, class_at] = 0.0                    # the class is never one of them
    e = np.zeros((n_hold, MAX_DIFFER + 1))
    e[:, 0] = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        for j in range(k):                      # sums over which facts are left out
            for d in range(MAX_DIFFER, 0, -1):
                e[:, d] += e[:, d - 1] * x[:, j]
        chance = m * np.exp(log_pi)[:, None] * np.cumsum(e, axis=1)
    chance = np.where(np.isfinite(chance), chance, np.inf)
    near = (chance < GATE).sum(axis=1) - 1
    allowed = np.where(near >= 0, near, np.where(chance[:, 0] < WIDE, 0, -1))

    # The pairs: a training row and a holdout row that differ in at most U facts.
    # Found exactly: two rows that differ in d facts agree in every band of facts
    # but at most d, so they share the key that leaves those bands out. The bands
    # take the facts in turn, so that each has fine facts as well as coarse ones.
    rng = np.random.default_rng(12345)
    n_bands = min(k, 10)
    bands = [list(range(b, k, n_bands)) for b in range(n_bands)]
    band_sum = []
    for band in bands:
        total = np.zeros(n, dtype="uint64")
        for j in band:
            total += rng.integers(1, 2 ** 62, size=int(facts[j].max()) + 1,
                                  dtype="uint64")[facts[j]]
        band_sum.append(total)
    whole = np.zeros(n, dtype="uint64")
    for total in band_sum:
        whole += total
    found = []
    for left_out in range(0, min(MAX_DIFFER, n_bands - 1) + 1):
        queries = np.zeros(n, dtype=bool)
        queries[hold[allowed >= left_out]] = True
        if not queries.any():
            break
        for dropped in itertools.combinations(range(n_bands), left_out):
            key = whole.copy()
            for b in dropped:
                key -= band_sum[b]
            found.append(_cross_pairs(key.view("int64"), m, queries))
    best = np.full(n_hold, k + 1, dtype="int64")    # fewest facts to a training row of its class
    if found:
        pair = np.unique(np.concatenate(found))
        t, h = pair // n, pair % n
        differ = np.zeros(len(pair), dtype="int64")
        for f in facts:
            differ += f[t] != f[h]
        if classification:
            differ[facts[class_at][t] != facts[class_at][h]] = k + 1
        order = np.argsort(-differ, kind="stable")
        best[(h - m)[order]] = differ[order]        # the last one written is the smallest

    # 5. what is there beyond what chance would give, level by level
    at = np.arange(n_hold)

    def beyond(rows, level):
        d = np.minimum(np.maximum(allowed, 0), level)
        by_chance = 1.0 - np.exp(-chance[at, d])
        expected = float(by_chance[rows].sum())
        spread = float(np.sqrt((by_chance[rows] * (1.0 - by_chance[rows])).sum()))
        counted = int(((best <= d) & rows).sum())
        return {"level": level, "rows": int(rows.sum()), "counted": counted,
                "chance": expected, "spread": spread,
                "beyond": max(0, counted - int(round(expected + MARGIN * spread)))}

    is_near, is_exact = near >= 0, (near < 0) & (allowed >= 0)
    levels = [beyond(is_near, level) for level in range(MAX_DIFFER + 1)]
    chosen = max(levels, key=lambda r: (r["beyond"], -r["level"]))
    widest = levels[-1]
    exact_only = beyond(is_exact, 0)
    del exact_only["level"]
    return {"count": chosen["beyond"] + exact_only["beyond"],
            "counted": widest["counted"] + exact_only["counted"],
            "chance": widest["chance"] + exact_only["chance"],
            "spread": float(np.hypot(widest["spread"], exact_only["spread"])),
            "level": chosen["level"], "levels": levels, "exact_only": exact_only,
            "facts": keep, "settled": settled}


# ------------------------------------------------------------ the entry point

def _verdict(count, n_holdout):
    count = int(count)
    share = count / n_holdout
    return {"count": count, "share": share, "stop": count >= STOP_ROWS and share > STOP_SHARE}


def count_repeats(frame, target, task, seed=None, holdout=None, exclude=()):
    """The three rules on one table.

    frame     a pandas DataFrame, one row per observation.
    target    the name of the target column.
    task      "classification" (exactly two classes) or "regression".
    seed      a whole number: the holdout is a random quarter of the rows that
              have a target, rounded up, in proportion to the classes; or
    holdout   the positions (0 for the first row of `frame`) of the holdout rows.
              Every other row that has a target is a training row.
    exclude   names of columns to leave out of the comparison, beyond those left
              out by the way a column is read (README.md).

    Returns a dictionary:
      "d49", "first_rule", "calibrated"   each with "count" (holdout rows the
          rule counts), "share" (of the holdout rows) and "stop" (more than 1%
          of the holdout rows and at least 10). "calibrated" also has "counted"
          (holdout rows with a training row as near as step 4 allows them),
          "chance" (how many of those chance alone gives), "spread" (the spread
          of that number), "level", "levels", "exact_only", "facts", "settled".
      "rows"      given, without a target, training, holdout.
      "holdout"   the positions of the holdout rows, as an array.
      "columns"   the names compared, and the names left out with the reason.
      "classes"   the two classes in order (the second is the positive one), or None.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'")
    if (seed is None) == (holdout is None):
        raise ValueError("pass a seed or the holdout row positions: one of the two")
    if not frame.columns.is_unique:
        raise ValueError("the column names must be distinct")
    if target not in frame.columns:
        raise ValueError(f"there is no column {target!r}")
    classification = task == "classification"
    frame = frame.reset_index(drop=True)
    y, usable, classes = _read_target(frame[target], target, task)
    rows = np.flatnonzero(usable)
    if holdout is None:
        picked = random_holdout(len(rows), y[rows] if classification else None, seed)
        holdout_rows = rows[picked]
    else:
        given = np.unique(np.asarray(holdout, dtype="int64"))
        if len(given) and (given[0] < 0 or given[-1] >= len(frame)):
            raise ValueError(f"holdout positions must lie in 0..{len(frame) - 1}")
        holdout_rows = given[usable[given]]
    train_rows = np.setdiff1d(rows, holdout_rows)
    if not len(train_rows) or not len(holdout_rows):
        raise ValueError("the split leaves no training row or no holdout row")

    compared, left_out = {}, {}
    for name in frame.columns:
        if name == target:
            continue
        if name in exclude:
            left_out[name] = "excluded by the caller"
            continue
        values, reason = _read_column(frame[name], usable, len(rows))
        if values is None:
            left_out[name] = reason
        else:
            compared[name] = values
    if not compared:
        raise ValueError("no column is left to compare: " + "; ".join(
            f"{name} ({reason})" for name, reason in left_out.items()))

    columns, names = list(compared.values()), list(compared) + [target]
    n_holdout = len(holdout_rows)
    copies = exact_copies(columns, y, train_rows, holdout_rows)
    repeats = first_rule(columns, y, classification, train_rows, holdout_rows)
    found = calibrated(columns, y, classification, train_rows, holdout_rows)
    found["facts"] = [names[i] for i in found["facts"]]
    found["settled"] = [names[i] for i in found["settled"]]
    return {
        "d49": _verdict(copies.sum(), n_holdout),
        "first_rule": _verdict(repeats.sum(), n_holdout),
        "calibrated": {**_verdict(found["count"], n_holdout),
                       **{key: value for key, value in found.items() if key != "count"}},
        "rows": {"given": len(frame), "no_target": int(len(frame) - len(rows)),
                 "training": len(train_rows), "holdout": n_holdout},
        "holdout": holdout_rows,
        "columns": {"compared": list(compared), "left_out": left_out},
        "classes": classes,
    }


# ----------------------------------------------------------------- an example

def example_table(n=6000, seed=0):
    """Independent rows of the kind of a customer file: no row repeats another
    but by chance, and in a table of this size many do."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "plan": rng.choice(np.array(["basic", "standard", "premium"], dtype=object), n,
                           p=[0.5, 0.3, 0.2]),
        "region": rng.choice(np.array(["north", "south", "east", "west"], dtype=object), n),
        "monthly_spend": np.round(np.maximum(5.0, rng.normal(52, 14, n))),
        "support_calls": np.minimum(9, np.floor(rng.exponential(1.8, n))).astype(int),
        "age": rng.integers(18, 80, n),
        "churned": np.where(rng.random(n) < 0.3, "yes", "no"),
    })


def with_copies(frame, share, seed=0):
    """`frame` with `share` of its rows written a second time, in shuffled order,
    and a column of unique values added, which hides the copies from `d49`."""
    rng = np.random.default_rng(seed)
    again = rng.choice(len(frame), round(share * len(frame)), replace=False)
    both = pd.concat([frame, frame.iloc[np.sort(again)]], ignore_index=True)
    both = both.iloc[rng.permutation(len(both))].reset_index(drop=True)
    both.insert(0, "reading", rng.normal(size=len(both)).round(6))
    return both


def _show(label, found):
    cal = found["calibrated"]
    print(f"{label}: {found['rows']['holdout']} holdout rows")
    for rule in ("d49", "first_rule", "calibrated"):
        r = found[rule]
        more = f"   ({cal['counted']} counted, chance gives {cal['chance']:.1f})" \
            if rule == "calibrated" else ""
        print(f"    {rule:<11} count {r['count']:>4}   share {r['share']:.3f}   "
              f"{'stop' if r['stop'] else 'run '}{more}")


if __name__ == "__main__":
    clean = example_table()
    _show("independent rows", count_repeats(clean, "churned", "classification", seed=0))
    _show("5% of the rows twice, a column of unique values added",
          count_repeats(with_copies(clean, 0.05), "churned", "classification", seed=0))
    _show("every row twice, a column of unique values added",
          count_repeats(with_copies(clean, 1.0), "churned", "classification", seed=0))
