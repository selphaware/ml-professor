"""Step 5: the column roles that need no target (spec 3 step 5, 4.3, 4.5, D19, D28, D34).

Roles are judged on the usable rows; the counts `n_missing`, `n_coerced` and
`n_infinite` are kept over every loaded row; `n_infinite` of a numeric column
also counts the values too large for the tree models (D48). Whole numbers
are judged by value, never by type, and those of an integer-typed column are
compared as integers (D50). A number written with thousands separators is
never parsed, and dates are recognised by a fixed list of forms, never by
pandas' parser.
"""
import datetime
import re

import numpy as np
import pandas as pd

from mlprofessor import errors
from mlprofessor.load import _is_empty_cell, missing_mask, read_number_columns, usable_rows

COLUMN_KEYS = [
    "name", "position", "dtype", "role", "kind", "reason", "leak_rule",
    "n_missing", "n_coerced", "n_infinite", "n_imputed",
]

DATE_SHARE = 0.95          # rule 4: share of values in a date form
NUMERIC_SHARE = 0.90       # rules 4a and 5: share of values that parse (or match)
ID_TEXT_SHARE = 0.95       # rule 6a: distinct values over usable rows
FREE_TEXT_SHARE = 0.50     # rule 6b: distinct values over non-missing values

# The largest 32-bit float (D48): the tree models read features as 32-bit
# floats, so a feature value of greater magnitude is read as an infinite one.
FEATURE_LIMIT = float(np.finfo(np.float32).max)

SEPARATED_NUMBER = re.compile(r"[+-]?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?")

# ------------------------------------------------------------ date forms (4.5)

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_Y = r"(?P<y>[0-9]{4})"
_A = r"(?P<a>[0-9]{1,2})"
_B = r"(?P<b>[0-9]{1,2})"
_MON = r"(?P<mon>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")"
_TIME = (r"(?:[T ](?P<hh>[0-9]{1,2}):(?P<mm>[0-9]{2})"
         r"(?::(?P<ss>[0-9]{2})(?:\.[0-9]+)?)?"
         r"(?:Z|[+-](?P<oh>[0-9]{2}):?(?P<om>[0-9]{2}))?)?")

# Each form: (pattern, how the numeric groups a and b may be read as (month, day)).
_YMD, _DMY_OR_MDY, _DMY, _MDY = "ymd", "either", "dmy", "mdy"
_DATE_FORMS = [
    (_Y + "-" + _A + "-" + _B, _YMD),
    (_Y + "/" + _A + "/" + _B, _YMD),
    (_A + "/" + _B + "/" + _Y, _DMY_OR_MDY),
    (_A + "-" + _B + "-" + _Y, _DMY_OR_MDY),
    (_A + r"\." + _B + r"\." + _Y, _DMY),
    (_A + "-" + _MON + "-" + _Y, _DMY),
    (_A + " " + _MON + " " + _Y, _DMY),
    (_MON + " " + _A + ",? " + _Y, _MDY),
]
_DATE_PATTERNS = [(re.compile(form + _TIME, re.IGNORECASE), order)
                  for form, order in _DATE_FORMS]


def _valid_date(year, month, day):
    try:
        datetime.date(year, month, day)
    except ValueError:
        return False
    return True


def _valid_time(match):
    if match.group("hh") is None:
        return True
    if int(match.group("hh")) > 23 or int(match.group("mm")) > 59:
        return False
    if match.group("ss") is not None and int(match.group("ss")) > 60:
        return False
    if match.group("oh") is not None:
        if int(match.group("oh")) > 23 or int(match.group("om")) > 59:
            return False
    return True


def is_date_like(text):
    """True when the stripped text is wholly one of the date forms of 4.5."""
    for pattern, order in _DATE_PATTERNS:
        match = pattern.fullmatch(text)
        if match is None or not _valid_time(match):
            continue
        groups = match.groupdict()
        year, a = int(groups["y"]), int(groups["a"])
        if groups.get("mon") is not None:
            if _valid_date(year, _MONTHS[groups["mon"].lower()], a):
                return True
            continue
        b = int(groups["b"])
        if order == _YMD and _valid_date(year, a, b):
            return True
        if order in (_DMY, _DMY_OR_MDY) and _valid_date(year, b, a):
            return True
        if order == _DMY_OR_MDY and _valid_date(year, a, b):
            return True
    return False


# ------------------------------------------------------------ reading types

def type_family(series):
    """`numeric`, `datetime`, `text` or `other`, from the type and the values."""
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if isinstance(dtype, pd.CategoricalDtype) or isinstance(dtype, pd.StringDtype):
        return "text"
    if pd.api.types.is_complex_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
        return "other"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if dtype == object:
        present = series[series.notna()]
        if all(isinstance(value, str) for value in present.to_numpy()):
            return "text"
    return "other"


def stripped_text(series):
    """The column as stripped text (object Series), with None where missing."""
    present = series.notna().to_numpy(dtype=bool)
    values = series.astype(object).to_numpy()
    out = np.empty(len(series), dtype=object)
    out[:] = None
    out[present] = [str(value).strip() for value in values[present]]
    return pd.Series(out, index=series.index, dtype=object)


def parse_numbers(text):
    """`pandas.to_numeric(..., errors="coerce")` of stripped text, as floats.

    A number with thousands separators is never parsed: `to_numeric` is given
    no option that would accept it.
    """
    parsed = pd.to_numeric(text, errors="coerce")
    return pd.Series(parsed, index=text.index).astype(float).to_numpy()


def numeric_values(series):
    """A numeric-typed column as floats, NaN where missing (infinity kept)."""
    return series.to_numpy(dtype=float, na_value=np.nan)


def beyond_limit(values):
    """True where a number is infinite or of magnitude above `FEATURE_LIMIT` (D48)."""
    values = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.abs(values) > FEATURE_LIMIT


def feature_numbers(values):
    """Floats with NaN for every missing, infinite or too-large cell (step 5, D48)."""
    values = np.array(values, dtype=float, copy=True)
    values[beyond_limit(values)] = np.nan
    return values


def _whole(values):
    return bool(np.all(np.isfinite(values)) and np.all(np.floor(values) == values))


# ------------------------------------------------------------ the rules

def _dense(n_values, n_distinct, smallest, largest):
    """The clause of D74: more than 95% of the values distinct, and the distinct
    values filling at least half of their range, counted in Python integers."""
    span = int(largest) - int(smallest) + 1
    return 100 * n_distinct > 95 * n_values and 2 * n_distinct >= span


def _numeric_rule(values):
    """Rules 8 to 8c on the usable rows' values; returns (role, kind, reason)."""
    values = values[np.isfinite(values) & ~beyond_limit(values)]
    if values.size == 0:
        return "excluded", "numeric", "all_missing"
    distinct = np.unique(values)
    if distinct.size == 1:
        return "excluded", "numeric", "constant"
    if _whole(values):
        smallest, largest = int(distinct[0]), int(distinct[-1])
        if distinct.size == values.size:
            steps = np.diff(values)
            consecutive = largest - smallest + 1 == values.size
            ordered = bool(np.all(steps > 0) or np.all(steps < 0))
            if consecutive or ordered:
                return "excluded", "numeric", "id_like"
        if _dense(values.size, distinct.size, smallest, largest):
            return "excluded", "numeric", "id_like"
    return "feature", "numeric", None


def _integer_rule(series):
    """Rules 8 to 8c for an integer-typed column, judged on the integers themselves.

    As floats, distinct whole numbers above 2**53 can become one; the steps
    between neighbours are compared, never subtracted, and the range is taken
    in Python integers, so that nothing merges or overflows.
    """
    present = series[series.notna()]
    if len(present) == 0:
        return "excluded", "numeric", "all_missing"
    unsigned = pd.api.types.is_unsigned_integer_dtype(series.dtype)
    values = present.to_numpy(dtype=np.uint64 if unsigned else np.int64)
    distinct = np.unique(values)
    if distinct.size == 1:
        return "excluded", "numeric", "constant"
    smallest, largest = int(distinct[0]), int(distinct[-1])
    if distinct.size == values.size:
        rising = bool(np.all(values[1:] > values[:-1]))
        falling = bool(np.all(values[1:] < values[:-1]))
        consecutive = largest - smallest + 1 == values.size
        if consecutive or rising or falling:
            return "excluded", "numeric", "id_like"
    if _dense(values.size, distinct.size, smallest, largest):
        return "excluded", "numeric", "id_like"
    return "feature", "numeric", None


def _text_rule(text_all, usable, n_usable):
    """Rules 4 to 6c on the usable rows. Returns (role, kind, reason, numeric)."""
    text = text_all.iloc[usable]
    present = text[text.notna()]
    n_present = len(present)
    distinct = pd.unique(present.to_numpy())

    date_by_value = {value: is_date_like(value) for value in distinct}
    n_dates = int(present.map(date_by_value).sum())
    if n_dates >= DATE_SHARE * n_present:
        return "excluded", "categorical", "datetime", False

    parses = ~np.isnan(parse_numbers(present))
    separated = present.str.fullmatch(SEPARATED_NUMBER.pattern).to_numpy(dtype=bool)
    if (len(distinct) > 1 and separated.any()
            and int((parses | separated).sum()) >= NUMERIC_SHARE * n_present):
        return "excluded", "categorical", "numeric_with_separators", False
    if int(parses.sum()) >= NUMERIC_SHARE * n_present:
        return None, "numeric", None, True

    if len(distinct) == 1:
        return "excluded", "categorical", "constant", False
    if len(distinct) > ID_TEXT_SHARE * n_usable:
        return "excluded", "categorical", "id_like", False
    if len(distinct) > FREE_TEXT_SHARE * n_present:
        return "excluded", "categorical", "free_text", False
    return "feature", "categorical", None, False


# ------------------------------------------------------------ rule 2a: dates as YYYYMMDD

FIRST_DATE_NUMBER, LAST_DATE_NUMBER = 19000101, 21001231
EIGHT_DIGITS = re.compile(r"[0-9]{8}")
_DAYS_IN_MONTH = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int64)


def is_date_number(number):
    """True when the whole number reads as a calendar date YYYYMMDD, 1900 to 2100 (D66)."""
    number = int(number)
    if not FIRST_DATE_NUMBER <= number <= LAST_DATE_NUMBER:
        return False
    return _valid_date(number // 10000, number // 100 % 100, number % 100)


def _date_number_mask(numbers):
    """For int64 whole numbers: True where one reads as a date YYYYMMDD, 1900 to 2100."""
    numbers = np.asarray(numbers, dtype=np.int64)
    inside = (numbers >= FIRST_DATE_NUMBER) & (numbers <= LAST_DATE_NUMBER)
    year, month, day = numbers // 10000, numbers // 100 % 100, numbers % 100
    month_ok = (month >= 1) & (month <= 12)
    length = _DAYS_IN_MONTH[np.where(month_ok, month, 0)]
    leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    length = length + ((month == 2) & leap).astype(np.int64)
    return inside & month_ok & (day >= 1) & (day <= length)


def _written_number(value):
    """A number as it is written: a whole number in digits, anything else as Python writes it."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    if np.isfinite(value) and value == np.floor(value):
        return str(int(value))
    return repr(value)


def _distinct_values_as_dates(series):
    """Rule 2a's reading of the non-missing values of `series`.

    Returns `(written, is_date, n_cells)`, one entry for each distinct value: how
    it is written, whether it reads as a date YYYYMMDD, and how many cells hold
    it. None for a column that is neither numeric (never boolean) nor text.
    """
    family = type_family(series)
    if family == "numeric":
        if pd.api.types.is_bool_dtype(series.dtype):
            return None
        present = series[series.notna()]
        if pd.api.types.is_integer_dtype(series.dtype):
            unsigned = pd.api.types.is_unsigned_integer_dtype(series.dtype)
            values = present.to_numpy(dtype=np.uint64 if unsigned else np.int64)
            distinct, n_cells = np.unique(values, return_counts=True)
            candidates = (distinct >= FIRST_DATE_NUMBER) & (distinct <= LAST_DATE_NUMBER)
        else:
            values = present.to_numpy(dtype=float, na_value=np.nan)
            values = values[~np.isnan(values)] + 0.0
            distinct, n_cells = np.unique(values, return_counts=True)
            # A fraction and an infinite cell are values, and no dates.
            with np.errstate(invalid="ignore"):
                candidates = (np.isfinite(distinct) & (np.floor(distinct) == distinct)
                              & (distinct >= FIRST_DATE_NUMBER)
                              & (distinct <= LAST_DATE_NUMBER))
        is_date = np.zeros(len(distinct), dtype=bool)
        is_date[candidates] = _date_number_mask(distinct[candidates].astype(np.int64))
        return [_written_number(v) for v in distinct.tolist()], is_date, n_cells
    if family == "text":
        text = stripped_text(series)
        counted = text[text.notna()].value_counts(sort=False)
        distinct = pd.Series(counted.index.astype(str), dtype=object)
        eight = distinct.str.fullmatch(EIGHT_DIGITS.pattern).to_numpy(dtype=bool)
        is_date = np.zeros(len(distinct), dtype=bool)
        if eight.any():
            numbers = np.array([int(v) for v in distinct[eight]], dtype=np.int64)
            is_date[eight] = _date_number_mask(numbers)
        return ([repr(v) for v in distinct.tolist()], is_date,
                counted.to_numpy(dtype=np.int64))
    return None


def _many_dates(n_dates, n_others):
    """Rule 2a's counts (D69, D72): at least two distinct dates, and at least nine for
    each distinct value that is no date, or at least five beside at most two."""
    return n_dates >= 2 and (n_dates >= 9 * n_others or (n_others <= 2 and n_dates >= 5))


def values_that_are_no_dates(series):
    """For a column of dates written YYYYMMDD by the counts of rule 2a, judged on every
    non-missing value of `series`: its values that are no dates, as `(written,
    n_cells)`, most cells first ([] for a clean column of dates). None when the
    column is no column of dates (step 3; D69)."""
    found = _distinct_values_as_dates(series)
    if found is None:
        return None
    written, is_date, n_cells = found
    n_dates = int(is_date.sum())
    if not _many_dates(n_dates, len(is_date) - n_dates):
        return None
    others = [(written[k], int(n_cells[k])) for k in np.flatnonzero(~is_date)]
    return sorted(others, key=lambda pair: -pair[1])


def _dates_as_numbers(series, usable):
    """Rule 2a of step 5 on the usable rows: dates written as whole numbers YYYYMMDD."""
    return values_that_are_no_dates(series.iloc[usable]) is not None


def _judge_column(series, usable):
    """One column's (role, kind, reason, n_coerced, extra infinities)."""
    usable_missing = series.iloc[usable].isna().to_numpy(dtype=bool)
    if usable_missing.all():
        return "excluded", None, "all_missing", 0, 0

    family = type_family(series)
    if family == "datetime":
        return "excluded", None, "datetime", 0, 0
    if _dates_as_numbers(series, usable):
        kind = "numeric" if family == "numeric" else "categorical"
        return "excluded", kind, "datetime", 0, 0
    if family == "other":
        return "excluded", None, "unsupported_type", 0, 0
    if family == "numeric":
        values = numeric_values(series)
        n_too_large = int((np.isfinite(values) & beyond_limit(values)).sum())
        if pd.api.types.is_integer_dtype(series.dtype):
            role, kind, reason = _integer_rule(series.iloc[usable])
        else:
            role, kind, reason = _numeric_rule(values[usable])
        return role, kind, reason, 0, n_too_large

    text_all = stripped_text(series)
    role, kind, reason, read_as_numeric = _text_rule(text_all, usable, len(usable))
    if not read_as_numeric:
        return role, kind, reason, 0, 0
    parsed = parse_numbers(text_all)
    n_coerced = int((text_all.notna().to_numpy(dtype=bool) & np.isnan(parsed)).sum())
    n_parsed_infinite = int(beyond_limit(parsed).sum())
    role, kind, reason = _numeric_rule(parsed[usable])
    return role, kind, reason, n_coerced, n_parsed_infinite


def column_roles(frame, target, time_col=None, group_col=None):
    """Step 5 alone: one record per column of `frame`, in order (spec 2.7).

    Object columns of numbers are read as numbers first (step 1a); `dtype` is
    the type each column came with.
    """
    if isinstance(group_col, str) and group_col not in frame.columns:
        raise errors.ColumnNotFoundError(
            f"No column named {group_col!r}. The columns are: "
            f"{', '.join(repr(str(c)) for c in frame.columns)}.")
    loaded_dtypes = {name: str(frame[name].dtype) for name in frame.columns}
    return roles_of_read_columns(read_number_columns(frame), target, time_col, loaded_dtypes,
                                 group_col)


# ------------------------------------------------------------ the entity column (D57)

def _entity_cell_key(value):
    """The key one object cell of an entity column is compared by; None for an empty cell.

    Text without the white space round it; whole numbers exactly; 0.0 and -0.0
    one value; anything else as the value it is. Raises `TypeError` for a value
    that cannot be compared for equality (a list, a dictionary, a set).
    """
    if _is_empty_cell(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) + 0.0
    if isinstance(value, (complex, np.complexfloating)):
        return complex(value) + 0j
    hash(value)
    return value


def read_entities(values, name=None):
    """Step 5, "How an entity column is read": one code per cell, -1 for an empty cell.

    Cells that are equal share a code; codes number the entities in the order
    they first appear. Raises `GroupColumnError` when a value cannot be compared.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    series = series.reset_index(drop=True)
    label = "the entity column" if name is None else f"the entity column {name!r}"
    dtype = series.dtype
    if isinstance(dtype, pd.SparseDtype):
        series = series.sparse.to_dense()
        dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        present = series.notna().to_numpy(dtype=bool)
        cells = series.astype(object).to_numpy()
        series = pd.Series([str(v).strip() if p else None for v, p in zip(cells, present)],
                           dtype=object)
        dtype = series.dtype
    if dtype == object:
        keys, empty = [], np.zeros(len(series), dtype=bool)
        for position, value in enumerate(series.to_numpy()):
            try:
                key = _entity_cell_key(value)
            except TypeError as exc:
                raise errors.GroupColumnError(
                    f"{label} holds a value that cannot be compared for equality "
                    f"(a {type(value).__name__}); an entity column holds values that can "
                    "be told equal or not, such as names or numbers.") from exc
            if key is None:
                empty[position] = True
            keys.append(key)
        codes = np.full(len(series), -1, dtype=np.int64)
        seen = {}
        for position, key in enumerate(keys):
            if not empty[position]:
                codes[position] = seen.setdefault(key, len(seen))
        return codes
    if pd.api.types.is_string_dtype(dtype):
        series = series.str.strip()
        series = series.where(series.str.len() > 0)
    elif pd.api.types.is_complex_dtype(dtype):
        series = series + 0j
    elif pd.api.types.is_float_dtype(dtype):
        series = series + 0.0
    codes, _ = pd.factorize(series, use_na_sentinel=True)
    return np.asarray(codes, dtype=np.int64)


def roles_of_read_columns(frame, target, time_col, loaded_dtypes, group_col=None):
    """Step 5 on a table already read by step 1a; `loaded_dtypes` names each column's type as loaded."""
    group_col = group_col if isinstance(group_col, str) else None
    usable = usable_rows(frame, target, time_col)["usable"]
    records = []
    for position, name in enumerate(frame.columns):
        series = frame[name]
        n_missing = int(series.isna().sum())
        n_infinite = int((missing_mask(series) & ~series.isna().to_numpy(dtype=bool)).sum())
        if name == target:
            role, kind, reason, n_coerced = "target", None, None, 0
        elif time_col is not None and name == time_col:
            role, kind, reason, n_coerced = "time", None, None, 0
        elif group_col is not None and name == group_col:
            # Never a feature, and no rule of step 5 is applied to it.
            role, kind, reason, n_coerced, n_infinite = "group", None, None, 0, 0
        else:
            role, kind, reason, n_coerced, extra_infinite = _judge_column(series, usable)
            n_infinite += extra_infinite
            if reason == "datetime":
                # Rule 2a: a column of dates holds no number to count (4.3).
                n_infinite = 0
        records.append({
            "name": str(name), "position": position, "dtype": loaded_dtypes[name],
            "role": role, "kind": kind, "reason": reason, "leak_rule": None,
            "n_missing": n_missing, "n_coerced": n_coerced,
            "n_infinite": n_infinite, "n_imputed": 0,
        })
    roles = pd.DataFrame.from_records(records, columns=COLUMN_KEYS)
    roles["leak_rule"] = pd.Series([None] * len(roles), dtype=object)
    return roles
