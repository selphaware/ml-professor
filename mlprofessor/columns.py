"""Step 5: the column roles that need no target (spec 3 step 5, 4.3, 4.5, D19, D28, D34).

Roles are judged on the usable rows; the counts `n_missing`, `n_coerced` and
`n_infinite` are kept over every loaded row; `n_infinite` of a numeric column
also counts the values too large for the tree models (D48). Whole numbers are judged by value,
never by type. A number written with thousands separators is never parsed,
and dates are recognised by a fixed list of forms, never by pandas' parser.
"""
import datetime
import re

import numpy as np
import pandas as pd

from mlprofessor.load import missing_mask, usable_rows

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

def _numeric_rule(values):
    """Rules 8 to 8c on the usable rows' values; returns (role, kind, reason)."""
    values = values[np.isfinite(values) & ~beyond_limit(values)]
    if values.size == 0:
        return "excluded", "numeric", "all_missing"
    distinct = np.unique(values)
    if distinct.size == 1:
        return "excluded", "numeric", "constant"
    if distinct.size == values.size and _whole(values):
        steps = np.diff(values)
        consecutive = int(values.max()) - int(values.min()) + 1 == values.size
        ordered = bool(np.all(steps > 0) or np.all(steps < 0))
        if consecutive or ordered:
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


def _judge_column(series, usable):
    """One column's (role, kind, reason, n_coerced, extra infinities)."""
    usable_missing = series.iloc[usable].isna().to_numpy(dtype=bool)
    if usable_missing.all():
        return "excluded", None, "all_missing", 0, 0

    family = type_family(series)
    if family == "datetime":
        return "excluded", None, "datetime", 0, 0
    if family == "other":
        return "excluded", None, "unsupported_type", 0, 0
    if family == "numeric":
        values = numeric_values(series)
        n_too_large = int((np.isfinite(values) & beyond_limit(values)).sum())
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


def column_roles(frame, target, time_col=None):
    """Step 5 alone: one record per column of `frame`, in order (spec 2.7)."""
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
        else:
            role, kind, reason, n_coerced, extra_infinite = _judge_column(series, usable)
            n_infinite += extra_infinite
        records.append({
            "name": str(name), "position": position, "dtype": str(series.dtype),
            "role": role, "kind": kind, "reason": reason, "leak_rule": None,
            "n_missing": n_missing, "n_coerced": n_coerced,
            "n_infinite": n_infinite, "n_imputed": 0,
        })
    roles = pd.DataFrame.from_records(records, columns=COLUMN_KEYS)
    roles["leak_rule"] = pd.Series([None] * len(roles), dtype=object)
    return roles
