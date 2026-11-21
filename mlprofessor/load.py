"""Step 1 and step 2: load one table, count the rows that can be used (spec 3, 6).

A CSV path is read with `pandas.read_csv(path, low_memory=False)` and nothing
else. Before that the file is read once with the `csv` module, so that what
pandas does silently is caught: duplicate header names (pandas renames them)
and data rows longer than the header (pandas turns their leading fields into
an index). Row positions 0..n-1 are fixed here and never change.
"""
import csv
import math
import os
from decimal import Decimal

import numpy as np
import pandas as pd

from mlprofessor import errors
from mlprofessor.task import read_target_values

DATAFRAME_SOURCE = "<DataFrame>"
_CSV_FIELD_LIMIT = 2 ** 31 - 1   # the csv module refuses long cells otherwise


def load_table(data):
    """Return `(frame, info)` for a DataFrame or a CSV path (spec 2.7)."""
    if isinstance(data, pd.DataFrame):
        return _load_dataframe(data)
    if isinstance(data, (str, os.PathLike)):
        return _load_csv(data)
    raise errors.InputError(
        f"data must be a pandas DataFrame or the path of a CSV file; "
        f"got {type(data).__name__}.")


def _load_dataframe(source_frame):
    names = [str(name) for name in source_frame.columns]
    _stop_on_duplicates(names, "The DataFrame")
    frame = source_frame.copy(deep=True)
    frame.columns = names
    frame = frame.reset_index(drop=True)
    info = {"kind": "dataframe", "source": DATAFRAME_SOURCE, "warnings": []}
    return frame, info


def _load_csv(path_like):
    path = os.fspath(path_like)
    if not os.path.isfile(path):
        raise errors.InputError(
            f"No file at {path!r}. Give the path of an existing CSV file.")
    header = _check_csv_shape(path)

    try:
        frame = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise errors.InputError(f"{path!r} holds no columns: {exc}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise errors.InputError(
            f"{path!r} could not be read as a CSV file: {exc}") from exc

    loaded_names = [str(name) for name in frame.columns]
    if len(loaded_names) != len(header):
        raise errors.InputError(
            f"{path!r}: the header names {len(header)} columns but "
            f"{len(loaded_names)} were read. Check the header line.")
    for position, cell in enumerate(header):
        if cell != "" and loaded_names[position] != cell:
            raise errors.DuplicateColumnsError(
                f"{path!r}: the column {cell!r} was renamed to "
                f"{loaded_names[position]!r} on loading, because another "
                "column has the same name. Give every column its own name.")
    _stop_on_duplicates(loaded_names, repr(path))
    frame.columns = loaded_names
    frame = frame.reset_index(drop=True)

    warnings = [
        {"code": "blank_header", "column": loaded_names[position],
         "message": (f"Header cell {position + 1} is blank; the column is kept "
                     f"under the name {loaded_names[position]!r}.")}
        for position, cell in enumerate(header) if cell == ""
    ]
    return frame, {"kind": "csv", "source": path, "warnings": warnings}


def _check_csv_shape(path):
    """Read the header and every row with `csv`; return the header cells.

    Raises on an empty file, on duplicate header names and on any data row
    with more fields than the header.
    """
    callers_limit = csv.field_size_limit(_CSV_FIELD_LIMIT)
    try:
        return _read_header(path)
    finally:
        csv.field_size_limit(callers_limit)


def _read_header(path):
    header = None
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if header is None:
                    if _is_blank(row):
                        continue
                    header = row
                    _stop_on_duplicates(
                        [cell for cell in header if cell != ""], repr(path))
                    continue
                if len(row) > len(header):
                    raise errors.InputError(
                        f"{path!r}: line {reader.line_num} has {len(row)} fields "
                        f"but the header has {len(header)}. A row number or an "
                        "extra field with no name in the header would be lost "
                        "or shift the columns. Name every field in the header.")
    except UnicodeDecodeError as exc:
        raise errors.InputError(f"{path!r} is not UTF-8 text: {exc}") from exc
    except csv.Error as exc:
        raise errors.InputError(
            f"{path!r} could not be read as a CSV file: {exc}") from exc
    if header is None:
        raise errors.InputError(f"{path!r} is empty: it has no header line.")
    return header


def _is_blank(row):
    return len(row) == 0 or (len(row) == 1 and row[0].strip() == "")


def _stop_on_duplicates(names, where):
    seen, repeated = set(), []
    for name in names:
        if name in seen and name not in repeated:
            repeated.append(name)
        seen.add(name)
    if repeated:
        raise errors.DuplicateColumnsError(
            f"{where} has more than one column named "
            f"{', '.join(repr(name) for name in repeated)}. "
            "Give every column its own name.")


# ------------------------------------------------------------ usable rows

def missing_mask(values):
    """True where a value is missing: a missing cell, or an infinite number."""
    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    missing = series.isna().to_numpy(dtype=bool)
    if pd.api.types.is_bool_dtype(series.dtype):
        return missing
    if pd.api.types.is_numeric_dtype(series.dtype):
        if pd.api.types.is_complex_dtype(series.dtype):
            numbers = series.to_numpy()
            return missing | np.isinf(numbers)
        numbers = series.to_numpy(dtype=float, na_value=np.nan)
        return missing | np.isinf(numbers)
    if series.dtype == object:
        infinite = np.fromiter(
            (_is_infinite_number(value) for value in series.to_numpy()),
            dtype=bool, count=len(series))
        return missing | infinite
    return missing


def _is_infinite_number(value):
    # A whole number is never infinite; `numpy.isinf` raises on one beyond 64 bits.
    if isinstance(value, (float, np.floating)):
        return bool(np.isinf(value))
    return False


# ------------------------------------------------------------ whole numbers (step 1a)

INT64_MIN, INT64_MAX = -2 ** 63, 2 ** 63 - 1
UINT64_MAX = 2 ** 64 - 1
_BEYOND_64_BITS = 2 ** 65   # stands for a Decimal whole number of more than 20 digits


def _is_empty_cell(value):
    """True for a missing cell: None, NaN, pd.NA, NaT, or a Decimal not-a-number."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, Decimal):
        return value.is_nan()
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


def _whole_number(value):
    """The whole number a cell holds exactly, as an int; None if it holds none."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, Decimal) and value.is_finite():
        if value.adjusted() > 20:
            return _BEYOND_64_BITS if value > 0 else -_BEYOND_64_BITS
        if value == value.to_integral_value():
            return int(value)
    return None


def beyond_64_bits(values):
    """True for an object column whose non-missing cells are all whole numbers
    that do not all fit in 64 bits, signed or unsigned (step 1a; D50)."""
    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    if series.dtype != object:
        return False
    low = high = None
    for value in series.to_numpy():
        if _is_empty_cell(value):
            continue
        number = _whole_number(value)
        if number is None:
            return False
        low = number if low is None else min(low, number)
        high = number if high is None else max(high, number)
    if low is None:
        return False
    fits_signed = INT64_MIN <= low and high <= INT64_MAX
    fits_unsigned = 0 <= low and high <= UINT64_MAX
    return not (fits_signed or fits_unsigned)


def _as_float(value):
    """A number as a 64-bit float; one beyond the largest double is infinite, with its sign."""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _without_decimal_nans(series, empty):
    """The column as it is, with a Decimal not-a-number read as None (pandas raises on sNaN)."""
    cells = series.to_numpy()
    decimal_nan = empty & np.fromiter(
        (isinstance(value, Decimal) for value in cells), dtype=bool, count=len(cells))
    if not decimal_nan.any():
        return series
    cells = cells.copy()
    cells[decimal_nan] = None
    return pd.Series(cells, index=series.index, name=series.name, dtype=object)


def read_numbers(values):
    """Step 1a (D51): an object column of numbers as the typed column it stands for.

    All booleans: a nullable boolean column. All whole numbers held exactly and
    fitting in 64 bits: a nullable Int64 (or UInt64) column, exact. Numbers of
    any other mixture: float64, NaN where a cell is empty. Anything else, and
    a column of any other type, is returned as it is.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if series.dtype != object:
        return series
    cells = series.to_numpy()
    empty = np.fromiter((_is_empty_cell(value) for value in cells),
                        dtype=bool, count=len(cells))
    kinds, wholes = set(), []
    for value in cells[~empty]:
        if isinstance(value, (bool, np.bool_)):
            kinds.add("boolean")
            continue
        number = _whole_number(value)
        if number is not None:
            kinds.add("whole")
            wholes.append(number)
        elif isinstance(value, (float, np.floating, Decimal)):
            kinds.add("float")
        else:
            return _without_decimal_nans(series, empty)
    if not kinds or ("boolean" in kinds and kinds != {"boolean"}):
        return _without_decimal_nans(series, empty)

    present = ~empty
    if kinds == {"boolean"}:
        data = np.zeros(len(cells), dtype=bool)
        data[present] = [bool(value) for value in cells[present]]
        array = pd.arrays.BooleanArray(data, empty.copy())
    elif kinds == {"whole"}:
        low, high = min(wholes), max(wholes)
        if INT64_MIN <= low and high <= INT64_MAX:
            dtype = np.int64
        elif 0 <= low and high <= UINT64_MAX:
            dtype = np.uint64
        else:
            return _without_decimal_nans(series, empty)   # too large for 64 bits (D50)
        data = np.zeros(len(cells), dtype=dtype)
        data[present] = np.array(wholes, dtype=dtype)
        array = pd.arrays.IntegerArray(data, empty.copy())
    else:
        array = np.full(len(cells), np.nan, dtype=np.float64)
        array[present] = [_as_float(value) for value in cells[present]]
    return pd.Series(array, index=series.index, name=series.name)


def read_number_columns(frame):
    """Step 1a for every column of a table; columns of other types are left as they are."""
    read = {name: read_numbers(frame[name]) for name in frame.columns
            if frame[name].dtype == object}
    if not read:
        return frame
    out = frame.copy(deep=False)
    for name, series in read.items():
        out[name] = series
    return out


def usable_rows(frame, target, time_col=None):
    """Step 2 as a count: the positions of the usable rows and the two drops."""
    for name in (target, time_col):
        if name is not None and name not in frame.columns:
            raise errors.ColumnNotFoundError(
                f"No column named {name!r}. The columns are: "
                f"{', '.join(repr(str(c)) for c in frame.columns)}.")
    # A text target of only white space is a missing target (D48).
    no_target = missing_mask(read_target_values(read_numbers(frame[target])))
    if time_col is None:
        no_time = np.zeros(len(frame), dtype=bool)
    else:
        no_time = missing_mask(read_numbers(frame[time_col])) & ~no_target
    usable = np.flatnonzero(~no_target & ~no_time).astype(np.int64)
    return {
        "usable": usable,
        "dropped_missing_target": int(no_target.sum()),
        "dropped_missing_time": int(no_time.sum()),
    }
