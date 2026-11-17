"""Step 1 and step 2: load one table, count the rows that can be used (spec 3, 6).

A CSV path is read with `pandas.read_csv(path, low_memory=False)` and nothing
else. Before that the file is read once with the `csv` module, so that what
pandas does silently is caught: duplicate header names (pandas renames them)
and data rows longer than the header (pandas turns their leading fields into
an index). Row positions 0..n-1 are fixed here and never change.
"""
import csv
import os

import numpy as np
import pandas as pd

from mlprofessor import errors

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
    csv.field_size_limit(_CSV_FIELD_LIMIT)
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
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isinf(value))
    return False


def usable_rows(frame, target, time_col=None):
    """Step 2 as a count: the positions of the usable rows and the two drops."""
    for name in (target, time_col):
        if name is not None and name not in frame.columns:
            raise errors.ColumnNotFoundError(
                f"No column named {name!r}. The columns are: "
                f"{', '.join(repr(str(c)) for c in frame.columns)}.")
    no_target = missing_mask(frame[target])
    if time_col is None:
        no_time = np.zeros(len(frame), dtype=bool)
    else:
        no_time = missing_mask(frame[time_col]) & ~no_target
    usable = np.flatnonzero(~no_target & ~no_time).astype(np.int64)
    return {
        "usable": usable,
        "dropped_missing_target": int(no_target.sum()),
        "dropped_missing_time": int(no_time.sum()),
    }
