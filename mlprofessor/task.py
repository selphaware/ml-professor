"""Step 4: decide the task from the target values (D4, D21)."""
import numpy as np
import pandas as pd

from mlprofessor import errors

TASKS = ("auto", "classification", "regression")
MIN_DISTINCT_REGRESSION = 21
WHOLE_NUMBER_WARNING = "whole_number_regression_target"


def _native(value):
    """A plain Python value for a numpy scalar, so it compares and serialises cleanly."""
    return value.item() if isinstance(value, np.generic) else value


def _is_text_like(dtype):
    """A column whose cells may hold text: object, pandas string, or category."""
    return (dtype == object or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype))


def read_target_values(values):
    """Target values as step 4 reads them (D46, D48).

    A text value loses the white space round it, as a categorical level does,
    and a text value of only white space is missing (None). A value that is not
    text is as it is; a column of no text type is returned unchanged.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if not _is_text_like(series.dtype):
        return series
    cells = series.to_numpy(dtype=object, na_value=None)
    read = np.empty(len(cells), dtype=object)
    for position, value in enumerate(cells):
        if isinstance(value, str):
            value = value.strip() or None
        read[position] = value
    return pd.Series(read, index=series.index, dtype=object)


def _is_numeric(target_values):
    """Integer or float type; a boolean or complex type is not numeric here (D48)."""
    dtype = target_values.dtype
    return (pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
            and not pd.api.types.is_complex_dtype(dtype))


def _sorted_classes(distinct_values):
    """The distinct values in sorted order: by value, or by their text if they cannot be compared."""
    native_values = [_native(v) for v in distinct_values]
    try:
        return sorted(native_values)
    except TypeError:
        return sorted(native_values, key=str)


def _first_unhashable(present_values):
    """The type name of the first value that cannot be hashed, or None (D53)."""
    if present_values.dtype != object:
        return None
    for value in present_values:
        try:
            hash(value)
        except TypeError:
            return type(value).__name__
    return None


def class_value(value):
    """A class value as JSON holds it: a boolean, a number or text (D53).

    Anything else (bytes, a date, a duration, a period, an interval, a complex
    number, a tuple) is its text form.
    """
    value = _native(value)
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, complex):
        return value
    return str(value)


def _all_whole_numbers(present_values):
    numbers = np.asarray(present_values, dtype=float)
    return bool(np.all(np.isfinite(numbers)) and np.all(numbers == np.floor(numbers)))


def decide_task(values, task="auto", rows=None):
    """Decide classification or regression for the target values given.

    Text is read without the white space round it, and missing values are
    ignored, text of only white space among them. Returns the type, how it was decided, the
    number of distinct values, the classes in sorted order with the second as
    the positive class, and a list of warning codes. With `rows`, the column
    is read (step 1a) over all its values and the task decided on those rows.
    """
    if task not in TASKS:
        raise errors.InputError(
            f"task must be one of {', '.join(TASKS)}; got {task!r}.")

    # load reads targets through this module, so it is imported here.
    from mlprofessor.load import beyond_64_bits, read_numbers

    values = read_numbers(values)
    if beyond_64_bits(values):
        raise errors.TaskUnsupportedError(
            "The target holds whole numbers too large for 64 bits, which are not read as "
            "numbers. To use them as labels, pass them as text.")
    target_values = read_target_values(values)
    if rows is not None:
        target_values = target_values.iloc[np.asarray(rows, dtype=np.int64)]
    present_values = target_values.dropna()
    unhashable = _first_unhashable(present_values)
    if unhashable is not None:
        raise errors.TaskUnsupportedError(
            f"The target holds values of type {unhashable}, which cannot be compared for "
            "equality, so its distinct values cannot be counted. Encode the target as "
            "text or numbers.")
    distinct_values = present_values.unique()
    n_distinct = len(distinct_values)
    numeric = _is_numeric(target_values)

    if n_distinct < 2:
        raise errors.TargetConstantError(
            f"The target has {n_distinct} distinct value(s) among the usable rows; "
            "at least two are needed.")

    if task == "classification":
        if n_distinct != 2:
            raise errors.TaskUnsupportedError(
                f"Classification needs exactly two distinct target values; "
                f"the target has {n_distinct}.")
        task_type, decided_by = "classification", "declared"
    elif task == "regression":
        if not numeric:
            raise errors.TaskUnsupportedError(
                f"Regression needs a numeric target; the target has type "
                f"{target_values.dtype}.")
        task_type, decided_by = "regression", "declared"
    elif n_distinct == 2:
        task_type, decided_by = "classification", "rule"
    elif numeric and n_distinct >= MIN_DISTINCT_REGRESSION:
        task_type, decided_by = "regression", "rule"
    else:
        kind = "numeric" if numeric else "not numeric"
        message = (
            f"The target is {kind} with {n_distinct} distinct values. Three or more "
            "classes are not supported in this version. What would run: the rows of "
            "two of the classes only, or the classes merged into two.")
        if numeric:
            message += (
                " A target which is a quantity, not a set of classes, can be run with "
                "task='regression'.")
        raise errors.TaskAmbiguousError(message)

    warnings = []
    classes = positive_class = None
    if task_type == "classification":
        classes = _sorted_classes(distinct_values)
        positive_class = classes[1]
    elif decided_by == "rule" and _all_whole_numbers(present_values):
        warnings.append(WHOLE_NUMBER_WARNING)

    return {
        "type": task_type,
        "decided_by": decided_by,
        "n_distinct": n_distinct,
        "classes": classes,
        "positive_class": positive_class,
        "warnings": warnings,
    }
