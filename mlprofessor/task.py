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


def _all_whole_numbers(present_values):
    numbers = np.asarray(present_values, dtype=float)
    return bool(np.all(np.isfinite(numbers)) and np.all(numbers == np.floor(numbers)))


def decide_task(values, task="auto"):
    """Decide classification or regression for the target values given.

    Text is read without the white space round it, and missing values are
    ignored, text of only white space among them. Returns the type, how it was decided, the
    number of distinct values, the classes in sorted order with the second as
    the positive class, and a list of warning codes.
    """
    if task not in TASKS:
        raise errors.InputError(
            f"task must be one of {', '.join(TASKS)}; got {task!r}.")

    target_values = read_target_values(values)
    present_values = target_values.dropna()
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
        kind = "numeric" if numeric else "non-numeric"
        raise errors.TaskAmbiguousError(
            f"The target is {kind} with {n_distinct} distinct values; the rule "
            "decides only two values (classification) or a numeric target with "
            f"more than {MIN_DISTINCT_REGRESSION - 1} (regression). Declare the "
            "task, or recode the target.")

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
