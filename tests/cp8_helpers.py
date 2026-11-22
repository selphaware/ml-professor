"""Shared helpers of the seventh pass (CP8). Builds on `mlp_helpers`, `cp3_helpers`,
`cp5_helpers`, `cp6_helpers` and `cp7_helpers`; holds no test.

Like `cp7_helpers`, nothing here makes a statement the call did not make (D63): the
tests of this pass call through `cp7_helpers.run`, `raised` and `run_cli`.

Nothing here imports `mlprofessor` at module level, but for `roles`, which imports it
inside its body.
"""
import datetime as _dt
import math
import re

import numpy as np
import pandas as pd

import cp3_helpers as c
import cp6_helpers as g
import cp7_helpers as s
import mlp_helpers as h

CAP = 2000
CAP_CODE = "fast_profile_group_too_large"
SENTINEL = 99991231
OTHERS = (99991231, 0, 99999999)

# The fixed passage of spec 5.4 that the `same_features_grouped` warning holds (D69).
GROUP_MEASURES = (
    "With duplicates='group' the headline measures whether a combination of feature "
    "values that the model has not seen can be predicted. On a table of few-valued "
    "columns that can give negative skill where duplicates='keep' finds signal."
)
RANDOM_CAVEAT = h.CAVEAT
GROUPED_CAVEAT = g.GROUPED_CAVEAT

MISSING = "(missing)"


# ------------------------------------------------------------------ readers

def roles(frame, target="y", **kw):
    """`stages.column_roles` as a mapping from a column's name to its record."""
    from mlprofessor import stages

    return {r["name"]: r for r in h.records(stages.column_roles(frame, target, **kw))}


def is_date(record):
    return record["role"] == "excluded" and record["reason"] == "datetime"


def is_number(record):
    return record["role"] == "feature" and record["kind"] == "numeric"


def rows_reached(result, log):
    """For an `h.Log`: the holdout rows together with every row a model was fitted on
    or asked to predict, which are the rows a cap kept."""
    rows = set(h.holdout_rows(result))
    for event in log.events:
        if event["kind"] == "model" and event["action"] in ("fit", "predict"):
            rows.update(event["rows"])
    return rows


def _key_cell(value, kind):
    if value is None or value is pd.NA or value is pd.NaT:
        return MISSING
    if kind == "numeric":
        try:
            number = float(value)
        except (TypeError, ValueError):
            return MISSING
        if math.isnan(number) or math.isinf(number):
            return MISSING
        return number + 0.0
    if isinstance(value, float) and math.isnan(value):
        return MISSING
    text = str(value).strip()
    return text if text else MISSING


def feature_keys(result, frame):
    """For every row of `frame`, the tuple of its values in the columns the result gives
    the role `feature`, as step 8a compares them: a number as a float (0.0 and -0.0
    alike), text without the white space round it, an empty cell as `(missing)`."""
    columns = [r for r in h.records(result.columns) if r["role"] == "feature"]
    assert columns, "the result has no feature column"
    cells = [[_key_cell(v, r["kind"]) for v in frame[r["name"]].tolist()] for r in columns]
    return [tuple(column[i] for column in cells) for i in range(len(frame))]


def whole_numbers_in(text):
    """The whole numbers written in `text`, each run of digits as one."""
    return [int(t) for t in re.findall(r"\d+", str(text))]


def holds_count_and_share(text, n_rows):
    """Whether `text` holds two whole numbers a <= b, b at least a quarter of `n_rows` and
    fewer than `n_rows`, with some number in it within 0.1 of 100 x a / b."""
    wholes = sorted(set(whole_numbers_in(text)))
    numbers = c.numbers_in(str(text))
    for b in wholes:
        if not (4 * b >= n_rows and b < n_rows):
            continue
        for a in wholes:
            if 1 <= a <= b and any(abs(x - 100.0 * a / b) <= 0.1 for x in numbers):
                return True
    return False


# ------------------------------------------------------------------- tables

def days(n_dates, start="2021-03-01"):
    """That many consecutive days, each as the whole number YYYYMMDD."""
    first = _dt.date.fromisoformat(start)
    return [s.date_number(first + _dt.timedelta(days=k)) for k in range(int(n_dates))]


def _like_column(column, value):
    """`value` as a cell of a column of the type of `column`."""
    if pd.api.types.is_string_dtype(column.dtype) or column.dtype == object:
        return str(value)
    if pd.api.types.is_float_dtype(column.dtype):
        return float(value)
    return int(value)


def planted(frame, values, cells=1, share=None, seed=0, column="day"):
    """A copy of `frame` with `values` planted in `column`: `cells` cells for each value,
    or, with `share`, that share of the rows dealt out among the values in turn. The
    cells are drawn from `seed`; the column keeps its type."""
    values = list(values)
    n = len(frame)
    total = int(round(share * n)) if share is not None else int(cells) * len(values)
    assert 0 < total <= n
    where = np.random.default_rng(seed).choice(n, size=total, replace=False)
    out = frame.copy()
    original = out[column]
    cells_now = original.tolist()
    for k, row in enumerate(where):
        cells_now[int(row)] = _like_column(original, values[k % len(values)])
    out[column] = pd.Series(cells_now, index=out.index).astype(original.dtype)
    return out


def dates_with(n_dates, others=(), n=300, cells=2, seed=0):
    """A table whose int64 column `day` holds exactly `n_dates` distinct dates, each many
    times, and each of `others` in `cells` cells; a noise column `x` and a 0/1 target
    `y`. Rows shuffled from `seed`."""
    rng = np.random.default_rng(seed)
    others = [int(v) for v in others]
    n_other = len(others) * int(cells)
    n_dated = n - n_other
    assert n_dated >= 2 * n_dates
    dated = days(n_dates)
    column = [dated[k % n_dates] for k in range(n_dated)]
    for value in others:
        column.extend([value] * int(cells))
    order = rng.permutation(n)
    frame = pd.DataFrame({
        "day": np.array(column, dtype=np.int64)[order],
        "x": rng.normal(size=n),
        "y": rng.integers(0, 2, size=n),
    })
    assert frame["day"].nunique() == n_dates + len(set(others))
    return frame


def with_copy_of_target(frame, name="status", target="y"):
    """`frame` with a text column before the target that copies a 0/1 target: `closed`
    for 1 and `open` for 0."""
    out = frame.copy()
    values = ["closed" if int(v) == 1 else "open" for v in out[target].tolist()]
    out.insert(list(out.columns).index(target), name, pd.array(values, dtype="str"))
    return out


ALIKE = (0.125, -0.5, 0.75, -1.25)


def one_row_alike_in_every_entity(n_entities=24, per=20, seed=0):
    """`who`, four noise measures `n1` to `n4`, a column `leak` that is the 0/1 target
    plus (row number + 1) / (2 x rows), and `y`. The first row of each entity holds the
    measures of `ALIKE`, with the classes 0 and 1 in turn. So no two rows agree in every
    column, and without `leak` those rows, one in each entity, agree in every feature
    that is left. Rows shuffled from `seed`."""
    rng = np.random.default_rng(seed)
    n = n_entities * per
    who, y = [], []
    measures = rng.normal(size=(n, 4))
    for k in range(n_entities):
        who.extend([f"e{k:03d}"] * per)
        labels = rng.integers(0, 2, size=per)
        labels[0] = k % 2
        labels[1] = 1 - k % 2
        y.extend(int(v) for v in labels)
        measures[k * per] = ALIKE
    frame = pd.DataFrame({"who": who})
    for j in range(4):
        frame[f"n{j + 1}"] = measures[:, j]
    frame["y"] = y
    order = rng.permutation(n)
    frame = frame.iloc[order].reset_index(drop=True)
    leak = frame["y"].to_numpy(dtype=float) + (np.arange(n) + 1.0) / (2.0 * n)
    frame.insert(list(frame.columns).index("y"), "leak", leak)
    return frame


def large_entities(large, small=300, per=5, seed=0, labels=None):
    """`s.sized_entities` with the large entities first (`e000`, `e001`, ... hold the rows
    given in `large`) and `small` more of `per` rows each."""
    sizes = [int(v) for v in large] + [int(per)] * int(small)
    return s.sized_entities(sizes, seed=seed, labels=labels)
