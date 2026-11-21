"""Shared helpers of the fourth pass (CP5). Builds on `mlp_helpers`,
`cp3_helpers` and `cp4_helpers`; holds no test.

Nothing here imports `mlprofessor` at module level: each test imports the
application inside its own body.
"""
import math

import numpy as np
import pandas as pd
import pytest

import cp3_helpers as c
import mlp_helpers as h

START = 1_700_000_000_000_000_000


# ------------------------------------------------------------------ running

def flat_family(log, n_rows, value=0.5):
    """One seam family predicting `value` for every row: quick, never fails."""
    return [h.family("flat", log, estimator=h.LookupEstimator,
                     values=np.full(int(n_rows), float(value)))]


def run_kept(frame, target="y", seed=0, **kw):
    """`c.run` with the flat family and a keeping step. Returns `(result, log)`."""
    log = c.KeepLog()
    result = c.run(frame, target, families=flat_family(log, len(frame)),
                   steps=[c.keep_step_factory(log)], seed=seed, **kw)
    return result, log


def _largest_step_fit(log):
    fits = c.KeepLog.calls(log, "step", "fit")
    assert fits, "no step was fitted"
    return max(fits, key=lambda ev: len(ev["rows"]))


def training_rows(log):
    """The rows of the largest `fit` a step was handed: the final fit's rows."""
    return list(_largest_step_fit(log)["rows"])


def raised(frame, target="y", families=None, steps=None, **kw):
    """The `ProfileError` that `profile()` raises (fast, one core by default).

    Fails if it returns a result; any other exception passes through.
    """
    import mlprofessor
    from mlprofessor import errors, extensions

    args = dict(h.FAST)
    args.update(kw)
    try:
        with extensions.use(families=families, steps=steps):
            result = mlprofessor.profile(frame, target, **args)
    except errors.ProfileError as exc:
        return exc
    pytest.fail(f"profile() returned a result; a named error was expected "
                f"(rows {h.rows_of(result)})")


# ------------------------------------------------------ copies, by hand

def _cell(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return None if math.isnan(value) else value + 0.0
    if isinstance(value, np.generic):
        return value.item()
    return value


def _same_class(value, positive):
    if value == positive:
        return True
    if isinstance(value, str) and isinstance(positive, str):
        return value.strip() == positive.strip()
    try:
        return float(value) == float(positive)
    except (TypeError, ValueError):
        return False


def copies_by_hand(result, log):
    """The test's own count for step 8a, from what the steps were handed."""
    doc = result.to_dict()
    fit = _largest_step_fit(log)
    train_frame = fit["frame"]
    columns = list(train_frame.columns)
    train_y = [float(v) + 0.0 for v in fit["y"]]

    holdout = h.holdout_rows(result)
    wanted = set(holdout)
    transforms = [ev for ev in c.KeepLog.calls(log, "step", "transform")
                  if set(ev["rows"]) == wanted]
    assert transforms, "no step transform was handed exactly the holdout rows"
    hold_frame = transforms[-1]["frame"]
    assert list(hold_frame.columns) == columns

    if doc["task"]["type"] == "classification":
        positive = doc["task"]["positive_class"]
        hold_y = [1.0 if _same_class(v, positive) else 0.0
                  for v in result.predictions["y_true"]]
    else:
        hold_y = [float(v) + 0.0 for v in result.predictions["y_true"]]
    by_row = dict(zip(h.holdout_rows(result), hold_y))

    def key(frame, position, target):
        return tuple(_cell(v) for v in frame.iloc[position].tolist()) + (target,)

    seen = {key(train_frame, i, train_y[i]) for i in range(len(train_frame))}
    count = 0
    for i, row in enumerate(hold_frame.index):
        if key(hold_frame, i, by_row[int(row)]) in seen:
            count += 1
    return count


def plant_copies(frame, k, target="y", seed=0, **kw):
    """A copy of `frame` in which exactly `k` holdout rows copy a training row.

    Returns `(frame, number of holdout rows)`. The split depends only on the
    rows, the seed and the classes, so the run on the result splits alike.
    """
    result, log = run_kept(frame, target, seed=seed, **kw)
    train = training_rows(log)
    holdout = h.holdout_rows(result)
    is_clf = result.to_dict()["task"]["type"] == "classification"
    features = [col for col in frame.columns if col != target]
    out = frame.copy()
    used = set()
    for row in holdout[:k]:
        source = next(r for r in train if r not in used
                      and (not is_clf or frame[target].iloc[r] == frame[target].iloc[row]))
        used.add(source)
        for col in features:
            out.iloc[row, out.columns.get_loc(col)] = frame[col].iloc[source]
        if not is_clf:
            out.iloc[row, out.columns.get_loc(target)] = frame[target].iloc[source]
    return out, len(holdout)


# ------------------------------------------------------------------ tables

def twice(frame, seed=0):
    """Every row present twice, in an order shuffled from `seed`."""
    both = pd.concat([frame, frame], ignore_index=True)
    order = np.random.default_rng(seed).permutation(len(both))
    return both.iloc[order].reset_index(drop=True)


def noise_frame(n, seed=0, n_columns=4, task="classification"):
    """Normal columns n1.. and a target `y` that depends on none of them."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"n{i + 1}": rng.normal(size=n) for i in range(n_columns)})
    if task == "classification":
        frame["y"] = rng.integers(0, 2, size=n)
    else:
        frame["y"] = rng.normal(size=n)
    return frame


def nanosecond_walk(n=600, start=START, step=100, seed=0):
    """A random walk with its previous value, a noise column and int64 stamps `t`."""
    rng = np.random.default_rng(seed)
    level = np.cumsum(rng.normal(size=n))
    prev = np.concatenate([[0.0], level[:-1]])
    x1 = rng.normal(size=n)
    t = np.array([start + step * k for k in range(n)], dtype=np.int64)
    frame = pd.DataFrame({"t": t, "x1": x1, "level_prev": prev, "level": level})
    order = rng.permutation(n)
    return frame.iloc[order].reset_index(drop=True)
