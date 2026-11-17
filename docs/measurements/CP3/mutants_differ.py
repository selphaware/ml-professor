"""CP3: each of the five small faults M1 to M5 changes what the tool does.

    python mutants_differ.py

For each fault, one crafted table on which the reference implementation with
that fault gives a different answer from the reference without it, with both
values. The tables are the ones the guard tests are built on (plan/test-plan.md,
"Second pass"). Runs in memory; writes nothing.

The reference implementation is imported from the judge's folder and its switch
is set in the running process, one fault at a time.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "verify" / "reference"))

import mlprofessor  # noqa: E402
from mlprofessor import core, extensions, stages  # noqa: E402

FAST = dict(profile="fast", n_jobs=1, seed=0)


class Recorder:
    """A seam estimator: keeps the matrix it is fitted on, predicts chosen values."""

    def __init__(self, task, fits, values=None):
        self.task, self.fits, self.values = task, fits, values

    def fit(self, X, y):
        self.fits.append({"rows": list(X.index), "columns": list(X.columns),
                          "values": X.to_numpy(dtype=float).copy()})
        self.mean = float(np.mean(y))
        return self

    def predict(self, X):
        if self.values is not None:
            return np.asarray(self.values, dtype=float)[np.asarray(X.index, dtype=int)]
        return np.full(len(X), self.mean)

    def predict_proba(self, X):
        v = np.clip(self.predict(X), 0, 1)
        return np.column_stack([1 - v, v])


def run(frame, target="y", values=None, **kw):
    fits = []
    family = {"name": "rec", "grid": [{}],
              "make": lambda task, params, seed, n_jobs: Recorder(task, fits, values)}
    with extensions.use(families=[family]):
        result = mlprofessor.profile(frame, target, **dict(FAST, **kw))
    return result, fits


def both(fault, measure):
    """`measure()` on the good reference, then with `fault` switched on."""
    out = []
    for switch in (None, fault):
        core.BROKEN = switch
        try:
            out.append(measure())
        finally:
            core.BROKEN = None
    return out


def clf_frame(n, seed):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    frame = pd.DataFrame({"x1": x1, "z1": rng.normal(size=n)})
    gappy = rng.normal(size=n)
    gappy[rng.random(n) < 0.15] = np.nan
    frame["gappy"] = gappy
    frame["colour"] = rng.choice(["red", "green", "blue"], size=n)
    frame["y"] = (rng.random(n) < 1 / (1 + np.exp(-2.5 * x1))).astype(int)
    return frame


def final_fit(fits, training):
    return next(f for f in fits if set(f["rows"]) == set(training))


def m1():
    """Replace the holdout's feature values by extremes: does the final fit's matrix move?"""
    frame = clf_frame(240, 41)
    first, fits_a = run(frame)
    hold = first.predictions["row"].tolist()
    training = sorted(set(range(len(frame))) - set(hold))
    changed = frame.copy()
    changed.loc[hold, ["x1", "gappy", "z1"]] = [1e6, -1e6, 5e5]
    _, fits_b = run(changed)
    a, b = final_fit(fits_a, training), final_fit(fits_b, training)
    column = a["columns"].index("x1")
    moved = np.abs(a["values"] - b["values"]).max()
    return (f"largest change in the final fit's matrix {moved:.4g}; mean of its x1 column "
            f"with the extremes in place {b['values'][:, column].mean():+.4f}")


def m2():
    """A level only holdout rows hold: does the final fit's matrix have a column for it?"""
    frame = clf_frame(240, 42)
    split = stages.split_random(len(frame), classes=frame["y"].to_numpy(), seed=0)
    hold = np.asarray(split["holdout"]).tolist()
    frame.loc[hold[:8], "colour"] = "holdout_only"
    result, fits = run(frame)
    training = sorted(set(range(len(frame))) - set(result.predictions["row"].tolist()))
    columns = final_fit(fits, training)["columns"]
    return f"the final fit's colour columns {[c for c in columns if c.startswith('colour=')]}"


def screened(frame, column, **kw):
    """The column's record, and every single-column skill the screen measured, in order."""
    measured, real = [], core._screen_column

    def keep(*args, **kwargs):
        measured.append(real(*args, **kwargs))
        return measured[-1]

    core._screen_column = keep
    try:
        result, _ = run(frame, **kw)
    finally:
        core._screen_column = real
    record = next(c for c in result.to_dict()["columns"] if c["name"] == column)
    skills = ", ".join("none" if s is None else f"{s:.3f}" for s in measured)
    return (f"`{column}` is {record['role']}, leak rule {record['leak_rule']}; single-column "
            f"skills measured, in column order: {skills}")


def m3_missing():
    """Empty for 70% of class 0 in the training rows, for 97% once the holdout is counted."""
    n = 400
    rng = np.random.default_rng(51)
    y = np.ones(n, dtype=int)
    y[[5, 30, 55, 80, 105, 130, 160, 190, 220, 260]] = 0
    y[300:390] = 0
    late = rng.normal(size=n)
    late[y == 0] = np.nan
    late[[80, 130, 260]] = rng.normal(size=3)
    frame = pd.DataFrame({"t": np.arange(n, dtype=float), "x1": rng.normal(size=n),
                          "late": late, "y": y})
    return screened(frame, "late", time_col="t")


def m3_trees():
    """A column a tree can read only once it has been fitted on holdout rows."""
    n = 400
    rng = np.random.default_rng(52)
    y = np.tile([0, 1], n // 2)
    y[:300] = y[np.concatenate([rng.permutation(50) + 50 * b for b in range(6)])]
    block = np.r_[np.repeat(np.arange(1, 7), 50), np.repeat(np.arange(2, 7), 20)]
    high = np.where(block % 2 == 1, y, 1 - y)
    frame = pd.DataFrame({"t": np.arange(n, dtype=float), "x1": rng.normal(size=n),
                          "drift": block + 0.25 + 0.5 * high, "y": y})
    return screened(frame, "drift", time_col="t")


def m4():
    """Squared errors that alternate 0 and 1 over 64 holdout timestamps."""
    n = 256
    rng = np.random.default_rng(61)
    frame = pd.DataFrame({"t": np.arange(n, dtype=float), "x1": rng.normal(size=n)})
    frame["y"] = np.round(rng.normal(size=n) * 8.0) / 8.0 + 3.0
    values = frame["y"].to_numpy() + rng.normal(size=n)
    values[192:] = frame["y"].to_numpy()[192:] + (np.arange(64) % 2)
    result, _ = run(frame, values=values, time_col="t", task="regression")
    h = result.headline
    return (f"score {h['score']!r}, interval {h['interval_low']!r} to {h['interval_high']!r} "
            f"({h['interval_method']}, block length {h['block_length']})")


def m5():
    """A model that predicts 0.5 for every row."""
    rng = np.random.default_rng(62)
    x1 = rng.normal(size=240)
    y = (rng.random(240) < 1 / (1 + np.exp(-2.5 * x1))).astype(int)
    frame = pd.DataFrame({"x1": x1, "y": y})
    result, _ = run(frame, values=np.full(240, 0.5))
    h = result.headline
    return f"score {h['score']!r}, skill {h['skill']!r}, p-value {h['p_value']!r}"


CASES = [("M1", "final fit's statistics see the holdout", m1),
         ("M2", "levels learned from every row", m2),
         ("M3", "missing-by-class counts every row", m3_missing),
         ("M3", "the screen's trees are fitted on holdout rows too", m3_trees),
         ("M4", "timestamps drawn one by one", m4),
         ("M5", "ties do not count in the p-value", m5)]

if __name__ == "__main__":
    for fault, what, measure in CASES:
        good, broken = both(fault, measure)
        print(f"{fault} ({what}): {(measure.__doc__ or '').strip()}")
        print(f"   good: {good}")
        print(f"   {fault}:   {broken}")
        print("   DIFFERENT" if good != broken else "   THE SAME: the fault changes nothing here")
