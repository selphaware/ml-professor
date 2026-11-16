"""The documented seam (spec.md, "Seam for tests").

    with extensions.use(families=[...], steps=[...]):
        result = mlprofessor.profile(...)

families: a list of {"name": str, "grid": [params, ...], "make": callable}.
    They replace the profile's own families while the block is active.
    make(task, params, seed, n_jobs) returns an unfitted estimator with
    fit(X, y) and predict_proba(X) (classification) or predict(X) (regression).
steps: a list of zero-argument callables, each returning an object with
    fit(X, y) and transform(X). A fresh one is made for every fit.
Both receive X as a DataFrame indexed by the rows' positions in the source.
"""
from contextlib import contextmanager

_active = {"families": None, "steps": None}


def active():
    """The current overrides (a copy)."""
    return dict(_active)


@contextmanager
def use(families=None, steps=None):
    """Replace the model families and/or add preprocessing steps for one block."""
    previous = dict(_active)
    _active["families"] = list(families) if families is not None else None
    _active["steps"] = list(steps) if steps is not None else None
    try:
        yield
    finally:
        _active.update(previous)
