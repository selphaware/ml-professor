"""Test seam (spec 2.6): replace model families or preprocessing steps in a block.

A `None` argument leaves the outer block's override in place, so nested blocks
can replace one part and keep the other. The previous state is restored on
exit, including after an exception.
"""
from contextlib import contextmanager

_overrides = {"families": None, "steps": None}


def current_overrides():
    """Return a copy of the overrides in force: `{"families", "steps"}`."""
    return dict(_overrides)


@contextmanager
def use(families=None, steps=None):
    """Replace the model families and/or the preprocessing steps for the block."""
    previous = dict(_overrides)
    if families is not None:
        _overrides["families"] = list(families)
    if steps is not None:
        _overrides["steps"] = list(steps)
    try:
        yield
    finally:
        _overrides.clear()
        _overrides.update(previous)
