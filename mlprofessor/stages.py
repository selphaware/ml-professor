"""Step functions (spec 2.7): the data-handling steps, callable on their own.

A thin facade: each name points at the code `profile()` itself uses.
"""
import pandas as pd

from mlprofessor import columns as _columns
from mlprofessor.load import load_table, usable_rows
from mlprofessor.metrics import mse, previous_value, roc_auc, skill
from mlprofessor.preprocess import Preprocessor
from mlprofessor.split import read_time, split_random, split_time
from mlprofessor.task import decide_task

# Made by step 12a, after every step a step function covers: null here.
_STATEMENT_KEYS = ("shuffled_score", "shuffled_skill", "skill_fall")


def column_roles(frame, target, time_col=None, group_col=None):
    """Step 5's roles of every column of `frame` (spec 2.7): the fourteen columns of
    `profile.json["columns"]`, the statement of step 12a null throughout."""
    roles = _columns.column_roles(frame, target, time_col=time_col, group_col=group_col)
    for key in _STATEMENT_KEYS:
        roles[key] = pd.Series([None] * len(roles), index=roles.index, dtype=object)
    return roles


__all__ = ["load_table", "usable_rows", "column_roles", "decide_task", "read_time",
           "split_random", "split_time", "roc_auc", "mse", "skill", "previous_value",
           "Preprocessor"]
