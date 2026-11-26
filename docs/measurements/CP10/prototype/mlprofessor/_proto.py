"""Prototype switches for the CP10 plan (D77). Each proposed rule of sections A to E can
be turned off or given a variant through an environment variable, so that old and new
are measured on the same code. Not part of the judge: the prototype lives outside the
repo.

    MLP_A        1 (default) | 0     the statement of what the headline rests on
    MLP_A_K      5                   shuffles for each feature column
    MLP_A_WARN   0.98                headline skill at or above which the warning is raised
    MLP_A_MIN    0.05                a column is named when its fall is at least this
    MLP_A_TOP    5                   and at most this many are named
    MLP_B        levels (default) | none
                                     the leak screen's reading of a column by its levels
    MLP_B_UNSEEN base (default) | mean
                                     what a level not seen in a fold's training rows is given
    MLP_B_ROWS   5                   a numeric column is read by its levels when it has at
                                     least this many training rows for each distinct value
    MLP_B_MAX    0                   ... and at most this many distinct values (0: any number)
    MLP_C        pairs (default) | old | mean | median | random
                                     the order reading among rows of equal column value
    MLP_D        median (default) | unmarked | old | always
                                     rule 2's unscored rows when more rows are marked than
                                     can be set aside
    MLP_E        0.5                 time order: a holdout share at or above this is named (0: off)
    MLP_DEBUG    0 | 1               keep a log of every reading in LOG
"""
import os


def _get(name, default):
    return os.environ.get(name, default)


A = _get("MLP_A", "1") == "1"
A_K = int(_get("MLP_A_K", "5"))
A_WARN = float(_get("MLP_A_WARN", "0.98"))
A_MIN = float(_get("MLP_A_MIN", "0.05"))
A_TOP = int(_get("MLP_A_TOP", "5"))
B = _get("MLP_B", "levels")
B_UNSEEN = _get("MLP_B_UNSEEN", "skip")        # D78: not scored
# D78: the fewest scored validation rows for a reading by levels to have a value
B_MIN = int(_get("MLP_B_MIN", "20"))
# ... and, for a classification, the fewest of them of each class
B_MIN_CLASS = int(_get("MLP_B_MIN_CLASS", "10"))
# D78: exact copies are shuffled together and share one fall
A_COPIES = _get("MLP_A_COPIES", "matrix")    # matrix | values | none
# D78 (3): two numeric columns of the matrix agree when within this of each other
COPY_TOLERANCE = float(_get("MLP_COPY_TOLERANCE", "1e-9"))
B_ROWS = float(_get("MLP_B_ROWS", "5"))
B_MAX = int(_get("MLP_B_MAX", "0"))
C = _get("MLP_C", "pairs")
C_FORMS = ("old", "mean", "median", "random", "pairs")
D = _get("MLP_D", "median")
E = float(_get("MLP_E", "0.5"))
DEBUG = _get("MLP_DEBUG", "0") == "1"
LOG = {}

# D80 (1): the floor of the list: the half-width of the headline skill's interval, and
# where there is none a stand-in (a number, `rows`, or `none` for A_MIN alone)
A_HALF = _get("MLP_A_HALF", "1") == "1"
A_STANDIN = _get("MLP_A_STANDIN", "rows")
# D80 (4): copies up to sign
COPY_SIGN = _get("MLP_COPY_SIGN", "1") == "1"
