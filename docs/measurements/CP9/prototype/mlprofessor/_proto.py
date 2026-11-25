"""Prototype switches for the CP9 plan. Each rule of section A can be turned on
or off, or given a variant, through an environment variable, so that old and new
are measured on the same code. Not part of the judge: the prototype lives outside
the repo and is thrown away.

    MLP_A1      1 (default) | 0     previous value at repeated timestamps
    MLP_LINE    theil_ls (default) | none | ls | ls2 | theil
                                    the straight line beside the screen's tree
    MLP_LINE    theil (default) | none | ls | ls2 | theil_ls
    MLP_TRIM    thresh (default) | count | none
                                    which pooled validation rows are left out of the
                                    screen's skill
    MLP_TRIM_C  100                 thresh: an error above this many times the
                                    baseline's median squared error is outlying
    MLP_TRIM_CAP 0.05               thresh: at most this share of the rows is left out
    MLP_TRIM_SHARE 0.01             count: this share, by each error
    MLP_A3      1 (default) | 0     the largest row's share of the squared error
    MLP_SHARE   0.5                 the cut-off of the warning
    MLP_A5      dense (default) | old | dense_or_ordered
    MLP_DENSE   0.5                 share of its range an identifier fills
    MLP_A6      units (default) | old | units_pooled
"""
import os


def _get(name, default):
    return os.environ.get(name, default)


A1 = _get("MLP_A1", "1") == "1"
LINE = _get("MLP_LINE", "theil")
TRIM = _get("MLP_TRIM", "thresh")
TRIM_C = float(_get("MLP_TRIM_C", "100"))
TRIM_K = float(_get("MLP_TRIM_K", "10"))     # median deviations, or median steps
TRIM_CAP = float(_get("MLP_TRIM_CAP", "0.01"))
TRIM_SHARE = float(_get("MLP_TRIM_SHARE", "0.01"))
# which rows a column's own prediction may leave unscored: count | thresh | none
DROP = _get("MLP_DROP", "count")
A3 = _get("MLP_A3", "1") == "1"
SHARE = float(_get("MLP_SHARE", "0.5"))
A5 = _get("MLP_A5", "dense_or_ordered")
DENSE = float(_get("MLP_DENSE", "0.5"))
# units: a fold that trains on fewer rows than it validates on is not scored
# floor: a fold that trains on fewer than A6_FLOOR rows is not scored
# floor_first: floor, and when the scored folds validate on fewer than A6_FLOOR
#   rows in all, no choice is made: the first configuration is used
# pooled: every fold scored, and a configuration is ranked by its squared error
#   over all validation rows together (regression)
# nochoice: every fold is scored as it is; when a fold trains on fewer than A6_FLOOR
#   rows (the other folds then validate on those same few rows), no choice is
#   made: the first configuration is used
A6 = _get("MLP_A6", "nochoice")
A6_FLOOR = int(_get("MLP_A6_FLOOR", "50"))
A3CV = _get("MLP_A3CV", "1") == "1"      # the same statement over the selection folds
DEBUG = _get("MLP_DEBUG", "0") == "1"
VARIANTS = _get("MLP_VARIANTS", "0") == "1"  # log the screen under every variant
LOG = {}
LAST = {}
# classification: beside the tree, the column read directly (one split with no
# minimum leaf; the share of each level)
DIRECT = _get("MLP_DIRECT", "1") == "1"
# which skill the warning tier (0.90) reads: every_row | as_decided
TELL = _get("MLP_TELL", "every_row")
LAST_TELL = None
# D74 (3): a value far outside all the other values of its column
EXTREME = _get("MLP_EXTREME", "1") == "1"
EXTREME_K = float(_get("MLP_EXTREME_K", "30"))
EXTREME_M = int(_get("MLP_EXTREME_M", "10"))
EXTREME_J = int(_get("MLP_EXTREME_J", "3"))
# D74 (2): each column also read with the chosen configuration, for the warning only
CHOSEN = _get("MLP_CHOSEN", "0") == "1"
# proposed, off by default: a warning when the target follows the order of a column
ORDER = _get("MLP_ORDER", "1") == "1"
ORDER_CUT = float(_get("MLP_ORDER_CUT", "0.10"))
