"""The leak screen (spec step 8): run on training rows only, before any model.

Rule 1, `missing_by_class`: a column filled in for one class and empty for the
other. Rules 2 and 3 read one measurement per column, its single-column
cross-validated skill over the selection folds, by the better of two readers
(a tree, and a robust line or a direct reading beside it): rule 2, with the
rows missed by most left unscored, excludes at 0.98 (`predicts_target`);
rule 3, every row scored, keeps the column with a `strong_single_column`
warning at 0.90. In a regression the tier has a second reading, by the
column's order, which warns and never excludes. Training rows whose target is
wildly off are set aside from the whole screen when they are few.
"""
import math

import numpy as np

from mlprofessor import metrics
from mlprofessor.columns import feature_numbers
from mlprofessor.errors import NonFiniteScoreError

LEAK_SKILL = 0.98
WARN_SKILL = 0.90
MISSING_MOST = 0.95
MISSING_FEW = 0.05
TREE_PARAMS = {"max_depth": 6, "min_samples_leaf": 5}
DIRECT_TREE_PARAMS = {"max_depth": 1, "min_samples_leaf": 1}
# Rows set aside: further than this many median deviations (or median steps).
FAR = 10
SET_ASIDE_SHARE = 0.01
# Rule 2 leaves unscored this share of the pooled rows, for the reader and the baseline.
UNSCORED_SHARE = 0.01
# The order reading.
ORDER_AGAINST = 0.10
ORDER_PAIRS = 20


def missing_by_class(missing, y, training_rows):
    """Rule 1: missing for more than 95% of one class and less than 5% of the other."""
    missing_train = np.asarray(missing, dtype=bool)[training_rows]
    y_train = np.asarray(y)[training_rows]
    shares = []
    for label in (0, 1):
        in_class = y_train == label
        if not in_class.any():
            return False
        shares.append(float(missing_train[in_class].mean()))
    low, high = sorted(shares)
    return high > MISSING_MOST and low < MISSING_FEW


def _levels(values):
    """A categorical column as the screen compares its levels: text."""
    return np.asarray(values, dtype=object).astype(str)


def _column_matrix(values, kind, train_rows, predict_rows):
    """One column as the tree sees it: numbers as they are, or codes of sorted training levels."""
    if kind == "numeric":
        column = feature_numbers(values)
        return column[train_rows].reshape(-1, 1), column[predict_rows].reshape(-1, 1)
    column = _levels(values)
    levels = np.unique(column[train_rows])
    codes = {level: float(code) for code, level in enumerate(levels)}

    def encode(rows):
        return np.array([codes.get(v, -1.0) for v in column[rows]], dtype=float).reshape(-1, 1)

    return encode(train_rows), encode(predict_rows)


def _make_tree(task_type, seed, params=TREE_PARAMS):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    if task_type == "classification":
        return DecisionTreeClassifier(random_state=seed, **params)
    return DecisionTreeRegressor(random_state=seed, **params)


def _not_finite(what, value, cause):
    return NonFiniteScoreError(
        f"The leak screen's {what} over the pooled validation rows of the selection "
        f"folds is not finite ({value!r}): {cause}")


TARGET_TOO_LARGE = ("a target value is likely too large for squared error (its square is "
                    "beyond the largest double, about 1.8e308).")


# ------------------------------------------------------------ the folds and baselines

def _scored_folds(folds, scored):
    """The (training rows, validation rows) of every fold that takes part."""
    return [(np.asarray(fold["training"]), np.asarray(fold["validation"]))
            for fold, is_scored in zip(folds, scored) if is_scored]


def _baseline_predictions(y, fold_rows, previous=None, previous_by_entity=False):
    """Pooled truth and each baseline's pooled predictions over `fold_rows`."""
    y = np.asarray(y, dtype=float)
    truth, mean_base, previous_base = [], [], []
    with np.errstate(over="ignore", invalid="ignore"):
        for train_rows, validation_rows in fold_rows:
            truth.append(y[validation_rows])
            fold_mean = float(np.mean(y[train_rows]))
            mean_base.append(np.full(len(validation_rows), fold_mean))
            if previous is not None:
                fold_previous = np.asarray(previous, dtype=float)[validation_rows]
                if previous_by_entity:
                    fold_previous = np.where(np.isnan(fold_previous), fold_mean, fold_previous)
                previous_base.append(fold_previous)
    predictions = {"training_mean": np.concatenate(mean_base)}
    if previous is not None:
        previous_pred = np.concatenate(previous_base)
        if np.isnan(previous_pred).any():
            raise RuntimeError(
                "the previous-value baseline has no value for a validation row of the "
                "selection folds; the split or the baseline is wrong")
        predictions["previous_value"] = previous_pred
    return np.concatenate(truth), predictions


def baseline_errors(y, folds, scored, previous=None, previous_by_entity=False):
    """The screen's regression baselines over the pooled validation rows of the scored folds.

    Returns `{"training_mean": error}`, with `"previous_value"` too when
    `previous` is given. Raises `NonFiniteScoreError` when an error is not
    finite. With `previous_by_entity` (an entity column), a validation row
    with no previous value is given its fold's training mean in its place;
    otherwise a missing one raises `RuntimeError`: every row of every
    validation block has one (spec step 8).
    """
    truth, predictions = _baseline_predictions(
        y, _scored_folds(folds, scored), previous, previous_by_entity)
    errors = {name: metrics.mse(truth, pred) for name, pred in predictions.items()}
    bad = [name for name, error in errors.items() if not np.isfinite(error)]
    if bad:
        raise NonFiniteScoreError(
            "The leak screen's " + " and ".join(
                f"{name} baseline ({errors[name]!r})" for name in bad)
            + " squared error over the pooled validation rows of the selection folds is "
            "not finite: " + TARGET_TOO_LARGE)
    return errors


def rows_set_aside(y, training_rows, previous=None):
    """`(rows, every_row_scored)`: the training rows set aside from the screen.

    A row is marked when its target lies more than 10 median deviations from
    the median target, or more than 10 median steps from its previous value
    (where `previous` is given and the row has one). The marked rows are set
    aside when they are at most 1% of the training rows (one being always
    allowed). When more are marked, none is, and `every_row_scored` is True:
    rule 2 then leaves no row unscored either.
    """
    training_rows = np.asarray(training_rows)
    y_train = np.asarray(y, dtype=float)[training_rows]
    marked = np.zeros(len(training_rows), dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        centre = np.median(y_train)
        distance = np.abs(y_train - centre)
        spread = np.median(distance)
        if spread > 0:
            marked |= distance > FAR * spread
        if previous is not None:
            previous_train = np.asarray(previous, dtype=float)[training_rows]
            has = ~np.isnan(previous_train)
            if has.any():
                steps = np.abs(y_train[has] - previous_train[has])
                step = np.median(steps)
                if step > 0:
                    marked[has] |= steps > FAR * step
    allowed = max(1, int(math.floor(SET_ASIDE_SHARE * len(training_rows))))
    n_marked = int(marked.sum())
    if n_marked <= allowed:
        return training_rows[marked], False
    return training_rows[:0], True


def _without_rows(fold_rows, rows):
    """`fold_rows` with `rows` out of every part; a fold left with an empty part is dropped."""
    if len(rows) == 0:
        return fold_rows
    kept = []
    for train_rows, validation_rows in fold_rows:
        train_rows = train_rows[~np.isin(train_rows, rows)]
        validation_rows = validation_rows[~np.isin(validation_rows, rows)]
        if len(train_rows) and len(validation_rows):
            kept.append((train_rows, validation_rows))
    return kept


# ------------------------------------------------------------ the readers

def _tree_reader(values, kind, y, fold_rows, task_type, seed):
    predicted = []
    for train_rows, validation_rows in fold_rows:
        train_x, predict_x = _column_matrix(values, kind, train_rows, validation_rows)
        tree = _make_tree(task_type, seed)
        tree.fit(train_x, y[train_rows])
        predicted.append(_tree_predictions(tree, predict_x, task_type))
    return np.concatenate(predicted)


def _tree_predictions(tree, predict_x, task_type):
    if task_type != "classification":
        return np.asarray(tree.predict(predict_x), dtype=float)
    proba = tree.predict_proba(predict_x)
    classes = list(tree.classes_)
    if 1 not in classes:
        return np.zeros(len(predict_x))
    return np.asarray(proba[:, classes.index(1)], dtype=float)


def robust_line(x, y):
    """The screen's line: (slope, level), or None when there is none.

    Rows sorted by `x` then `y`; with n rows and m = n // 2, row i is paired
    with row i + n - m. The slope is the median of the pairs' slopes (a pair
    of equal `x` has none), the level the median of y - slope x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return None
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    m = n // 2
    low, high = np.arange(m), np.arange(m) + (n - m)
    dx = x[high] - x[low]
    has_slope = dx != 0
    if not has_slope.any():
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        slope = float(np.median((y[high] - y[low])[has_slope] / dx[has_slope]))
        level = float(np.median(y - slope * x))
    if not (np.isfinite(slope) and np.isfinite(level)):
        return None
    return slope, level


def _line_reader(values, y, fold_rows):
    """The line's pooled predictions, or None when the line is no reader of the column."""
    column = feature_numbers(values)
    predicted = []
    with np.errstate(over="ignore", invalid="ignore"):
        for train_rows, validation_rows in fold_rows:
            fold_mean = float(np.mean(y[train_rows]))
            x_train = column[train_rows]
            present = ~np.isnan(x_train)
            line = robust_line(x_train[present], y[train_rows][present])
            x_valid = column[validation_rows]
            if line is None:
                predicted.append(np.full(len(validation_rows), fold_mean))
                continue
            slope, level = line
            predicted.append(np.where(np.isnan(x_valid), fold_mean, level + slope * x_valid))
    predicted = np.concatenate(predicted)
    if not np.all(np.isfinite(predicted)):
        return None
    return predicted


def _direct_reader(values, kind, y, fold_rows, seed):
    """A classification's column read directly, with no minimum leaf."""
    predicted = []
    if kind == "numeric":
        column = feature_numbers(values)
        for train_rows, validation_rows in fold_rows:
            tree = _make_tree("classification", seed, DIRECT_TREE_PARAMS)
            tree.fit(column[train_rows].reshape(-1, 1), y[train_rows])
            predicted.append(_tree_predictions(
                tree, column[validation_rows].reshape(-1, 1), "classification"))
        return np.concatenate(predicted)
    column = _levels(values)
    for train_rows, validation_rows in fold_rows:
        positive = np.asarray(y[train_rows]) == 1
        fold_share = float(positive.mean())
        train_levels = column[train_rows]
        shares = {}
        for level in np.unique(train_levels):
            shares[level] = float(positive[train_levels == level].mean())
        predicted.append(np.array([shares.get(v, fold_share) for v in column[validation_rows]],
                                  dtype=float))
    return np.concatenate(predicted)


# ------------------------------------------------------------ skill

def _ratio_skill(own_sum, base_sum):
    if base_sum == 0:
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return float(1.0 - np.float64(own_sum) / np.float64(base_sum))


def _scored_mask(own_errors, base_errors, every_row=False):
    """Rule 2's rows: all but the k missed by most by each of the two (k = ceil 1%)."""
    n = len(own_errors)
    keep = np.ones(n, dtype=bool)
    k = max(1, int(math.ceil(UNSCORED_SHARE * n)))
    if every_row or 2 * k >= n:
        return keep
    keep[np.argsort(-own_errors, kind="stable")[:k]] = False
    keep[np.argsort(-base_errors, kind="stable")[:k]] = False
    return keep


def _lowest(values):
    """The lower skill counts; none when any baseline gives none."""
    if any(v is None for v in values):
        return None
    return min(values)


def _regression_readings(truth, predicted, baselines, every_row=False):
    """(rule 3's skill, every row scored; rule 2's, the rows missed by most unscored)."""
    with np.errstate(over="ignore", invalid="ignore"):
        own = (truth - predicted) ** 2
        every, trimmed = [], []
        for base_pred in baselines.values():
            base = (truth - base_pred) ** 2
            every.append(_ratio_skill(own.sum(), base.sum()))
            keep = _scored_mask(own, base, every_row)
            trimmed.append(_ratio_skill(own[keep].sum(), base[keep].sum()))
    return _lowest(every), _lowest(trimmed)


def _better(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _check_skill(skill, column):
    if skill is not None and not np.isfinite(skill):
        raise _not_finite(
            f"single-column skill of {column}", skill,
            "the baseline's error is too small beside the column's own for their ratio "
            "to be held.")


def column_readings(values, kind, y, fold_rows, task_type, seed, truth, baselines=None,
                    name=None, every_row=False):
    """(rule 3's skill, rule 2's skill) of one column over `fold_rows`.

    Each is that of the better reader; None when it has no value. With
    `every_row`, rule 2 scores every row, as rule 3 does.
    """
    y = np.asarray(y)
    column = "a column" if name is None else f"column {name!r}"
    tree_pred = _tree_reader(values, kind, y, fold_rows, task_type, seed)
    if task_type == "classification":
        readers = [tree_pred, _direct_reader(values, kind, y, fold_rows, seed)]
        skill = _better([metrics.skill("roc_auc", metrics.roc_auc(truth, pred), 0.5)
                         for pred in readers])
        return skill, skill
    own_error = metrics.mse(truth, tree_pred)
    if not np.isfinite(own_error):
        raise _not_finite(f"{column}'s own squared error", own_error, TARGET_TOO_LARGE)
    readings = [_regression_readings(truth, tree_pred, baselines, every_row)]
    if kind == "numeric":
        line_pred = _line_reader(values, np.asarray(y, dtype=float), fold_rows)
        if line_pred is not None and np.isfinite(metrics.mse(truth, line_pred)):
            readings.append(_regression_readings(truth, line_pred, baselines, every_row))
    every = _better([r[0] for r in readings])
    trimmed = _better([r[1] for r in readings])
    _check_skill(every, column)
    _check_skill(trimmed, column)
    return every, trimmed


def share_against_order(x, y):
    """The smaller share of rises and falls of `y` between neighbours in `x`'s order.

    Rows missing either are left out; ties of `x` are sorted by `y` and are no
    pair. None with fewer than 20 pairs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    present = ~(np.isnan(x) | np.isnan(y))
    x, y = x[present], y[present]
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    pair = np.diff(x) != 0
    total = int(pair.sum())
    if total < ORDER_PAIRS:
        return None
    step = np.diff(y)[pair]
    return min(int((step > 0).sum()), int((step < 0).sum())) / total


def _order_share(values, y, training_rows, previous_share):
    """The order reading's share for a column, or None when it names nothing."""
    share = share_against_order(feature_numbers(values)[training_rows], y[training_rows])
    if share is None or share > ORDER_AGAINST:
        return None
    if previous_share is not None and previous_share <= ORDER_AGAINST:
        return None
    return share


def screen(features, kinds, missing, y, training_rows, folds, scored, task_type, seed,
           previous=None, previous_by_entity=False):
    """Step 8 for every feature, in file order.

    Returns one record per feature: `{"name", "leak_rule", "skill", "warn",
    "order_share"}`, `leak_rule` being `missing_by_class`, `predicts_target`
    or None; `warn` is True for a column the rule-3 reading names, and
    `order_share` is the share against the order for one the order reading
    names (else None). `previous_by_entity` says `previous` is each entity's
    own, NaN where an entity has none.
    """
    records = []
    training_rows = np.asarray(training_rows)
    fold_rows = _scored_folds(folds, scored)
    truth = baselines = None
    previous_share = None
    every_row = False
    if task_type == "classification":
        truth = np.concatenate([np.asarray(y)[v] for _, v in fold_rows])
    else:
        y = np.asarray(y, dtype=float)
        # Measured before any column, every row in, so a target too large stops first.
        baseline_errors(y, folds, scored, previous=previous,
                        previous_by_entity=previous_by_entity)
        set_aside, every_row = rows_set_aside(y, training_rows, previous)
        fold_rows = _without_rows(fold_rows, set_aside)
        truth, baselines = _baseline_predictions(y, fold_rows, previous, previous_by_entity)
        if previous is not None:
            previous_share = share_against_order(
                np.asarray(previous, dtype=float)[training_rows], y[training_rows])
    for name in features.columns:
        kind = kinds[name]
        record = {"name": name, "leak_rule": None, "skill": None, "warn": False,
                  "order_share": None}
        records.append(record)
        if task_type == "classification" and missing_by_class(missing[name], y, training_rows):
            record["leak_rule"] = "missing_by_class"
            continue
        values = features[name].to_numpy()
        every, trimmed = column_readings(values, kind, y, fold_rows, task_type, seed, truth,
                                         baselines, name=name, every_row=every_row)
        if every is None:
            # A baseline is exact on the rows scored: no skill, so no rule applies.
            continue
        if trimmed is not None and trimmed >= LEAK_SKILL:
            record["skill"] = float(trimmed)
            record["leak_rule"] = "predicts_target"
            continue
        record["skill"] = float(every)
        if every >= WARN_SKILL:
            record["warn"] = True
        elif task_type != "classification" and kind == "numeric":
            record["order_share"] = _order_share(values, y, training_rows, previous_share)
    return records


__all__ = ["screen", "column_readings", "baseline_errors", "rows_set_aside", "robust_line",
           "share_against_order", "missing_by_class", "LEAK_SKILL", "WARN_SKILL"]
