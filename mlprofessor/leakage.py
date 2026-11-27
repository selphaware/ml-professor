"""The leak screen (spec step 8): run on training rows only, before any model.

Rule 1, `missing_by_class`: a column filled in for one class and empty for the
other. Rules 2 and 3 read one measurement per column, its single-column
cross-validated skill over the selection folds, by the better of two readers
(a tree, and a robust line or a direct reading beside it): rule 2, with the
rows missed by most left unscored, excludes at 0.98 (`predicts_target`);
rule 3, every row scored, keeps the column with a `strong_single_column`
warning at 0.90. In a regression the tier has a second reading, by the
column's order, which warns and never excludes. Training rows whose target is
wildly off are set aside from the whole screen when they are few; when they
are many, none is, and rule 2 is also read against each fold's training median.
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
# The reading by levels: rows a value for a numeric column to be read so, and the floor.
LEVEL_ROWS = 5
LEVEL_SCORED = 20
LEVEL_SCORED_CLASS = 10


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


def _median_predictions(y, fold_rows):
    """Each validation row given the median target of its fold's training rows, pooled."""
    y = np.asarray(y, dtype=float)
    pooled = [np.full(len(validation_rows), float(np.median(y[train_rows])))
              for train_rows, validation_rows in fold_rows]
    return np.concatenate(pooled) if pooled else np.zeros(0)


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
    """`(rows, many_marked)`: the training rows set aside from the screen.

    A row is marked when its target lies more than 10 median deviations from
    the median target, or more than 10 median steps from its previous value
    (where `previous` is given and the row has one). The marked rows are set
    aside when they are at most 1% of the training rows (one being always
    allowed). When more are marked, none is, and `many_marked` is True: rule 2
    is then also read against each fold's training median.
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


def _direct_reader(values, y, fold_rows, seed):
    """A numeric column of a classification read directly, with no minimum leaf."""
    predicted = []
    column = feature_numbers(values)
    for train_rows, validation_rows in fold_rows:
        tree = _make_tree("classification", seed, DIRECT_TREE_PARAMS)
        tree.fit(column[train_rows].reshape(-1, 1), y[train_rows])
        predicted.append(_tree_predictions(
            tree, column[validation_rows].reshape(-1, 1), "classification"))
    return np.concatenate(predicted)


def reads_by_levels(values, kind, training_rows):
    """Whether the column is read by its levels: every categorical column, and a numeric
    one whose training rows, where it is not missing, number at least five for each
    distinct value among them (rows >= 5 x distinct values, as d >= 9 x of step 5)."""
    if kind != "numeric":
        return True
    column = feature_numbers(values)[np.asarray(training_rows)]
    column = column[~np.isnan(column)]
    if len(column) == 0:
        return False
    return len(column) >= LEVEL_ROWS * len(np.unique(column))


def _level_codes(values, kind):
    """One whole-number code per level; -1 for a numeric cell that is missing (no level)."""
    if kind != "numeric":
        _, codes = np.unique(_levels(values), return_inverse=True)
        return np.asarray(codes, dtype=np.int64).ravel()
    column = feature_numbers(values)
    codes = np.full(len(column), -1, dtype=np.int64)
    present = ~np.isnan(column)
    if present.any():
        _, inverse = np.unique(column[present], return_inverse=True)
        codes[present] = np.asarray(inverse).ravel()
    return codes


def _level_reader(values, kind, y, fold_rows, task_type):
    """(pooled predictions, scored): each validation row is given the mean target of its
    fold's training rows of its level; a row whose level they do not hold is not scored."""
    codes = _level_codes(values, kind)
    n_levels = int(codes.max()) + 1 if len(codes) else 0
    if task_type == "classification":
        target = (np.asarray(y) == 1).astype(float)
    else:
        target = np.asarray(y, dtype=float)
    predicted, scored = [], []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for train_rows, validation_rows in fold_rows:
            train_codes = codes[train_rows]
            held = train_codes >= 0
            counts = np.bincount(train_codes[held], minlength=n_levels)
            sums = np.bincount(train_codes[held], weights=target[train_rows][held],
                               minlength=n_levels)
            valid_codes = codes[validation_rows]
            is_scored = valid_codes >= 0
            is_scored[is_scored] = counts[valid_codes[is_scored]] > 0
            fold_pred = np.zeros(len(validation_rows), dtype=float)
            at = valid_codes[is_scored]
            fold_pred[is_scored] = sums[at] / counts[at]
            predicted.append(fold_pred)
            scored.append(is_scored)
    if not predicted:
        return np.zeros(0), np.zeros(0, dtype=bool)
    return np.concatenate(predicted), np.concatenate(scored)


def _level_floor(truth, scored, task_type):
    """The floor of the reading by levels: 20 pooled rows scored and, in a classification,
    10 of each class scored or every pooled row of that class."""
    if int(scored.sum()) < LEVEL_SCORED:
        return False
    if task_type != "classification":
        return True
    for label in (0, 1):
        in_class = truth == label
        n_scored = int((in_class & scored).sum())
        if n_scored < LEVEL_SCORED_CLASS and n_scored != int(in_class.sum()):
            return False
    return True


# ------------------------------------------------------------ skill

def _ratio_skill(own_sum, base_sum):
    if base_sum == 0:
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return float(1.0 - np.float64(own_sum) / np.float64(base_sum))


def _scored_mask(own_errors, base_errors):
    """Rule 2's rows: all but the k missed by most by each of the two (k = ceil 1%)."""
    n = len(own_errors)
    keep = np.ones(n, dtype=bool)
    k = max(1, int(math.ceil(UNSCORED_SHARE * n)))
    if 2 * k >= n:
        return keep
    keep[np.argsort(-own_errors, kind="stable")[:k]] = False
    keep[np.argsort(-base_errors, kind="stable")[:k]] = False
    return keep


def _lowest(values):
    """The lower skill counts; none when any baseline gives none."""
    if any(v is None for v in values):
        return None
    return min(values)


def _regression_readings(truth, predicted, baselines, rule2_baselines=None):
    """(rule 3's skill, every row scored; rule 2's, the rows missed by most unscored).

    Rule 2 is read against `baselines` and against `rule2_baselines` as well.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        own = (truth - predicted) ** 2
        every, trimmed = [], []
        for base_pred in baselines.values():
            base = (truth - base_pred) ** 2
            every.append(_ratio_skill(own.sum(), base.sum()))
        for base_pred in list(baselines.values()) + list((rule2_baselines or {}).values()):
            base = (truth - base_pred) ** 2
            keep = _scored_mask(own, base)
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


def _auc_skill(truth, predicted):
    return metrics.skill("roc_auc", metrics.roc_auc(truth, predicted), 0.5)


def column_readings(values, kind, y, fold_rows, task_type, seed, truth, baselines=None,
                    name=None, rule2_baselines=None, by_levels=False):
    """(rule 3's skill, rule 2's skill) of one column over `fold_rows`.

    Each is that of the better reader; None when it has no value. In a
    regression rule 2 is also read against `rule2_baselines` (pooled
    predictions in the order of `truth`). With `by_levels` the column is also
    read by its levels, over the rows that reading scores.
    """
    y = np.asarray(y)
    truth = np.asarray(truth)
    column = "a column" if name is None else f"column {name!r}"
    tree_pred = _tree_reader(values, kind, y, fold_rows, task_type, seed)
    level_pred = scored = None
    if by_levels:
        level_pred, scored = _level_reader(values, kind, y, fold_rows, task_type)
        if not _level_floor(truth, scored, task_type):
            level_pred = None
    if task_type == "classification":
        skills = [_auc_skill(truth, tree_pred)]
        if kind == "numeric":
            skills.append(_auc_skill(truth, _direct_reader(values, y, fold_rows, seed)))
        if level_pred is not None and len(np.unique(truth[scored])) == 2:
            skills.append(_auc_skill(truth[scored], level_pred[scored]))
        skill = _better(skills)
        return skill, skill
    own_error = metrics.mse(truth, tree_pred)
    if not np.isfinite(own_error):
        raise _not_finite(f"{column}'s own squared error", own_error, TARGET_TOO_LARGE)
    rule2_baselines = rule2_baselines or {}
    readings = [_regression_readings(truth, tree_pred, baselines, rule2_baselines)]
    if kind == "numeric":
        line_pred = _line_reader(values, np.asarray(y, dtype=float), fold_rows)
        if line_pred is not None and np.isfinite(metrics.mse(truth, line_pred)):
            readings.append(_regression_readings(truth, line_pred, baselines, rule2_baselines))
    if level_pred is not None:
        own = level_pred[scored]
        if np.all(np.isfinite(own)) and np.isfinite(metrics.mse(truth[scored], own)):
            readings.append(_regression_readings(
                truth[scored], own,
                {base: pred[scored] for base, pred in baselines.items()},
                {base: pred[scored] for base, pred in rule2_baselines.items()}))
    every = _better([r[0] for r in readings])
    trimmed = _better([r[1] for r in readings])
    _check_skill(every, column)
    _check_skill(trimmed, column)
    return every, trimmed


def share_against_order(x, y):
    """The smaller average share of rises and falls of `y` between rows of neighbouring
    distinct values of `x`.

    Rows missing either are left out. For each two neighbouring distinct values
    every row of the one is compared with every row of the next (equal targets
    count in neither share); each pair of values counts once. Rows of equal `x`
    are never compared. None with fewer than 20 pairs of values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    present = ~(np.isnan(x) | np.isnan(y))
    x, y = x[present], y[present]
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    if len(x) == 0:
        return None
    starts = np.flatnonzero(np.r_[True, np.diff(x) != 0])
    ends = np.r_[starts[1:], len(x)]
    pairs = len(starts) - 1
    if pairs < ORDER_PAIRS:
        return None
    sizes = ends - starts
    group = np.repeat(np.arange(len(starts), dtype=np.int64), sizes)
    # Inside each value the targets are sorted, so (value, rank of target) is one
    # sorted key, and each row's place among the next value's rows is a search.
    rank = np.unique(y, return_inverse=True)[1].astype(np.int64)
    width = int(rank.max()) + 1
    key = group * width + rank
    low_rows = group < len(starts) - 1
    probe = (group[low_rows] + 1) * width + rank[low_rows]
    nxt = group[low_rows] + 1
    below = np.searchsorted(key, probe, side="left") - starts[nxt]
    above = ends[nxt] - np.searchsorted(key, probe, side="right")
    totals = (sizes[:-1] * sizes[1:]).astype(float)
    rises = np.bincount(group[low_rows], weights=above, minlength=pairs)[:pairs]
    falls = np.bincount(group[low_rows], weights=below, minlength=pairs)[:pairs]
    rise_sum = float(np.sum(rises / totals))
    fall_sum = float(np.sum(falls / totals))
    return min(rise_sum, fall_sum) / pairs


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
    rule2_baselines = None
    if task_type == "classification":
        truth = np.concatenate([np.asarray(y)[v] for _, v in fold_rows])
    else:
        y = np.asarray(y, dtype=float)
        # Measured before any column, every row in, so a target too large stops first.
        baseline_errors(y, folds, scored, previous=previous,
                        previous_by_entity=previous_by_entity)
        set_aside, many_marked = rows_set_aside(y, training_rows, previous)
        fold_rows = _without_rows(fold_rows, set_aside)
        truth, baselines = _baseline_predictions(y, fold_rows, previous, previous_by_entity)
        if many_marked:
            # Rows far off cannot move a median: rule 2 is read against it as well.
            rule2_baselines = {"training_median": _median_predictions(y, fold_rows)}
        if previous is not None:
            previous_share = share_against_order(
                np.asarray(previous, dtype=float)[training_rows], y[training_rows])
    for name in features.columns:
        kind = kinds[name]
        record = {"name": name, "leak_rule": None, "skill": None, "warn": False,
                  "order_share": None, "by_levels": False}
        records.append(record)
        if task_type == "classification" and missing_by_class(missing[name], y, training_rows):
            record["leak_rule"] = "missing_by_class"
            continue
        values = features[name].to_numpy()
        record["by_levels"] = reads_by_levels(values, kind, training_rows)
        every, trimmed = column_readings(values, kind, y, fold_rows, task_type, seed, truth,
                                         baselines, name=name,
                                         rule2_baselines=rule2_baselines,
                                         by_levels=record["by_levels"])
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
