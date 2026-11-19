"""The leak screen (spec step 8): run on training rows only, before any model.

Rule 1, `missing_by_class`: a column filled in for one class and empty for the
other. Rules 2 and 3 share one number per column, its single-column
cross-validated skill over the selection folds: at least 0.98 excludes it
(`predicts_target`); at least 0.90 keeps it with a `strong_single_column`
warning.
"""
import numpy as np
import pandas as pd

from mlprofessor import metrics

LEAK_SKILL = 0.98
WARN_SKILL = 0.90
MISSING_MOST = 0.95
MISSING_FEW = 0.05
TREE_PARAMS = {"max_depth": 6, "min_samples_leaf": 5}


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


def _column_matrix(values, kind, train_rows, predict_rows):
    """One column as the tree sees it: numbers as they are, or codes of sorted training levels."""
    if kind == "numeric":
        column = np.asarray(values, dtype=float)
        return column[train_rows].reshape(-1, 1), column[predict_rows].reshape(-1, 1)
    column = np.asarray(values, dtype=object).astype(str)
    levels = np.unique(column[train_rows])
    codes = {level: float(code) for code, level in enumerate(levels)}

    def encode(rows):
        return np.array([codes.get(v, -1.0) for v in column[rows]], dtype=float).reshape(-1, 1)

    return encode(train_rows), encode(predict_rows)


def _make_tree(task_type, seed):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    if task_type == "classification":
        return DecisionTreeClassifier(random_state=seed, **TREE_PARAMS)
    return DecisionTreeRegressor(random_state=seed, **TREE_PARAMS)


def single_column_skill(values, kind, y, folds, scored, task_type, seed, previous=None):
    """Pooled cross-validated skill of one column over the scored selection folds.

    `previous`, when given (regression, time order, unique timestamps), holds
    for every row position the previous-value baseline; the baseline used is
    whichever of it and each fold's training mean has the lower pooled error.
    None when that baseline's pooled error is exactly zero: skill has no value.
    """
    y = np.asarray(y)
    truth, predicted, mean_base, previous_base = [], [], [], []
    for fold, is_scored in zip(folds, scored):
        if not is_scored:
            continue
        train_rows, validation_rows = fold["training"], fold["validation"]
        train_x, predict_x = _column_matrix(values, kind, train_rows, validation_rows)
        tree = _make_tree(task_type, seed)
        tree.fit(train_x, y[train_rows])
        if task_type == "classification":
            proba = tree.predict_proba(predict_x)
            positive = list(tree.classes_).index(1) if 1 in tree.classes_ else None
            values_pred = (proba[:, positive] if positive is not None
                           else np.zeros(len(validation_rows)))
        else:
            values_pred = tree.predict(predict_x)
            mean_base.append(np.full(len(validation_rows), float(np.mean(y[train_rows]))))
            if previous is not None:
                previous_base.append(previous[validation_rows])
        truth.append(y[validation_rows])
        predicted.append(np.asarray(values_pred, dtype=float))
    truth = np.concatenate(truth)
    predicted = np.concatenate(predicted)
    if task_type == "classification":
        return metrics.skill("roc_auc", metrics.roc_auc(truth, predicted), 0.5)
    baseline_error = metrics.mse(truth, np.concatenate(mean_base))
    if previous_base:
        previous_pred = np.concatenate(previous_base)
        if np.all(np.isfinite(previous_pred)):
            baseline_error = min(baseline_error, metrics.mse(truth, previous_pred))
    return metrics.skill("mse", metrics.mse(truth, predicted), baseline_error)


def screen(features, kinds, missing, y, training_rows, folds, scored, task_type, seed,
           previous=None):
    """Step 8 for every feature, in file order.

    Returns one record per feature: `{"name", "leak_rule", "skill", "warn"}`,
    `leak_rule` being `missing_by_class`, `predicts_target` or None.
    """
    records = []
    for name in features.columns:
        kind = kinds[name]
        record = {"name": name, "leak_rule": None, "skill": None, "warn": False}
        if task_type == "classification" and missing_by_class(missing[name], y, training_rows):
            record["leak_rule"] = "missing_by_class"
            records.append(record)
            continue
        values = features[name].to_numpy()
        skill = single_column_skill(values, kind, y, folds, scored, task_type, seed,
                                    previous=previous)
        if skill is None:
            # The baseline is exact on the pooled rows: no skill, so no rule applies.
            records.append(record)
            continue
        record["skill"] = float(skill)
        if skill >= LEAK_SKILL:
            record["leak_rule"] = "predicts_target"
        elif skill >= WARN_SKILL:
            record["warn"] = True
        records.append(record)
    return records


__all__ = ["screen", "single_column_skill", "missing_by_class", "LEAK_SKILL", "WARN_SKILL"]
