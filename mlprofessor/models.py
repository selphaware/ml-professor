"""Model families (spec 4.1), predictions on a preprocessed matrix, and selection (step 9).

A family is a mapping `{"name", "grid", "make"}`; `make(task, params, seed, n_jobs)`
returns a new unfitted estimator. The same shape serves the built-in families
and those given through `extensions.use`.
"""
import numpy as np

from mlprofessor import extensions

CLASSIFICATION = "classification"
REGRESSION = "regression"


def _make_linear(task, params, seed, n_jobs):
    from sklearn.linear_model import LogisticRegression, Ridge

    if task == CLASSIFICATION:
        return LogisticRegression(max_iter=1000, **params)
    return Ridge(**params)


def _make_tree(task, params, seed, n_jobs):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    if task == CLASSIFICATION:
        return DecisionTreeClassifier(random_state=seed, **params)
    return DecisionTreeRegressor(random_state=seed, **params)


def _make_forest(task, params, seed, n_jobs):
    """A forest fitted with `n_jobs` threads and predicting on one."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    base = RandomForestClassifier if task == CLASSIFICATION else RandomForestRegressor

    class _Forest(base):
        def predict(self, X):
            self.n_jobs = 1
            return super().predict(X)

        def predict_proba(self, X):
            self.n_jobs = 1
            return super().predict_proba(X)

    return _Forest(n_estimators=100, random_state=seed, n_jobs=n_jobs, **params)


def forest_maker():
    """The `make` of the built-in forest family: its shuffles are predicted side by side."""
    return _make_forest


def _make_boosting(task, params, seed, n_jobs):
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)

    cls = (HistGradientBoostingClassifier if task == CLASSIFICATION
           else HistGradientBoostingRegressor)
    return cls(random_state=seed, early_stopping=False, **params)


# The grids of spec 4.1, index 0 (the family's default) first.
_LINEAR_GRID = {
    CLASSIFICATION: (("C",), [(1.0,), (0.001,), (0.01,), (0.1,), (10.0,), (100.0,)]),
    REGRESSION: (("alpha",), [(1.0,), (0.01,), (0.1,), (10.0,), (100.0,), (1000.0,)]),
}
_FOREST_GRID = {
    CLASSIFICATION: [(1, "sqrt"), (1, 1 / 3), (5, "sqrt"), (5, 1 / 3), (20, "sqrt"), (20, 1 / 3)],
    REGRESSION: [(1, 1 / 3), (1, "sqrt"), (5, 1 / 3), (5, "sqrt"), (20, 1 / 3), (20, "sqrt")],
}
_BOOSTING_GRID = [(31, 0.1, 100), (31, 0.05, 300), (7, 0.1, 100),
                  (7, 0.05, 300), (63, 0.1, 100), (63, 0.05, 300)]


def _grid(names, rows):
    """One params mapping per grid row."""
    return [dict(zip(names, row)) for row in rows]


def builtin_families(task, profile):
    """The profile's own families, in tie-break order."""
    linear_default = {"C": 1.0} if task == CLASSIFICATION else {"alpha": 1.0}
    if profile == "fast":
        return [
            {"name": "linear", "grid": [linear_default], "make": _make_linear},
            {"name": "tree", "grid": [{}], "make": _make_tree},
        ]
    if profile == "full":
        return [
            {"name": "linear", "grid": _grid(*_LINEAR_GRID[task]), "make": _make_linear},
            {"name": "forest", "grid": _grid(("min_samples_leaf", "max_features"),
                                             _FOREST_GRID[task]), "make": _make_forest},
            {"name": "boosting", "grid": _grid(("max_leaf_nodes", "learning_rate", "max_iter"),
                                               _BOOSTING_GRID), "make": _make_boosting},
        ]
    raise NotImplementedError(f"the {profile!r} profile's families are not built yet")


def active_families(task, profile):
    """The families in force: those of `extensions.use` when given, else the profile's."""
    override = extensions.current_overrides()["families"]
    if override is not None:
        return [dict(family) for family in override]
    return builtin_families(task, profile)


def configurations(families):
    """One record per configuration, in family order then grid order."""
    configs = []
    for family in families:
        for index, params in enumerate(family["grid"]):
            configs.append({
                "config": f"{family['name']}-{index}",
                "family": family["name"],
                "params": dict(params),
                "make": family["make"],
            })
    return configs


def predict_values(estimator, matrix, task):
    """Classification: probability of the positive class. Regression: the prediction.

    Raises `ValueError` when the estimator returns a wrong shape or a value
    that is not finite, so the configuration is recorded as failed.
    """
    if task == CLASSIFICATION:
        proba = np.asarray(estimator.predict_proba(matrix), dtype=float)
        if proba.ndim != 2 or proba.shape != (len(matrix), 2):
            raise ValueError(
                f"predict_proba returned shape {proba.shape}; expected ({len(matrix)}, 2)")
        values = proba[:, 1].copy()
    else:
        values = np.asarray(estimator.predict(matrix), dtype=float).reshape(-1)
        if values.shape != (len(matrix),):
            raise ValueError(
                f"predict returned {values.size} values for {len(matrix)} rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("the model predicted a value that is not finite")
    return values


def ranked_configurations(model_records, greater_is_better):
    """Indices of the fitted configurations, best `cv_mean` first; ties keep table order."""
    fitted = [i for i, record in enumerate(model_records)
              if record["status"] == "fitted" and record["cv_mean"] is not None]
    sign = -1.0 if greater_is_better else 1.0
    return sorted(fitted, key=lambda i: (sign * model_records[i]["cv_mean"], i))
