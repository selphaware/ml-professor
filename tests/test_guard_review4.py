"""T39: guards of the fifth pass (spec steps 2 and 5, 2.3, 2.6; D57). Green today.

Each holds what the fourth review pinned and the application already does. A
failure means the application is wrong, not the test.
"""
import decimal

import numpy as np
import pandas as pd

import cp3_helpers as c
import cp6_helpers as g
import mlp_helpers as h


def _rec():
    return [h.family("rec", h.Log())]


def test_m20_infinite_cells_of_the_target_and_the_time_column_are_counted():
    """M20: the infinite cells of the target and of the time column count in `n_infinite`."""
    rng = np.random.default_rng(401)
    n = 240
    frame = pd.DataFrame({
        "t": np.arange(n, dtype=float),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    frame["y"] = 2.0 * frame["x1"] + rng.normal(size=n)
    frame.loc[[11, 97], "y"] = np.inf
    frame.loc[[30, 150, 201], "t"] = -np.inf
    result = g.run(frame, families=_rec(), time_col="t")
    cols = h.columns_of(result)
    assert (cols["y"]["n_infinite"], cols["y"]["n_missing"]) == (2, 0), cols["y"]
    assert (cols["t"]["n_infinite"], cols["t"]["n_missing"]) == (3, 0), cols["t"]
    rows = h.rows_of(result)
    assert rows["dropped_missing_target"] == 2, rows
    assert rows["dropped_missing_time"] == 3, rows
    assert g.count_code(result, "infinite_values") == 0, h.warnings_of(result)


def test_an_infinite_float_in_an_object_target_is_a_missing_target():
    frame = h.clf_frame(200, seed=402, labels=("a", "b"))
    target = list(frame["y"])
    target[5], target[50], target[120] = float("inf"), np.float64("inf"), -np.inf
    target[77] = None
    frame["y"] = pd.Series(target, dtype=object)
    result = g.run(frame, families=_rec())
    assert h.rows_of(result)["dropped_missing_target"] == 4
    record = h.columns_of(result)["y"]
    assert record["n_infinite"] == 3, record
    assert record["n_missing"] == 1, record
    assert record["dtype"] == "object", record
    assert result.to_dict()["task"]["classes"] == ["a", "b"]


def test_an_object_target_of_numbers_with_an_infinite_one():
    frame = h.reg_frame(200, seed=403)
    values = [decimal.Decimal(str(round(float(v), 6))) for v in frame["y"]]
    values[42] = decimal.Decimal("-Infinity")
    frame["y"] = pd.Series(values, dtype=object)
    result = g.run(frame, families=_rec())
    assert result.to_dict()["task"]["type"] == "regression"
    assert h.rows_of(result)["dropped_missing_target"] == 1
    assert h.columns_of(result)["y"]["n_infinite"] == 1


def test_a_class_beside_infinite_floats_alone_is_a_constant_target():
    frame = h.clf_frame(200, seed=404)
    target = ["a" if k % 3 else float("inf") for k in range(200)]
    frame["y"] = pd.Series(target, dtype=object)
    err = g.raised(frame, families=_rec())
    assert err.code == "target_constant", f"{type(err).__name__}: {err}"


def test_a_result_can_be_made_from_its_own_document():
    import mlprofessor

    result = g.run(h.clf_frame(200, seed=405), families=_rec())
    again = mlprofessor.ProfileResult(result.to_dict(), result.predictions)
    assert again.to_dict() == result.to_dict()
    assert again.headline == result.headline
    pd.testing.assert_frame_equal(again.predictions.reset_index(drop=True),
                                  result.predictions.reset_index(drop=True))


class _ColumnEstimator(h.LookupEstimator):
    def predict(self, X):
        return super().predict(X).reshape(-1, 1)


class _ShortEstimator(h.LookupEstimator):
    def predict(self, X):
        return super().predict(X)[:-1]


def test_a_regression_model_may_return_a_column_of_values():
    frame = h.reg_frame(200, seed=406)
    values = frame["y"].to_numpy() + np.random.default_rng(407).normal(size=200)
    log = h.Log()
    families = [
        h.family("lookup", log, estimator=h.LookupEstimator, values=values),
        h.family("column", log, estimator=_ColumnEstimator, values=values),
        h.family("short", log, estimator=_ShortEstimator, values=values),
    ]
    result = g.run(frame, families=families)
    status = {m["config"]: m["status"] for m in h.models_of(result)}
    assert status == {"lookup-0": "fitted", "column-0": "fitted", "short-0": "failed"}, status
    assert c.count_code(result, "model_failed") == 1
