"""Shared helpers of the third pass (CP4). Builds on `mlp_helpers` and
`cp3_helpers`; holds no test.

Nothing here imports `mlprofessor` at module level: each test imports the
application inside its own body.

The point of this pass (D44) is a number that is not finite turned into null,
which a helper that reads NaN as None would hide. So `assert_raw_numbers`
reads `result.to_dict()` and `result.headline` exactly as the application
returns them, and converts nothing first.
"""
import json
import math

import numpy as np
import pandas as pd

import cp3_helpers as c
import mlp_helpers as h

LARGEST_DOUBLE = 1.7976931348623157e308
LARGEST_FLOAT32 = 3.4028234663852886e38


# ---------------------------------------------------------------- numbers

def is_number(value):
    """An int or a float (numpy's included), not a bool, and finite."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return math.isfinite(float(value))


def _non_finite_floats(node, path="$"):
    """Paths of every float in `node` that is NaN or infinite."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(_non_finite_floats(value, f"{path}.{key}"))
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            found.extend(_non_finite_floats(value, f"{path}[{i}]"))
    elif isinstance(node, (float, np.floating)) and not math.isfinite(float(node)):
        found.append(f"{path} = {node!r}")
    return found


def _nonempty_text(value):
    return isinstance(value, str) and value.strip() != ""


def _assert_headline(head, doc, where):
    for key in ["score", "baseline_score", "interval_level"]:
        assert is_number(head[key]), f"{where}.{key} = {head[key]!r} is not a number"

    skill_codes = [w for w in doc["warnings"] if w["code"] == "skill_undefined"]
    if head["skill"] is None:
        assert head["baseline_score"] == 0, (
            f"{where}.skill is null, and baseline_score is {head['baseline_score']!r}, not 0")
        assert len(skill_codes) == 1, f"{where}.skill is null with {len(skill_codes)} skill_undefined warnings"
    else:
        assert is_number(head["skill"]), f"{where}.skill = {head['skill']!r} is not a number"

    if head["interval_method"] is not None:
        for key in ["interval_low", "interval_high"]:
            assert is_number(head[key]), f"{where}.{key} = {head[key]!r} is not a number"
    else:
        assert head["interval_low"] is None and head["interval_high"] is None, (
            f"{where}: no interval method, yet interval_low/high are "
            f"{head['interval_low']!r}, {head['interval_high']!r}")
        assert _nonempty_text(head["interval_reason"]), f"{where}: no interval and no interval_reason"
        assert head["skill_low"] is None and head["skill_high"] is None, (
            f"{where}: no interval method, yet skill_low/high are not null")

    lo, hi = head["skill_low"], head["skill_high"]
    assert (lo is None and hi is None) or (is_number(lo) and is_number(hi)), (
        f"{where}: skill_low, skill_high = {lo!r}, {hi!r}: both numbers or both null")

    if head["p_value"] is not None:
        assert is_number(head["p_value"]), f"{where}.p_value = {head['p_value']!r} is not a number"
        assert is_number(head["skill"]), f"{where}: a p-value is reported without a skill"
    else:
        assert head["n_permutations"] is None, f"{where}: no p-value, yet n_permutations is set"
        assert _nonempty_text(head["p_value_reason"]), f"{where}: no p-value and no p_value_reason"


def assert_raw_numbers(result):
    """Every number of a finished run is finite; a null stands only where spec 5.2 allows it."""
    doc = result.to_dict()

    bad = _non_finite_floats(doc)
    assert bad == [], f"numbers that are not finite in to_dict(): {bad}"
    json.dumps(doc, allow_nan=False)

    _assert_headline(doc["headline"], doc, "to_dict()['headline']")
    _assert_headline(result.headline, doc, "result.headline")
    bad = _non_finite_floats(dict(result.headline))
    assert bad == [], f"numbers that are not finite in result.headline: {bad}"

    for record in doc["baselines"]:
        assert is_number(record["score"]), f"baseline {record['name']}: score {record['score']!r}"

    scored = [bool(f["scored"]) for f in doc["validation"]["folds"]]
    for model in doc["models"]:
        name = model["config"]
        if model["status"] == "fitted":
            assert is_number(model["cv_mean"]), f"{name} is fitted with cv_mean {model['cv_mean']!r}"
            assert len(model["fold_scores"]) == len(scored)
            for k, (score, is_scored) in enumerate(zip(model["fold_scores"], scored)):
                if is_scored:
                    assert is_number(score), f"{name} fold {k + 1} is scored with {score!r}"
                else:
                    assert score is None, f"{name} fold {k + 1} is not scored, yet holds {score!r}"
            if sum(scored) >= 2:
                assert is_number(model["cv_std"]), f"{name} is fitted with cv_std {model['cv_std']!r}"
        else:
            assert model["status"] == "failed", f"{name}: status {model['status']!r}"
            assert _nonempty_text(model["error"]), f"{name} failed with no error text"

    assert is_number(doc["cv"]["chosen_mean"]), f"cv.chosen_mean = {doc['cv']['chosen_mean']!r}"
    assert is_number(doc["validation"]["holdout_share"]), (
        f"validation.holdout_share = {doc['validation']['holdout_share']!r}")

    y_pred = result.predictions["y_pred"].to_numpy(dtype=float)
    assert np.isfinite(y_pred).all(), "a y_pred that is not finite"


# ------------------------------------------------------- the console summary

def shown4(value, text):
    """Spec 2.2 (D46): some number printed in `text` is within 0.0005 x |value| of it.

    A value of exactly zero asks for a printed number equal to zero.
    """
    value = float(value)
    assert math.isfinite(value)
    tolerance = 0.0005 * abs(value)
    return any(abs(number - value) <= tolerance for number in c.numbers_in(text))


# ------------------------------------------------------------------- tables

def reg_table(n=400, seed=0):
    """`h.reg_frame` with 20.0 added to the target and a first column `t` of 0.0, 1.0, ..."""
    frame = h.reg_frame(n=n, seed=seed)
    frame["y"] = frame["y"] + 20.0
    frame.insert(0, "t", np.arange(n, dtype=float))
    return frame


def panel(n_times, per_time, seed=0):
    """`per_time` rows on each of `n_times` numeric timestamps `t`; y = x1 + noise.

    x1 and the noise come from one generator, one after the other.
    """
    rng = np.random.default_rng(seed)
    n = n_times * per_time
    x1 = rng.normal(size=n)
    noise = rng.normal(size=n)
    return pd.DataFrame({
        "t": np.repeat(np.arange(n_times, dtype=float), per_time),
        "x1": x1,
        "y": x1 + noise,
    })
