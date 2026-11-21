"""T41: no part of the output holds a number that is not finite (spec 2.3;
"Numbers that are not finite", rule 6; the fourth review, finding 4), and the
three-class message (step 4; D57).

Finding 5 (step 9, "What can fail a configuration, and what cannot") has no
test: no input reaches the tool's own scoring with predictions that are not
already one finite number for each row.
"""
import copy
import math

import numpy as np
import pandas as pd
import pytest

import cp6_helpers as g
import mlp_helpers as h


def _plant_task(doc):
    doc["task"]["classes"] = [0.0, math.inf]


def _plant_columns(doc):
    doc["columns"][0]["n_missing"] = math.nan


def _plant_input(doc):
    doc["input"]["gap"] = -math.inf


def _plant_rows(doc):
    doc["rows"][0]["count"] = math.nan


def _plant_meta(doc):
    doc["meta"]["timing"]["seconds"] = math.inf


def _plant_warnings(doc):
    doc["warnings"].append({"code": "planted", "column": None, "message": math.nan})


def _plant_headline(doc):
    doc["headline"]["score"] = math.nan


def _plant_validation(doc):
    doc["validation"]["holdout_share"] = np.float64("nan")


PLANTS = {
    "task": _plant_task, "columns": _plant_columns, "input": _plant_input,
    "rows": _plant_rows, "meta": _plant_meta, "warnings": _plant_warnings,
    "headline": _plant_headline, "validation": _plant_validation,
}


@pytest.fixture(scope="module")
def finished():
    result = g.run(h.clf_frame(200, seed=601), families=[h.family("rec", h.Log())])
    return result.to_dict(), result.predictions.copy()


@pytest.mark.parametrize("part", list(PLANTS))
def test_no_part_of_the_output_can_hold_a_number_that_is_not_finite(part, finished):
    """M18: a float that is not finite, in any part of the document, stops the constructor."""
    import mlprofessor
    from mlprofessor import errors

    doc, predictions = finished
    again = mlprofessor.ProfileResult(copy.deepcopy(doc), predictions.copy())
    assert again.to_dict() == doc

    planted = copy.deepcopy(doc)
    PLANTS[part](planted)
    with pytest.raises(errors.ProfileError) as caught:
        mlprofessor.ProfileResult(planted, predictions.copy())
    assert isinstance(caught.value, errors.NonFiniteScoreError), type(caught.value).__name__
    assert caught.value.code == "non_finite_score"


def test_the_three_class_message_does_not_suggest_declaring_the_task():
    frame = h.clf_frame(240, seed=602)
    frame["y"] = pd.Series([["red", "green", "blue"][k % 3] for k in range(240)], dtype=object)
    err = g.raised(frame, families=[h.family("rec", h.Log())])
    assert err.code == "task_ambiguous", f"{type(err).__name__}: {err}"
    text = str(err)
    assert "hree or more classes are not supported" in text, text
    assert "3" in text, text
    assert "task=" not in text, text
    assert "eclare" not in text, text


def test_a_numeric_target_of_few_values_still_names_regression():
    frame = h.clf_frame(240, seed=603)
    frame["y"] = np.arange(240) % 5
    err = g.raised(frame, families=[h.family("rec", h.Log())])
    assert err.code == "task_ambiguous", f"{type(err).__name__}: {err}"
    text = str(err)
    assert "task='regression'" in text, text
    assert "hree or more classes are not supported" in text, text
