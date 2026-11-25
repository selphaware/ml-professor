"""T66: no choice between configurations where the folds cannot choose (spec step 9, 2.5,
4.2, 5.3, 7, 10.10, 11; D38, D74 point 6).

Under the grouped scheme, when some fold has fewer than 50 training rows and fewer
training rows than validation rows, every fold is still scored and reported, the
configurations are ranked in configuration order, and one `selection_not_made`
warning is raised. The counts are read from `validation.folds`; which units land in
which fold is open (spec 11), so each table meets the condition, or fails it, however
the units are dealt.
"""
import numpy as np
import pandas as pd
import pytest

import cp3_helpers as c
import cp7_helpers as s
import cp8_helpers as e
import cp9_helpers as q
import mlp_helpers as h

CODE = "selection_not_made"


def _guess(frame, seed, spread):
    y = frame["y"].to_numpy(dtype=float)
    return y + spread * np.random.default_rng([seed, int(spread * 1000)]).normal(size=len(y))


def _families(frame, *spreads, log=None, seed=0):
    """One lookup family for each spread, `a`, `b`, ...: the target plus that noise."""
    log = log or h.Log()
    return [h.family("abcdefgh"[k], log, estimator=h.LookupEstimator,
                     values=_guess(frame, seed + k, spread))
            for k, spread in enumerate(spreads)]


def _starved(result):
    """The folds with fewer than 50 training rows and fewer than validation rows."""
    return [f for f in result.to_dict()["validation"]["folds"]
            if f["training_rows"] < q.SELECTION_FLOOR
            and f["training_rows"] < f["validation_rows"]]


def _by_name(result):
    return {r["config"]: r for r in result.to_dict()["models"]}


def _assert_no_choice(result, first="a-0", better="b-0"):
    doc = result.to_dict()
    starved = _starved(result)
    assert len(starved) == 1, doc["validation"]["folds"]
    assert starved[0]["validation_rows"] >= 1490
    assert doc["cv"]["n_scored_folds"] == 5
    models = _by_name(result)
    for record in models.values():
        assert record["status"] == "fitted"
        assert len(record["fold_scores"]) == 5
        assert all(v is not None for v in record["fold_scores"])
    assert models[better]["cv_mean"] < 0.1 * models[first]["cv_mean"]
    assert doc["headline"]["model"] == first and doc["cv"]["chosen"] == first
    assert models[first]["chosen"] and not models[better]["chosen"]
    messages = q.told(result, CODE)
    assert len(messages) == 1, s.codes(result)
    assert [r["column"] for r in s.warning_records(result, CODE)] == [None]
    assert q.holds_row(messages[0], starved[0]["training_rows"]), messages[0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_entity_of_nearly_all_training_rows_and_no_choice_is_made(seed):
    """F28: beside one entity of nearly all the training rows the folds choose nothing:
    the first configuration is used."""
    frame = q.dominant_entity(seed=seed)
    result = s.run(frame, families=_families(frame, 3.0, 0.1), time_col=False,
                   group_col="who")
    assert result.to_dict()["validation"]["scheme"] == "grouped"
    _assert_no_choice(result)


def test_the_first_configuration_that_fits_is_used():
    """F28: `a` raises on the final fit; the next in configuration order is used, not the
    one with the best fold scores."""
    frame = q.dominant_entity(seed=3)
    statements = {"time_col": False, "group_col": "who"}
    held = h.holdout_rows(s.run(frame, families=[h.family("rec", h.Log())], **statements))
    log = h.Log()
    failing = h.family("a", log, estimator=c.FailOnCommand, values=_guess(frame, 1, 3.0),
                       fail_predict_rows=held)
    families = [failing]
    families += [h.family("b", log, estimator=h.LookupEstimator, values=_guess(frame, 2, 2.0)),
                 h.family("c", log, estimator=h.LookupEstimator, values=_guess(frame, 3, 0.1))]
    result = s.run(frame, families=families, **statements)
    models = _by_name(result)
    assert models["a-0"]["status"] == "failed" and models["a-0"]["fold_scores"][0] is not None
    assert models["c-0"]["cv_mean"] < models["b-0"]["cv_mean"]
    assert result.to_dict()["headline"]["model"] == "b-0"
    assert models["b-0"]["chosen"] and not models["c-0"]["chosen"]
    assert len(q.told(result, CODE)) == 1


def test_a_configuration_with_no_fold_scores_is_not_the_first():
    """F28: a family that fails in every fold comes first; the first configuration that
    has a `cv_mean` is used."""
    frame = q.dominant_entity(seed=4)
    log = h.Log()
    families = [h.family("bad", log, estimator=h.FailingEstimator)]
    families += _families(frame, 3.0, 0.1, log=log)
    result = s.run(frame, families=families, time_col=False, group_col="who")
    models = _by_name(result)
    assert models["bad-0"]["status"] == "failed" and models["bad-0"]["cv_mean"] is None
    assert result.to_dict()["headline"]["model"] == "a-0"
    assert models["b-0"]["cv_mean"] < models["a-0"]["cv_mean"]
    assert len(q.told(result, CODE)) == 1


def test_rows_that_agree_in_every_feature_are_units_of_the_rule_too():
    """F28: under `duplicates="group"` 1,490 rows alike in every feature are one unit."""
    frame = q.dominant_entity(seed=5)
    measures = ["n1", "n2", "n3", "n4"]
    first = frame.groupby("who")[measures].transform("first")
    frame[measures] = first
    frame = frame.drop(columns=["who"])
    assert frame.groupby(measures).size().max() == 1490
    result = s.run(frame, families=_families(frame, 3.0, 0.1), duplicates="group", **s.SAID)
    assert result.to_dict()["validation"]["scheme"] == "grouped"
    _assert_no_choice(result)


def test_a_small_fold_that_trains_on_as_many_rows_as_it_validates_on_is_chosen_on():
    """M48: a fold of fewer than 50 training rows that trains on at least as many rows as
    it validates on does not meet the rule: the choice is made."""
    frame = s.sized_entities([30] + [1] * 74, seed=9021)
    result = s.run(frame, families=_families(frame, 3.0, 0.1), time_col=False,
                   group_col="who")
    folds = result.to_dict()["validation"]["folds"]
    assert sum(f["validation_rows"] for f in folds) == h.rows_of(result)["training"] == 78
    big = [f for f in folds if f["validation_rows"] >= 30]
    assert len(big) == 1 and 39 <= big[0]["training_rows"] <= 48, folds
    assert big[0]["training_rows"] >= big[0]["validation_rows"]
    assert _starved(result) == []
    assert CODE not in s.codes(result)
    assert result.to_dict()["cv"]["chosen"] == "b-0"


def test_an_ordinary_table_of_entities_is_chosen_on_as_before():
    frame = s.sized_entities([5] * 120, seed=9031)
    result = s.run(frame, families=_families(frame, 3.0, 0.1), time_col=False,
                   group_col="who")
    assert _starved(result) == []
    assert CODE not in s.codes(result)
    assert result.to_dict()["cv"]["chosen"] == "b-0"


def test_the_rule_is_for_the_grouped_scheme_only():
    """Under time order a first fold that trains on fewer rows than it validates on, and
    under the random scheme a small table: the choice is made."""
    rng = np.random.default_rng(9041)
    stamps = list(range(13)) + [13 + k for k in range(77) for _ in range(5)]
    timed = pd.DataFrame({"t": np.array(stamps, dtype=np.int64),
                          "n1": rng.normal(size=len(stamps)),
                          "y": rng.normal(size=len(stamps))})
    result = s.run(timed, families=_families(timed, 3.0, 0.1), time_col="t", group_col=False)
    first = result.to_dict()["validation"]["folds"][0]
    assert first["training_rows"] < q.SELECTION_FLOOR
    assert first["training_rows"] < first["validation_rows"]
    assert CODE not in s.codes(result)
    assert result.to_dict()["cv"]["chosen"] == "b-0"

    small = h.reg_frame(n=104, seed=9042)
    result = s.run(small, families=_families(small, 3.0, 0.1), **s.SAID)
    assert result.to_dict()["validation"]["scheme"] == "random"
    assert CODE not in s.codes(result)
    assert result.to_dict()["cv"]["chosen"] == "b-0"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_dominant_shape_is_reported_at_about_zero(seed):
    """F28: with the built-in families, one warning, the first configuration, and skill
    inside -0.03 to 0.02 on noise."""
    frame = q.dominant_entity(seed=100 + seed)
    result = s.run(frame, time_col=False, group_col="who")
    doc = result.to_dict()
    assert len(q.told(result, CODE)) == 1
    assert doc["models"][0]["chosen"] and doc["cv"]["chosen"] == doc["models"][0]["config"]
    assert -0.03 <= doc["headline"]["skill"] <= 0.02, doc["headline"]["skill"]
    assert e.CAP == 2000 and h.rows_of(result)["dropped_row_cap"] == 0
