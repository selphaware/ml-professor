"""T21: a configuration that fails after it is chosen (spec 3 steps 9 to 11, 2.5, 2.6, 5.2; D38).

Red today: `test_failure_after_the_choice_keeps_fold_scores_and_falls_back`.
Green today: the other three.

Four seam families of one configuration each predict values the test chose.
Each reveals the target on a share of the rows (0.8, 0.5, 0.3, 0.1) and gives
shared noise elsewhere; a larger share reveals a superset of the rows a
smaller one does, so on any set of rows its ROC AUC is at least as high. Their
`cv_mean` therefore ranks them best, second, third, last under any split.
"""
import numpy as np
import pandas as pd
import pytest

import cp3_helpers as c
import mlp_helpers as h

N = 240
SHARES = {"best": 0.8, "second": 0.5, "third": 0.3, "last": 0.1}


def _table():
    rng = np.random.default_rng(91)
    x1 = rng.normal(size=N)
    y = (rng.random(N) < 1.0 / (1.0 + np.exp(-2.0 * x1))).astype(int)
    frame = pd.DataFrame({"x1": x1, "z1": rng.normal(size=N), "y": y})
    reveal = rng.random(N)
    noise = rng.uniform(0.05, 0.95, size=N)
    values = {name: np.where(reveal < share, y.astype(float), noise)
              for name, share in SHARES.items()}
    return frame, values


def _families(log, values, names, fail_fit=(), fail_holdout=(), training=None, holdout=None):
    out = []
    for name in names:
        extra = {"values": values[name]}
        if name in fail_fit:
            extra["fail_fit_rows"] = training
        if name in fail_holdout:
            extra["fail_predict_rows"] = holdout
        out.append(h.family(name, log, c.FailOnCommand, **extra))
    return out


def _split_rows(frame, values):
    """Holdout and training rows from a clean first run."""
    clean = c.run(frame, families=_families(h.Log(), values, ["best"]))
    holdout = h.holdout_rows(clean)
    training = sorted(set(range(N)) - set(holdout))
    return holdout, training


def _in_turn(log, hold):
    """The families whose predict calls hold holdout rows, in call order, with the calls of
    one family in a row counted once.

    D59, D78 point 7: until the ninth pass the configuration that scored the holdout
    predicted it once. Since then it predicts it again, five times for each set of
    columns, to state what the headline rests on; a family's calls in a row are one turn,
    so what is asserted is still the order in which the holdout reaches configurations.
    """
    predicted = [ev["family"] for ev in log.events
                 if ev["action"] == "predict" and set(ev["rows"]) & hold]
    return [f for k, f in enumerate(predicted) if k == 0 or f != predicted[k - 1]]


@pytest.mark.parametrize("way", ["final_fit", "holdout"])
def test_failure_after_the_choice_keeps_fold_scores_and_falls_back(way):
    """D38: a chosen configuration that then fails keeps its fold scores; the next best by cv_mean takes its place."""
    frame, values = _table()
    clean = c.run(frame, families=_families(h.Log(), values, ["best", "second"]))
    holdout = h.holdout_rows(clean)
    training = sorted(set(range(N)) - set(holdout))
    clean_best = {m["config"]: m for m in h.models_of(clean)}["best-0"]
    assert clean_best["chosen"] is True
    assert None not in clean_best["fold_scores"]

    failing = {"fail_fit": ["best"]} if way == "final_fit" else {"fail_holdout": ["best"]}
    result = c.run(frame, families=_families(h.Log(), values, ["best", "second"],
                                             training=training, holdout=holdout, **failing))
    assert h.holdout_rows(result) == holdout
    models = {m["config"]: m for m in h.models_of(result)}
    best, second = models["best-0"], models["second-0"]

    assert best["status"] == "failed"
    assert best["chosen"] is False
    assert c.FailOnCommand.message_for("best") in best["error"]
    assert best["fold_scores"] == pytest.approx(clean_best["fold_scores"], rel=1e-12)
    assert best["cv_mean"] is not None and best["cv_std"] is not None
    assert best["cv_mean"] == pytest.approx(clean_best["cv_mean"], rel=1e-12)
    assert best["cv_std"] == pytest.approx(clean_best["cv_std"], rel=1e-12)

    assert second["status"] == "fitted"
    assert second["chosen"] is True
    assert second["error"] is None
    assert c.count_code(result, "model_failed") == 1

    doc = result.to_dict()
    assert result.headline["model"] == "second-0"
    assert doc["cv"]["chosen"] == "second-0"
    assert doc["cv"]["chosen_mean"] == pytest.approx(second["cv_mean"], rel=1e-12)
    y = frame["y"].to_numpy()
    expected = h.auc_by_hand(y[holdout], values["second"][holdout])
    assert result.headline["score"] == pytest.approx(expected, rel=1e-12)
    assert np.allclose(result.predictions["y_pred"].to_numpy(dtype=float),
                       values["second"][holdout], rtol=0, atol=1e-12)


def test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration():
    """The fallback goes down the cv_mean ranking, not the table order; one configuration gives the headline."""
    frame, values = _table()
    holdout, training = _split_rows(frame, values)
    order = ["last", "third", "best", "second"]
    result = c.run(frame, families=_families(h.Log(), values, order, fail_holdout=["best"],
                                             fail_fit=["second"], training=training,
                                             holdout=holdout))
    models = h.models_of(result)
    assert [m["config"] for m in models] == [f"{name}-0" for name in order]
    assert [m["status"] for m in models] == ["fitted", "fitted", "failed", "failed"]
    assert [m["config"] for m in models if m["chosen"]] == ["third-0"]
    assert c.count_code(result, "model_failed") == 2

    y = frame["y"].to_numpy()
    expected = h.auc_by_hand(y[holdout], values["third"][holdout])
    assert result.headline["model"] == "third-0"
    assert result.headline["score"] == pytest.approx(expected, rel=1e-12)
    assert result.headline["skill"] == pytest.approx(2 * expected - 1, rel=1e-9)


def test_holdout_rows_reach_one_configuration_at_a_time():
    """Holdout rows go to one configuration at a time, in ranking order, and to none once one has predicted them."""
    frame, values = _table()
    holdout, training = _split_rows(frame, values)
    hold = set(holdout)
    order = ["last", "third", "best", "second"]

    log = h.Log()
    result = c.run(frame, families=_families(log, values, order, fail_holdout=["best", "second"],
                                             training=training, holdout=holdout))
    reached = _in_turn(log, hold)
    assert reached == ["best", "second", "third"]
    statuses = {m["family"]: m["status"] for m in h.models_of(result)}
    assert statuses["best"] == "failed" and statuses["second"] == "failed"
    assert not any(set(ev["rows"]) & hold for ev in log.events if ev["action"] == "fit")

    log = h.Log()
    c.run(frame, families=_families(log, values, order))
    reached = _in_turn(log, hold)
    assert reached == ["best"]


def test_every_configuration_failing_after_selection_raises_all_models_failed(tmp_path):
    """Every configuration failing after the choice is `all_models_failed`, carrying each error text."""
    from mlprofessor.errors import AllModelsFailedError

    frame, values = _table()
    holdout, training = _split_rows(frame, values)
    out = tmp_path / "out"
    with pytest.raises(AllModelsFailedError) as caught:
        c.run(frame, families=_families(h.Log(), values, ["best", "second"], fail_fit=["best"],
                                        fail_holdout=["second"], training=training,
                                        holdout=holdout),
              out_dir=out)
    message = str(caught.value)
    assert c.FailOnCommand.message_for("best") in message
    assert c.FailOnCommand.message_for("second") in message
    assert h.files_in(out) == []
