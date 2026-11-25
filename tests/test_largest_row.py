"""T64: the share of a squared error that its largest row carries, on the holdout and over
the selection folds (spec steps 9 and 11, 2.5, 5.2, 5.3, 11; D74 point 3).

Four statements, each a share and a row: the model and the headline's baseline on the
holdout (`headline`), the chosen configuration and the baseline over the selection
folds (`cv`), and every configuration over the selection folds (`models`). One
`largest_row_dominates` warning for each statement above one half. Nothing changes a
number. Every share is counted again here with `cp9_helpers.largest_row`.
"""
import math

import numpy as np
import pandas as pd
import pytest

import cp7_helpers as s
import cp9_helpers as q
import mlp_helpers as h

DOMINATES = "largest_row_dominates"


def _table():
    """`h.reg_frame(n=400, seed=8701)` with the target of its first six rows empty, so
    that a row's number is not its place among the rows split."""
    frame = h.reg_frame(n=400, seed=8701)
    frame.loc[:5, "y"] = np.nan
    return frame


def _guesses(frame, seed, spread):
    y = np.nan_to_num(frame["y"].to_numpy(dtype=float))
    return y + spread * np.random.default_rng(seed).normal(size=len(frame))


def _lookup(name, log, values):
    return h.family(name, log, estimator=h.LookupEstimator, values=values)


def _two_families(a, b, log=None):
    """Two lookup families on one log, configurations `a-0` and `b-0`."""
    log = log or h.Log()
    return log, [_lookup("a", log, a), _lookup("b", log, b)]


def _folds(log, family, n_train):
    """`(fit rows, predicted rows)` of each selection fold of `family`, in order."""
    out = []
    for record in log.objects("model"):
        if record["family"] == family and len(record["fit"]) < n_train:
            out.append((list(record["fit"]), sorted(record["predict"])))
    return out


def _training(result, frame):
    held = set(h.holdout_rows(result))
    present = frame["y"].notna().to_numpy()
    return [r for r in range(len(frame)) if present[r] and r not in held]


def _close(a, b):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9)


def _assert_pair(share, row, want):
    assert (share is None) == (want[0] is None), (share, row, want)
    if want[0] is None:
        assert row is None
    else:
        assert _close(share, want[0]), (share, want)
        assert row == want[1], (row, want)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_the_keys_are_in_every_part(task, tmp_path):
    """M45 on the regression: the four keys in `headline` and `cv`, the two in every
    record of `models`, in the object and in the file."""
    frame = _table() if task == "regression" else h.clf_frame(n=300, seed=8711)
    result = s.run(frame, out_dir=tmp_path, **s.SAID)
    doc = h.load_json_strict(tmp_path / "profile.json")
    assert result.models.shape[0] > 1
    for part in (result.to_dict(), doc):
        for key in q.HEADLINE_LARGEST:
            assert key in part["headline"] and key in part["cv"], key
        for record in part["models"]:
            for key in q.MODEL_LARGEST:
                assert key in record, key
    for key in q.MODEL_LARGEST:
        assert key in result.models.columns
    values = ([doc["headline"][k] for k in q.HEADLINE_LARGEST]
              + [doc["cv"][k] for k in q.HEADLINE_LARGEST]
              + [r[k] for r in doc["models"] if r["status"] == "fitted" for k in q.MODEL_LARGEST])
    if task == "classification":
        assert all(v is None for v in values), values
        assert all(r[k] is None for r in doc["models"] for k in q.MODEL_LARGEST)
        assert DOMINATES not in s.codes(result)
    else:
        assert all(v is not None for v in values), values
        for k, v in enumerate(values):
            if k % 2 == 0:
                assert isinstance(v, float) and 0 < v <= 1, v
            else:
                assert isinstance(v, int) and not isinstance(v, bool), v
    assert doc["headline"] == h.plain(result.to_dict()["headline"])


def test_the_holdout_shares_are_those_of_the_predictions_file(tmp_path):
    """F26: the shares on the holdout are of the squared errors in `predictions.csv`."""
    frame = _table()
    _, families = _two_families(_guesses(frame, 1, 1.0), _guesses(frame, 2, 3.0))
    result = s.run(frame, families=families, out_dir=tmp_path, **s.SAID)
    head = result.to_dict()["headline"]
    preds = pd.read_csv(tmp_path / "predictions.csv")
    rows = preds["row"].tolist()
    _assert_pair(head["largest_row_share"], head["largest_row"],
                 q.largest_row(rows, preds["y_true"], preds["y_pred"]))
    y = frame["y"].to_numpy(dtype=float)
    mean = math.fsum(y[r] for r in _training(result, frame)) / len(_training(result, frame))
    assert head["baseline"] == "training_mean"
    _assert_pair(head["baseline_largest_row_share"], head["baseline_largest_row"],
                 q.largest_row(rows, preds["y_true"], [mean] * len(rows)))


def test_under_time_order_the_baseline_is_the_one_the_headline_used():
    """F26: under time order the baseline's shares are those of the previous value."""
    frame = q.walk(n=400, seed=8721)
    _, families = s.rec()
    result = s.run(frame, families=families, time_col="t", group_col=False)
    doc = result.to_dict()
    assert doc["headline"]["baseline"] == "previous_value"
    y = frame["y"].to_numpy(dtype=float)
    held = h.holdout_rows(result)
    _assert_pair(doc["headline"]["baseline_largest_row_share"],
                 doc["headline"]["baseline_largest_row"],
                 q.largest_row(held, y[held], y[np.asarray(held) - 1]))
    folds = list(range(50, 300))
    _assert_pair(doc["cv"]["baseline_largest_row_share"], doc["cv"]["baseline_largest_row"],
                 q.largest_row(folds, y[folds], y[np.asarray(folds) - 1]))


def test_every_configuration_has_its_share_over_the_selection_folds():
    """M45: each record of `models` has its own share, counted from the rows its folds
    predicted; `cv` holds the chosen one's and the baseline's."""
    frame = _table()
    a, b = _guesses(frame, 3, 1.0), _guesses(frame, 4, 2.0)
    log, families = _two_families(a, b)
    result = s.run(frame, families=families, **s.SAID)
    doc = result.to_dict()
    y = frame["y"].to_numpy(dtype=float)
    n_train = h.rows_of(result)["training"]
    by_name = {r["config"]: r for r in doc["models"]}
    for name, family, guess in (("a-0", "a", a), ("b-0", "b", b)):
        folds = _folds(log, family, n_train)
        assert len(folds) == 5
        rows = [r for _, predicted in folds for r in predicted]
        want = q.largest_row(rows, y[rows], guess[rows])
        _assert_pair(by_name[name]["largest_row_share"], by_name[name]["largest_row"], want)
        if by_name[name]["chosen"]:
            _assert_pair(doc["cv"]["largest_row_share"], doc["cv"]["largest_row"], want)
    rows, means = [], []
    for fit, predicted in _folds(log, "a", n_train):
        mean = math.fsum(y[r] for r in fit) / len(fit)
        rows.extend(predicted)
        means.extend([mean] * len(predicted))
    _assert_pair(doc["cv"]["baseline_largest_row_share"], doc["cv"]["baseline_largest_row"],
                 q.largest_row(rows, y[rows], means))


def test_a_configuration_that_failed_has_no_share():
    """A configuration that failed during selection has a null share and row."""
    frame = _table()
    log = h.Log()
    families = [h.family("a", log, estimator=h.FailingEstimator),
                _lookup("b", log, _guesses(frame, 5, 1.0))]
    result = s.run(frame, families=families, **s.SAID)
    by_name = {r["config"]: r for r in result.to_dict()["models"]}
    assert by_name["a-0"]["status"] == "failed"
    assert by_name["a-0"]["largest_row_share"] is None
    assert by_name["a-0"]["largest_row"] is None
    assert isinstance(by_name["b-0"]["largest_row_share"], float)
    assert isinstance(by_name["b-0"]["largest_row"], int)


def test_a_sum_of_zero_has_no_largest_row():
    """A model that is exact, and a baseline exact on a flat holdout (D36), have none."""
    frame = _table()
    exact = np.nan_to_num(frame["y"].to_numpy(dtype=float))
    _, families = _two_families(exact, _guesses(frame, 6, 1.0))
    doc = s.run(frame, families=families, **s.SAID).to_dict()
    assert doc["headline"]["model"] == "a-0" and doc["headline"]["score"] == 0
    for part in (doc["headline"], doc["cv"]):
        assert part["largest_row_share"] is None and part["largest_row"] is None
        assert isinstance(part["baseline_largest_row_share"], float)
    record = [r for r in doc["models"] if r["config"] == "a-0"][0]
    assert record["largest_row_share"] is None and record["largest_row"] is None

    flat = q.walk(n=400, seed=8731)
    flat.loc[280:, "y"] = 1.5
    _, families = s.rec()
    doc = s.run(flat, families=families, time_col="t", group_col=False).to_dict()
    head = doc["headline"]
    assert head["baseline"] == "previous_value" and head["baseline_score"] == 0
    assert head["skill"] is None
    assert head["baseline_largest_row_share"] is None and head["baseline_largest_row"] is None
    assert isinstance(head["largest_row_share"], float)


def test_no_warning_without_a_row_that_dominates():
    """On a clean table no statement is above one half."""
    frame = _table()
    _, families = _two_families(_guesses(frame, 7, 1.0), _guesses(frame, 8, 2.0))
    result = s.run(frame, families=families, **s.SAID)
    assert DOMINATES not in s.codes(result)


def _first_run(frame):
    _, families = s.rec()
    return s.run(frame, families=families, **s.SAID)


def test_one_holdout_row_the_model_misses_is_one_record():
    """F26: one prediction 500 off is one record, naming `holdout`, `a-0`, the row and
    the share."""
    frame = _table()
    row = h.holdout_rows(_first_run(frame))[7]
    guess = _guesses(frame, 9, 10.0)
    guess[row] += 500.0
    log = h.Log()
    result = s.run(frame, families=[_lookup("a", log, guess)], **s.SAID)
    head = result.to_dict()["headline"]
    held = h.holdout_rows(result)
    y = frame["y"].to_numpy(dtype=float)
    _assert_pair(head["largest_row_share"], head["largest_row"],
                 q.largest_row(held, y[held], guess[held]))
    assert head["largest_row"] == row and head["largest_row_share"] > 0.5
    # Of the absolute errors the row would carry less than half (F26).
    absolute = np.abs(y[held] - guess[held])
    assert absolute.max() < 0.5 * absolute.sum()
    messages = q.told(result, DOMINATES)
    assert len(messages) == 1, messages
    text = messages[0]
    assert "holdout" in text and "a-0" in text
    assert q.holds_row(text, row) and q.holds_share(text, head["largest_row_share"])
    assert [r["column"] for r in s.warning_records(result, DOMINATES)] == [None]


def test_a_wild_holdout_target_is_a_record_for_the_model_and_one_for_the_baseline():
    """A holdout target of 999999 carries the model's error and the baseline's."""
    frame = _table()
    row = h.holdout_rows(_first_run(frame))[3]
    guess = _guesses(frame, 10, 0.5)
    frame.loc[row, "y"] = 999999.0
    result = s.run(frame, families=[_lookup("a", h.Log(), guess)], **s.SAID)
    head = result.to_dict()["headline"]
    assert head["largest_row"] == row and head["baseline_largest_row"] == row
    messages = q.told(result, DOMINATES)
    assert len(messages) == 2, messages
    assert all("holdout" in m and q.holds_row(m, row) for m in messages)
    assert sum("training_mean" in m for m in messages) == 1
    assert sum("a-0" in m for m in messages) >= 1


def test_a_wild_training_target_is_one_record_for_the_configurations_and_one_for_the_baseline():
    """M45: a training target of 999999 is one record for the configurations, however
    many have a share above one half, and one for the baseline over the folds."""
    frame = _table()
    row = q.training_rows(_first_run(frame), frame)[50]
    a, b = _guesses(frame, 11, 0.5), _guesses(frame, 12, 1.0)
    frame.loc[row, "y"] = 999999.0
    _, families = _two_families(a, b)
    result = s.run(frame, families=families, **s.SAID)
    doc = result.to_dict()
    assert all(r["largest_row_share"] > 0.5 and r["largest_row"] == row
               for r in doc["models"])
    messages = q.told(result, DOMINATES)
    assert len(messages) == 2, messages
    assert all("selection folds" in m and q.holds_row(m, row) for m in messages)
    baseline = [m for m in messages if "training_mean" in m]
    assert len(baseline) == 1
    assert q.holds_share(baseline[0], doc["cv"]["baseline_largest_row_share"])


def test_exactly_one_half_is_not_above_one_half():
    """Two rows that carry the error equally: a share of exactly one half, and no record."""
    frame = _table()
    held = sorted(h.holdout_rows(_first_run(frame)))
    first, second = held[4], held[9]
    frame.loc[first, "y"] = 1.0
    frame.loc[second, "y"] = 2.0
    guess = np.nan_to_num(frame["y"].to_numpy(dtype=float)).copy()
    guess[first], guess[second] = 9.0, 10.0
    result = s.run(frame, families=[_lookup("a", h.Log(), guess)], **s.SAID)
    head = result.to_dict()["headline"]
    assert head["largest_row_share"] == 0.5
    assert head["largest_row"] == first
    assert DOMINATES not in s.codes(result)


def test_the_statements_leave_nothing_out_of_the_score(tmp_path):
    """The score, the baseline and skill keep the row the statement names."""
    frame = _table()
    row = h.holdout_rows(_first_run(frame))[3]
    guess = _guesses(frame, 13, 0.5)
    frame.loc[row, "y"] = 999999.0
    result = s.run(frame, families=[_lookup("a", h.Log(), guess)], out_dir=tmp_path,
                   **s.SAID)
    head = result.to_dict()["headline"]
    preds = pd.read_csv(tmp_path / "predictions.csv")
    assert row in preds["row"].tolist()
    errors = (preds["y_true"] - preds["y_pred"]) ** 2
    assert math.isclose(head["score"], math.fsum(errors) / len(errors), rel_tol=1e-9)
    y = frame["y"].to_numpy(dtype=float)
    training = _training(result, frame)
    mean = math.fsum(y[r] for r in training) / len(training)
    baseline = math.fsum((t - mean) ** 2 for t in preds["y_true"]) / len(preds)
    assert math.isclose(head["baseline_score"], baseline, rel_tol=1e-9)
    assert math.isclose(head["skill"], 1 - head["score"] / head["baseline_score"],
                        rel_tol=1e-9)
    assert head["largest_row"] == row
