"""T65: under time order the rows of a fit are handed over in time order (spec 2.6, steps
9 and 10, 7, 11; D74 point 4).

Every fit of a step supplied through the seam and of a model, in each selection fold
and in the final fit, is handed its rows in time order, the rows of one timestamp in
the order they have in the file, with the target in the same order. With no time
column the order is open (spec 11), and nothing is asserted about it here.
"""
import numpy as np
import pytest

import cp6_helpers as g
import cp7_helpers as s
import cp9_helpers as q
import mlp_helpers as h

TABLES = {
    "unique_timestamps": (lambda: q.walk(n=300, seed=8801, shuffle=True), False),
    "repeated_timestamps": (lambda: q.walk(n=100, seed=8802, per=3, noise=0.5, shuffle=True),
                            False),
    "entities": (lambda: g.walks(n_series=6, n_times=60, seed=8803), "who"),
}


def _recorded_run(frame, group_col):
    log = h.Log()
    result = s.run(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)],
                   time_col="t", group_col=group_col)
    return log, result


def _fits(log):
    return [ev for ev in log.events if ev["action"] == "fit"]


def _assert_in_time_order(frame, event):
    keys = frame["t"].tolist()
    rows = event["rows"]
    assert rows == sorted(rows, key=lambda r: (keys[r], r)), (event["kind"], rows[:12])
    if event["y"] is not None:
        assert event["y_index_ok"] is True
        want = frame["y"].to_numpy(dtype=float)[rows]
        assert np.allclose(np.asarray(event["y"], dtype=float), want)


@pytest.mark.parametrize("case", list(TABLES))
def test_every_fit_is_handed_its_rows_in_time_order(case):
    """M49: every fit of a step and of a model is handed its rows in time order."""
    make, group_col = TABLES[case]
    frame = make()
    log, result = _recorded_run(frame, group_col)
    assert result.to_dict()["validation"]["scheme"] == "time_ordered"
    fits = _fits(log)
    assert sum(ev["kind"] == "model" for ev in fits) == 6
    assert any(ev["kind"] == "step" for ev in fits)
    for event in fits:
        _assert_in_time_order(frame, event)


def test_rows_of_one_timestamp_keep_their_file_order():
    """M49: a table written latest first, three rows a timestamp: each fit is in time
    order, and within a timestamp in file order."""
    frame = q.walk(n=120, seed=8812, per=3, noise=0.5)
    order = np.argsort(-frame["t"].to_numpy(), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    keys = frame["t"].tolist()
    assert keys[0] > keys[-1]
    log, _ = _recorded_run(frame, False)
    for event in _fits(log):
        rows = event["rows"]
        _assert_in_time_order(frame, event)
        for a, b in zip(rows, rows[1:]):
            if keys[a] == keys[b]:
                assert a < b


class _FirstThird(h._FakeEstimator):
    """Fitted on the first third of the rows it is handed, by position."""

    def fit(self, X, y):
        from sklearn.linear_model import Ridge

        k = max(2, len(X) // 3)
        self.inner = Ridge(alpha=1.0).fit(X.to_numpy(dtype=float)[:k],
                                          np.asarray(y, dtype=float)[:k])
        return self

    def predict(self, X):
        return self.inner.predict(X.to_numpy(dtype=float))


def _positional_run(frame):
    families = [h.family("third", h.Log(), estimator=_FirstThird)]
    return s.run(frame, families=families, time_col="t", group_col=False)


def test_the_order_of_the_file_changes_no_number_under_time_order():
    """M49: a model that reads its rows by position gives the same numbers whatever the
    order of the rows in the file."""
    ordered = q.walk(n=300, seed=8821)
    ordered = q.with_column(ordered, "lag", np.concatenate(
        [[np.nan], ordered["y"].to_numpy(dtype=float)[:-1]]))
    permuted = ordered.iloc[np.random.default_rng(8821).permutation(len(ordered))]
    permuted = permuted.reset_index(drop=True)
    first, second = _positional_run(ordered), _positional_run(permuted)
    a, b = first.to_dict(), second.to_dict()
    assert a["headline"]["score"] == pytest.approx(b["headline"]["score"], rel=1e-9)
    assert a["models"][0]["fold_scores"] == pytest.approx(b["models"][0]["fold_scores"],
                                                          rel=1e-9)

    def by_time(result, frame):
        stamps = frame["t"].tolist()
        return {stamps[int(r)]: float(p) for r, p in
                zip(result.predictions["row"], result.predictions["y_pred"])}

    one, two = by_time(first, ordered), by_time(second, permuted)
    assert set(one) == set(two)
    for stamp in one:
        assert one[stamp] == pytest.approx(two[stamp], rel=1e-9, abs=1e-12)


def test_with_no_time_column_the_run_is_as_it_was():
    """With no time column: six model fits, the final one on the training rows, and no
    holdout row in any fit."""
    frame = h.reg_frame(n=300, seed=8831)
    log = h.Log()
    result = s.run(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)],
                   **s.SAID)
    held = set(h.holdout_rows(result))
    training = set(range(len(frame))) - held
    models = [ev for ev in _fits(log) if ev["kind"] == "model"]
    assert len(models) == 6
    assert set(models[-1]["rows"]) == training
    for event in _fits(log):
        assert not set(event["rows"]) & held
        if event["y"] is not None:
            assert event["y_index_ok"] is True
