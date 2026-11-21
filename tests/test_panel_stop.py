"""T44: a panel with nothing said about its entities stops and asks (spec 2.4,
2.5, step 11; D58).
"""
import numpy as np

import cp6_helpers as g
import mlp_helpers as h

CODE = "entities_undeclared"


def _persistent():
    frame = h.panel_frame(60, 5, seed=13, persistent=True)
    means = frame.groupby("date")["y"].mean().to_numpy()[:45]
    assert h.lag1_autocorr(means) > 0.8
    return frame


def test_a_persistent_panel_with_no_entity_column_stops(tmp_path):
    """M19: a persistent panel with nothing said about entities stops before any fit."""
    from mlprofessor import errors

    log = h.Log()
    out = tmp_path / "out"
    err = g.raised(_persistent(), families=[h.family("rec", log)], steps=[h.step_factory(log)],
                   time_col="date", out_dir=str(out))
    assert err.code == CODE, f"{type(err).__name__}: {err}"
    assert isinstance(err, errors.EntitiesUndeclaredError)
    for needle in ("group_col", "--group-col", "group_col=False", "--no-entities"):
        assert needle in str(err), (needle, str(err))
    assert h.files_in(out) == []
    assert log.events == [], "a step or a model was fitted before the stop"


def test_the_statement_lets_the_run_go_ahead_as_before():
    result = g.run(_persistent(), families=[h.family("rec", h.Log())], time_col="date",
                   group_col=False)
    doc = result.to_dict()
    assert doc["input"]["group_col"] is False
    assert doc["validation"]["group_col"] is None
    assert [b["name"] for b in doc["baselines"]] == ["training_mean"]
    flagged = [c for code, c in h.warnings_of(result) if code == "autocorrelated_panel_target"]
    assert flagged == ["y"]


def test_a_panel_that_does_not_persist_runs_without_a_word():
    frame = h.panel_frame(60, 5, seed=13, persistent=False)
    result = g.run(frame, families=[h.family("rec", h.Log())], time_col="date")
    assert g.count_code(result, "autocorrelated_panel_target") == 0
    assert result.to_dict()["input"]["group_col"] is None


def test_a_classification_panel_is_not_asked():
    frame = _persistent()
    frame["y"] = np.random.default_rng(1001).integers(0, 2, size=len(frame))
    result = g.run(frame, families=[h.family("rec", h.Log())], time_col="date")
    assert result.to_dict()["validation"]["scheme"] == "time_ordered"


def test_the_command_line_says_there_are_no_entities(tmp_path):
    path = tmp_path / "panel.csv"
    _persistent().to_csv(path, index=False)
    common = ["profile", "--data", path, "--target", "y", "--profile", "fast", "--n-jobs", "1",
              "--time-col", "date"]

    out = tmp_path / "asked"
    proc = h.run_cli([*common, "--out", out])
    assert proc.returncode == 3, proc.stderr
    assert CODE in proc.stderr, proc.stderr
    assert h.files_in(out) == []

    out = tmp_path / "stated"
    proc = h.run_cli([*common, "--out", out, "--no-entities"])
    assert proc.returncode == 0, proc.stderr
    assert "autocorrelated_panel_target" in proc.stdout, proc.stdout
