"""T20: colliding feature names stop the run before any fit (spec 3 step 5, 2.4, 2.7; D37, D43).

Red today, apart from `test_a_level_only_in_rows_with_no_target_does_not_count`.
Levels here are written without white space round them; the guard of that is
in `test_guard_final_fit.py`.
"""
import numpy as np
import pandas as pd
import pytest

import cp3_helpers as c
import mlp_helpers as h


def _base(n=200, seed=81):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    frame = pd.DataFrame({"x1": x1})
    frame["y"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-2.5 * x1))).astype(int)
    return frame, rng


def _expect_collision(frame, out_dir, must_name, **kw):
    """Run with a recording family and step; the error names the column and nothing ran."""
    from mlprofessor.errors import FeatureNameCollisionError

    log = h.Log()
    with pytest.raises(FeatureNameCollisionError) as caught:
        c.run(frame, families=[h.family("rec", log)], steps=[h.step_factory(log)],
              out_dir=out_dir, **kw)
    assert must_name in str(caught.value)
    assert log.events == [], "the run stops before any fit"
    assert h.files_in(out_dir) == []
    return caught.value


def test_error_class_and_code():
    from mlprofessor.errors import FeatureNameCollisionError, ProfileError

    assert issubclass(FeatureNameCollisionError, ProfileError)
    assert FeatureNameCollisionError.code == "feature_name_collision"


def test_numeric_column_named_like_a_level_stops_the_run(tmp_path):
    frame, rng = _base()
    frame.insert(1, "colour", rng.choice(["red", "green", "blue"], size=len(frame)))
    frame.insert(2, "colour=red", rng.normal(size=len(frame)))
    _expect_collision(frame, tmp_path / "out", "colour=red")


def test_two_categorical_columns_that_combine_to_one_name_stop_the_run(tmp_path):
    frame, rng = _base()
    frame.insert(1, "site", rng.choice(["a=b", "c", "d"], size=len(frame)))
    frame.insert(2, "site=a", rng.choice(["b", "e", "f"], size=len(frame)))
    _expect_collision(frame, tmp_path / "out", "site=a")


def test_a_rare_level_counts(tmp_path):
    frame, rng = _base(n=240)
    levels = [f"level{k:02d}" for k in range(39)]
    colour = np.array([levels[i % 39] for i in range(len(frame))], dtype=object)
    colour[17] = "red"
    frame.insert(1, "colour", colour)
    frame.insert(2, "colour=red", rng.normal(size=len(frame)))
    assert frame["colour"].nunique() == 40
    assert (frame["colour"] == "red").sum() == 1
    _expect_collision(frame, tmp_path / "out", "colour=red")


def test_the_missing_level_counts(tmp_path):
    frame, rng = _base()
    colour = rng.choice(["red", "green", "blue"], size=len(frame)).astype(object)
    colour[11] = None
    frame.insert(1, "colour", colour)
    frame.insert(2, "colour=(missing)", rng.normal(size=len(frame)))
    assert frame["colour"].isna().sum() == 1
    _expect_collision(frame, tmp_path / "out", "colour=(missing)")


def test_a_level_only_in_rows_with_no_target_does_not_count():
    """A level held only by rows with no target names no matrix column: no collision."""
    frame, rng = _base(n=220)
    colour = rng.choice(["red", "green", "blue"], size=len(frame)).astype(object)
    no_target = [7, 150]
    colour[no_target] = "purple"
    frame.insert(1, "colour", colour)
    frame.insert(2, "colour=purple", rng.normal(size=len(frame)))
    frame["y"] = frame["y"].astype(float)
    frame.loc[no_target, "y"] = np.nan

    log = c.KeepLog()
    result = c.run(frame, families=[h.family("keep", log, c.MatrixEstimator)])
    cols = h.columns_of(result)
    assert cols["colour"]["role"] == "feature"
    assert cols["colour=purple"]["role"] == "feature"
    fits = log.calls("model", "fit")
    assert fits
    for fit in fits:
        assert fit["columns"].count("colour=purple") == 1
        assert len(fit["columns"]) == len(set(fit["columns"]))


def test_a_level_in_rows_the_fast_profile_leaves_out_still_counts(tmp_path):
    """The rule is decided before the row cap: a level only in capped-out rows still collides."""
    n = 2300
    rng = np.random.default_rng(82)
    x1 = rng.normal(size=n)
    colour = rng.choice(["green", "blue"], size=n).astype(object)
    colour[0] = "red"
    frame = pd.DataFrame({
        "t": np.arange(n, dtype=float),
        "x1": x1,
        "colour": colour,
        "colour=red": rng.normal(size=n),
        "y": 2.0 * x1 + rng.normal(size=n),
    })
    _expect_collision(frame, tmp_path / "out", "colour=red", time_col="t", profile="fast")


def test_preprocessor_fit_raises_on_colliding_names():
    from mlprofessor import stages
    from mlprofessor.errors import FeatureNameCollisionError

    kinds = {"colour": "categorical", "colour=red": "numeric"}
    holds_red = pd.DataFrame({"colour": ["red", "green", "blue", "red"],
                              "colour=red": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(FeatureNameCollisionError):
        stages.Preprocessor(kinds).fit(holds_red)

    no_red = pd.DataFrame({"colour": ["green", "blue", "yellow", "green"],
                           "colour=red": [1.0, 2.0, 3.0, 4.0]})
    out = stages.Preprocessor(kinds).fit(no_red).transform(no_red)
    assert list(out.index) == list(no_red.index)
    assert len(out.columns) == 4
    assert len(set(out.columns)) == 4


def test_command_line_reports_the_collision(tmp_path):
    frame, rng = _base()
    frame.insert(1, "colour", rng.choice(["red", "green", "blue"], size=len(frame)))
    frame.insert(2, "colour=red", rng.normal(size=len(frame)))
    path = tmp_path / "collide.csv"
    frame.to_csv(path, index=False)
    out = tmp_path / "out"
    proc = h.run_cli(["profile", "--data", path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1"])
    assert proc.returncode == 3, proc.stderr
    assert any(line.startswith("error: feature_name_collision:")
               for line in proc.stderr.splitlines())
    assert h.files_in(out) == []
