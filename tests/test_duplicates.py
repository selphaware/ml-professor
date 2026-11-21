"""T32: copies of a holdout row among the training rows (spec 2.1, 2.2, 2.4, 2.5, step 8a, 5.2; D49, D53, D54).

Red today, every test: the application has no `duplicates` argument, makes no
count and never stops for copies. The count is worked out here from what the
steps were handed (`cp5_helpers.copies_by_hand`), never trusted.
"""
import inspect
import json

import numpy as np
import pandas as pd
import pytest

import cp3_helpers as c
import cp5_helpers as p
import mlp_helpers as h

CODE = "holdout_copies_in_training"
KEPT = ("holdout_copies_kept", None)


def _validation(result):
    return result.to_dict()["validation"]


def test_duplicates_is_the_tenth_parameter_and_defaults_to_error():
    """The signature is exactly the ten parameters of spec 2.1 (D54)."""
    import mlprofessor

    params = inspect.signature(mlprofessor.profile).parameters
    assert list(params) == ["data", "target", "task", "time_col", "gap", "profile", "seed",
                            "out_dir", "n_jobs", "duplicates"]
    assert params["duplicates"].default == "error"
    assert params["duplicates"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_error_class_and_code():
    from mlprofessor import errors

    assert issubclass(errors.HoldoutCopiesError, errors.ProfileError)
    assert errors.HoldoutCopiesError.code == CODE


@pytest.mark.parametrize("value", ["drop", "KEEP", "", True, None, 1],
                         ids=["drop", "KEEP", "empty", "True", "None", "one"])
def test_any_other_value_is_an_input_error(value):
    import mlprofessor
    from mlprofessor import errors

    ordinary = h.clf_frame(n=120)
    clashing = pd.DataFrame(np.zeros((120, 3)), columns=["x1", "x1", "y"])
    for frame in (ordinary, clashing):
        with pytest.raises(errors.ProfileError) as caught:
            mlprofessor.profile(frame, "y", profile="fast", n_jobs=1, duplicates=value)
        assert isinstance(caught.value, errors.InputError), type(caught.value).__name__
        assert caught.value.code == "input_error"


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_every_row_twice_stops_the_run(task, tmp_path):
    """F9: noise with every row present twice is stopped, not scored as predictability."""
    from mlprofessor import errors

    frame = p.twice(p.noise_frame(300, task=task))
    assert len(frame) == 600
    out = tmp_path / "out"
    err = p.raised(frame, task=task, out_dir=out)
    assert isinstance(err, errors.HoldoutCopiesError), f"{type(err).__name__}: {err}"
    assert err.code == CODE
    assert h.files_in(out) == []

    result, log = p.run_kept(frame, task=task, duplicates="keep")
    holdout = h.holdout_rows(result)
    assert len(holdout) == 150
    count = p.copies_by_hand(result, log)
    assert 90 <= count <= 135, count
    validation = _validation(result)
    assert validation["n_holdout_copies"] == count
    assert abs(validation["holdout_copy_share"] - count / 150) <= 1e-12

    message = str(err)
    numbers = c.numbers_in(message)
    assert count in numbers, message
    assert 150 in numbers, message
    assert any(abs(x - 100.0 * count / 150) <= 0.1 for x in numbers), message
    assert "keep" in message, message
    assert "time_col" not in message, message


def test_keep_runs_and_says_so():
    frame = p.twice(p.noise_frame(300))
    result, log = p.run_kept(frame, duplicates="keep")
    doc = result.to_dict()
    assert doc["input"]["duplicates"] == "keep"
    assert doc["validation"]["scheme"] == "random"
    count = p.copies_by_hand(result, log)
    assert count >= 10
    assert doc["validation"]["n_holdout_copies"] == count
    pairs = h.warnings_of(result)
    assert pairs.count(KEPT) == 1
    assert c.count_code(result, "holdout_copies_kept") == 1
    assert c.count_code(result, "duplicate_rows") == 1
    c.assert_strict_json(result)


RULE_CASES = {
    "9_of_100": (400, 9, False),
    "10_of_100": (400, 10, True),
    "10_of_1000": (4000, 10, False),
    "11_of_1000": (4000, 11, True),
    "0_of_1000": (4000, 0, False),
}


@pytest.mark.parametrize("case", list(RULE_CASES))
def test_the_rule_needs_a_share_and_a_count(case):
    """F9: the rule is more than 1% of the holdout rows and at least 10 of them (D53)."""
    from mlprofessor import errors

    n_rows, planted, stops = RULE_CASES[case]
    frame, n_holdout = p.plant_copies(h.clf_frame(n=n_rows), planted, profile="full")
    assert n_holdout == n_rows // 4

    kept, log = p.run_kept(frame, profile="full", duplicates="keep")
    assert p.copies_by_hand(kept, log) == planted
    validation = _validation(kept)
    assert validation["n_holdout_copies"] == planted
    assert abs(validation["holdout_copy_share"] - planted / n_holdout) <= 1e-12
    assert h.warnings_of(kept).count(KEPT) == (1 if stops else 0)
    assert c.count_code(kept, "holdout_copies_kept") == (1 if stops else 0)

    if stops:
        err = p.raised(frame, families=p.flat_family(h.Log(), len(frame)), profile="full")
        assert isinstance(err, errors.HoldoutCopiesError), f"{type(err).__name__}: {err}"
        assert err.code == CODE
    else:
        default, _ = p.run_kept(frame, profile="full")
        doc = default.to_dict()
        assert doc["input"]["duplicates"] == "error"
        assert doc["validation"]["n_holdout_copies"] == planted
        assert c.count_code(default, "holdout_copies_kept") == 0
        assert (h.without(doc, "meta.timing", "input.duplicates")
                == h.without(kept.to_dict(), "meta.timing", "input.duplicates"))
        pd.testing.assert_frame_equal(default.predictions, kept.predictions)


def test_the_rule_in_regression():
    from mlprofessor import errors

    frame, n_holdout = p.plant_copies(h.reg_frame(n=400), 12, task="regression")
    assert n_holdout == 100
    err = p.raised(frame, task="regression", families=p.flat_family(h.Log(), len(frame)))
    assert isinstance(err, errors.HoldoutCopiesError), f"{type(err).__name__}: {err}"

    result, log = p.run_kept(frame, task="regression", duplicates="keep")
    assert p.copies_by_hand(result, log) == 12
    assert _validation(result)["n_holdout_copies"] == 12

    holdout = h.holdout_rows(result)
    changed = frame.copy()
    for row in holdout[:6]:
        changed.iloc[row, changed.columns.get_loc("y")] += 1.0
    result, log = p.run_kept(changed, task="regression", duplicates="keep")
    assert h.holdout_rows(result) == holdout
    assert p.copies_by_hand(result, log) == 6
    assert _validation(result)["n_holdout_copies"] == 6


def test_values_are_compared_as_a_step_receives_them():
    """Spec step 8a, "How values are compared": numbers as floats, levels stripped, the class as read."""
    gen = np.random.default_rng(84)
    n = 400
    base = pd.DataFrame({"x1": gen.normal(size=n), "x2": gen.normal(size=n),
                         "y": gen.integers(0, 2, size=n)})
    first, first_log = p.run_kept(base, profile="full")
    holdout = h.holdout_rows(first)
    train = p.training_rows(first_log)
    labels = base["y"].to_numpy()

    x1 = [f"{v:.6f}" for v in base["x1"]]
    x2 = base["x2"].tolist()
    colour = ["red" if k % 2 else "blue" for k in range(n)]
    record = [f"rec{k:05d}" for k in range(n)]
    y = ["yes" if v == 1 else "no" for v in labels]

    pairs, used = [], set()
    for row in holdout[:12]:
        source = next(r for r in train if r not in used and labels[r] == labels[row])
        used.add(source)
        pairs.append((row, source))
        x1[row], x2[row], colour[row] = x1[source], x2[source], colour[source]

    (h0, s0), (h1, s1), (h2, s2), (h3, _), (h4, _), (h5, _) = pairs[:6]
    x1[s0], x1[h0] = "0.0", "-0.0"            # equal as floats
    x1[s1], x1[h1] = None, "abc"              # both missing: an empty and a coerced cell
    x1[s2], x1[h2] = "1.5", " 1.50 "          # the same number
    colour[h3] = f" {colour[h3]} "            # the same level
    y[h4] = y[h4] + " "                       # the same class
    colour[h5] = colour[h5].capitalize()      # another level: not a copy

    table = pd.DataFrame({
        "x1": pd.Series(x1, dtype=object), "x2": x2,
        "colour": pd.Series(colour, dtype=object), "record": pd.Series(record, dtype=object),
        "y": pd.Series(y, dtype=object),
    })
    result, log = p.run_kept(table, profile="full", duplicates="keep")
    assert h.holdout_rows(result) == holdout, "the split moved"
    cols = h.columns_of(result)
    assert (cols["x1"]["role"], cols["x1"]["kind"]) == ("feature", "numeric")
    assert (cols["colour"]["role"], cols["colour"]["kind"]) == ("feature", "categorical")
    assert cols["record"]["role"] == "excluded"
    assert p.copies_by_hand(result, log) == 11
    assert _validation(result)["n_holdout_copies"] == 11


def test_the_rule_is_judged_before_any_model_is_fitted():
    """Spec 2.4, step 8a: a table that breaks the rule stops before any step or model is fitted."""
    from mlprofessor import errors

    frame = p.twice(p.noise_frame(300))
    log = h.Log()
    families = [h.family("fail", log, estimator=h.FailingEstimator)]
    err = p.raised(frame, families=families, steps=[h.step_factory(log)])
    assert isinstance(err, errors.HoldoutCopiesError), f"{type(err).__name__}: {err}"
    assert log.events == []

    err = p.raised(frame, families=families, steps=[h.step_factory(log)], duplicates="keep")
    assert isinstance(err, errors.AllModelsFailedError), f"{type(err).__name__}: {err}"


def test_time_order_has_no_rule():
    gen = np.random.default_rng(85)
    n = 400
    frame = pd.DataFrame({
        "t": np.arange(n, dtype=float),
        "x1": np.array([0.5, 1.5, 2.5, 3.5])[np.arange(n) % 4],
        "x2": (np.arange(n) % 2).astype(float),
        "y": gen.integers(0, 2, size=n),
    })
    for value in ("error", "keep"):
        result, _ = p.run_kept(frame, time_col="t", duplicates=value)
        doc = result.to_dict()
        assert doc["validation"]["scheme"] == "time_ordered"
        assert doc["validation"]["n_holdout_copies"] is None
        assert doc["validation"]["holdout_copy_share"] is None
        assert doc["input"]["duplicates"] == value
        assert c.count_code(result, "holdout_copies_kept") == 0
    err = p.raised(frame.drop(columns=["t"]), families=p.flat_family(h.Log(), n))
    assert err.code == CODE, f"{type(err).__name__}: {err}"


def test_the_message_names_an_undeclared_date_column():
    """D53: copies that differ only in an undeclared date column; the message names it."""
    gen = np.random.default_rng(86)
    n = 400
    frame = pd.DataFrame({
        "when": pd.Series([d.strftime("%Y-%m-%d") for d in h.dates(n)], dtype=object),
        "x1": np.repeat(gen.normal(size=50), 8),
        "y": gen.integers(0, 2, size=n),
    })
    err = p.raised(frame, families=p.flat_family(h.Log(), n))
    assert err.code == CODE, f"{type(err).__name__}: {err}"
    assert "when" in str(err) and "time_col" in str(err), str(err)

    result, _ = p.run_kept(frame, time_col="when")
    assert _validation(result)["n_holdout_copies"] is None

    result, _ = p.run_kept(frame, duplicates="keep")
    pairs = h.warnings_of(result)
    assert ("date_column_undeclared", "when") in pairs
    assert pairs.count(KEPT) == 1


def test_command_line_takes_duplicates(tmp_path):
    frame = p.twice(p.noise_frame(300))
    path = tmp_path / "twice.csv"
    frame.to_csv(path, index=False)
    common = ["profile", "--data", path, "--target", "y", "--profile", "fast", "--n-jobs", "1"]

    out = tmp_path / "default"
    proc = h.run_cli([*common, "--out", out])
    assert proc.returncode == 3, proc.stderr
    lines = proc.stderr.strip().splitlines()
    assert len(lines) == 1, proc.stderr
    assert lines[0].startswith(f"error: {CODE}: "), proc.stderr
    assert h.files_in(out) == []

    out = tmp_path / "keep"
    proc = h.run_cli([*common, "--out", out, "--duplicates", "keep"])
    assert proc.returncode == 0, proc.stderr
    doc = h.load_json_strict(out / "profile.json")
    assert doc["input"]["duplicates"] == "keep"
    assert doc["validation"]["n_holdout_copies"] >= 10
    assert "holdout_copies_kept" in proc.stdout

    out = tmp_path / "error"
    proc = h.run_cli([*common, "--out", out, "--duplicates", "error"])
    assert proc.returncode == 3, proc.stderr
    assert h.files_in(out) == []

    proc = h.run_cli([*common, "--out", tmp_path / "drop", "--duplicates", "drop"])
    assert proc.returncode == 2, proc.stderr

    proc = h.run_cli(["profile", "--help"])
    assert proc.returncode == 0
    assert "--duplicates" in proc.stdout
    json.dumps(doc, allow_nan=False)
