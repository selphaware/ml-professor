"""T68: five guards, a second test by another route for F22, M33, M34, M37 and M39 (spec 8
and 8.1, steps 6, 7, 8a and 12, 2.2, 2.5, 5.4). All green before this pass is built: a
guard that fails means the application is wrong, not the test.
"""
import numpy as np
import pandas as pd
import pytest

import cp7_helpers as s
import cp8_helpers as e
import cp9_helpers as q
import mlp_helpers as h


def _lookup(values, log=None):
    return h.family("look", log or h.Log(), estimator=h.LookupEstimator, values=values)


def _bootstrap_table(case):
    rng = np.random.default_rng([7221, len(case)])
    if case == "bootstrap_rows":
        frame = h.clf_frame(n=400, seed=7221)
        guess = 1.0 / (1.0 + np.exp(-2.0 * frame["x1"].to_numpy()))
        return frame, guess, dict(s.SAID)
    if case == "bootstrap_groups":
        frame = s.sized_entities([3] * 200, seed=7222)
        guess = frame["y"].to_numpy(dtype=float) + rng.normal(size=len(frame))
        return frame, guess, {"time_col": False, "group_col": "who"}
    frame = q.walk(n=320, seed=7223)
    guess = frame["y"].to_numpy(dtype=float) + rng.normal(size=len(frame))
    return frame, guess, {"time_col": "t", "group_col": False}


@pytest.mark.parametrize("case", ["bootstrap_rows", "bootstrap_groups", "bootstrap_blocks"])
def test_f22_the_interval_is_the_same_at_any_number_of_cores(case):
    """F22: the resamples do not depend on the number of cores."""
    frame, guess, statements = _bootstrap_table(case)
    one = s.run(frame, families=[_lookup(guess)], n_jobs=1, **statements).to_dict()
    three = s.run(frame, families=[_lookup(guess)], n_jobs=3, **statements).to_dict()
    assert one["headline"]["interval_method"] == case
    assert one["meta"]["n_jobs"] == 1 and three["meta"]["n_jobs"] == 3
    for key in ("interval_low", "interval_high", "skill_low", "skill_high", "p_value"):
        assert one["headline"][key] == three["headline"][key], key
    assert one["headline"]["interval_low"] is not None
    assert h.without(one, "meta.n_jobs", "meta.timing") == \
        h.without(three, "meta.n_jobs", "meta.timing")


def test_m33_three_rows_alike_after_a_removed_leak_stop_a_regression():
    """M33: under `duplicates="group"`, three holdout rows that agree with training rows in
    every feature left after the leak screen stop the run: there is no threshold."""
    from mlprofessor import stages

    rng = np.random.default_rng(7331)
    n = 400
    frame = pd.DataFrame({"n1": rng.normal(size=n), "n2": rng.normal(size=n),
                          "y": rng.normal(size=n)})
    frame = q.with_column(frame, "copy", 3.7 * frame["y"].to_numpy() + 12.0)
    split = stages.split_random(n, None, 0)
    held, training = list(split["holdout"]), list(split["training"])
    for k in range(3):
        frame.loc[held[10 * k], ["n1", "n2"]] = frame.loc[training[10 * k], ["n1", "n2"]].values
    log, families = s.rec()
    err = s.raised(frame, families=families, duplicates="group", **s.SAID)
    assert err.code == "holdout_copies_in_training", f"{type(err).__name__}: {err}"
    assert "copy" in str(err)
    assert 3 in e.whole_numbers_in(str(err)), str(err)
    assert not [ev for ev in log.events if ev["kind"] == "model"]


@pytest.mark.parametrize("profile", ["fast", "full"])
def test_m34_the_command_line_gives_the_full_profiles_own_error(profile, tmp_path):
    """M34: where the full profile would stop too, its own error is the command line's."""
    frame = e.large_entities([2500, 300, 300, 300], small=0, seed=7341)
    csv_path = tmp_path / "entities.csv"
    frame.to_csv(csv_path, index=False)
    out = tmp_path / "out"
    proc = s.run_cli(["profile", "--data", csv_path, "--target", "y", "--out", out,
                      "--profile", profile, "--n-jobs", "1", "--no-time-order",
                      "--group-col", "who"])
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert proc.stderr.strip().startswith("error: too_few_groups:"), proc.stderr
    assert h.files_in(out) == []


def test_m37_the_file_says_what_group_measures(tmp_path):
    """M37: in `profile.json` the `same_features_grouped` warning says what `"group"`
    measures."""
    frame = s.few_valued(600, (4, 4), seed=7371)
    csv_path = tmp_path / "few.csv"
    frame.to_csv(csv_path, index=False)
    out = tmp_path / "out"
    proc = s.run_cli(["profile", "--data", csv_path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1", "--duplicates", "group",
                      "--no-time-order", "--no-entities"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = h.load_json_strict(out / "profile.json")
    found = [w["message"] for w in doc["warnings"] if w["code"] == "same_features_grouped"]
    assert len(found) == 1, [w["code"] for w in doc["warnings"]]
    assert e.GROUP_MEASURES in found[0]


def _cell(case, a, b):
    if case == "dictionary":
        return {"k": int(a), "v": int(b)}
    return (int(a), [int(b), int(a) + int(b)])


@pytest.mark.parametrize("case", ["dictionary", "tuple_with_a_list"])
def test_m39_rows_with_a_dictionary_cell_are_counted_as_repeats(case):
    """M39: rows whose cell cannot be hashed are compared by its text: twelve repeats."""
    rng = np.random.default_rng([7391, len(case)])
    n = 200
    a, b = rng.integers(0, 5, size=n), rng.integers(5, 9, size=n)
    x1, y = rng.normal(size=n), rng.normal(size=n)
    rows = [(_cell(case, a[r], b[r]), x1[r], y[r]) for r in range(n)]
    rows += [(_cell(case, a[r], b[r]), x1[r], y[r]) for r in range(30, 42)]
    order = rng.permutation(len(rows))
    rows = [rows[k] for k in order]
    frame = pd.DataFrame({"tags": pd.Series([r[0] for r in rows], dtype=object),
                          "x1": [r[1] for r in rows], "y": [r[2] for r in rows]})
    seen, repeats = set(), 0
    for r in rows:
        key = (str(r[0]), r[1], r[2])
        repeats += key in seen
        seen.add(key)
    assert repeats == 12
    result = s.run(frame, families=[_lookup(np.zeros(len(frame)))], duplicates="keep",
                   task="regression", **s.SAID)
    assert result.to_dict()["input"]["duplicate_rows"] == 12
