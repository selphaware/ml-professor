"""T76: a second route for the small faults of the ninth pass (spec 8.1; D77, D78, D80).

Each fault of M52 to M69 that the files of this pass hold with one test, or on one table,
is held here once more, by another task, scheme or table. A guard that fails means the
application is wrong, not the test.
"""
import math

import numpy as np
import pandas as pd
import pytest

import cp10_helpers as t
import cp6_helpers as g
import cp7_helpers as s
import cp9_helpers as q
import mlp_helpers as h


def _shuffles(log, result):
    """The recorded matrices of the shuffles: every holdout call after the headline's."""
    calls = t.holdout_calls(log, h.holdout_rows(result))
    return calls[0], calls[1:]


def _key(matrix):
    return (tuple(str(c) for c in matrix.columns), tuple(int(i) for i in matrix.index),
            np.asarray(matrix, dtype=float).tobytes())


def test_m52_the_shuffled_matrices_are_the_same_at_any_n_jobs():
    """M52: by the seam, a classification: the same shuffled matrices at n_jobs 1 and 4."""
    frame = h.clf_frame(n=300, seed=9801)
    got = []
    for jobs in (1, 4):
        log, families = t.recording()
        result = s.run(frame, families=families, n_jobs=jobs, **t.RANDOM)
        _, shuffles = _shuffles(log, result)
        assert len(shuffles) == t.REST_SHUFFLES * 3
        got.append((sorted(_key(m) for _, m, _ in shuffles),
                    [tuple(r[k] for k in t.STATEMENT_KEYS) for r in t.doc(result)["columns"]]))
    assert got[0] == got[1]


def test_m53_under_time_order_no_column_outside_the_models_has_a_statement():
    """M53: the time column, a constant, free text and a suspected leak have no statement."""
    rng = np.random.default_rng(9802)
    frame = h.time_frame(n=400, seed=9802)
    n = len(frame)
    words = [f"note {v} on the {c} item" for v, c in
             zip(rng.integers(0, 1000, size=n), rng.choice(["red", "blue"], size=n))]
    frame.insert(1, "flat", 2.5)
    frame.insert(2, "words", words)
    frame.insert(3, "copy", frame["y"].to_numpy() * 1.0)
    result = t.run(frame, {"time_col": "date", "group_col": False})
    assert t.record(result, "words")["reason"] == "free_text"
    assert t.leaks(result) == ["copy"]
    for name in ("date", "flat", "words", "copy", "y"):
        assert t.statement(result, name) == (None, None, None), name
    for name in ("x1", "z1"):
        assert None not in t.statement(result, name), name
    t.assert_rule_17(result)


def test_m54_a_classification_of_noise_lists_nothing():
    """M54: a classification of noise: no fall reaches the floor, the list is empty."""
    rng = np.random.default_rng(9803)
    n = 400
    frame = pd.DataFrame({f"n{k}": rng.normal(size=n) for k in range(1, 5)})
    frame["y"] = rng.integers(0, 2, size=n)
    result = t.run(frame)
    assert t.rests_on(result) == [] == t.list_by_hand(result)
    assert t.floor(result) >= t.REST_FALL


def test_m56_a_relabelled_categorical_copy_shares_one_fall():
    """M56: a categorical column with its levels renamed, and 100 x - 7, share one fall."""
    rng = np.random.default_rng(9804)
    n = 400
    x1 = rng.normal(size=n)
    k = rng.choice(["north", "south", "east"], size=n)
    effect = {"north": 1.5, "south": -1.5, "east": 0.0}
    frame = pd.DataFrame({"k": k, "x1": x1, "x1_far": 100.0 * x1 - 7.0,
                          "k_new": [{"north": "N", "south": "S", "east": "E"}[v] for v in k],
                          "z1": rng.normal(size=n),
                          "y": x1 + np.array([effect[v] for v in k]) + rng.normal(size=n)})
    result = t.run(frame, {"time_col": False, "group_col": False}, task="regression")
    assert t.same(t.statement(result, "x1_far"), t.statement(result, "x1"))
    assert t.same(t.statement(result, "k_new"), t.statement(result, "k"))
    assert not t.same(t.statement(result, "k"), t.statement(result, "x1"))


def test_m69_a_negated_copy_is_moved_with_what_it_copies():
    """M69: by the seam, -3 x1 + 1 is moved by the permutation that moves x1."""
    frame = h.reg_frame(n=300, seed=9805)
    frame.insert(1, "x1_neg", -3.0 * frame["x1"].to_numpy() + 1.0)
    log, families = t.recording()
    result = s.run(frame, families=families, **t.RANDOM)
    (_, base, _), shuffles = _shuffles(log, result)
    assert len(shuffles) == t.REST_SHUFFLES * 4
    a0, b0 = np.asarray(base["x1"], float), np.asarray(base["x1_neg"], float)
    slope, cut = np.polyfit(a0, b0, 1)
    assert slope < 0
    moved = 0
    for _, matrix, _ in shuffles:
        a, b = np.asarray(matrix["x1"], float), np.asarray(matrix["x1_neg"], float)
        assert np.array_equal(a, a0) == np.array_equal(b, b0)
        assert np.allclose(b, slope * a + cut, atol=1e-9)
        moved += not np.array_equal(a, a0)
    assert moved == t.REST_SHUFFLES
    assert t.same(t.statement(result, "x1_neg"), t.statement(result, "x1"))


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_m66_a_model_that_can_predict_no_shuffle_leaves_every_feature_unstated(task):
    """M66: a model that fails on every shuffled matrix: the run finishes, nothing stated."""
    frame = h.reg_frame(n=300, seed=9806) if task == "regression" else h.clf_frame(
        n=300, seed=9806)
    _, plain = t.plain_family()
    expected = s.run(frame, families=plain, **t.RANDOM)
    _, families = t.failing_on_shuffle(column=None)
    result = s.run(frame, families=families, **t.RANDOM)
    for key in ("score", "skill", "skill_low", "skill_high", "model"):
        assert result.headline[key] == expected.headline[key], key
    names = t.features(result)
    for name in names:
        assert t.statement(result, name) == (None, None, None), name
    named = [r["column"] for r in t.warning_records(result, t.NOT_COMPUTED)]
    assert sorted(named) == sorted(names), named
    assert t.rests_on(result) == []
    t.assert_rule_17(result)


def test_m67_a_classification_with_a_short_holdout():
    """M67: 200 rows: the floor is the half-width of the skill interval, above 0.05."""
    result = t.run(h.clf_frame(n=200, seed=9807))
    head = result.headline
    half = (head["skill_high"] - head["skill_low"]) / 2.0
    assert half > t.REST_FALL + 0.01, half
    assert math.isclose(t.floor(result), half, rel_tol=1e-9)
    assert t.rests_on(result) == t.list_by_hand(result)


def test_m68_a_grouped_classification_with_no_interval():
    """M68: 40 entities each of one class, no skill interval: the floor is a stand-in."""
    rng = np.random.default_rng(9808)
    labels = rng.permutation([0] * 20 + [1] * 20)
    who = t.entities(40, 10, seed=9808)
    y = np.array([labels[int(w[1:])] for w in who])
    frame = pd.DataFrame({"who": who, "x1": y + rng.normal(size=len(y)),
                          "z1": rng.normal(size=len(y)), "y": y})
    result = t.run(frame, {"time_col": False, "group_col": "who"})
    head = result.headline
    assert head["skill"] is not None and head["skill_low"] is None
    ours = t.standin_half_width(result)
    assert ours > t.REST_FALL + 0.02, ours
    assert t.floor(result) > t.REST_FALL
    assert abs(t.floor(result) - ours) <= ours / 3.0, (t.floor(result), ours)


def test_m65_with_no_skill_the_summary_still_has_its_line(tmp_path):
    """M65, D36: with no headline skill the summary still prints its line, with `none`."""
    rng = np.random.default_rng(9809)
    n = 240
    x1 = rng.normal(size=n)
    y = 2.0 * x1 + rng.normal(size=n)
    y[150:] = 0.0
    frame = pd.DataFrame({"t": np.arange(n, dtype=float), "x1": x1, "y": y})
    path = t.write_csv(frame, tmp_path)
    out = tmp_path / "out"
    proc = s.run_cli(["profile", "--data", path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1", "--time-col", "t",
                      "--no-entities"])
    assert proc.returncode == 0, proc.stderr
    doc = h.load_json_strict(out / "profile.json")
    assert doc["headline"]["skill"] is None and doc["headline"]["rests_on"] == []
    lines = t.summary_line(proc.stdout)
    assert len(lines) == 1 and "none" in lines[0], proc.stdout


def test_m58_under_the_grouped_scheme_unseen_levels_are_not_scored():
    """M58: entities of five; 40% of the rows hold a level of their own: excluded."""
    frame = s.sized_entities([5] * 120, seed=9810)
    rng = np.random.default_rng(9810)
    n = len(frame)
    own = np.zeros(n, dtype=bool)
    own[rng.choice(n, size=int(0.4 * n), replace=False)] = True
    y = frame["y"].to_numpy()
    band = np.zeros(n, dtype=int)
    band[~own] = t.bands_of(y[~own], 30)
    code = np.array(t.coded(band, 30, seed=9810), dtype=object)
    code[own] = [f"u{v:05d}" for v in rng.permutation(100_000)[:int(own.sum())]]
    frame.insert(3, "code", list(code))
    result = t.run(frame, {"time_col": False, "group_col": "who"})
    assert t.leaks(result) == ["code"], t.leaks(result)
    messages = t.told(result, t.LEAK, "code")
    assert len(messages) == 1 and t.GROUP_COL_SENTENCE in messages[0]


def test_m59_too_few_scored_rows_of_the_negative_class():
    """M59: fewer than ten negative rows scored: the reading has no value, nothing excluded."""
    result = t.run(t.few_scored_table(rare=0, seed=9811))
    assert t.leaks(result) == [], t.leaks(result)
    assert t.record(result, "code")["role"] == "feature"


def test_m60_a_classification_code_of_six_rows_a_value():
    """M60: a class code of six rows a value, 4.5 among the training rows: not read by levels."""
    rng = np.random.default_rng(9812)
    values = 100
    label = rng.permutation([0] * 50 + [1] * 50)
    value = rng.permutation(np.repeat(np.arange(values), 6))
    frame = pd.DataFrame({"x1": rng.normal(size=len(value)),
                          "code": (value * 7 + 11).astype(np.int64), "y": label[value]})
    result = t.run(frame)
    training = t.training_rows_of(result, frame)
    assert frame.loc[training, "code"].value_counts().min() < t.LEVEL_ROWS
    assert t.quiet_feature(result, "code"), s.codes(result)


def _falling(seed):
    rng = np.random.default_rng(seed)
    who = t.entities(400, 5, seed=seed)
    n = len(who)
    value = rng.integers(0, 24, size=n)
    return pd.DataFrame({"who": who, "v": value.astype(np.int64),
                         "z1": rng.normal(size=n), "y": -0.3 * value + rng.normal(size=n)})


def test_m62_a_falling_trend_under_the_grouped_scheme():
    """M62: 24 values whose means fall with the target, and the rows not: not named.

    Which entities the split holds out is open (spec 7, 11), so the table is the first
    of a fixed series whose means are in order on this build's own training rows
    (2.7: the run's rows are `split_random(n, None, seed, groups=...)`)."""
    from mlprofessor import stages

    for seed in range(9813, 9913):
        frame = _falling(seed)
        held = set(int(r) for r in stages.split_random(
            len(frame), None, 0, groups=frame["who"].to_numpy())["holdout"])
        kept = [r for r in range(len(frame)) if r not in held]
        v, y = frame.loc[kept, "v"].tolist(), frame.loc[kept, "y"].tolist()
        if (t.against_by_pairs(v, y) > 2 * q.ORDER_AGAINST
                and t.against_by_means(v, y) <= q.ORDER_AGAINST):
            break
    else:
        pytest.fail("no table of the series has its means in order on the training rows")
    result = t.run(frame, {"time_col": False, "group_col": "who"})
    training = t.training_rows_of(result, frame)
    v, y = frame.loc[training, "v"].tolist(), frame.loc[training, "y"].tolist()
    assert t.against_by_pairs(v, y) > 2 * q.ORDER_AGAINST
    assert t.against_by_means(v, y) <= q.ORDER_AGAINST
    assert t.quiet_feature(result, "v"), s.codes(result)


@pytest.mark.parametrize("seed", [9713, 9714])
def test_m63_a_target_of_noise_with_seven_sentinel_cells(seed):
    """M63: seven of the 450 training targets set to -999: no column excluded."""
    rng = np.random.default_rng(seed)
    n = 600
    frame = pd.DataFrame({f"n{k}": rng.normal(size=n) for k in range(1, 5)})
    frame["y"] = rng.normal(size=n)
    training = t.training_of(frame)
    rows = rng.choice(training, size=7, replace=False)
    frame.loc[rows, "y"] = -999.0
    result = t.run(frame)
    assert set(int(r) for r in rows) <= set(t.training_rows_of(result, frame))
    assert t.leaks(result) == [], t.leaks(result)


def test_m63_under_time_order_with_entities():
    """M63: six walks; 1.5% of the training targets set to -999: no column excluded."""
    frame = g.walks(n_series=6, n_times=100, seed=9815).drop(columns=["prev"])
    rng = np.random.default_rng(9815)
    frame.insert(2, "n1", rng.normal(size=len(frame)))
    frame.insert(3, "n2", rng.normal(size=len(frame)))
    training = [r for r in range(len(frame)) if frame.loc[r, "t"] < 75]
    rows = rng.choice(training, size=int(round(0.015 * len(training))), replace=False)
    frame.loc[rows, "y"] = -999.0
    result = t.run(frame, {"time_col": "t", "group_col": "who"})
    assert sorted(t.training_rows_of(result, frame)) == training
    assert t.leaks(result) == [], t.leaks(result)
