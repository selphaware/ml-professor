"""T69: what the headline rests on (spec step 12a, 2.2, 4.2, 4.3, 5.2, 5.3 rules 17 to 19;
D77 point 1, D78, D80).

On every finished run each feature's record holds `shuffled_score` (the chosen
configuration's score on the holdout with that feature's holdout values shuffled, the
mean of five shuffles), `shuffled_skill` and `skill_fall`; copies as the models receive
them are shuffled together; `headline.rests_on` lists the features whose fall reaches
`headline.rests_on_floor`, at most five, largest first, ties in file order; the console
summary prints one line for it. Which permutations are drawn is open (spec 11): a
shuffled score is recomputed only through a seam model whose predictions the test
records.
"""
import math

import numpy as np
import pandas as pd
import pytest

import cp10_helpers as t
import cp7_helpers as s
import mlp_helpers as h


def _carriers(n=400, seed=0):
    """`x1` and `x2` carry the target, `z1` and `z2` are noise (`mlp_helpers.reg_frame`)."""
    return h.reg_frame(n=n, seed=seed)


def test_every_feature_has_a_statement_and_no_other_column_has_one():
    """F29, M53: each feature's statement is of a shuffle; no other column has one."""
    frame = _carriers(seed=9101)
    n = len(frame)
    frame.insert(0, "rid", np.arange(n, dtype=np.int64))
    frame.insert(1, "flat", 3.0)
    frame.insert(len(frame.columns) - 1, "copy", frame["y"].to_numpy() * 1.0)
    result = t.run(frame)
    assert t.leaks(result) == ["copy"]
    head = result.headline
    for name in ("x1", "x2", "z1", "z2"):
        score, shuffled, dropped = t.statement(result, name)
        assert score is not None and shuffled is not None and dropped is not None, name
    for name in ("x1", "x2"):
        score, _, dropped = t.statement(result, name)
        assert score > head["score"], (name, score, head["score"])
        assert dropped > head["rests_on_floor"], (name, dropped, head["rests_on_floor"])
    for name in ("y", "rid", "flat", "copy"):
        assert t.statement(result, name) == (None, None, None), name
    t.assert_rule_17(result)


def test_a_classification_states_its_falls_in_skill_of_roc_auc():
    """The statement of a classification is in skill of ROC AUC (5.3 rule 17)."""
    frame = h.clf_frame(n=400, seed=9102)
    result = t.run(frame)
    head = result.headline
    for name in ("x1", "z1", "z2"):
        score, shuffled, dropped = t.statement(result, name)
        assert 0.0 <= score <= 1.0
        assert math.isclose(shuffled, 2.0 * score - 1.0, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(dropped, head["skill"] - shuffled, rel_tol=1e-9, abs_tol=1e-12)
    assert t.fall(result, "x1") > head["rests_on_floor"]
    t.assert_rule_17(result)


def _copies_frame(n=400, seed=9103):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    k = rng.choice(["red", "green", "blue"], size=n)
    effect = {"red": 1.0, "green": -1.0, "blue": 0.0}
    y = 2.0 * x1 + x2 + np.array([effect[v] for v in k]) + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x1_scaled": 2.0 * x1 + 3.0, "x1_neg": -x1, "x2": x2,
                         "k": k, "z1": rng.normal(size=n), "y": y})


def _parts(matrix, name):
    return [c for c in matrix.columns if str(c) == name or str(c).startswith(f"{name}=")]


def test_the_shuffles_move_one_set_of_columns_among_the_holdout_rows():
    """F29, F31, M56, M69: after the headline's call, five shuffles of each set of copies."""
    frame = _copies_frame()
    log, families = t.recording()
    result = s.run(frame, families=families, **t.RANDOM)
    hold = h.holdout_rows(result)
    calls = t.holdout_calls(log, hold)
    sets = [("x1", "x1_scaled", "x1_neg"), ("x2",), ("k",), ("z1",)]
    assert len(calls) == 1 + t.REST_SHUFFLES * len(sets), len(calls)
    oid, base, base_pred = calls[0]
    assert all(c[0] == oid for c in calls), "one fitted configuration predicts every time"
    truth = pd.Series(np.asarray(result.predictions["y_true"], dtype=float),
                      index=[int(r) for r in result.predictions["row"]])
    guess = dict(zip((int(r) for r in result.predictions["row"]), result.predictions["y_pred"]))
    assert np.allclose(base_pred, [guess[int(i)] for i in base.index], rtol=1e-9, atol=0)
    slope = np.polyfit(np.asarray(base["x1"], float), np.asarray(base["x1_neg"], float), 1)[0]
    assert slope < 0
    scores = {group: [] for group in sets}
    for _, matrix, pred in calls[1:]:
        assert list(matrix.index) == list(base.index), "the holdout rows, as for the headline"
        moved = [c for c in matrix.columns
                 if not np.array_equal(np.asarray(matrix[c], float), np.asarray(base[c], float))]
        touched = [g for g in sets if any(c in moved for name in g for c in _parts(base, name))]
        assert len(touched) == 1, (moved, touched)
        group = touched[0]
        own = [c for name in group for c in _parts(base, name)]
        assert set(moved) <= set(own), (moved, group)
        for c in own:
            assert sorted(np.asarray(matrix[c], float)) == sorted(np.asarray(base[c], float))
        if len(group) > 1:
            for name in group[1:]:
                a = np.asarray(matrix["x1"], float)
                b = np.asarray(matrix[name], float)
                b0 = np.asarray(base[name], float)
                a0 = np.asarray(base["x1"], float)
                fit = np.polyfit(a0, b0, 1)
                assert np.allclose(b, fit[0] * a + fit[1], atol=1e-9), name
        y = truth.loc[[int(i) for i in matrix.index]].to_numpy()
        scores[group].append(float(np.mean((y - pred) ** 2)))
    for group, values in scores.items():
        assert len(values) == t.REST_SHUFFLES, (group, len(values))
        for name in group:
            assert math.isclose(t.statement(result, name)[0], float(np.mean(values)),
                                rel_tol=1e-9), name


def test_copies_are_shuffled_together_and_share_one_fall():
    """F31, M55, M56, M69: an exact, a rescaled and a negated copy, and renamed levels."""
    rng = np.random.default_rng(9104)
    n = 400
    x1 = rng.normal(size=n)
    k = rng.choice(["red", "green", "blue", "grey"], size=n)
    effect = {"red": 1.5, "green": -1.5, "blue": 0.5, "grey": -0.5}
    renamed = {"red": "r", "green": "g", "blue": "b", "grey": "e"}
    y = 2.0 * x1 + np.array([effect[v] for v in k]) + rng.normal(size=n)
    frame = pd.DataFrame({"x1": x1, "c_same": x1.copy(), "b_scaled": 2.0 * x1 + 3.0,
                          "a_neg": -x1, "k": k, "k_renamed": [renamed[v] for v in k],
                          "z1": rng.normal(size=n), "y": y})
    result = t.run(frame)
    first = t.statement(result, "x1")
    for name in ("c_same", "b_scaled", "a_neg"):
        assert t.same(t.statement(result, name), first), (name, t.statement(result, name), first)
    assert t.same(t.statement(result, "k_renamed"), t.statement(result, "k"))
    assert not t.same(t.statement(result, "z1"), first)
    assert first[2] > result.headline["rests_on_floor"]
    assert t.listed(result)[:4] == ["x1", "c_same", "b_scaled", "a_neg"], t.rests_on(result)
    assert t.rests_on(result) == t.list_by_hand(result)


def test_a_column_that_differs_in_one_holdout_row_is_no_copy():
    """A column equal to x1 but in one holdout cell is no copy: each falls by less."""
    frame = _carriers(seed=9105)
    together = frame.copy()
    together.insert(1, "x1_alt", frame["x1"].to_numpy().copy())
    joint = t.run(together)
    assert t.same(t.statement(joint, "x1_alt"), t.statement(joint, "x1"))
    row = h.holdout_rows(joint)[0]
    apart = together.copy()
    apart.loc[row, "x1_alt"] = apart.loc[row, "x1_alt"] + 0.5
    result = t.run(apart)
    assert row in h.holdout_rows(result)
    assert not t.same(t.statement(result, "x1"), t.statement(result, "x1_alt"))
    both = t.fall(joint, "x1")
    assert t.fall(result, "x1") < both and t.fall(result, "x1_alt") < both


def test_the_statement_is_the_same_at_any_n_jobs():
    """M52: the shuffles are drawn from a stream of their own, whatever n_jobs."""
    frame = _carriers(seed=9106)
    one = t.run(frame, n_jobs=1)
    three = t.run(frame, n_jobs=3)
    for rec_one, rec_three in zip(t.doc(one)["columns"], t.doc(three)["columns"]):
        assert [rec_one[k] for k in t.STATEMENT_KEYS] == [rec_three[k] for k in t.STATEMENT_KEYS]
    assert t.rests_on(one) == t.rests_on(three)
    assert t.floor(one) == t.floor(three)


def test_the_list_holds_the_columns_whose_fall_reaches_the_floor():
    """M54: two columns carry the headline and two are noise: the list names the two."""
    result = t.run(_carriers(seed=9107))
    assert t.listed(result) == ["x1", "x2"], t.rests_on(result)
    assert t.rests_on(result) == t.list_by_hand(result)
    for name in ("z1", "z2"):
        assert t.fall(result, name) < t.floor(result)


def test_at_most_five_are_listed_largest_first():
    """Seven columns of one weight each: five listed, by fall from the largest."""
    rng = np.random.default_rng(9108)
    n = 600
    frame = pd.DataFrame({f"w{k}": rng.normal(size=n) for k in range(1, 8)})
    frame["y"] = frame.sum(axis=1) + rng.normal(size=n)
    result = t.run(frame)
    got = t.rests_on(result)
    assert len(got) == t.REST_MOST
    falls = sorted((t.fall(result, f"w{k}") for k in range(1, 8)), reverse=True)
    assert [v for _, v in got] == falls[:t.REST_MOST]
    assert all(v >= t.floor(result) for v in falls), falls
    assert got == t.list_by_hand(result)


def test_equal_falls_go_in_file_order():
    """M55: two exact copies share one fall and are listed in file order."""
    frame = _carriers(seed=9109).rename(columns={"x1": "z_first"})
    frame.insert(1, "a_second", frame["z_first"].to_numpy().copy())
    result = t.run(frame)
    names = t.listed(result)
    assert t.fall(result, "z_first") == t.fall(result, "a_second")
    assert names.index("z_first") + 1 == names.index("a_second"), names
    assert t.rests_on(result) == t.list_by_hand(result)


def test_the_floor_is_the_half_width_of_the_skill_interval():
    """M67: a skill interval whose half-width is above 0.05: the floor is that half-width."""
    result = t.run(_carriers(seed=9110))
    head = result.headline
    half = (head["skill_high"] - head["skill_low"]) / 2.0
    assert half > t.REST_FALL + 0.01, half
    assert math.isclose(t.floor(result), half, rel_tol=1e-9)
    assert t.floor(result) == pytest.approx(t.floor_by_hand(head), rel=1e-12)
    assert t.rests_on(result) == t.list_by_hand(result)


def _grouped_regression(n_entities=40, per=10, seed=9111):
    rng = np.random.default_rng(seed)
    who = t.entities(n_entities, per, seed=seed)
    n = len(who)
    x1 = rng.normal(size=n)
    return pd.DataFrame({"who": who, "x1": x1, "z1": rng.normal(size=n),
                         "y": 2.0 * x1 + 1.5 * rng.normal(size=n)})


def test_where_the_headline_has_no_interval_the_floor_stands_in():
    """M68: fewer than 30 holdout entities, no skill interval: the floor is a stand-in."""
    frame = _grouped_regression()
    result = t.run(frame, {"time_col": False, "group_col": "who"})
    head = result.headline
    assert t.doc(result)["validation"]["scheme"] == "grouped"
    assert head["skill"] is not None and head["skill_low"] is None
    ours = t.standin_half_width(result, frame)
    assert ours > t.REST_FALL + 0.02, ours
    assert t.floor(result) > t.REST_FALL
    assert abs(t.floor(result) - ours) <= ours / 3.0, (t.floor(result), ours)
    assert t.rests_on(result) == t.list_by_hand(result)


def _flat_holdout(n=240, seed=9112):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    y = 2.0 * x1 + rng.normal(size=n)
    y[150:] = 0.0
    return pd.DataFrame({"t": np.arange(n, dtype=float), "x1": x1,
                         "z1": rng.normal(size=n), "y": y})


def test_with_no_headline_skill_a_statement_holds_the_score_alone():
    """D36: with no headline skill a feature has its score and null skill and fall."""
    result = t.run(_flat_holdout(), {"time_col": "t", "group_col": False})
    assert result.headline["skill"] is None
    for name in ("x1", "z1"):
        score, shuffled, dropped = t.statement(result, name)
        assert isinstance(score, float) and math.isfinite(score)
        assert shuffled is None and dropped is None
    assert t.rests_on(result) == []
    assert t.floor(result) is None
    t.assert_rule_17(result)


def test_the_summary_prints_one_line_for_the_list(tmp_path):
    """M65: one line `rests on`, the floor and each listed column with its fall; or `none`."""
    path = t.write_csv(_carriers(seed=9113), tmp_path)
    out = tmp_path / "out"
    proc = s.run_cli(["profile", "--data", path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1", "--no-time-order",
                      "--no-entities"])
    assert proc.returncode == 0, proc.stderr
    doc = h.load_json_strict(out / "profile.json")
    lines = t.summary_line(proc.stdout)
    assert len(lines) == 1, proc.stdout
    line = lines[0]
    head = doc["headline"]
    assert head["rests_on"], "the table is built to rest on columns"
    assert t.holds_value(line, head["rests_on_floor"]), line
    at = -1
    for entry in head["rests_on"]:
        quoted = repr(entry["column"])
        assert quoted in line[at + 1:], (quoted, line)
        at = line.index(quoted, at + 1)
        assert t.holds_value(line, entry["skill_fall"]), (entry, line)

    rng = np.random.default_rng(9114)
    noise = pd.DataFrame({"n1": rng.normal(size=300), "n2": rng.normal(size=300),
                          "y": rng.normal(size=300)})
    path = t.write_csv(noise, tmp_path, "noise.csv")
    out = tmp_path / "noise_out"
    proc = s.run_cli(["profile", "--data", path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1", "--no-time-order",
                      "--no-entities"])
    assert proc.returncode == 0, proc.stderr
    doc = h.load_json_strict(out / "profile.json")
    assert doc["headline"]["rests_on"] == []
    lines = t.summary_line(proc.stdout)
    assert len(lines) == 1 and "none" in lines[0], proc.stdout


@pytest.mark.parametrize("infinite", [False, True])
def test_a_model_that_cannot_predict_a_shuffled_matrix_leaves_that_column_unstated(infinite):
    """M66: a model that fails on a shuffle of x2: the run finishes, x2 is unstated."""
    frame = _carriers(seed=9115)
    _, plain = t.plain_family()
    expected = s.run(frame, families=plain, **t.RANDOM)
    _, families = t.failing_on_shuffle(column="x2", infinite=infinite)
    result = s.run(frame, families=families, **t.RANDOM)
    for key in ("score", "skill", "skill_low", "skill_high", "baseline_score", "model"):
        assert result.headline[key] == expected.headline[key], key
    assert t.statement(result, "x2") == (None, None, None)
    named = t.warning_records(result, t.NOT_COMPUTED)
    assert [r["column"] for r in named] == ["x2"], named
    for name in ("x1", "z1", "z2"):
        assert t.same(t.statement(result, name), t.statement(expected, name)), name
    t.assert_rule_17(result)
