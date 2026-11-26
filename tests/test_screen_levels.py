"""T71: the leak screen reads a column by its levels (spec step 8, 2.5, 4.2, 5.4, 7, 11;
D77 point 3, D78 points 4 to 6, D80 point 3).

Every categorical column, and a numeric column whose training rows hold at least five
rows for each distinct value, is read by its levels: a validation row is given the mean
target of the fold's training rows of its level, a row whose level those rows do not
hold is not scored, and the reading has a value only with at least 20 rows scored and,
in a classification, 10 of each class or all of one. A column excluded so has the fixed
sentence of 5.4 that names `group_col` in its `suspected_leak` message.
"""
import numpy as np
import pandas as pd
import pytest

import cp10_helpers as t
import cp7_helpers as s
import mlp_helpers as h


def _assert_excluded_by_levels(result, name="code", others=("x1", "z1")):
    assert t.leaks(result) == [name], t.leaks(result)
    assert t.record(result, name)["leak_rule"] == "predicts_target"
    messages = t.told(result, t.LEAK, name)
    assert len(messages) == 1 and t.GROUP_COL_SENTENCE in messages[0], messages
    for other in others:
        assert t.record(result, other)["role"] == "feature", other


@pytest.mark.parametrize("task,text", [("regression", True), ("regression", False),
                                       ("classification", True),
                                       ("classification", False)])
def test_a_coded_copy_of_the_target_is_excluded(task, text):
    """F32, M61: the target in 30 bands, or 40 levels each of one class, labels at random."""
    frame = t.coded_copy_table(task, n=600, seed=9301, text=text)
    result = t.run(frame)
    _assert_excluded_by_levels(result)


def test_under_the_grouped_scheme_too():
    """F32: a text code of a regression's target, the rows of 120 entities of five."""
    frame = s.sized_entities([5] * 120, seed=9302)
    band = t.bands_of(frame["y"], 30)
    frame.insert(3, "code", t.coded(band, 30, seed=9302))
    result = t.run(frame, {"time_col": False, "group_col": "who"})
    assert t.doc(result)["validation"]["scheme"] == "grouped"
    _assert_excluded_by_levels(result, others=("m1", "m2"))


def test_a_column_excluded_by_another_reading_has_no_sentence_on_group_col():
    """An exact numeric copy of the target, every value distinct: no sentence on group_col."""
    frame = h.reg_frame(n=400, seed=9303)
    frame.insert(2, "copy", frame["y"].to_numpy() * 1.0)
    result = t.run(frame)
    assert t.leaks(result) == ["copy"]
    messages = t.told(result, t.LEAK, "copy")
    assert len(messages) == 1 and t.GROUP_COL_SENTENCE not in messages[0], messages


def _unseen_table(task, n=600, seed=9304):
    """60% of the rows hold a coded copy of the target, 40% a level of their own."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    own = np.zeros(n, dtype=bool)
    own[rng.choice(n, size=int(0.4 * n), replace=False)] = True
    names = rng.permutation(100_000)[:n]
    if task == "regression":
        y = x1 + rng.normal(size=n)
        band = np.zeros(n, dtype=int)
        band[~own] = t.bands_of(y[~own], 30)
        k = 30
    else:
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-1.5 * x1))).astype(int)
        band = t.level_of_class(y, 40, rng)
        k = 40
    code = np.array(t.coded(band, k, seed=seed + 1), dtype=object)
    code[own] = [f"u{v:05d}" for v in names[own]]
    return pd.DataFrame({"x1": x1, "z1": rng.normal(size=n), "code": list(code), "y": y})


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_rows_whose_level_the_fold_has_not_seen_are_not_scored(task):
    """M58: 40% of the rows hold a level of their own; on the others the column is the target."""
    result = t.run(_unseen_table(task))
    _assert_excluded_by_levels(result)


def test_a_reading_by_levels_on_too_few_rows_of_a_class_has_no_value():
    """M59: fewer than ten positive rows scored: the reading has no value, nothing excluded."""
    frame = t.few_scored_table(rare=1)
    result = t.run(frame)
    assert t.leaks(result) == [], t.leaks(result)
    assert t.record(result, "code")["role"] == "feature"


def test_few_values_are_counted_over_the_training_rows():
    """M60: six rows for each value, 4.5 among the training rows: not read by its levels."""
    rng = np.random.default_rng(9306)
    n = 600
    x1 = rng.normal(size=n)
    y = x1 + rng.normal(size=n)
    band = t.bands_of(y, 100)
    frame = pd.DataFrame({"x1": x1, "z1": rng.normal(size=n),
                          "code": t.coded(band, 100, seed=9306, text=False), "y": y})
    result = t.run(frame)
    training = t.training_rows_of(result, frame)
    counts = pd.Series(frame.loc[training, "code"]).value_counts()
    assert counts.min() < t.LEVEL_ROWS, counts.min()
    assert t.quiet_feature(result, "code"), s.codes(result)


def test_numeric_bands_with_enough_rows_are_read_by_their_levels():
    """F32: ten rows for each value, at least five of each among the training rows: excluded."""
    rng = np.random.default_rng(9311)
    n = 600
    x1 = rng.normal(size=n)
    y = x1 + rng.normal(size=n)
    band = t.bands_of(y, 60)
    frame = pd.DataFrame({"x1": x1, "z1": rng.normal(size=n),
                          "code": t.coded(band, 60, seed=9311, text=False), "y": y})
    result = t.run(frame)
    training = t.training_rows_of(result, frame)
    counts = pd.Series(frame.loc[training, "code"]).value_counts()
    assert counts.min() >= t.LEVEL_ROWS, counts.min()
    _assert_excluded_by_levels(result)


def test_eight_bands_are_warned_about_and_kept():
    """D78 point 4: a coded copy of eight bands is a feature with strong_single_column."""
    frame = t.coded_copy_table("regression", n=600, seed=9308, k=8)
    result = t.run(frame)
    assert t.leaks(result) == []
    assert t.record(result, "code")["role"] == "feature"
    assert len(t.told(result, t.STRONG, "code")) == 1, s.codes(result)


def _honest(task, numeric=False, n=600, seed=9309):
    """A column of ten levels that explains about half of the target, and noise of 60
    levels."""
    rng = np.random.default_rng(seed)
    level = rng.integers(0, 10, size=n)
    effect = rng.permutation(np.linspace(-1.6, 1.6, 10))
    signal = effect[level]
    if task == "regression":
        y = signal + rng.normal(size=n) * float(np.std(signal))
    else:
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-2.0 * signal))).astype(int)
    ten = (level * 3 + 2).astype(np.int64) if numeric else [f"t{v}" for v in level]
    noise = [f"n{v:02d}" for v in rng.integers(0, 60, size=n)]
    return pd.DataFrame({"ten": ten, "sixty": noise, "x1": rng.normal(size=n), "y": y})


def test_an_honest_column_of_ten_levels_is_a_quiet_feature():
    """Ten levels that explain about half of the target, and noise of 60 levels: no rule."""
    result = t.run(_honest("regression"))
    assert t.quiet_feature(result, "ten") and t.quiet_feature(result, "sixty"), s.codes(result)


def test_in_a_classification_too():
    """The same in a classification: no rule names either column."""
    result = t.run(_honest("classification"))
    assert t.quiet_feature(result, "ten") and t.quiet_feature(result, "sixty"), s.codes(result)


def test_a_numeric_column_of_ten_values_is_a_quiet_feature():
    """Ten whole-number values that explain about half of the target: no rule."""
    result = t.run(_honest("regression", numeric=True))
    assert t.quiet_feature(result, "ten") and t.quiet_feature(result, "sixty"), s.codes(result)
