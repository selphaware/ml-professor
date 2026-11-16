"""T5: `stages.split_random` (spec step 7, random scheme)."""
import math

import numpy as np
import pytest


def _arr(a):
    return np.asarray(a, dtype=int)


def _check_increasing(a):
    a = _arr(a)
    assert np.all(np.diff(a) > 0)


@pytest.mark.parametrize("n", [100, 101, 103, 400])
def test_holdout_and_training_partition_rows(n):
    from mlprofessor import stages

    s = stages.split_random(n, seed=0)
    train, hold = _arr(s["training"]), _arr(s["holdout"])
    _check_increasing(train)
    _check_increasing(hold)
    assert len(hold) == math.ceil(n / 4)
    assert not set(train) & set(hold)
    assert sorted(set(train) | set(hold)) == list(range(n))


def _labels(n_a, n_b, seed=0):
    labels = np.array(["a"] * n_a + ["b"] * n_b)
    return np.random.default_rng(seed).permutation(labels)


@pytest.mark.parametrize("n_a,n_b", [(100, 100), (360, 40), (183, 20)])
def test_stratified_holdout(n_a, n_b):
    from mlprofessor import stages

    classes = _labels(n_a, n_b)
    n = n_a + n_b
    s = stages.split_random(n, classes=classes, seed=1)
    hold = _arr(s["holdout"])
    assert len(hold) == math.ceil(n / 4)
    for label, count in [("a", n_a), ("b", n_b)]:
        held = int((classes[hold] == label).sum())
        assert abs(held - count / 4) <= 1, (label, held, count)


@pytest.mark.parametrize("stratified", [False, True])
def test_folds_partition_training_rows(stratified):
    from mlprofessor import stages

    n = 203
    classes = _labels(163, 40) if stratified else None
    s = stages.split_random(n, classes=classes, seed=2)
    train = set(_arr(s["training"]).tolist())
    folds = s["folds"]
    assert len(folds) == 5
    seen = []
    for fold in folds:
        val, fit = _arr(fold["validation"]), _arr(fold["training"])
        _check_increasing(val)
        _check_increasing(fit)
        assert set(val) <= train
        assert set(fit) == train - set(val)
        seen.extend(val.tolist())
    assert sorted(seen) == sorted(train)
    sizes = [len(f["validation"]) for f in folds]
    assert max(sizes) - min(sizes) <= 1
    if stratified:
        train_arr = np.array(sorted(train))
        for label in ["a", "b"]:
            per_class = int((classes[train_arr] == label).sum())
            for fold in folds:
                in_fold = int((classes[_arr(fold["validation"])] == label).sum())
                assert abs(in_fold - per_class / 5) <= 1, (label, in_fold, per_class)


def test_same_seed_same_split_other_seed_other_split():
    from mlprofessor import stages

    a = stages.split_random(200, seed=7)
    b = stages.split_random(200, seed=7)
    c = stages.split_random(200, seed=8)
    assert _arr(a["holdout"]).tolist() == _arr(b["holdout"]).tolist()
    for fa, fb in zip(a["folds"], b["folds"]):
        assert _arr(fa["validation"]).tolist() == _arr(fb["validation"]).tolist()
    assert _arr(a["holdout"]).tolist() != _arr(c["holdout"]).tolist()


def test_holdout_is_drawn_at_random():
    from mlprofessor import stages

    n = 400
    hold = _arr(stages.split_random(n, seed=0)["holdout"])
    assert hold.tolist() != list(range(n // 4))
    assert hold.tolist() != list(range(n - n // 4, n))


@pytest.mark.parametrize("stratified", [False, True])
def test_f1_no_fold_training_part_holds_a_holdout_row(stratified):
    """F1: the selection folds never reach into the holdout."""
    from mlprofessor import stages

    n = 240
    classes = _labels(150, 90) if stratified else None
    s = stages.split_random(n, classes=classes, seed=3)
    hold = set(_arr(s["holdout"]).tolist())
    for fold in s["folds"]:
        assert not hold & set(_arr(fold["training"]).tolist())
        assert not hold & set(_arr(fold["validation"]).tolist())
