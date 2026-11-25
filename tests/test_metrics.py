"""T7: `stages.roc_auc`, `mse`, `skill`, `previous_value` (spec step 11, D5, D10)."""
import numpy as np
import pytest

import mlp_helpers as h


@pytest.mark.parametrize("y,score,expected", [
    # no ties: pairs (pos, neg): 0.35>0.1, 0.35<0.4, 0.8>0.1, 0.8>0.4 -> 3/4
    ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 3 / 4),
    # ties between classes: pos {0.2, 0.7, 0.5}, neg {0.2, 0.5}
    # 0.2: tie + loss = 0.5; 0.7: 2 wins; 0.5: win + tie = 1.5 -> 4/6
    ([0, 1, 0, 1, 1], [0.2, 0.2, 0.5, 0.7, 0.5], 4 / 6),
    # all scores equal
    ([0, 1, 0, 1, 0, 1], [0.3] * 6, 0.5),
    # perfect and inverted
    ([0, 0, 0, 1, 1, 1, 1], [0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9], 1.0),
    ([0, 0, 0, 1, 1, 1, 1], [0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1], 0.0),
    # ten elements, one tie across classes at 0.5:
    # pos {0.9,0.8,0.5,0.45,0.2}, neg {0.7,0.5,0.4,0.3,0.1}
    # 0.9:5, 0.8:5, 0.5:3.5, 0.45:3, 0.2:1 -> 17.5/25
    ([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [0.9, 0.7, 0.8, 0.5, 0.5, 0.4, 0.45, 0.3, 0.2, 0.1], 17.5 / 25),
])
def test_roc_auc_by_hand(y, score, expected):
    from mlprofessor import stages

    assert h.auc_by_hand(y, score) == pytest.approx(expected)
    assert stages.roc_auc(np.array(y), np.array(score)) == pytest.approx(expected, abs=1e-12)


def test_f4_constant_prediction_on_rare_class_has_no_skill():
    """F4: accuracy would read 0.95 here; ROC AUC reads 0.5 and skill 0."""
    from mlprofessor import stages

    y = np.array([1] * 5 + [0] * 95)
    pred = np.zeros(100)
    accuracy = float(np.mean((pred > 0.5).astype(int) == y))
    assert accuracy == pytest.approx(0.95)
    auc = stages.roc_auc(y, pred)
    assert auc == pytest.approx(0.5)
    assert stages.skill("roc_auc", auc, 0.5) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        stages.skill("accuracy", 0.95, 0.5)


def test_one_class_raises():
    from mlprofessor import errors, stages

    with pytest.raises(errors.OneClassError):
        stages.roc_auc(np.zeros(8, dtype=int), np.linspace(0, 1, 8))


def test_mse_and_skill_by_hand():
    from mlprofessor import stages

    assert stages.mse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 5.0])) == pytest.approx(5 / 3)
    assert stages.mse(np.array([2.0, 4.0]), np.array([3.0, 3.0])) == pytest.approx(1.0)
    assert stages.skill("mse", 1.0, 4.0) == pytest.approx(0.75)
    assert stages.skill("mse", 6.0, 4.0) == pytest.approx(-0.5)
    assert stages.skill("roc_auc", 0.8, 0.5) == pytest.approx(0.6)
    assert stages.skill("roc_auc", 0.3, 0.5) == pytest.approx(-0.4)


def _previous_by_hand(y, keys, gap):
    order = np.argsort(keys)
    out = np.full(len(y), np.nan)
    for pos in range(len(order)):
        if pos - (gap + 1) >= 0:
            out[order[pos]] = y[order[pos - (gap + 1)]]
    return out


@pytest.mark.parametrize("gap", [0, 2])
@pytest.mark.parametrize("key_kind", ["float", "datetime"])
def test_previous_value_on_shuffled_keys(gap, key_kind):
    from mlprofessor import stages

    rng = np.random.default_rng(0)
    n = 12
    perm = rng.permutation(n)
    keys = (np.arange(n, dtype=float) * 3.0)[perm]
    if key_kind == "datetime":
        keys = (np.datetime64("2024-01-01", "ns") + (keys.astype(int) * np.timedelta64(1, "D")))
    y = rng.normal(size=n)
    got = np.asarray(stages.previous_value(y, keys, gap=gap), dtype=float)
    exp = _previous_by_hand(y, keys, gap)
    assert np.isnan(got).sum() == gap + 1
    assert np.array_equal(np.isnan(got), np.isnan(exp))
    assert np.allclose(got[~np.isnan(exp)], exp[~np.isnan(exp)])


def test_f8_previous_value_refuses_repeated_keys():
    """F8: the persistence baseline is defined only on unique timestamps."""
    from mlprofessor import stages

    # Until D74 repeated keys were refused. Since D74 the previous value of a row is
    # the mean target of the timestamp before. Either is accepted here (D59);
    # the new behaviour is pinned by tests/test_repeated_stamps.py.
    try:
        got = np.asarray(stages.previous_value(
            np.arange(5.0), np.array([1.0, 2.0, 2.0, 3.0, 4.0])), dtype=float)
    except ValueError:
        return
    assert np.isnan(got[0])
    assert np.allclose(got[1:], [0.0, 0.0, 1.5, 3.0])


def test_f8_previous_value_beats_the_mean_on_a_random_walk():
    """F8: on a random walk the training mean is the wrong yardstick."""
    from mlprofessor import stages

    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=300))
    keys = np.arange(300, dtype=float)
    prev = np.asarray(stages.previous_value(y, keys), dtype=float)
    hold = np.arange(225, 300)
    pv_err = h.mse_by_hand(y[hold], prev[hold])
    mean_err = h.mse_by_hand(y[hold], np.full(len(hold), y[:225].mean()))
    assert pv_err < 0.2 * mean_err
