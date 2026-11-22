"""T51: the grouped holdout holds a quarter and no more than it must; a unit of a quarter
or more is named (spec 2.5, 2.7, step 7, 5.4, 11; D64, D66).

Which units are held out is open, so these tests assert property 3 of step 7 from the
result alone (`cp7_helpers.assert_property_three`), and which units the two warnings
name, which does not depend on the seed. Property 3 is checked through
`stages.split_random(..., groups=...)` for seeds 0 to 9, which fits nothing.
"""
import numpy as np
import pytest

import cp7_helpers as s
import mlp_helpers as h

SEEDS = range(10)


def _split(frame, seed, classes=None):
    from mlprofessor import stages

    groups = frame["who"].to_numpy(dtype=object)
    return stages.split_random(len(frame), classes=classes, seed=seed, groups=groups)


def _units(frame, column="who"):
    return frame[column].tolist()


def _holdout_names(frame, holdout_rows, column="who"):
    names = frame[column].tolist()
    return {names[int(r)] for r in holdout_rows}


def _grouped(frame, **kw):
    return s.run(frame, families=[h.family("rec", h.Log())], time_col=False,
                 group_col="who", **kw)


def _one_large(size, rest_size=5, total=400, seed=0, labels=None):
    sizes = [size] + [rest_size] * ((total - size) // rest_size)
    assert sum(sizes) == total
    return s.sized_entities(sizes, seed=seed, labels=labels)


def _assert_named(result, code, rows, share_percent, passage):
    found = s.warning_records(result, code)
    assert len(found) == 1, s.codes(result)
    record = found[0]
    assert record["column"] is None
    text = str(record["message"])
    assert float(rows) in s.numbers_in(text), text
    assert s.holds_number(text, share_percent), text
    assert passage in text, text


@pytest.mark.parametrize("large", [100, 120, 240])
def test_a_large_entity_is_never_held_out_when_smaller_ones_can_fill_a_quarter(large):
    """F13: a large entity is never the holdout while smaller ones can fill a quarter."""
    frame = _one_large(large, seed=1600 + large)
    units = _units(frame)
    for seed in SEEDS:
        split = _split(frame, seed)
        holdout = [int(r) for r in split["holdout"]]
        s.assert_property_three(holdout, units, where=f"seed {seed}")
        assert "e000" not in _holdout_names(frame, holdout), f"seed {seed}"
        assert 100 <= len(holdout) <= 104, (seed, len(holdout))

    result = _grouped(frame, seed=3)
    assert sorted(h.holdout_rows(result)) == [int(r) for r in _split(frame, 3)["holdout"]]
    assert result.to_dict()["validation"]["holdout_share"] == pytest.approx(0.25)


def _long_tailed(seed=1611):
    rng = np.random.default_rng(seed)
    sizes = []
    while sum(sizes) < 600:
        sizes.append(int(min(rng.zipf(1.8), 150)))
    sizes[-1] -= sum(sizes) - 600
    sizes = [size for size in sizes if size > 0]
    assert sum(sizes) == 600
    return sizes


@pytest.mark.parametrize("case", ["long_tailed", "a_handful_of_large", "two_classes"])
def test_the_holdout_holds_a_quarter_and_needs_every_unit_it_has(case):
    classes = None
    if case == "long_tailed":
        frame = s.sized_entities(_long_tailed(), seed=1612)
    elif case == "a_handful_of_large":
        frame = s.sized_entities([80, 70, 60, 50, 50, 40, 30, 20], seed=1613)
    else:
        sizes = [90] + [5] * 10 + [5] * 20
        labels = [1] * 11 + [0] * 20
        frame = s.sized_entities(sizes, seed=1614, labels=labels)
        classes = frame["y"].to_numpy()
    units = _units(frame)
    for seed in SEEDS:
        holdout = [int(r) for r in _split(frame, seed, classes)["holdout"]]
        if classes is None:
            s.assert_property_three(holdout, units, where=f"seed {seed}")
            continue
        for label in (0, 1):
            rows = [r for r in range(len(frame)) if classes[r] == label]
            s.assert_property_three([r for r in holdout if classes[r] == label],
                                    {r: units[r] for r in rows},
                                    where=f"seed {seed}, class {label}")


def test_what_the_holdout_can_spare_goes_back():
    """M26: units the completed holdout can spare go back to the training rows."""
    frame = s.sized_entities([160, 160] + [16] * 5, seed=1621)
    sizes = frame["who"].value_counts().to_dict()
    for seed in SEEDS:
        holdout = [int(r) for r in _split(frame, seed)["holdout"]]
        names = _holdout_names(frame, holdout)
        assert len(holdout) == 160 and len(names) == 1, (seed, len(holdout), sorted(names))
        assert names <= {"e000", "e001"}, (seed, names)
        assert len(sizes) - len(names) == 6


def test_a_large_unit_kept_in_training_is_named():
    """M27: a unit of a quarter or more kept among the training rows gets its warning."""
    frame = _one_large(240, seed=1631)
    result = _grouped(frame)
    _assert_named(result, s.KEPT, 240, 60.0, s.LARGE_KEPT)
    assert s.IS_HOLDOUT not in s.codes(result)
    assert "e000" not in _holdout_names(frame, h.holdout_rows(result))
    assert result.to_dict()["validation"]["holdout_share"] == pytest.approx(0.25)


def test_a_unit_too_large_for_a_sound_split_is_the_holdout_and_is_named():
    frame = _one_large(360, seed=1641)
    result = _grouped(frame)
    assert _holdout_names(frame, h.holdout_rows(result)) == {"e000"}
    assert len(h.holdout_rows(result)) == 360
    v = result.to_dict()["validation"]
    assert v["holdout_share"] == pytest.approx(0.9)
    assert (v["n_groups_training"], v["n_groups_holdout"]) == (8, 1)
    _assert_named(result, s.IS_HOLDOUT, 360, 90.0, s.LARGE_IS_HOLDOUT)
    assert s.KEPT not in s.codes(result)
    assert result.headline["interval_method"] is None
    assert "no_interval" in s.codes(result)


def test_one_warning_for_each_large_unit_and_a_share_of_its_own_class():
    frame = s.sized_entities([160, 160] + [16] * 5, seed=1651)
    result = _grouped(frame)
    _assert_named(result, s.KEPT, 160, 40.0, s.LARGE_KEPT)
    _assert_named(result, s.IS_HOLDOUT, 160, 40.0, s.LARGE_IS_HOLDOUT)

    sizes = [60] + [5] * 8 + [5] * 32
    labels = [1] * 9 + [0] * 32
    frame = s.sized_entities(sizes, seed=1652, labels=labels)
    assert len(frame) == 260
    result = _grouped(frame)
    assert result.to_dict()["task"]["type"] == "classification"
    _assert_named(result, s.KEPT, 60, 60.0, s.LARGE_KEPT)
    assert s.IS_HOLDOUT not in s.codes(result)
    assert "e000" not in _holdout_names(frame, h.holdout_rows(result))


def test_no_large_unit_no_warning():
    frame = s.sized_entities([5] * 80, seed=1661)
    result = _grouped(frame)
    found = s.codes(result)
    assert s.KEPT not in found and s.IS_HOLDOUT not in found
    s.assert_property_three(h.holdout_rows(result), _units(frame), where="the run")


def test_the_rule_holds_for_rows_that_agree_in_every_feature():
    rng = np.random.default_rng(1671)
    c1 = ["a0"] * 240
    c2 = ["b0"] * 240
    for i in range(1, 5):
        for j in range(1, 5):
            c1.extend([f"a{i}"] * 10)
            c2.extend([f"b{j}"] * 10)
    import pandas as pd

    frame = pd.DataFrame({"c1": c1, "c2": c2, "y": rng.normal(size=400)})
    frame = frame.iloc[rng.permutation(400)].reset_index(drop=True)
    result = s.run(frame, families=[h.family("rec", h.Log())], duplicates="group", **s.SAID)
    units = [f"{a}|{b}" for a, b in zip(frame["c1"], frame["c2"])]
    holdout = h.holdout_rows(result)
    s.assert_property_three(holdout, units, where="combinations")
    assert "a0|b0" not in {units[r] for r in holdout}
    _assert_named(result, s.KEPT, 240, 60.0, s.LARGE_KEPT)
    assert result.to_dict()["validation"]["n_rows_same_features"] == 400
