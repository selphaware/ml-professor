"""T57: the fast profile's cap with an entity column and a large entity (spec 2.4, steps 6
and 7, 4.2, 7, 11; D70, D72).

With an entity column and no time column the cap keeps whole entities, and always every
entity of a quarter or more of the usable rows; when it cannot, the run stops with
`fast_profile_group_too_large`, unless the full profile would stop too. Which of the
smaller entities are kept is open: only the four properties of step 6 are asserted.
"""
import numpy as np
import pandas as pd
import pytest

import cp5_helpers as p
import cp7_helpers as s
import cp8_helpers as e
import mlp_helpers as h

GROUPS = {"time_col": False, "group_col": "who"}


def _flat(frame):
    return p.flat_family(h.Log(), len(frame))


def _entity_stop(frame, **kw):
    return s.raised(frame, families=_flat(frame), **{**GROUPS, **kw})


def _group_error_class():
    from mlprofessor import errors

    cls = getattr(errors, "FastProfileGroupError", None)
    assert cls is not None, "mlprofessor.errors has no FastProfileGroupError"
    return cls


def _assert_too_large(err):
    cls = _group_error_class()
    assert type(err) is cls, f"{type(err).__name__}: {err}"
    assert err.code == e.CAP_CODE
    message = str(err)
    assert "2000" in message or "2,000" in message, message
    assert "full" in message, message


def _entity_rows(frame, name):
    return {int(r) for r in np.flatnonzero(frame["who"].to_numpy() == name)}


def _kept(result):
    rows = h.rows_of(result)
    return rows["training"] + rows["holdout"]


def test_the_error_is_named():
    from mlprofessor import errors

    cls = _group_error_class()
    assert issubclass(cls, errors.ProfileError)
    assert cls.code == "fast_profile_group_too_large"
    assert errors.FastProfileRowsError.code == "fast_profile_too_few_rows"
    assert errors.FastProfileError.code == "fast_profile_too_few_timestamps"


def test_an_entity_larger_than_the_cap_stops_the_fast_profile(tmp_path):
    """F15: the fast profile leaves out an entity larger than its cap and reports on the rest."""
    frame = e.large_entities([2500], seed=5701)
    assert len(frame) == 4000
    for seed in (0, 1, 2):
        out = tmp_path / f"out{seed}"
        err = _entity_stop(frame, seed=seed, out_dir=out)
        _assert_too_large(err)
        assert h.files_in(out) == [], seed


def test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept():
    """M35: an entity of a quarter or more of the rows that fits under the cap is always kept."""
    frame = e.large_entities([1200], seed=5702)
    assert len(frame) == 2700 and 4 * 1200 >= 2700
    large = _entity_rows(frame, "e000")
    for seed in range(6):
        log, families = s.rec()
        result = s.run(frame, families=families, seed=seed, **GROUPS)
        reached = e.rows_reached(result, log)
        assert large <= reached, f"seed {seed}: {len(large - reached)} rows of e000 left out"
        assert _kept(result) == e.CAP, (seed, h.rows_of(result))
        assert len(reached) == e.CAP, seed
        assert h.rows_of(result)["dropped_row_cap"] == 700, seed


def test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run():
    frame = e.large_entities([1100, 1100], small=360, seed=5703)
    assert len(frame) == 4000
    for seed in (0, 1):
        _assert_too_large(_entity_stop(frame, seed=seed))

    frame = e.large_entities([900, 900], seed=5704)
    assert len(frame) == 3300 and 4 * 900 >= 3300
    large = _entity_rows(frame, "e000") | _entity_rows(frame, "e001")
    for seed in range(4):
        log, families = s.rec()
        result = s.run(frame, families=families, seed=seed, **GROUPS)
        reached = e.rows_reached(result, log)
        assert large <= reached, f"seed {seed}: {len(large - reached)} rows of e000, e001 left out"
        assert _kept(result) == e.CAP, (seed, h.rows_of(result))


@pytest.mark.parametrize("seed", [0, 1])
def test_the_cap_itself_is_the_boundary(seed):
    from mlprofessor import errors

    frame = e.large_entities([2000], seed=5705)
    err = _entity_stop(frame, seed=seed)
    assert type(err) is errors.FastProfileRowsError, f"{type(err).__name__}: {err}"
    assert err.code == "fast_profile_too_few_rows"
    assert "full" in str(err), str(err)

    frame = e.large_entities([2001], seed=5706)
    err = _entity_stop(frame, seed=seed)
    _assert_too_large(err)


def test_when_the_full_profile_would_stop_too_its_error_is_raised():
    """M34: the fast profile's stop for a large entity is not raised where the full profile would stop too."""
    few = e.large_entities([2500, 300, 300, 300], small=0, seed=5707)
    many = e.large_entities([2500], seed=5708,
                            labels=[0, 1] + [0] * 299)
    cases = [("too_few_groups", few, {}), ("too_few_rows", many, {"task": "classification"})]
    for code, frame, kw in cases:
        full = _entity_stop(frame, profile="full", **kw)
        assert full.code == code, f"{type(full).__name__}: {full}"
        fast = _entity_stop(frame, **kw)
        assert type(fast) is type(full), f"{code}: {type(fast).__name__}: {fast}"
        assert fast.code == code


def _seven_hundred_and_small():
    sizes = [700] + [int(v) for v in np.random.default_rng(2008).integers(1, 12, size=320)]
    return sizes, s.sized_entities(sizes, seed=5709)


def test_the_cap_keeps_whole_entities_every_large_one_and_no_more_than_it_can():
    sizes, frame = _seven_hundred_and_small()
    total = sum(sizes)
    assert total > 2500 and 4 * 700 >= total
    names = frame["who"].tolist()
    size_of = pd.Series(names).value_counts().to_dict()
    holdouts = []
    for seed in range(4):
        log, families = s.rec()
        result = s.run(frame, families=families, seed=seed, **GROUPS)
        reached = e.rows_reached(result, log)
        rows = h.rows_of(result)
        assert len(reached) <= e.CAP, seed
        assert len(reached) == rows["training"] + rows["holdout"], (seed, rows)
        assert rows["dropped_row_cap"] == total - len(reached), (seed, rows)
        kept_count = pd.Series([names[r] for r in reached]).value_counts().to_dict()
        for name, k in kept_count.items():
            assert k == size_of[name], f"seed {seed}: {name} has {k} of its {size_of[name]} rows"
        assert "e000" in kept_count, f"seed {seed}: e000 was left out"
        left_out = [size_of[name] for name in size_of if name not in kept_count]
        assert left_out, f"seed {seed}: no entity was left out"
        assert min(left_out) > e.CAP - len(reached), (
            f"seed {seed}: an entity of {min(left_out)} rows was left out beside "
            f"{len(reached)} kept rows")
        holdouts.append(tuple(sorted(h.holdout_rows(result))))
    assert len(set(holdouts)) > 1, "the four seeds hold out the same rows"


def test_the_full_profile_runs_the_same_table():
    frame = e.large_entities([2500], seed=5701)
    result = s.run(frame, families=_flat(frame), profile="full", **GROUPS)
    rows = h.rows_of(result)
    assert result.to_dict()["validation"]["scheme"] == "grouped"
    assert rows["dropped_row_cap"] == 0
    assert rows["training"] + rows["holdout"] == 4000


def test_it_is_for_an_entity_column_with_no_time_column_only():
    frame = e.large_entities([2500], seed=5701)
    frame.insert(0, "t", np.arange(len(frame), dtype=float))
    result = s.run(frame, families=_flat(frame), time_col="t", group_col="who")
    rows = h.rows_of(result)
    assert result.to_dict()["validation"]["scheme"] == "time_ordered"
    assert rows["dropped_row_cap"] == 2000, rows
    assert rows["training"] + rows["holdout"] + rows["dropped_gap"] == 2000, rows

    frame = s.few_valued(40_000, (4, 4), seed=5710)
    result = s.run(frame, families=_flat(frame), duplicates="group", **s.SAID)
    rows = h.rows_of(result)
    v = result.to_dict()["validation"]
    assert rows["training"] + rows["holdout"] == 2000, rows
    assert rows["dropped_row_cap"] == 38_000, rows
    assert v["scheme"] == "grouped"
    assert v["n_groups_training"] + v["n_groups_holdout"] == 16, v


def test_the_command_line_exits_three(tmp_path):
    frame = e.large_entities([2500], seed=5701)
    csv_path = tmp_path / "table.csv"
    frame.to_csv(csv_path, index=False)
    out = tmp_path / "out"
    proc = s.run_cli(["profile", "--data", csv_path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1", "--group-col", "who",
                      "--no-time-order"])
    assert proc.returncode == 3, proc.stdout + proc.stderr
    lines = proc.stderr.strip().splitlines()
    assert len(lines) == 1, proc.stderr
    assert lines[0].startswith(f"error: {e.CAP_CODE}: "), lines[0]
    assert h.files_in(out) == []
