"""T46: the type matrix, the entity role (spec section 7, "The type matrix",
"The entity role"; step 5, "How an entity column is read"; D52, D57).

Every kind and every extreme it allows (`two` among them, which the first
matrix tries only as the target), as the declared
entity column `who` beside `x1` and `z1` with a 0/1 target and no time column
(`<kind>-group-<extreme>`); and every kind with its ordinary values once more
with a numeric time column and a numeric target
(`<kind>-group-ordinary-timed`). Each case ends in a result or in a named
error; any other exception propagates and fails the case. Which of the two is
not pinned here: `test_groups_random.py` and `test_groups_time.py` do that.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import mlp_helpers as h
import test_type_matrix as tm


def _column(kind, extreme, c):
    if extreme == "one_value":
        column = tm.KINDS[kind]("feature", "ordinary", c)
        return None if column is None else tm.one_value(column)
    return tm.KINDS[kind]("feature", extreme, c)


def build_frame(kind, extreme, timed=False):
    """The table of one case, or (None, None) when the kind does not allow the extreme."""
    c = tm.Ctx()
    column = _column(kind, extreme, c)
    if column is None:
        return None, None
    frame = pd.DataFrame({"x1": c.x1, "z1": c.z1})
    frame["who"] = column
    if timed:
        frame.insert(0, "t", np.arange(tm.N, dtype=float))
        frame["y"] = 2.0 * c.x1 + c.noise
        return frame, dict(target="y", time_col="t", group_col="who")
    frame["y"] = (c.x1 + 0.5 * c.noise > 0).astype(int)
    return frame, dict(target="y", group_col="who")


def _case_ids():
    found = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kind in tm.KINDS:
            for extreme in tm.EXTREMES:
                frame, _ = build_frame(kind, extreme)
                if frame is not None:
                    found.append(f"{kind}-group-{extreme}")
            frame, _ = build_frame(kind, "ordinary", timed=True)
            if frame is not None:
                found.append(f"{kind}-group-ordinary-timed")
    return found


CASE_IDS = _case_ids()


@pytest.mark.parametrize("case", CASE_IDS)
def test_a_result_or_a_named_error_as_the_entity_column(case, tmp_path):
    from mlprofessor import errors
    import mlprofessor

    parts = case.split("-")
    kind, extreme, timed = parts[0], parts[2], parts[-1] == "timed"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame, args = build_frame(kind, extreme, timed=timed)
    names = [str(n) for n in frame.columns]
    data = frame
    if kind == "csv":
        data = str(tmp_path / "table.csv")
        frame.to_csv(data, index=False)
    out = tmp_path / "out"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = h.stated_profile(data, profile="fast", seed=0, n_jobs=1,
                                      out_dir=str(out), **args)
    except errors.ProfileError as exc:
        code = getattr(type(exc), "code", None)
        assert isinstance(code, str) and code, type(exc).__name__
        assert str(exc).strip(), f"{type(exc).__name__} with an empty message"
        assert h.files_in(out) == [], f"{code}: files written {h.files_in(out)}"
    else:
        tm._assert_result(result, names, out)
        assert h.columns_of(result)["who"]["role"] == "group"
        assert result.to_dict()["validation"]["group_col"] == "who"
