"""T22: the output folder, process settings, the no-features message (spec 2.1, 2.4; D39).

Red today: the unusable `out_dir`, the CSV field size limit, the no-features message.
Green today: a new `out_dir` is created; the thread limits are left as found.
"""
import csv

import numpy as np
import pandas as pd
import pytest

import cp3_helpers as c
import mlp_helpers as h


# ------------------------------------------------------------- out_dir

def _blocked_out_dir(tmp_path, case):
    blocker = tmp_path / "a_file.txt"
    blocker.write_text("keep me\n", encoding="utf-8")
    out = blocker if case == "names_a_file" else blocker / "below"
    return blocker, out


@pytest.mark.parametrize("case", ["names_a_file", "below_a_file"])
def test_unusable_out_dir_is_an_input_error_before_anything_is_loaded(tmp_path, case):
    import mlprofessor
    from mlprofessor.errors import ProfileError

    blocker, out = _blocked_out_dir(tmp_path, case)

    duplicated = pd.DataFrame([[1.0, 2.0, 0], [2.0, 3.0, 1]] * 60, columns=["a", "a", "y"])
    with pytest.raises(ProfileError) as caught:
        mlprofessor.profile(duplicated, "y", out_dir=out, **h.FAST)
    assert caught.value.code == "input_error", "the out_dir is judged before the table is loaded"

    log = h.Log()
    with pytest.raises(ProfileError) as caught:
        c.run(h.clf_frame(n=200, seed=101), families=[h.family("rec", log)],
              steps=[h.step_factory(log)], out_dir=out)
    assert caught.value.code == "input_error"
    assert log.events == []
    assert blocker.read_text(encoding="utf-8") == "keep me\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a_file.txt"]


def test_out_dir_that_does_not_exist_yet_is_created(tmp_path):
    out = tmp_path / "one" / "two" / "three"
    c.run(h.clf_frame(n=200, seed=102), out_dir=out)
    assert h.files_in(out) == ["predictions.csv", "profile.json"]


# ------------------------------------------------------ process settings

def _good_csv(tmp_path):
    path = tmp_path / "good.csv"
    h.clf_frame(n=160, seed=103).to_csv(path, index=False)
    return path


def _duplicate_csv(tmp_path):
    path = tmp_path / "dup.csv"
    rows = "".join(f"{i},{i % 5},{i * 0.5},{i % 2}\n" for i in range(160))
    path.write_text("a,b,a,y\n" + rows, encoding="utf-8")
    return path


def _call_profile(path):
    import mlprofessor
    return h.stated_profile(str(path), "y", **h.FAST)


def _call_load(path):
    from mlprofessor import stages
    return stages.load_table(str(path))


CASES = {
    "profile_good": (_call_profile, _good_csv, False),
    "load_table_good": (_call_load, _good_csv, False),
    "profile_duplicate_columns": (_call_profile, _duplicate_csv, True),
    "load_table_duplicate_columns": (_call_load, _duplicate_csv, True),
}


@pytest.mark.parametrize("case", list(CASES))
def test_csv_field_size_limit_is_left_as_found(tmp_path, case):
    from mlprofessor.errors import DuplicateColumnsError

    call, make, raises = CASES[case]
    path = make(tmp_path)
    original = csv.field_size_limit(100_000)
    try:
        if raises:
            with pytest.raises(DuplicateColumnsError):
                call(path)
        else:
            call(path)
        assert csv.field_size_limit() == 100_000
    finally:
        csv.field_size_limit(original)


def _thread_state():
    from threadpoolctl import threadpool_info

    return sorted((p.get("user_api"), p.get("internal_api"), p.get("num_threads"))
                  for p in threadpool_info())


def test_thread_limits_are_left_as_found():
    from mlprofessor.errors import AllModelsFailedError

    frame = h.clf_frame(n=200, seed=104)
    c.run(frame)  # every library is loaded from here on
    before = _thread_state()
    c.run(frame, n_jobs=2)
    assert _thread_state() == before

    with pytest.raises(AllModelsFailedError):
        c.run(frame, families=[h.family("broken", h.Log(), h.FailingEstimator)], n_jobs=2)
    assert _thread_state() == before


# ------------------------------------------------------ no_features message

def test_no_features_message_names_each_excluded_column_and_its_reason():
    from mlprofessor.errors import NoFeaturesError

    n = 160
    rng = np.random.default_rng(105)
    y = (rng.random(n) < 0.5).astype(int)
    unusable = pd.DataFrame({
        "flat": np.full(n, 3.0),
        "serial": np.arange(n),
        "handle": [f"user{i:05d}" for i in range(n)],
        "y": y,
    })
    with pytest.raises(NoFeaturesError) as caught:
        c.run(unusable)
    message = str(caught.value)
    for name in ["flat", "serial", "handle", "constant", "id_like"]:
        assert name in message

    leaky = pd.DataFrame({"serial": np.arange(n), "mirror": y, "y": y})
    with pytest.raises(NoFeaturesError) as caught:
        c.run(leaky)
    message = str(caught.value)
    for name in ["mirror", "suspected_leak", "serial", "id_like"]:
        assert name in message
