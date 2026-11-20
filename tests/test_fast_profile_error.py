"""T28: the fast profile's error only when the full profile would run (spec steps 6, 7, 2.4; D46).

Red today: `test_fast_raises_the_full_profiles_error_when_full_would_stop_too`.
Green today: `test_fast_error_stands_when_the_full_profile_would_run` (it
fails on M7, where too few timestamps under the cap is always `too_few_rows`).

Errors are compared by exact class and by `code`: `pytest.raises` of a parent
class alone would accept the wrong one.
"""
import pytest

import cp3_helpers as c
import cp4_helpers as k
import mlp_helpers as h

FAST_CODE = "fast_profile_too_few_timestamps"


def _raised(frame, profile, gap=0):
    """The named error the run raises (it must raise one)."""
    from mlprofessor.errors import ProfileError

    with pytest.raises(ProfileError) as caught:
        c.run(frame, families=[h.family("rec", h.Log())], time_col="t", gap=gap,
              profile=profile, n_jobs=1)
    return caught.value


STOPS_TOO = {
    "five_dates_of_600": (5, 600, 0, "too_few_rows"),
    "three_dates_of_2500_cap_keeps_nothing": (3, 2500, 0, "too_few_rows"),
    "nine_dates_of_450_gap_2": (9, 450, 2, "gap_too_large"),
}


@pytest.mark.parametrize("case", list(STOPS_TOO))
def test_fast_raises_the_full_profiles_error_when_full_would_stop_too(case):
    """D46: the fast profile never sends the caller to a full profile that stops too."""
    n_times, per_time, gap, code = STOPS_TOO[case]
    frame = k.panel(n_times, per_time, seed=51)

    full = _raised(frame, "full", gap)
    assert full.code == code, f"full profile: {type(full).__name__}: {full}"

    fast = _raised(frame, "fast", gap)
    assert fast.code != FAST_CODE, f"fast profile says to use full, which stops too: {fast}"
    assert type(fast) is type(full), f"fast {type(fast).__name__} vs full {type(full).__name__}"
    assert fast.code == full.code == code


RUNS_UNDER_FULL = {
    "eight_dates_of_450": (8, 450),
    "twelve_dates_of_2500_latest_over_the_cap": (12, 2500),
}


@pytest.mark.parametrize("case", list(RUNS_UNDER_FULL))
def test_fast_error_stands_when_the_full_profile_would_run(case):
    """M7: the fast profile's own error is raised when full would get past the split."""
    from mlprofessor.errors import FastProfileError

    n_times, per_time = RUNS_UNDER_FULL[case]
    frame = k.panel(n_times, per_time, seed=52)

    fast = _raised(frame, "fast")
    assert type(fast) is FastProfileError, f"{type(fast).__name__}: {fast}"
    assert fast.code == FAST_CODE

    result = c.run(frame, families=[h.family("rec", h.Log())], time_col="t",
                   profile="full", n_jobs=1)
    assert result.to_dict()["validation"]["scheme"] == "time_ordered"
    assert h.rows_of(result)["dropped_row_cap"] == 0
    assert h.rows_of(result)["loaded"] == n_times * per_time
