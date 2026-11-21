"""T36: one to nine digits in a fraction of a second (spec step 3, 11; D52).

Red today, every test. Three digits were accepted before and the first pass
holds them, so they are left out here.
"""
import numpy as np
import pandas as pd
import pytest

import cp3_helpers as c
import cp5_helpers as p
import mlp_helpers as h

DIGITS = [1, 2, 4, 5, 6, 7, 8, 9]
ENDINGS = {"none": "", "Z": "Z", "plus_two": "+02:00"}
CASES = [(n, e) for n in DIGITS for e in ENDINGS]


@pytest.mark.parametrize("digits,ending", CASES, ids=[f"{n}-{e}" for n, e in CASES])
def test_one_to_nine_digits_are_read_to_the_nanosecond(digits, ending):
    from mlprofessor import stages

    fraction = "123456789"[:digits]
    tail = ENDINGS[ending]
    values = [f"2024-01-05T09:15:00.{fraction}{tail}", f"2024-01-05 09:15:01.{fraction}{tail}"]
    keys, form = stages.read_time(pd.Series(values))
    assert form == "iso8601"
    assert keys.dtype == np.dtype("datetime64[ns]"), keys.dtype
    first = np.datetime64("2024-01-05T09:15:00", "ns") + np.timedelta64(int(fraction.ljust(9, "0")), "ns")
    if ending == "plus_two":
        first = first - np.timedelta64(2, "h")
    assert keys[0] == first, (keys[0], first)
    assert keys[1] == first + np.timedelta64(1, "s")


def test_digit_counts_may_be_mixed_in_one_column():
    from mlprofessor import stages

    values = ["2024-01-05 09:15:00", "2024-01-05 09:15:00.5", "2024-01-05 09:15:00.25",
              "2024-01-05 09:15:00.125000", "2024-01-05 09:15:00.062500000", None]
    keys, form = stages.read_time(pd.Series(values, dtype=object))
    assert form == "iso8601"
    second = np.datetime64("2024-01-05T09:15:00", "ns")
    offsets = [0, 500_000_000, 250_000_000, 125_000_000, 62_500_000]
    for key, offset in zip(keys[:5], offsets):
        assert key == second + np.timedelta64(offset, "ns"), (key, offset)
    assert np.isnat(keys[5])


REFUSED = ["2024-01-05 09:15:00.1234567891", "2024-01-05T09:15:00.123456789012Z"]


def _text_dates(n, zone):
    stamps = pd.date_range("2024-01-05 09:00:00", periods=n, freq="min")
    if zone:
        return [f"{s:%Y-%m-%dT%H:%M:%S}Z" for s in stamps]
    return [f"{s:%Y-%m-%d %H:%M:%S}" for s in stamps]


def _assert_names_the_fraction(err, refused):
    from mlprofessor import errors

    assert isinstance(err, errors.TimeColumnError), f"{type(err).__name__}: {err}"
    message = str(err)
    assert refused in message, message
    assert "fraction" in message.lower(), message
    assert "not ISO 8601" not in message, message


@pytest.mark.parametrize("refused", REFUSED, ids=["ten_digits", "twelve_digits_Z"])
def test_ten_digits_are_refused_and_the_message_names_the_fraction(refused):
    from mlprofessor import errors, stages

    zone = refused.endswith("Z")
    good = _text_dates(1, zone)[0]
    with pytest.raises(errors.ProfileError) as caught:
        stages.read_time(pd.Series([good, refused]))
    _assert_names_the_fraction(caught.value, refused)

    frame = h.time_frame(n=200)
    dates = _text_dates(200, zone)
    dates[17] = refused
    frame["date"] = pd.Series(dates)
    _assert_names_the_fraction(p.raised(frame, time_col="date"), refused)

    day_first = pd.Series([f"{d:02d}/01/2024" for d in range(1, 32)])
    assert day_first.iloc[24] == "25/01/2024"
    with pytest.raises(errors.ProfileError) as caught:
        stages.read_time(day_first)
    assert isinstance(caught.value, errors.TimeColumnError), type(caught.value).__name__
    assert "fraction" not in str(caught.value).lower(), str(caught.value)


def test_a_csv_pandas_wrote_from_microsecond_timestamps_runs(tmp_path):
    """Finding 10: a CSV that pandas wrote from microsecond timestamps is read."""
    frame = h.time_frame(n=240)
    stamps = pd.date_range("2024-03-01 08:00:00.000001", periods=240, freq="1500us")
    frame["date"] = stamps
    path = tmp_path / "micro.csv"
    frame.to_csv(path, index=False)
    first_line = path.read_text(encoding="utf-8").splitlines()[1]
    assert first_line.split(",")[0] == "2024-03-01 08:00:00.000001"

    from_file = c.run(str(path), time_col="date")
    validation = from_file.to_dict()["validation"]
    assert validation["time_form"] == "iso8601"
    assert validation["timestamps_unique"] is True
    assert (validation["n_train_timestamps"], validation["n_holdout_timestamps"]) == (180, 60)
    assert h.holdout_rows(from_file) == list(range(180, 240))
    assert pd.Timestamp(validation["holdout_start"]) == stamps[180]

    in_memory = c.run(frame, time_col="date")
    assert h.holdout_rows(in_memory) == h.holdout_rows(from_file)
    a, b = from_file.headline["score"], in_memory.headline["score"]
    assert abs(a - b) <= 1e-9 * max(abs(a), abs(b)), (a, b)

    proc = h.run_cli(["profile", "--data", path, "--target", "y", "--time-col", "date",
                      "--out", tmp_path / "out", "--profile", "fast", "--n-jobs", "1"])
    assert proc.returncode == 0, proc.stderr
