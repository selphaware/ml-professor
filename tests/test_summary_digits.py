"""T29: the console summary prints four significant digits (spec 2.2; D46). Red today.

A value counts as shown when some number printed in the summary is within
0.0005 x |value| of the value in `profile.json`: a relative tolerance, so it
holds at any scale. One test over three runs, so that it is red in every case
today: the second and third runs alone would pass on some of their numbers.
"""
import numpy as np

import cp4_helpers as k
import mlp_helpers as h

KEYS = ["score", "baseline_score", "skill", "interval_low", "interval_high",
        "skill_low", "skill_high", "p_value"]


def _run(tmp_path, name, frame, *extra):
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    out = tmp_path / f"{name}_out"
    proc = h.run_cli(["profile", "--data", path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1", *extra])
    assert proc.returncode == 0, proc.stderr
    return h.load_json_strict(out / "profile.json")["headline"], proc.stdout


def _assert_every_number_shown(name, head, text):
    for key in KEYS:
        value = head[key]
        if value is None:
            continue
        assert k.is_number(value), f"{name}: {key} = {value!r}"
        assert k.shown4(value, text), f"{name}: {key} = {value!r} is not shown to four significant digits:\n{text}"


def test_summary_numbers_agree_with_profile_json_to_four_significant_digits(tmp_path):
    small = h.reg_frame(n=240, seed=61)
    small["y"] = small["y"] * 1e-4
    head, text = _run(tmp_path, "small", small)
    assert head["score"] < 1e-6, "the table is on the scale it is meant to be"
    _assert_every_number_shown("small", head, text)

    large = h.reg_frame(n=240, seed=61)
    large["y"] = large["y"] * 1e6
    head, text = _run(tmp_path, "large", large)
    assert head["score"] > 1e9, "the table is on the scale it is meant to be"
    _assert_every_number_shown("large", head, text)

    flat = h.time_frame(n=240, seed=62).drop(columns=["date"])
    flat.insert(0, "t", np.arange(len(flat), dtype=float))
    flat.loc[150:, "y"] = 0.0
    head, text = _run(tmp_path, "flat", flat, "--time-col", "t")
    assert head["baseline_score"] == 0 and k.is_number(head["baseline_score"])
    assert head["skill"] is None
    assert k.shown4(0.0, text), f"a baseline score of zero is not shown as a zero:\n{text}"
    _assert_every_number_shown("flat", head, text)
