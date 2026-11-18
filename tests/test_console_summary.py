"""T22: what the console summary shows (spec 2.2; D39). Red today.

Only what is shown is pinned: both intervals, the p-value or why there is
none, and one line per warning. The wording and layout are open.
"""
import numpy as np
import pandas as pd

import cp3_helpers as c
import mlp_helpers as h


def _run(tmp_path, frame, *extra):
    path = tmp_path / "table.csv"
    frame.to_csv(path, index=False)
    out = tmp_path / "out"
    proc = h.run_cli(["profile", "--data", path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1", *extra])
    assert proc.returncode == 0, proc.stderr
    doc = h.load_json_strict(out / "profile.json")
    return doc, proc.stdout


def _assert_every_warning_has_its_line(doc, text):
    assert doc["warnings"], "the table is built to raise warnings"
    missing = c.unmatched_warning_lines(doc["warnings"], text)
    assert missing == [], f"warnings without a line of their own: {missing}\n{text}"


def test_summary_shows_both_intervals_the_p_value_and_each_warning(tmp_path):
    n = 160
    frame = h.reg_frame(n=n, seed=111)
    frame.insert(0, "when", [d.strftime("%Y-%m-%d") for d in h.dates(n)])
    amount = np.random.default_rng(112).normal(size=n).round(3).astype(str).astype(object)
    amount[9] = "abc"
    frame.insert(1, "amount", amount)
    doc, text = _run(tmp_path, frame)

    codes = [w["code"] for w in doc["warnings"]]
    for code in ["date_column_undeclared", "values_coerced", "small_holdout"]:
        assert code in codes
    head = doc["headline"]
    for key in ["interval_low", "interval_high", "skill_low", "skill_high", "p_value"]:
        assert head[key] is not None
        assert c.shown(head[key], text), f"{key} = {head[key]} is not shown:\n{text}"
    _assert_every_warning_has_its_line(doc, text)


def test_summary_under_time_order_says_why_there_is_no_p_value(tmp_path):
    doc, text = _run(tmp_path, h.time_frame(n=200, seed=113), "--time-col", "date")
    head = doc["headline"]
    for key in ["interval_low", "interval_high", "skill_low", "skill_high"]:
        assert head[key] is not None
        assert c.shown(head[key], text), f"{key} = {head[key]} is not shown:\n{text}"
    assert head["p_value"] is None
    assert head["p_value_reason"] and head["p_value_reason"] in text
    _assert_every_warning_has_its_line(doc, text)


def test_summary_says_why_there_is_no_interval(tmp_path):
    panel = h.panel_frame(n_times=40, per_time=3, seed=114)
    doc, text = _run(tmp_path, panel, "--time-col", "date")
    head = doc["headline"]
    assert head["interval_method"] is None
    assert head["interval_reason"] and head["interval_reason"] in text
    assert head["p_value_reason"] and head["p_value_reason"] in text
    _assert_every_warning_has_its_line(doc, text)
