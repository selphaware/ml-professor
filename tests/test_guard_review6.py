"""T54: guards of the seventh pass (spec 8.1, steps 3, 6, 8, 9, 10, 11 and 12, 2.2, 2.5
and 5.2; D71, D72).

Eleven second tests, each by another route than the first, for faults that one test
each held alone (M1, M5, M6, M8, M10, M11, M12, M17, M21, M23, M28); two readings D72
pinned as the application has them (M38, M39); and a date beyond the nanosecond range
among ordinary ones. Every one holds behaviour the application already has: a guard
that fails means the application is wrong, not the test.
"""
import csv
import datetime as dt
import math
import os
import statistics
import subprocess
from fractions import Fraction
from math import comb

import numpy as np
import pandas as pd
import pytest

import cp3_helpers as c
import cp4_helpers as k
import cp7_helpers as s
import mlp_helpers as h


# ------------------------------------------------------------------ helpers

def _lookup(values):
    return h.family("lookup", h.Log(), estimator=h.LookupEstimator, values=values)


def _cli(csv_path, target, out, *more):
    return s.run_cli(["profile", "--data", csv_path, "--target", target, "--out", out,
                      "--profile", "fast", "--n-jobs", "1", *more])


def _files(out):
    doc = h.load_json_strict(out / "profile.json")
    with open(out / "predictions.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return doc, rows


def _rows(doc):
    return {r["item"]: int(r["count"]) for r in doc["rows"]}


# ------------------------------------------------------------------- M1

def _rising_and_gappy():
    rng = np.random.default_rng(6101)
    n = 240
    t = np.arange(n, dtype=float)
    rising = 0.05 * t + rng.normal(size=n)
    gappy = 0.05 * t + rng.normal(size=n)
    gappy[rng.random(n) < 0.15] = np.nan
    y = rng.normal(size=n)
    filled = [r for r in range(180) if not np.isnan(gappy[r])]
    if len(filled) % 2 == 0:
        gappy[filled[0]] = np.nan
    return pd.DataFrame({"t": t, "rising": rising, "gappy": gappy, "y": y})


def test_m1_the_final_fit_is_centred_on_its_own_rows():
    """M1: under time order the final fit's scaling and imputation come from the training rows only."""
    frame = _rising_and_gappy()
    log = c.KeepLog()
    result = s.run(frame, families=[h.family("keep", log, c.MatrixEstimator)],
                   time_col="t", group_col=False, task="regression")
    assert sorted(h.holdout_rows(result)) == list(range(180, 240))

    fits = log.calls("model", "fit")
    assert len(fits) == 6, [len(ev["rows"]) for ev in fits]
    for ev in fits:
        assert max(ev["rows"]) < 180, "a fit was handed a holdout row"
    final = [ev for ev in fits if sorted(ev["rows"]) == list(range(180))]
    assert len(final) == 1, [len(ev["rows"]) for ev in fits]

    for ev in fits:
        column = ev["values"][:, ev["columns"].index("rising")]
        assert abs(column.mean()) <= 1e-9, (len(ev["rows"]), column.mean())
        population, sample = column.std(), column.std(ddof=1)
        assert abs(population - 1.0) <= 1e-9 or abs(sample - 1.0) <= 1e-9, (
            len(ev["rows"]), population, sample)

    fit = final[0]
    gappy = frame["gappy"].to_numpy()
    filled = [r for r in range(180) if not np.isnan(gappy[r])]
    assert len(filled) % 2 == 1
    median = float(np.median(gappy[filled]))
    median_rows = [r for r in filled if gappy[r] == median]
    assert len(median_rows) == 1
    position = {r: i for i, r in enumerate(fit["rows"])}
    column = fit["values"][:, fit["columns"].index("gappy")]
    expected = column[position[median_rows[0]]]
    empty = [r for r in range(180) if np.isnan(gappy[r])]
    assert empty
    for r in empty:
        assert abs(column[position[r]] - expected) <= 1e-12, (r, column[position[r]], expected)

    record = h.columns_of(result)["gappy"]
    assert record["n_imputed"] == int(np.isnan(gappy).sum()), record


# ------------------------------------------------------------------- M5

def _tied_table():
    rng = np.random.default_rng(6501)
    y = rng.permutation(np.array([0] * 200 + [1] * 200))
    x1 = rng.normal(size=400)
    prediction = np.where(np.arange(400) % 25 == 0, 0.9, 0.1)
    prediction = prediction[rng.permutation(400)]
    return pd.DataFrame({"x1": x1, "y": y}), prediction


def _twice_pn_auc(j, n_high, positives, negatives):
    return (2 * j * (negatives - n_high + j) + j * (n_high - j)
            + (positives - j) * (negatives - n_high + j))


def test_m5_a_shuffle_that_ties_counts_where_only_some_of_them_tie():
    """M5: shuffles whose skill ties the observed one count in the p-value."""
    frame, prediction = _tied_table()
    gaps, ties = [], []
    for seed in range(40):
        result = s.run(frame, families=[_lookup(prediction)], seed=seed, **s.SAID)
        head = result.headline
        assert head["n_permutations"] == 200
        y_true = (result.predictions["y_true"].to_numpy(dtype=float) == 1.0).astype(int)
        y_pred = result.predictions["y_pred"].to_numpy(dtype=float)
        assert abs(head["score"] - h.auc_by_hand(y_true, y_pred)) <= 1e-12

        total = len(y_true)
        positives = int(y_true.sum())
        negatives = total - positives
        high = y_pred > 0.5
        n_high = int(high.sum())
        observed = int(y_true[high].sum())
        mark = _twice_pn_auc(observed, n_high, positives, negatives)
        at_least, equal = Fraction(0), Fraction(0)
        for j in range(max(0, n_high - negatives), min(n_high, positives) + 1):
            weight = Fraction(comb(positives, j) * comb(negatives, n_high - j),
                              comb(total, n_high))
            value = _twice_pn_auc(j, n_high, positives, negatives)
            if value >= mark:
                at_least += weight
            if value == mark:
                equal += weight
        gaps.append(head["p_value"] - (1 + 200 * float(at_least)) / 201)
        ties.append(float(equal))
    assert np.mean(ties) > 0.15, np.mean(ties)
    assert abs(np.mean(gaps)) < 0.05, (np.mean(gaps), gaps)


# ------------------------------------------------------------------- M6

def test_m6_an_infinite_holdout_prediction_in_a_regression_falls_back(tmp_path):
    """M6: an infinite prediction on the holdout fails its configuration; the next one stands."""
    frame = h.reg_frame(n=240, seed=6601)
    rng = np.random.default_rng(6602)
    y = frame["y"].to_numpy()
    sharp = y + 0.3 * rng.normal(size=len(y))
    sound = y + 1.5 * rng.normal(size=len(y))

    first = s.run(frame, families=[_lookup(sound)], task="regression", **s.SAID)
    holdout = sorted(h.holdout_rows(first))
    clean = s.run(frame, families=[_lookup(sharp)], task="regression", **s.SAID)
    clean_model = h.models_of(clean)[0]

    wild = sharp.copy()
    wild[holdout[0]] = np.inf
    wild[holdout[1]] = -np.inf
    log = h.Log()
    families = [h.family("sharp", log, estimator=h.LookupEstimator, values=wild),
                h.family("sound", log, estimator=h.LookupEstimator, values=sound)]
    out = tmp_path / "out"
    result = s.run(frame, families=families, task="regression", out_dir=out, **s.SAID)

    models = {m["config"]: m for m in h.models_of(result)}
    bad, good = models["sharp-0"], models["sound-0"]
    assert bad["status"] == "failed" and bad["error"] and not bad["chosen"], bad
    assert len(bad["fold_scores"]) == len(clean_model["fold_scores"])
    for got, want in zip(bad["fold_scores"], clean_model["fold_scores"]):
        assert math.isclose(got, want, rel_tol=1e-12), (got, want)
    assert math.isclose(bad["cv_mean"], clean_model["cv_mean"], rel_tol=1e-12)
    assert bad["cv_mean"] < good["cv_mean"], (bad["cv_mean"], good["cv_mean"])
    assert good["chosen"] and result.headline["model"] == "sound-0"
    assert s.codes(result).count("model_failed") == 1, s.codes(result)

    rows = h.holdout_rows(result)
    assert sorted(rows) == holdout
    y_pred = result.predictions["y_pred"].to_numpy(dtype=float)
    assert np.isfinite(y_pred).all()
    assert np.allclose(y_pred, sound[rows], rtol=0, atol=1e-12)
    y_true = result.predictions["y_true"].to_numpy(dtype=float)
    assert math.isclose(result.headline["score"], h.mse_by_hand(y_true, sound[rows]),
                        rel_tol=1e-12)
    k.assert_raw_numbers(result)
    _, written = _files(out)
    assert written and all(math.isfinite(float(r["y_pred"])) for r in written)


# ------------------------------------------------------------------- M8

def test_m8_the_interval_from_whole_entities_is_the_middle_95_percent():
    """M8: the interval of the entity bootstrap runs from its 2.5th to its 97.5th percentile."""
    frame = s.sized_entities([3] * 160, seed=6801)
    prediction = frame["y"].to_numpy() + np.random.default_rng(6802).normal(size=len(frame))
    who = frame["who"].tolist()
    own_rng = np.random.default_rng(6803)
    lows, highs = [], []
    for seed in range(60):
        result = s.run(frame, families=[_lookup(prediction)], seed=seed,
                       task="regression", time_col=False, group_col="who")
        head = result.headline
        assert head["interval_method"] == "bootstrap_groups"
        assert head["interval_level"] == 0.95
        assert head["n_resamples_used"] == 200
        rows = h.holdout_rows(result)
        y_true = result.predictions["y_true"].to_numpy(dtype=float)
        y_pred = result.predictions["y_pred"].to_numpy(dtype=float)
        errors = (y_true - y_pred) ** 2
        assert math.isclose(errors.mean(), head["score"], rel_tol=1e-9)
        sums = {}
        counts = {}
        for row, err in zip(rows, errors):
            sums[who[row]] = sums.get(who[row], 0.0) + err
            counts[who[row]] = counts.get(who[row], 0) + 1
        assert len(sums) >= 30 and set(counts.values()) == {3}, counts
        entity_sums = np.array(list(sums.values()))
        m = len(entity_sums)
        draws = own_rng.integers(0, m, size=(4000, m))
        means = entity_sums[draws].sum(axis=1) / (3.0 * m)
        low, high = np.percentile(means, [2.5, 97.5])
        sd = means.std()
        lows.append((head["interval_low"] - low) / sd)
        highs.append((head["interval_high"] - high) / sd)
    assert abs(np.mean(lows)) < 0.22, (np.mean(lows), np.mean(highs))
    assert abs(np.mean(highs)) < 0.22, (np.mean(lows), np.mean(highs))


# ------------------------------------------------------------------ M10

def test_m10_the_fast_cap_is_a_sample_whatever_the_order_of_the_file(tmp_path):
    """M10: the fast cap is a sample drawn from the seed, not the head of the file."""
    rng = np.random.default_rng(6001)
    frame = pd.DataFrame({"x1": rng.normal(size=3000), "z1": rng.normal(size=3000),
                          "y": ["a"] * 2000 + ["b"] * 1000})
    data = tmp_path / "ordered.csv"
    frame.to_csv(data, index=False)
    held = []
    for seed in (0, 1):
        out = tmp_path / f"out{seed}"
        proc = _cli(data, "y", out, "--seed", str(seed), "--no-time-order", "--no-entities")
        assert proc.returncode == 0, proc.stderr
        doc, rows = _files(out)
        counts = _rows(doc)
        assert (counts["dropped_row_cap"], counts["training"], counts["holdout"]) == (
            1000, 1500, 500), counts
        assert doc["task"]["classes"] == ["a", "b"]
        late = [r for r in rows if int(r["row"]) >= 2000]
        assert all(r["y_true"] == "b" for r in late)
        assert 140 <= len(late) <= 195, len(late)
        held.append(sorted(int(r["row"]) for r in rows))
    assert held[0] != held[1]


# ----------------------------------------------------------- M11 and M38

def _late_classes(first_mixed):
    rng = np.random.default_rng(6111)
    n = 400
    t = np.arange(n, dtype=float)
    y = rng.integers(0, 2, size=n)
    y[:50 * (first_mixed - 1)] = 0
    x1 = rng.normal(size=n)
    prediction = 0.3 * y + rng.uniform(0.0, 0.7, size=n)
    return pd.DataFrame({"t": t, "x1": x1, "y": y}), prediction


def test_m11_cv_std_of_two_scored_folds_is_their_sample_deviation():
    """M11: with two scored folds cv_std is |a - b| / sqrt(2), the sample deviation."""
    frame, prediction = _late_classes(4)
    result = s.run(frame, families=[_lookup(prediction)], time_col="t", group_col=False)
    doc = result.to_dict()
    assert doc["task"]["type"] == "classification"
    assert [f["scored"] for f in doc["validation"]["folds"]] == [False, False, False, True, True]
    assert doc["cv"]["n_scored_folds"] == 2
    model = doc["models"][0]
    assert model["fold_scores"][:3] == [None, None, None]
    a, b = model["fold_scores"][3:]
    assert a != b
    assert math.isclose(model["cv_mean"], (a + b) / 2, rel_tol=1e-12)
    assert math.isclose(model["cv_std"], abs(a - b) / math.sqrt(2), rel_tol=1e-9)
    assert math.isclose(model["cv_std"], statistics.stdev([a, b]), rel_tol=1e-9)
    assert doc["cv"]["chosen_std"] == model["cv_std"]


@pytest.mark.parametrize("first_mixed, not_scored", [
    pytest.param(4, [1, 2, 3], id="three_folds"),
    pytest.param(3, [1, 2], id="two_folds"),
])
def test_m38_one_fold_not_scored_record_for_each_fold_not_scored(first_mixed, not_scored):
    """M38: one fold_not_scored warning for each fold that is not scored."""
    frame, prediction = _late_classes(first_mixed)
    result = s.run(frame, families=[_lookup(prediction)], time_col="t", group_col=False)
    doc = result.to_dict()
    missed = [f["fold"] for f in doc["validation"]["folds"] if not f["scored"]]
    assert missed == not_scored
    found = s.warning_records(result, "fold_not_scored")
    assert len(found) == len(missed) >= 2, s.codes(result)
    assert all(r["column"] is None for r in found)
    assert doc["cv"]["n_scored_folds"] == 5 - len(missed)


# ------------------------------------------------------------------ M12

def _coin_flips(case):
    rng = np.random.default_rng(6121)
    if case == "blocks":
        n = 240
        frame = pd.DataFrame({"t": np.arange(n, dtype=float), "x1": rng.normal(size=n),
                              "y": rng.integers(0, 2, size=n)})
        return frame, {"time_col": "t", "group_col": False}, "bootstrap_blocks"
    n = 480
    frame = pd.DataFrame({"who": [f"u{q // 3:03d}" for q in range(n)],
                          "x1": rng.normal(size=n), "y": rng.integers(0, 2, size=n)})
    frame = frame.iloc[rng.permutation(n)].reset_index(drop=True)
    return frame, {"time_col": False, "group_col": "who"}, "bootstrap_groups"


@pytest.mark.parametrize("case", ["blocks", "entities"])
def test_m12_tied_predictions_in_a_resample_of_blocks_and_of_entities(case):
    """M12: in a resample of blocks or of entities, tied predictions share their rank."""
    frame, said, method = _coin_flips(case)
    result = s.run(frame, families=[_lookup(np.full(len(frame), 0.5))], **said)
    head = result.headline
    assert head["interval_method"] == method
    assert head["score"] == 0.5 and head["skill"] == 0.0
    assert head["n_resamples_used"] == 200
    assert head["interval_low"] == 0.5 and head["interval_high"] == 0.5, head
    assert head["skill_low"] == 0.0 and head["skill_high"] == 0.0, head
    assert head["p_value"] is None


# ------------------------------------------------------------------ M17

def test_m17_the_fast_cap_keeps_whole_entities_of_seven_rows(tmp_path):
    """M17: with an entity column the fast cap keeps whole entities, so 1,995 rows and not 2,000."""
    frame = s.sized_entities([7] * 400, seed=6171)
    data = tmp_path / "sevens.csv"
    frame.to_csv(data, index=False)
    out = tmp_path / "out"
    proc = _cli(data, "y", out, "--no-time-order", "--group-col", "who")
    assert proc.returncode == 0, proc.stderr
    doc, rows = _files(out)
    counts = _rows(doc)
    assert counts["training"] + counts["holdout"] == 1995, counts
    assert counts["dropped_row_cap"] == 805, counts
    v = doc["validation"]
    assert v["scheme"] == "grouped"
    assert v["n_groups_training"] + v["n_groups_holdout"] == 285
    who = frame["who"].tolist()
    held = {}
    for r in rows:
        name = who[int(r["row"])]
        held[name] = held.get(name, 0) + 1
    assert set(held.values()) == {7}, held
    assert len(held) == v["n_groups_holdout"]


# ------------------------------------------------------------------ M21

def test_m21_a_resample_of_whole_entities_has_one_score():
    """M21: under the grouped scheme a resample draws whole entities, so every one scores 0.5."""
    rng = np.random.default_rng(6211)
    n_entities = 160
    n = 2 * n_entities
    who = [f"u{q // 2:03d}" for q in range(n)]
    y = np.round(rng.normal(size=n) * 8.0) / 8.0 + 3.0
    x1 = rng.normal(size=n)
    prediction = y + np.where(np.arange(n) % 2 == 0, 0.0, 1.0)
    order = rng.permutation(n)
    frame = pd.DataFrame({"who": who, "x1": x1, "y": y}).iloc[order].reset_index(drop=True)
    prediction = prediction[order]
    errors = (frame["y"].to_numpy() - prediction) ** 2
    assert set(errors.tolist()) == {0.0, 1.0}

    result = s.run(frame, families=[_lookup(prediction)], task="regression",
                   time_col=False, group_col="who")
    doc = result.to_dict()
    assert doc["validation"]["n_groups_holdout"] >= 30
    head = result.headline
    assert head["interval_method"] == "bootstrap_groups" and head["block_length"] is None
    assert head["score"] == 0.5
    assert head["n_resamples_used"] == 200
    assert abs(head["interval_low"] - 0.5) <= 1e-12, head
    assert abs(head["interval_high"] - 0.5) <= 1e-12, head
    assert head["p_value"] is None


# ------------------------------------------------------------------ M23

def _sales_panel():
    rng = np.random.default_rng(6239)
    n_dates = 80
    first = dt.date(2022, 1, 3)
    sizes = [(2, 3, 4)[q % 3] for q in range(n_dates)]
    shocks = rng.normal(size=n_dates)
    level = np.zeros(n_dates)
    level[0] = shocks[0]
    for q in range(1, n_dates):
        level[q] = 0.68 * level[q - 1] + shocks[q]
    level = 3.0 * level
    date_of = np.repeat(np.arange(n_dates), sizes)
    n = len(date_of)
    x = rng.normal(size=n)
    sales = level[date_of] + 0.2 * rng.normal(size=n)
    frame = pd.DataFrame({
        "date": [(first + dt.timedelta(days=int(q))).isoformat() for q in date_of],
        "x": x, "sales": sales,
    })
    return frame.iloc[rng.permutation(n)].reset_index(drop=True)


def _training_dates(frame, column):
    counts = frame[column].value_counts()
    ordered = sorted(counts.index)
    held, total = 0, len(frame)
    cut = len(ordered)
    while 4 * held < total:
        cut -= 1
        held += int(counts[ordered[cut]])
    return ordered[:cut]


def test_m23_a_panel_between_the_two_cut_offs_is_warned_about_on_the_command_line(tmp_path):
    """M23: a panel autocorrelated at about 0.67 is warned about, as above 0.5 and not only above 0.8."""
    frame = _sales_panel()
    assert len(frame) == 239
    training = _training_dates(frame, "date")
    means = frame.groupby("date")["sales"].mean().loc[training].to_numpy()
    ratio = h.lag1_autocorr(means)
    pearson = float(np.corrcoef(means[:-1], means[1:])[0, 1])
    assert 0.62 <= ratio <= 0.74 and 0.62 <= pearson <= 0.74, (ratio, pearson, len(training))

    data = tmp_path / "sales.csv"
    frame.to_csv(data, index=False)
    out = tmp_path / "out"
    proc = _cli(data, "sales", out, "--time-col", "date", "--no-entities")
    assert proc.returncode == 0, proc.stderr
    doc, _ = _files(out)
    v = doc["validation"]
    assert v["scheme"] == "time_ordered"
    assert v["timestamps_unique"] is False
    assert v["n_train_timestamps"] == len(training)
    found = [w for w in doc["warnings"] if w["code"] == "autocorrelated_panel_target"]
    assert len(found) == 1 and found[0]["column"] == "sales", doc["warnings"]
    assert any("autocorrelated_panel_target" in line and "sales" in line
               for line in proc.stdout.splitlines()), proc.stdout


# ------------------------------------------------------------------ M28

def _text_day_panel():
    rng = np.random.default_rng(6281)
    first = dt.date(2023, 11, 1)
    parts = []
    for q in range(6):
        level = 10.0 * q + np.cumsum(rng.normal(size=61))
        parts.append(pd.DataFrame({
            "step": np.arange(60),
            "who": f"s{q:02d}",
            "prev": level[:-1],
            "y": level[1:],
        }))
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
    numbers = [s.date_number(first + dt.timedelta(days=int(v))) for v in frame["step"]]
    cells = [f" {v} " if r % 7 == 0 else str(v) for r, v in enumerate(numbers)]
    cells[11] = None
    frame = frame.drop(columns="step")
    frame.insert(0, "day", pd.Series(cells, dtype=object))
    return frame, numbers


def test_m28_eight_digit_text_orders_a_panel_and_read_time_reads_it():
    """M28: eight-digit text in an object column is accepted as the time column of a panel."""
    from mlprofessor import stages

    frame, numbers = _text_day_panel()
    assert frame["day"].dtype == object
    with_day = frame.drop(index=11)
    counts = pd.Series([numbers[r] for r in with_day.index]).value_counts()
    ordered = sorted(counts.index)
    held, cut = 0, len(ordered)
    while 4 * held < len(with_day):
        cut -= 1
        held += int(counts[ordered[cut]])
    holdout_days = set(ordered[cut:])
    own_holdout = sorted(r for r in with_day.index if numbers[r] in holdout_days)

    result = s.run(frame, families=[h.family("rec", h.Log())], time_col="day", group_col="who")
    doc = result.to_dict()
    assert sorted(h.holdout_rows(result)) == own_holdout
    v = doc["validation"]
    assert v["scheme"] == "time_ordered" and v["time_form"] == "numeric"
    assert h.rows_of(result)["dropped_missing_time"] == 1
    for name, want in (("train_end", ordered[cut - 1]), ("holdout_start", ordered[cut]),
                       ("holdout_end", ordered[-1])):
        assert not isinstance(v[name], str) and v[name] == want, (name, v[name], want)
    assert h.columns_of(result)["day"]["role"] == "time"
    assert [b["name"] for b in doc["baselines"]] == ["training_mean", "previous_value"]
    assert v["n_holdout_without_previous"] == 0

    keys, form = stages.read_time(frame["day"])
    assert form == "numeric"
    keys = np.asarray(keys)
    assert keys.dtype == np.float64
    assert np.isnan(keys[11])
    want = np.array([np.nan if r == 11 else float(numbers[r]) for r in range(len(frame))])
    mask = np.arange(len(frame)) != 11
    assert np.array_equal(keys[mask], want[mask])


# ------------------------------------------- a date beyond the nanosecond range

_SCRIPT = """
import sys
import numpy as np
import pandas as pd
import mlprofessor
from mlprofessor import errors

rng = np.random.default_rng(6902)
when = np.arange(np.datetime64("2015-01-01", "s"), np.datetime64("2015-01-01", "s")
                 + np.timedelta64(1500 * 86400, "s"), np.timedelta64(86400, "s"))
when[700] = np.datetime64("9999-12-31", "s")
frame = pd.DataFrame({"date": when, "x": rng.normal(size=1500), "y": rng.normal(size=1500)})
assert str(frame["date"].dtype) == "datetime64[s]", frame["date"].dtype
try:
    mlprofessor.profile(frame, "y", time_col="date", group_col=False, profile="fast",
                        n_jobs=1, out_dir=sys.argv[1])
except errors.ProfileError as exc:
    print("named:", exc.code)
else:
    print("finished")
"""


@pytest.mark.parametrize("case", ["iso_text", "datetime_in_seconds"])
def test_one_date_beyond_the_nanosecond_range_among_ordinary_ones_is_the_named_error(
        case, tmp_path):
    """No variant: one 9999-12-31 among 1,500 ordinary dates is time_column_invalid, in a process of its own."""
    out = tmp_path / "out"
    if case == "iso_text":
        rng = np.random.default_rng(6901)
        days = [(dt.date(2015, 1, 1) + dt.timedelta(days=q)).isoformat() for q in range(1500)]
        days[700] = "9999-12-31"
        frame = pd.DataFrame({"date": days, "x": rng.normal(size=1500),
                              "y": rng.normal(size=1500)})
        data = tmp_path / "far.csv"
        frame.to_csv(data, index=False)
        proc = _cli(data, "y", out, "--time-col", "date", "--no-entities")
        assert proc.returncode == 3, (proc.returncode, proc.stderr)
        lines = proc.stderr.strip().splitlines()
        assert len(lines) == 1 and lines[0].startswith("error: time_column_invalid:"), proc.stderr
    else:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run([h.PYTHON, "-B", "-c", _SCRIPT, str(out)], cwd=str(h.REPO_ROOT),
                              capture_output=True, text=True, timeout=600, env=env)
        assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
        assert "named: time_column_invalid" in proc.stdout, (proc.stdout, proc.stderr)
    assert h.files_in(out) == []


# ------------------------------------------------------------------ M39

def test_m39_rows_with_a_cell_that_cannot_be_hashed_are_counted_as_repeats():
    """M39: a row with a list cell is counted as a repeat when it matches another by the list's text."""
    rng = np.random.default_rng(6391)
    n = 200
    first = rng.integers(0, 5, size=n)
    second = rng.integers(5, 9, size=n)
    x1, z1 = rng.normal(size=n), rng.normal(size=n)
    y = rng.integers(0, 2, size=n)
    tags = [[int(a), int(b)] for a, b in zip(first, second)]
    rows = [(tags[r], x1[r], z1[r], int(y[r])) for r in range(n)]
    rows += [(list(tags[r]), x1[r], z1[r], int(y[r])) for r in range(10, 25)]
    rows += [(list(reversed(tags[r])), x1[r], z1[r], int(y[r])) for r in range(40, 45)]
    order = rng.permutation(len(rows))
    rows = [rows[q] for q in order]
    frame = pd.DataFrame({"tags": pd.Series([r[0] for r in rows], dtype=object),
                          "x1": [r[1] for r in rows], "z1": [r[2] for r in rows],
                          "y": [r[3] for r in rows]})
    seen, repeats = set(), 0
    for r in rows:
        key = (str(r[0]), r[1], r[2], r[3])
        repeats += key in seen
        seen.add(key)
    assert repeats == 15

    result = s.run(frame, families=[_lookup(np.full(len(frame), 0.5))], duplicates="keep",
                   **s.SAID)
    doc = result.to_dict()
    assert doc["input"]["duplicate_rows"] == 15, doc["input"]
    assert s.codes(result).count("duplicate_rows") == 1, s.codes(result)
    record = h.columns_of(result)["tags"]
    assert record["role"] == "excluded" and record["reason"] == "unsupported_type", record
