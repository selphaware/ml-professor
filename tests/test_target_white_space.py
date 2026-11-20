"""T26: white space round a target value (D46); a target cell of only white space (D48).

Spec: steps 2 and 4, 2.7, 5.1, 5.2. Red today.

A text target value is read without the white space round it: `yes`, ` yes`
and `yes ` are one class. A text cell holding nothing but white space is a
missing target, dropped and counted like an empty one. "White space" here is
spaces and tabs only.
"""
import csv

import numpy as np
import pandas as pd
import pytest

import cp3_helpers as c
import mlp_helpers as h

PADDINGS = [lambda v: "  " + v, lambda v: v + " ", lambda v: v + "\t", lambda v: " " + v + " \t"]


def _padded_every_fourth(clean):
    written = list(clean)
    for k, r in enumerate(range(0, len(written), 4)):
        written[r] = PADDINGS[k % len(PADDINGS)](written[r])
    return written


def _model_fits(log):
    return [ev for ev in log.events if ev["kind"] == "model" and ev["action"] == "fit"]


def test_a_target_written_with_stray_white_space_has_two_classes():
    from mlprofessor import stages

    frame = h.clf_frame(n=240, seed=21, labels=("no", "yes"))
    clean = [str(v) for v in frame["y"]]
    frame["y"] = pd.Series(_padded_every_fourth(clean), dtype=object)
    assert frame["y"].nunique() > 2, "as written the column has more than two values"

    log = h.Log()
    result = c.run(frame, families=[h.family("rec", log)], task="auto")
    task = result.to_dict()["task"]
    assert task["type"] == "classification"
    assert task["decided_by"] == "rule"
    assert task["n_distinct"] == 2
    assert task["classes"] == ["no", "yes"]
    assert task["positive_class"] == "yes"

    rows = h.holdout_rows(result)
    assert [str(v) for v in result.predictions["y_true"]] == [clean[r] for r in rows]

    split = stages.split_random(len(clean), np.asarray(clean, dtype=object), 0)
    assert rows == [int(r) for r in split["holdout"]]

    fits = _model_fits(log)
    assert fits, "the recording family was fitted"
    for ev in fits:
        assert ev["y"] == [1 if clean[r] == "yes" else 0 for r in ev["rows"]]


def test_every_cell_of_one_class_written_with_a_trailing_space(tmp_path):
    import json

    frame = h.clf_frame(n=240, seed=22, labels=("no", "yes"))
    frame["y"] = ["yes " if v == "yes" else v for v in frame["y"]]
    path = tmp_path / "trailing.csv"
    frame.to_csv(path, index=False)
    assert set(pd.read_csv(path, low_memory=False)["y"]) == {"no", "yes "}, "the file holds `yes ` as written"

    out = tmp_path / "out"
    result = c.run(str(path), families=[h.family("rec", h.Log())], out_dir=out)
    with open(out / "profile.json", encoding="utf-8") as handle:
        doc = json.load(handle)
    for task in (result.to_dict()["task"], doc["task"]):
        assert task["classes"] == ["no", "yes"]
        assert task["positive_class"] == "yes"

    with open(out / "predictions.csv", encoding="utf-8", newline="") as handle:
        y_true = [record["y_true"] for record in csv.DictReader(handle)]
    assert y_true, "predictions.csv holds the holdout rows"
    assert set(y_true) == {"no", "yes"}


BLANK_CELLS = {3: "  ", 50: "\t", 97: " \t ", 140: "   ", 199: " "}
EMPTY_CELLS = [20, 120]


def _table_with_blank_targets():
    frame = h.clf_frame(n=240, seed=23, labels=("no", "yes"))
    values = [str(v) for v in frame["y"]]
    target = pd.Series(values, dtype=object)
    for r, text in BLANK_CELLS.items():
        target[r] = text
    for r in EMPTY_CELLS:
        target[r] = None
    frame["y"] = target
    return frame


@pytest.mark.parametrize("source", ["dataframe", "csv"])
def test_a_target_cell_of_only_white_space_is_a_missing_target(tmp_path, source):
    from mlprofessor import stages

    frame = _table_with_blank_targets()
    if source == "csv":
        path = tmp_path / "blank_targets.csv"
        frame.to_csv(path, index=False)
        data = str(path)
    else:
        data = frame

    loaded, _ = stages.load_table(data)
    for r, text in BLANK_CELLS.items():
        cell = loaded["y"].iloc[r]
        assert isinstance(cell, str) and cell.strip(" \t") == "", f"row {r} loads as {cell!r}"

    dropped = sorted(list(BLANK_CELLS) + EMPTY_CELLS)
    usable = stages.usable_rows(loaded, "y")
    assert usable["dropped_missing_target"] == len(dropped)
    assert not set(dropped) & {int(r) for r in usable["usable"]}
    assert len(usable["usable"]) == len(frame) - len(dropped)

    result = c.run(data, families=[h.family("rec", h.Log())])
    assert h.rows_of(result)["dropped_missing_target"] == len(dropped)
    assert result.to_dict()["task"]["classes"] == ["no", "yes"]
    assert not set(dropped) & set(h.holdout_rows(result))


def test_decide_task_reads_text_without_the_white_space_round_it():
    from mlprofessor import stages

    values = pd.Series(["yes", "yes ", "no", " no", "  ", "\tyes", None] * 10, dtype=object)
    decided = stages.decide_task(values)
    assert decided["type"] == "classification"
    assert decided["decided_by"] == "rule"
    assert decided["n_distinct"] == 2
    assert list(decided["classes"]) == ["no", "yes"]
    assert decided["positive_class"] == "yes"


def test_split_random_compares_text_labels_without_white_space():
    from mlprofessor import stages

    rng = np.random.default_rng(24)
    labels = np.where(rng.random(200) < 0.3, "yes", "no").astype(object)
    pads = [" {}", "{}  ", "\t{}", " {} \t"]
    padded = np.array([pads[i % len(pads)].format(v) for i, v in enumerate(labels)], dtype=object)
    assert len(set(padded)) > 2

    plain = stages.split_random(200, labels, 7)
    spaced = stages.split_random(200, padded, 7)

    def same(a, b):
        return [int(x) for x in a] == [int(x) for x in b]

    assert same(plain["holdout"], spaced["holdout"])
    assert same(plain["training"], spaced["training"])
    assert len(plain["folds"]) == len(spaced["folds"]) == 5
    for fa, fb in zip(plain["folds"], spaced["folds"]):
        assert same(fa["training"], fb["training"])
        assert same(fa["validation"], fb["validation"])

    holdout = [int(r) for r in spaced["holdout"]]
    n_yes = int((labels == "yes").sum())
    held_yes = int((labels[holdout] == "yes").sum())
    assert abs(held_yes - n_yes / 4) <= 1
