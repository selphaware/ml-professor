"""T18 guards: the built-in preprocessing of every fit sees only that fit's rows (spec 3 step 8, 2.6, 8.1).

Green today. A failure here means the application is wrong, not the test.
M1: the final fit's imputation and scaling statistics learned with the holdout.
M2: one-hot levels learned from every row of the table.
"""
import numpy as np
import pandas as pd

import cp3_helpers as c
import mlp_helpers as h


def _table(n=240, seed=41):
    """A numeric feature with ~15% missing cells and a categorical of three levels."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    frame = pd.DataFrame({"x1": x1, "z1": rng.normal(size=n)})
    gappy = rng.normal(size=n)
    gappy[rng.random(n) < 0.15] = np.nan
    frame["gappy"] = gappy
    frame["colour"] = rng.choice(["red", "green", "blue"], size=n)
    frame["y"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-2.5 * x1))).astype(int)
    return frame


def _run_keep(frame, steps=False):
    log = c.KeepLog()
    families = [h.family("keep", log, c.MatrixEstimator)]
    result = c.run(frame, families=families,
                   steps=[c.keep_step_factory(log)] if steps else None)
    return result, log


def test_m1_final_fit_statistics_ignore_the_holdout():
    """M1 (small F1): the final fit's imputation and scaling never learn from holdout rows."""
    frame = _table()
    first, log_a = _run_keep(frame)
    hold = h.holdout_rows(first)
    training = sorted(set(range(len(frame))) - set(hold))
    assert frame.loc[training, "gappy"].isna().any(), "the median must be in play"

    changed = frame.copy()
    changed.loc[hold, ["x1", "gappy", "z1"]] = [1e6, -1e6, 5e5]
    second, log_b = _run_keep(changed)
    assert h.holdout_rows(second) == hold

    fits_a = log_a.calls("model", "fit")
    fits_b = log_b.calls("model", "fit")
    assert len(fits_a) == 6 and len(fits_b) == 6, "five folds and the final fit"
    assert sum(set(f["rows"]) == set(training) for f in fits_a) == 1

    by_rows_b = {tuple(f["rows"]): f for f in fits_b}
    for fit_a in fits_a:
        fit_b = by_rows_b[tuple(fit_a["rows"])]
        assert fit_a["columns"] == fit_b["columns"]
        assert fit_a["values"].shape == fit_b["values"].shape
        assert np.array_equal(fit_a["values"], fit_b["values"]), (
            "a fit's matrix moved when only holdout rows changed; largest change "
            f"{np.nanmax(np.abs(fit_a['values'] - fit_b['values']))}")


def test_m2_levels_come_from_the_rows_of_each_fit():
    """M2 (small F1): one-hot levels come from the rows of each fit, never from the whole table."""
    from mlprofessor import stages

    frame = _table(seed=42)
    split = stages.split_random(len(frame), classes=frame["y"].to_numpy(), seed=0)
    hold = [int(r) for r in split["holdout"]]
    fold0_validation = [int(r) for r in split["folds"][0]["validation"]]
    holdout_only = hold[:4]
    fold_only = fold0_validation[:4]
    frame.loc[holdout_only, "colour"] = "holdout_only"
    frame.loc[fold_only, "colour"] = "fold_only"

    result, log = _run_keep(frame)
    assert h.holdout_rows(result) == sorted(hold)
    training = set(range(len(frame))) - set(hold)

    fits = log.calls("model", "fit")
    assert len(fits) == 6
    with_fold_only = 0
    for fit in fits:
        columns = fit["columns"]
        assert len(columns) == len(set(columns)), "no two columns of one name"
        assert "colour=holdout_only" not in columns
        holds_level = bool(set(fit["rows"]) & set(fold_only))
        assert ("colour=fold_only" in columns) == holds_level
        with_fold_only += holds_level
    assert with_fold_only == 5, "four folds and the final fit"
    without = [f for f in fits if "colour=fold_only" not in f["columns"]]
    assert len(without) == 1
    assert set(without[0]["rows"]) == training - set(fold0_validation)

    # By effect: in the holdout prediction a level unseen in the fit is all zeros.
    holdout_predicts = [p for p in log.calls("model", "predict") if set(p["rows"]) & set(hold)]
    # D59, D78 point 7: since the ninth pass the chosen configuration predicts the holdout
    # again, shuffled, to state what the headline rests on. The first call is the
    # headline's, and it is the one read here.
    assert len(holdout_predicts) >= 1
    predict = holdout_predicts[0]
    colour_columns = [i for i, name in enumerate(predict["columns"]) if name.startswith("colour=")]
    assert colour_columns
    for row in holdout_only:
        position = predict["rows"].index(row)
        assert np.all(predict["values"][position, colour_columns] == 0.0)


def test_surrounding_white_space_is_not_part_of_a_level():
    """D42: `red`, ` red` and `red ` are one level in counting, the leak screen, the steps and the matrix."""
    n = 240
    rng = np.random.default_rng(43)
    x1 = rng.normal(size=n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-2.5 * x1))).astype(int)
    words = rng.choice(["red", "green", "blue"], size=n)

    def padded(word, r):
        lead, trail = r % 16, r // 16
        tail = " " * trail + ("\t" if r % 3 == 0 else "")
        return " " * lead + word + tail

    frame = pd.DataFrame({
        "x1": x1,
        "z1": rng.normal(size=n),
        "colour": [padded(w, r) for r, w in enumerate(words)],
        "echo": [padded("yes" if v == 1 else "no", r) for r, v in enumerate(y)],
        "one_level": [padded("same", r) for r in range(n)],
        "y": y,
    })
    for name in ["colour", "echo", "one_level"]:
        assert frame[name].nunique() == n, "no two cells of a column are the same text as written"

    log = c.KeepLog()
    result = c.run(frame, families=[h.family("keep", log, c.MatrixEstimator)],
                   steps=[c.keep_step_factory(log)])
    cols = h.columns_of(result)

    assert cols["colour"]["role"] == "feature"
    assert cols["colour"]["kind"] == "categorical"
    assert cols["one_level"]["role"] == "excluded"
    assert cols["one_level"]["reason"] == "constant"
    assert cols["echo"]["role"] == "excluded"
    assert cols["echo"]["reason"] == "suspected_leak"
    assert cols["echo"]["leak_rule"] == "predicts_target"

    step_fits = log.calls("step", "fit")
    assert step_fits
    for fit in step_fits:
        assert "echo" not in fit["columns"]
        assert set(fit["frame"]["colour"]) <= {"red", "green", "blue"}

    model_fits = log.calls("model", "fit")
    assert model_fits
    for fit in model_fits:
        assert sorted(name for name in fit["columns"] if "=" in name) == [
            "colour=blue", "colour=green", "colour=red"]
