"""T10: profile.json, predictions.csv, `to_dict`, `to_json`, the command line (spec 2.2, 2.3, 5)."""
import math
import sys

import numpy as np
import pandas as pd
import pytest

import mlp_helpers as h

META = {"tool", "schema_version", "profile", "seed", "n_jobs", "versions", "timing"}
INPUT = {"source", "kind", "target", "time_col", "task", "gap", "n_rows", "n_columns", "duplicate_rows"}
TASK = {"type", "decided_by", "n_distinct", "target_dtype", "classes", "positive_class"}
VALIDATION = {"scheme", "holdout_share", "n_folds", "stratified", "gap", "time_col", "time_form",
              "train_end", "holdout_start", "holdout_end", "n_train_timestamps",
              "n_holdout_timestamps", "timestamps_unique", "folds", "caveat"}
TIME_FIELDS = ["time_col", "time_form", "train_end", "holdout_start", "holdout_end",
               "n_train_timestamps", "n_holdout_timestamps", "timestamps_unique"]
HEADLINE = {"metric", "score", "model", "baseline", "baseline_score", "skill", "interval_method",
            "interval_level", "interval_low", "interval_high", "skill_low", "skill_high",
            "n_resamples", "n_resamples_used", "block_length", "interval_reason", "p_value",
            "n_permutations", "p_value_reason"}
CV = {"metric", "greater_is_better", "n_folds", "n_scored_folds", "n_configurations", "chosen",
      "chosen_mean", "chosen_std"}
MODEL = {"config", "family", "params", "status", "error", "fold_scores", "cv_mean", "cv_std", "chosen"}
FOLD = {"fold", "training_rows", "validation_rows", "gap_rows", "scored"}
TOP = ["meta", "input", "rows", "columns", "task", "validation", "headline", "baselines", "cv",
       "models", "warnings", "limitations"]


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _profile(data, out_dir, **kw):
    import mlprofessor

    args = dict(h.FAST, seed=3)
    args.update(kw)
    return mlprofessor.profile(data, "y", out_dir=out_dir, **args)


def _mixed_frame(n=200, seed=30, task="classification"):
    frame = h.clf_frame(n, seed=seed) if task == "classification" else h.reg_frame(n, seed=seed)
    rng = np.random.default_rng(seed)
    frame.insert(1, "colour", [["red", "green", "blue"][k] for k in rng.integers(0, 3, size=n)])
    gaps = rng.normal(size=n)
    gaps[rng.random(n) < 0.15] = np.nan
    frame.insert(2, "gappy", gaps)
    return frame


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("schema")
    out = {}
    for task in ["classification", "regression"]:
        frame = _mixed_frame(task=task)
        folder = base / task
        res = _profile(frame, folder)
        out[task] = {"frame": frame, "dir": folder, "res": res,
                     "doc": h.load_json_strict(folder / "profile.json")}
    return out


TASKS = ["classification", "regression"]


@pytest.mark.parametrize("task", TASKS)
def test_profile_json_has_every_key_with_its_type(runs, task):
    doc, frame = runs[task]["doc"], runs[task]["frame"]
    assert set(TOP) <= set(doc)

    meta = doc["meta"]
    assert META <= set(meta)
    assert meta["tool"] == "mlprofessor" and meta["schema_version"] == 1
    assert is_num(meta["timing"]["seconds"])
    assert {"python", "pandas", "numpy", "scikit-learn", "threadpoolctl"} <= set(meta["versions"])
    assert all(isinstance(v, str) for v in meta["versions"].values())

    inp = doc["input"]
    assert INPUT <= set(inp)
    assert inp["source"] == "<DataFrame>" and inp["kind"] == "dataframe"
    assert inp["target"] == "y" and inp["time_col"] is None and inp["task"] == "auto"
    assert inp["gap"] == 0
    assert inp["n_rows"] == len(frame) and inp["n_columns"] == frame.shape[1]
    assert is_int(inp["duplicate_rows"]) and inp["duplicate_rows"] == 0

    assert [r["item"] for r in doc["rows"]] == h.ROWS_ITEMS
    assert all(set(r) == {"item", "count"} and is_int(r["count"]) for r in doc["rows"])

    assert len(doc["columns"]) == frame.shape[1]
    for rec in doc["columns"]:
        assert set(h.COLUMN_KEYS) <= set(rec)
        assert isinstance(rec["name"], str) and is_int(rec["position"])
        assert isinstance(rec["dtype"], str)
        assert rec["role"] in {"target", "time", "feature", "excluded"}
        for key in ["n_missing", "n_coerced", "n_infinite", "n_imputed"]:
            assert is_int(rec[key])

    t = doc["task"]
    assert TASK <= set(t)
    assert t["type"] == task and t["decided_by"] == "rule" and isinstance(t["target_dtype"], str)
    if task == "classification":
        assert t["classes"] == [0, 1] and t["positive_class"] == 1 and t["n_distinct"] == 2
    else:
        assert t["classes"] is None and t["positive_class"] is None
        assert t["n_distinct"] == frame["y"].nunique()

    v = doc["validation"]
    assert VALIDATION <= set(v)
    assert v["scheme"] == "random" and v["n_folds"] == 5 and v["gap"] == 0
    assert v["stratified"] is (task == "classification")
    assert all(v[k] is None for k in TIME_FIELDS)
    rows = {r["item"]: r["count"] for r in doc["rows"]}
    assert v["holdout_share"] == pytest.approx(rows["holdout"] / (rows["training"] + rows["holdout"]))
    assert [f["fold"] for f in v["folds"]] == [1, 2, 3, 4, 5]
    for f in v["folds"]:
        assert FOLD <= set(f) and isinstance(f["scored"], bool)
        assert f["training_rows"] + f["validation_rows"] == rows["training"]
        assert f["gap_rows"] == 0
    assert sum(f["validation_rows"] for f in v["folds"]) == rows["training"]
    assert "caveat" in v

    hl = doc["headline"]
    assert HEADLINE <= set(hl)
    assert hl["metric"] == ("roc_auc" if task == "classification" else "mse")
    assert is_num(hl["score"]) and is_num(hl["skill"]) and is_num(hl["baseline_score"])

    used = [b for b in doc["baselines"] if b["used"]]
    assert len(used) == 1 and used[0]["name"] == hl["baseline"]
    assert all({"name", "score", "used"} <= set(b) for b in doc["baselines"])
    if task == "classification":
        assert [b["name"] for b in doc["baselines"]] == ["chance"]
        assert doc["baselines"][0]["score"] == pytest.approx(0.5)
    else:
        assert [b["name"] for b in doc["baselines"]] == ["training_mean"]

    cv = doc["cv"]
    assert CV <= set(cv)
    assert cv["metric"] == hl["metric"] and cv["greater_is_better"] is (task == "classification")
    assert cv["n_folds"] == 5 and cv["n_scored_folds"] == 5
    assert cv["n_configurations"] == len(doc["models"])
    assert cv["chosen"] == hl["model"]

    chosen = [m for m in doc["models"] if m["chosen"]]
    assert len(chosen) == 1 and chosen[0]["config"] == hl["model"] and chosen[0]["status"] == "fitted"
    assert cv["chosen_mean"] == pytest.approx(chosen[0]["cv_mean"])
    for m in doc["models"]:
        assert MODEL <= set(m)
        assert isinstance(m["params"], dict) and len(m["fold_scores"]) == 5
        assert m["status"] in {"fitted", "failed"}

    for w in doc["warnings"]:
        assert {"code", "column", "message"} <= set(w)
    # content of the interval fields and of `limitations` is checked in their own files
    assert "limitations" in doc


@pytest.mark.parametrize("task", TASKS)
def test_result_object_equals_the_file(runs, task, tmp_path):
    res, doc = runs[task]["res"], runs[task]["doc"]
    assert h.plain(res.to_dict()) == doc
    first = res.to_dict()
    first["meta"] = "changed"
    del first["rows"]
    assert h.plain(res.to_dict()) == doc
    res.to_json(tmp_path / "again.json")
    assert h.load_json_strict(tmp_path / "again.json") == doc
    assert h.plain(dict(res.headline)) == doc["headline"]
    for part in ["rows", "columns", "models", "warnings"]:
        assert h.records(getattr(res, part)) == doc[part], part
    for m in h.records(res.models):
        assert isinstance(m["params"], dict) and isinstance(m["fold_scores"], list)


@pytest.mark.parametrize("task", TASKS)
def test_predictions_file_reconciles(runs, task):
    res, folder, frame, doc = (runs[task][k] for k in ["res", "dir", "frame", "doc"])
    path = folder / "predictions.csv"
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "row,y_true,y_pred"
    pred = pd.read_csv(path)
    rows = pred["row"].tolist()
    assert rows == sorted(rows) and len(set(rows)) == len(rows)
    counts = {r["item"]: r["count"] for r in doc["rows"]}
    assert len(pred) == counts["holdout"]
    assert np.allclose(pred["y_true"].to_numpy(dtype=float), frame["y"].to_numpy(dtype=float)[rows])
    assert np.isfinite(pred["y_pred"].to_numpy(dtype=float)).all()
    assert rows == [int(r) for r in res.predictions["row"]]
    assert np.allclose(pred["y_pred"], res.predictions["y_pred"].to_numpy(dtype=float), rtol=1e-9)
    assert np.allclose(pred["y_true"].to_numpy(dtype=float), res.predictions["y_true"].to_numpy(dtype=float))


@pytest.mark.parametrize("task", TASKS)
def test_rows_columns_and_headline_reconcile(runs, task):
    frame, doc, folder = runs[task]["frame"], runs[task]["doc"], runs[task]["dir"]
    counts = {r["item"]: r["count"] for r in doc["rows"]}
    assert counts["loaded"] == len(frame)
    assert counts["loaded"] == sum(counts[k] for k in h.ROWS_ITEMS[1:])
    assert [c["name"] for c in doc["columns"]] == list(frame.columns)

    pred = pd.read_csv(folder / "predictions.csv")
    hl = doc["headline"]
    y_true = pred["y_true"].to_numpy(dtype=float)
    y_pred = pred["y_pred"].to_numpy(dtype=float)
    if task == "classification":
        auc = h.auc_by_hand(y_true == 1, y_pred)
        assert hl["score"] == pytest.approx(auc, abs=1e-9)
        assert hl["skill"] == pytest.approx(2 * auc - 1, abs=1e-9)
    else:
        score = h.mse_by_hand(y_true, y_pred)
        training = sorted(set(range(len(frame))) - set(pred["row"].tolist()))
        y = frame["y"].to_numpy(dtype=float)
        base = h.mse_by_hand(y_true, np.full(len(y_true), y[training].mean()))
        assert hl["score"] == pytest.approx(score, rel=1e-9)
        assert hl["baseline_score"] == pytest.approx(base, rel=1e-9)
        assert hl["skill"] == pytest.approx(1 - score / base, rel=1e-9)


def test_meta_records_the_run(runs):
    import numpy
    import pandas
    import sklearn
    import threadpoolctl

    meta = runs["classification"]["doc"]["meta"]
    assert meta["profile"] == "fast" and meta["seed"] == 3 and meta["n_jobs"] == 1
    v = meta["versions"]
    assert v["pandas"] == pandas.__version__ and v["numpy"] == numpy.__version__
    assert v["scikit-learn"] == sklearn.__version__ and v["threadpoolctl"] == threadpoolctl.__version__
    assert v["python"].startswith(f"{sys.version_info.major}.{sys.version_info.minor}")


def test_csv_and_dataframe_give_the_same_result(tmp_path, runs):
    frame = runs["classification"]["frame"]
    path = tmp_path / "table.csv"
    frame.to_csv(path, index=False)
    _profile(str(path), tmp_path / "from_csv")
    doc_csv = h.load_json_strict(tmp_path / "from_csv" / "profile.json")
    doc_df = runs["classification"]["doc"]
    assert doc_csv["input"]["kind"] == "csv" and doc_csv["input"]["source"] == str(path)
    drop = ("input.source", "input.kind", "meta.timing")
    assert h.without(doc_csv, *drop) == h.without(doc_df, *drop)
    assert ((tmp_path / "from_csv" / "predictions.csv").read_bytes()
            == (runs["classification"]["dir"] / "predictions.csv").read_bytes())


def test_same_seed_gives_identical_files(tmp_path, runs):
    frame = runs["regression"]["frame"]
    _profile(frame, tmp_path / "again")
    again = h.load_json_strict(tmp_path / "again" / "profile.json")
    assert h.without(again, "meta.timing") == h.without(runs["regression"]["doc"], "meta.timing")
    assert ((tmp_path / "again" / "predictions.csv").read_bytes()
            == (runs["regression"]["dir"] / "predictions.csv").read_bytes())


def test_command_line_matches_the_function(tmp_path):
    frame = _mixed_frame(seed=31)
    path = tmp_path / "t.csv"
    frame.to_csv(path, index=False)
    _profile(str(path), tmp_path / "fn")
    proc = h.run_cli(["profile", "--data", str(path), "--target", "y", "--out", tmp_path / "cli",
                      "--profile", "fast", "--n-jobs", "1", "--seed", "3"])
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip()
    assert h.files_in(tmp_path / "cli") == ["predictions.csv", "profile.json"]
    cli_doc = h.load_json_strict(tmp_path / "cli" / "profile.json")
    fn_doc = h.load_json_strict(tmp_path / "fn" / "profile.json")
    assert h.without(cli_doc, "meta.timing") == h.without(fn_doc, "meta.timing")
    assert ((tmp_path / "cli" / "predictions.csv").read_bytes()
            == (tmp_path / "fn" / "predictions.csv").read_bytes())


def test_command_line_named_error_and_bad_option(tmp_path):
    frame = h.clf_frame(150, seed=32)
    path = tmp_path / "t.csv"
    frame.to_csv(path, index=False)
    proc = h.run_cli(["profile", "--data", path, "--target", "nope", "--out", tmp_path / "out",
                      "--profile", "fast", "--n-jobs", "1"])
    assert proc.returncode == 3
    lines = [ln for ln in proc.stderr.splitlines() if ln.strip()]
    assert any(ln.startswith("error: column_not_found:") for ln in lines), proc.stderr
    assert h.files_in(tmp_path / "out") == []
    proc = h.run_cli(["profile", "--data", path, "--target", "y", "--out", tmp_path / "out2",
                      "--bogus-option"])
    assert proc.returncode == 2
    assert h.files_in(tmp_path / "out2") == []


def test_warnings_from_steps_one_to_eleven(tmp_path):
    rng = np.random.default_rng(33)
    n = 120
    x1 = rng.normal(size=n)
    coerced = [f"{v:.3f}" for v in rng.normal(size=n)]
    for i in [4, 50, 90]:
        coerced[i] = "abc"
    infin = rng.normal(size=n)
    infin[[7, 70]] = np.inf
    frame = pd.DataFrame({
        "x1": x1, "coerced": coerced, "infin": infin, "": rng.normal(size=n),
        "y": np.round(50 * x1 + 250 + 20 * rng.normal(size=n)).astype(int),
    })
    frame = pd.concat([frame, frame.iloc[[0, 1, 2]]], ignore_index=True)
    path = tmp_path / "w.csv"
    frame.to_csv(path, index=False)
    loaded = pd.read_csv(path)
    blank_name = loaded.columns[3]
    assert blank_name.startswith("Unnamed")
    n_dup = int(loaded.duplicated().sum())
    assert n_dup == 3

    res = _profile(str(path), tmp_path / "out")
    doc = h.load_json_strict(tmp_path / "out" / "profile.json")
    found = {(w["code"], w["column"]) for w in doc["warnings"]}
    for expected in [("values_coerced", "coerced"), ("infinite_values", "infin"),
                     ("blank_header", blank_name), ("duplicate_rows", None),
                     ("whole_number_regression_target", "y"), ("small_holdout", None)]:
        assert expected in found, expected
    assert doc["input"]["duplicate_rows"] == n_dup
    assert doc["task"]["type"] == "regression"
    cols = {c["name"]: c for c in doc["columns"]}
    assert cols["coerced"]["n_coerced"] == 3 and cols["infin"]["n_infinite"] == 2
    assert set(h.warnings_of(res)) == found


def test_numeric_with_separators_warning_has_one_form(tmp_path):
    rng = np.random.default_rng(34)
    frame = h.clf_frame(200, seed=34)
    frame.insert(0, "sep_all", [f"{v:,.2f}" for v in rng.uniform(10, 90_000, size=200)])
    few = [f"{v:.2f}" for v in rng.uniform(10, 900, size=200)]
    for i, text in zip([3, 60, 140], ["1,200", "4,800.25", "12,345"]):
        few[i] = text
    frame.insert(1, "sep_few", few)
    words = [f"{v:.2f}" for v in rng.uniform(10, 900, size=200)]
    words[10], words[20] = "abc", "12 units"
    frame.insert(2, "words", words)
    path = tmp_path / "s.csv"
    frame.to_csv(path, index=False)

    _profile(str(path), tmp_path / "out")
    doc = h.load_json_strict(tmp_path / "out" / "profile.json")
    nws = [w["column"] for w in doc["warnings"] if w["code"] == "numeric_with_separators"]
    coerced = [w["column"] for w in doc["warnings"] if w["code"] == "values_coerced"]
    cols = {c["name"]: c for c in doc["columns"]}
    excluded = sorted(c for c, r in cols.items() if r["reason"] == "numeric_with_separators")
    assert excluded == ["sep_all", "sep_few"]
    assert sorted(nws) == excluded
    assert "sep_all" not in coerced and "sep_few" not in coerced
    assert coerced.count("words") == 1 and "words" not in nws
    assert cols["words"]["role"] == "feature" and cols["words"]["n_coerced"] == 2
