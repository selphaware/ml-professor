"""T15: the full profile's grids, selection, and the same result at any n_jobs (spec 4.1, step 9, D17)."""
import os

import numpy as np
import pandas as pd
import pytest

import mlp_helpers as h

THIRD = 1 / 3
LINEAR = {"classification": [("C", c) for c in [1.0, 0.001, 0.01, 0.1, 10.0, 100.0]],
          "regression": [("alpha", a) for a in [1.0, 0.01, 0.1, 10.0, 100.0, 1000.0]]}
FOREST = {"classification": [(1, "sqrt"), (1, THIRD), (5, "sqrt"), (5, THIRD), (20, "sqrt"), (20, THIRD)],
          "regression": [(1, THIRD), (1, "sqrt"), (5, THIRD), (5, "sqrt"), (20, THIRD), (20, "sqrt")]}
BOOSTING = [(31, 0.1, 100), (31, 0.05, 300), (7, 0.1, 100), (7, 0.05, 300), (63, 0.1, 100), (63, 0.05, 300)]


def expected_grid(task):
    out = [(f"linear-{i}", "linear", {k: v}) for i, (k, v) in enumerate(LINEAR[task])]
    out += [(f"forest-{i}", "forest", {"min_samples_leaf": leaf, "max_features": mf})
            for i, (leaf, mf) in enumerate(FOREST[task])]
    out += [(f"boosting-{i}", "boosting",
             {"max_leaf_nodes": m, "learning_rate": lr, "max_iter": it})
            for i, (m, lr, it) in enumerate(BOOSTING)]
    return out


def same_params(actual, expected):
    if set(actual) != set(expected):
        return False
    for key, want in expected.items():
        got = actual[key]
        if isinstance(want, float):
            if isinstance(got, (bool, str)) or abs(got - want) > 1e-9:
                return False
        elif got != want:
            return False
    return True


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_eighteen_configurations_exactly(full_runs, task):
    res = full_runs["clf" if task == "classification" else "reg"]
    models = h.models_of(res)
    expected = expected_grid(task)
    assert len(models) == 18
    assert [m["config"] for m in models] == [e[0] for e in expected]
    assert [m["family"] for m in models] == [e[1] for e in expected]
    for m, (name, _, params) in zip(models, expected):
        assert same_params(m["params"], params), (name, m["params"])
    assert res.to_dict()["cv"]["n_configurations"] == 18


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_f3_headline_is_chosen_configuration_on_the_holdout(full_runs, task):
    """F3: with 18 configurations, the headline is still the holdout score of the best cv_mean."""
    key = "clf" if task == "classification" else "reg"
    res, frame = full_runs[key], full_runs[f"{key}_frame"]
    models = [m for m in h.models_of(res) if m["status"] == "fitted"]
    means = [m["cv_mean"] for m in models]
    best = max(means) if task == "classification" else min(means)
    expected = [m["config"] for m in models if m["cv_mean"] == best][0]
    assert res.headline["model"] == expected
    assert res.to_dict()["cv"]["chosen"] == expected

    pred = pd.read_csv(full_runs[f"{key}_dir"] / "predictions.csv")
    y_true = pred["y_true"].to_numpy(dtype=float)
    y_pred = pred["y_pred"].to_numpy(dtype=float)
    assert np.allclose(y_true, frame["y"].to_numpy(dtype=float)[pred["row"].to_numpy()])
    if task == "classification":
        score = h.auc_by_hand(y_true == 1, y_pred)
        assert res.headline["score"] == pytest.approx(score, abs=1e-9)
        assert res.headline["skill"] == pytest.approx(2 * score - 1, abs=1e-9)
    else:
        score = h.mse_by_hand(y_true, y_pred)
        assert res.headline["score"] == pytest.approx(score, rel=1e-9)
        assert res.headline["skill"] == pytest.approx(1 - score / res.headline["baseline_score"], rel=1e-9)


def test_same_result_at_one_and_two_jobs(full_runs, tmp_path):
    import mlprofessor

    mlprofessor.profile(full_runs["reg_csv"], "y", profile="full", n_jobs=2, seed=0,
                        out_dir=tmp_path / "two")
    two = h.load_json_strict(tmp_path / "two" / "profile.json")
    one = h.load_json_strict(full_runs["reg_dir"] / "profile.json")
    assert one["meta"]["n_jobs"] == 1 and two["meta"]["n_jobs"] == 2
    drop = ("meta.timing", "meta.n_jobs")
    assert h.without(one, *drop) == h.without(two, *drop)
    assert ((tmp_path / "two" / "predictions.csv").read_bytes()
            == (full_runs["reg_dir"] / "predictions.csv").read_bytes())


def test_n_jobs_minus_one_records_cpu_count():
    import mlprofessor
    from mlprofessor import extensions

    with extensions.use(families=[h.family("rec", h.Log())]):
        res = mlprofessor.profile(h.clf_frame(150, seed=3), "y", profile="fast", n_jobs=-1)
    assert res.to_dict()["meta"]["n_jobs"] == os.cpu_count()


def test_command_line_defaults_to_full(full_runs, tmp_path):
    proc = h.run_cli(["profile", "--data", full_runs["reg_csv"], "--target", "y",
                      "--out", tmp_path / "cli", "--n-jobs", "1"])
    assert proc.returncode == 0, proc.stderr
    cli_doc = h.load_json_strict(tmp_path / "cli" / "profile.json")
    fn_doc = h.load_json_strict(full_runs["reg_dir"] / "profile.json")
    assert cli_doc["meta"]["profile"] == "full"
    assert h.without(cli_doc, "meta.timing") == h.without(fn_doc, "meta.timing")
