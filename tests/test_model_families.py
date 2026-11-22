"""T14: the families under `full`, failures, and the seam's scoping (spec 2.6, 4.1, step 9)."""
import pytest

import mlp_helpers as h

DEFAULTS = {
    "classification": {
        "linear-0": {"C": 1.0},
        "forest-0": {"min_samples_leaf": 1, "max_features": "sqrt"},
        "boosting-0": {"max_leaf_nodes": 31, "learning_rate": 0.1, "max_iter": 100},
    },
    "regression": {
        "linear-0": {"alpha": 1.0},
        "forest-0": {"min_samples_leaf": 1, "max_features": 1 / 3},
        "boosting-0": {"max_leaf_nodes": 31, "learning_rate": 0.1, "max_iter": 100},
    },
}


def params_match(actual, expected):
    if set(actual) != set(expected):
        return False
    for key, want in expected.items():
        got = actual[key]
        if isinstance(want, float):
            if not (isinstance(got, (int, float)) and not isinstance(got, bool)
                    and abs(got - want) < 1e-9):
                return False
        elif got != want or type(got) is bool:
            return False
    return True


def _profile(frame, **kw):
    import mlprofessor

    args = {"n_jobs": 1, "seed": 0, "profile": "full"}
    args.update(kw)
    return h.stated_profile(frame, "y", **args)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_full_families_in_order_with_defaults(full_runs, task):
    res = full_runs["clf" if task == "classification" else "reg"]
    models = h.models_of(res)
    order = []
    for m in models:
        if m["family"] not in order:
            order.append(m["family"])
    assert order == ["linear", "forest", "boosting"]
    for fam in order:
        first = [m for m in models if m["family"] == fam][0]
        name = f"{fam}-0"
        assert first["config"] == name
        assert params_match(first["params"], DEFAULTS[task][name]), (name, first["params"])
        assert first["status"] == "fitted"


def test_f6_failing_family_is_recorded_not_dropped():
    """F6: a configuration that raises stays in the table, failed, with its error and a warning."""
    from mlprofessor import extensions

    log = h.Log()
    fams = [h.family("broken", log, h.FailingEstimator, grid=[{"k": 1}, {"k": 2}]),
            h.family("ok", log)]
    with extensions.use(families=fams):
        res = _profile(h.clf_frame(200, seed=1))
    models = {m["config"]: m for m in h.models_of(res)}
    assert list(models) == ["broken-0", "broken-1", "ok-0"]
    for name in ["broken-0", "broken-1"]:
        m = models[name]
        assert m["status"] == "failed"
        assert h.FailingEstimator.MESSAGE in m["error"]
        assert m["chosen"] is False
        assert m["fold_scores"] == [None] * 5 and m["cv_mean"] is None
    assert models["broken-1"]["params"] == {"k": 2}
    assert models["ok-0"]["status"] == "fitted" and models["ok-0"]["error"] is None
    assert res.headline["model"] == "ok-0"
    failed = [(c, col) for c, col in h.warnings_of(res) if c == "model_failed"]
    assert failed == [("model_failed", None)] * 2


def test_every_family_failing_raises():
    from mlprofessor import errors, extensions

    fams = [h.family("broken", h.Log(), h.FailingEstimator),
            h.family("also_broken", h.Log(), h.FailingEstimator)]
    with extensions.use(families=fams):
        with pytest.raises(errors.AllModelsFailedError) as info:
            _profile(h.clf_frame(200, seed=2))
    assert h.FailingEstimator.MESSAGE in str(info.value)


def _configs(frame, **kw):
    return [m["config"] for m in h.models_of(_profile(frame, **kw))]


def test_extensions_use_scopes_restores_and_nests():
    from mlprofessor import extensions

    frame = h.clf_frame(200, seed=3)
    fam_a = h.family("a", h.Log())
    fam_b = h.family("b", h.Log())
    with extensions.use(families=[fam_a]):
        assert _configs(frame) == ["a-0"]
        with extensions.use(families=[fam_b]):
            assert _configs(frame) == ["b-0"]
        assert _configs(frame) == ["a-0"]

    step_log = h.Log()
    with pytest.raises(KeyError):
        with extensions.use(families=[fam_b], steps=[h.step_factory(step_log)]):
            raise KeyError("leave the block by an exception")
    with extensions.use(families=[fam_a]):
        assert _configs(frame) == ["a-0"]
    assert step_log.events == []
    assert _configs(frame, profile="fast") == ["linear-0", "tree-0"]
