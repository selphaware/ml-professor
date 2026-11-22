"""Shared helpers for the suite: table builders, recording fakes, arithmetic by hand.

Nothing here imports `mlprofessor`: every test imports the application inside
its own body, so one missing module fails only the tests that need it.
"""
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

ROWS_ITEMS = [
    "loaded", "dropped_missing_target", "dropped_missing_time", "dropped_gap",
    "dropped_row_cap", "training", "holdout",
]
COLUMN_KEYS = [
    "name", "position", "dtype", "role", "kind", "reason", "leak_rule",
    "n_missing", "n_coerced", "n_infinite", "n_imputed",
]
CAVEAT = ("Random split: this number is optimistic if the rows are ordered in "
          "time or one entity appears on many rows.")
LIMITATIONS = [
    "Predictability here means how well these model families did on this data under this validation scheme.",
    "Low skill is not proof that the target cannot be predicted.",
    "High skill is not proof of cause.",
    "A random split is optimistic when rows are ordered in time or one entity appears on many rows.",
    "Look-ahead inside the caller's own features cannot be detected.",
    "Classification skill and regression skill are on different scales and are not comparable.",
    "A CSV file is read with pandas' types, so text identifiers can lose leading zeros; pass a DataFrame to control the types.",
    "Intervals are approximate: they tend to be too narrow when the holdout is short and, under time order, when errors are serially correlated.",
    "Negative skill means the models did worse than the baseline on the holdout; read it as no evidence of predictability.",
]
FAST = {"profile": "fast", "n_jobs": 1}


# ---------------------------------------------------------------- arithmetic

def auc_by_hand(y, score):
    """ROC AUC by counting pairs: a win is 1, a tie 1/2. No ranks, no sklearn."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pos, neg = score[y == 1], score[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins) / (len(pos) * len(neg))


def mse_by_hand(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean((y - pred) ** 2))


# ------------------------------------------------------------ normalisation

def plain(value):
    """Read a value as JSON would: NaN/None/NA/NaT as None, numpy as Python."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def records(frame):
    return plain(frame.to_dict(orient="records"))


def _reject_constant(name):
    raise ValueError(f"non-finite constant {name} in JSON")


def load_json_strict(path):
    """Parse JSON, failing on NaN, Infinity or -Infinity."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_constant)


def without(doc, *paths):
    """A deep copy of `doc` with dotted keys removed (e.g. "meta.timing")."""
    doc = json.loads(json.dumps(doc))
    for path in paths:
        parts = path.split(".")
        node = doc
        for part in parts[:-1]:
            node = node[part]
        node.pop(parts[-1], None)
    return doc


# ----------------------------------------------------------- result readers

def rows_of(result):
    return {str(i): int(c) for i, c in zip(result.rows["item"], result.rows["count"])}


def warnings_of(result):
    """List of (code, column) pairs, column None when null."""
    return [(r["code"], r["column"]) for r in records(result.warnings)]


def columns_of(result):
    return {r["name"]: r for r in records(result.columns)}


def models_of(result):
    return records(result.models)


def holdout_rows(result):
    return [int(r) for r in result.predictions["row"]]


# ------------------------------------------------------------------- tables

def clf_frame(n=200, seed=0, n_noise=2, signal=2.5, labels=(0, 1), target="y"):
    """x1 drives a two-class target; z1.. are noise."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    y01 = (rng.random(n) < 1.0 / (1.0 + np.exp(-signal * x1))).astype(int)
    frame = pd.DataFrame({"x1": x1})
    for i in range(n_noise):
        frame[f"z{i + 1}"] = rng.normal(size=n)
    if tuple(labels) == (0, 1):
        frame[target] = y01
    else:
        frame[target] = [labels[v] for v in y01]
    return frame


def reg_frame(n=200, seed=0, n_noise=2, target="y"):
    """y = 2*x1 + x2 + noise; z1.. are noise."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    frame = pd.DataFrame({"x1": x1, "x2": x2})
    for i in range(n_noise):
        frame[f"z{i + 1}"] = rng.normal(size=n)
    frame[target] = 2.0 * x1 + x2 + 1.5 * rng.normal(size=n)
    return frame


def dates(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="D")


def time_frame(n=200, seed=0, task="regression", walk=False):
    """Unique daily timestamps in column `date`; one row each."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"date": dates(n)})
    frame["x1"] = rng.normal(size=n)
    frame["z1"] = rng.normal(size=n)
    if walk:
        y = np.cumsum(rng.normal(size=n))
    elif task == "regression":
        y = 2.0 * frame["x1"].to_numpy() + rng.normal(size=n)
    else:
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-2.5 * frame["x1"].to_numpy()))).astype(int)
    frame["y"] = y
    return frame


def panel_frame(n_times=60, per_time=5, seed=0, persistent=True):
    """Repeated timestamps: `per_time` rows per date; regression target."""
    rng = np.random.default_rng(seed)
    if persistent:
        level = np.cumsum(rng.normal(size=n_times)) * 3.0
    else:
        level = rng.normal(size=n_times) * 3.0
    when = np.repeat(dates(n_times).to_numpy(), per_time)
    lev = np.repeat(level, per_time)
    n = n_times * per_time
    frame = pd.DataFrame({"date": when, "x1": rng.normal(size=n)})
    frame["y"] = lev + 0.5 * frame["x1"].to_numpy() + 0.3 * rng.normal(size=n)
    return frame


def lag1_autocorr(series):
    s = np.asarray(series, dtype=float)
    a, b = s[:-1] - s.mean(), s[1:] - s.mean()
    return float((a * b).sum() / ((s - s.mean()) ** 2).sum())


# ------------------------------------------------------------ recording fakes

class Log:
    """Every fit, predict and transform the fakes receive, in call order."""

    def __init__(self):
        self.events = []
        self.threads = []
        self._ids = itertools.count()

    def new_id(self):
        return next(self._ids)

    def add(self, owner, action, X, y=None):
        self.events.append({
            "oid": owner.oid,
            "kind": owner.kind,
            "family": getattr(owner, "family", None),
            "params": getattr(owner, "params", None),
            "action": action,
            "is_frame": isinstance(X, pd.DataFrame),
            "rows": [int(i) for i in X.index],
            "columns": [str(c) for c in X.columns],
            "y": None if y is None else list(np.asarray(y).tolist()),
            "y_index_ok": None if y is None else list(y.index) == list(X.index),
        })

    def objects(self, kind):
        """Calls grouped per object, in creation order."""
        out = {}
        for ev in self.events:
            if ev["kind"] != kind:
                continue
            rec = out.setdefault(ev["oid"], {
                "family": ev["family"], "params": ev["params"], "n_fit": 0,
                "fit": [], "predict": [], "transform": [], "fit_columns": [],
                "fit_y": [],
            })
            if ev["action"] == "fit":
                rec["n_fit"] += 1
                rec["fit"].extend(ev["rows"])
                rec["fit_columns"] = ev["columns"]
                rec["fit_y"] = ev["y"]
            else:
                rec[ev["action"]].extend(ev["rows"])
        return list(out.values())

    def all_fit_rows(self):
        rows = set()
        for ev in self.events:
            if ev["action"] == "fit":
                rows.update(ev["rows"])
        return rows


class _FakeEstimator:
    kind = "model"

    def __init__(self, log, family, task, params, seed, n_jobs):
        self.log, self.family, self.task = log, family, task
        self.params, self.seed, self.n_jobs = dict(params), seed, n_jobs
        self.oid = log.new_id()

    def _proba(self, v):
        v = np.clip(np.asarray(v, dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - v, v])


class WrapEstimator(_FakeEstimator):
    """Records its calls and delegates to a real linear model."""

    def __init__(self, *args, capture_threads=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_threads = capture_threads

    def fit(self, X, y):
        self.log.add(self, "fit", X, y)
        if self.capture_threads:
            from threadpoolctl import threadpool_info
            self.log.threads.append(threadpool_info())
        from sklearn.linear_model import LogisticRegression, Ridge
        if self.task == "classification":
            self.inner = LogisticRegression(max_iter=1000)
        else:
            self.inner = Ridge(alpha=1.0)
        self.inner.fit(X.to_numpy(dtype=float), np.asarray(y, dtype=float))
        return self

    def predict_proba(self, X):
        self.log.add(self, "predict", X)
        return self.inner.predict_proba(X.to_numpy(dtype=float))

    def predict(self, X):
        self.log.add(self, "predict", X)
        return self.inner.predict(X.to_numpy(dtype=float))


class LookupEstimator(_FakeEstimator):
    """Predicts values the test chose, looked up by source row position."""

    def __init__(self, *args, values=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.values = np.asarray(values, dtype=float)

    def fit(self, X, y):
        self.log.add(self, "fit", X, y)
        return self

    def predict_proba(self, X):
        self.log.add(self, "predict", X)
        return self._proba(self.values[np.asarray(X.index, dtype=int)])

    def predict(self, X):
        self.log.add(self, "predict", X)
        return self.values[np.asarray(X.index, dtype=int)].copy()


def _row_noise(rows):
    return np.array([np.random.default_rng(10_007 + int(r)).random() for r in rows])


class MemorizeEstimator(_FakeEstimator):
    """Perfect on every row it was fitted on, noise elsewhere."""

    def fit(self, X, y):
        self.log.add(self, "fit", X, y)
        self.known = dict(zip((int(i) for i in X.index), np.asarray(y, dtype=float)))
        return self

    def _values(self, X):
        rows = [int(i) for i in X.index]
        noise = _row_noise(rows)
        return np.array([self.known.get(r, noise[k]) for k, r in enumerate(rows)])

    def predict_proba(self, X):
        self.log.add(self, "predict", X)
        return self._proba(self._values(X))

    def predict(self, X):
        self.log.add(self, "predict", X)
        return self._values(X)


class FailingEstimator(_FakeEstimator):
    MESSAGE = "deliberate failure from the test family"

    def fit(self, X, y):
        self.log.add(self, "fit", X, y)
        raise RuntimeError(self.MESSAGE)

    def predict_proba(self, X):
        raise RuntimeError(self.MESSAGE)

    def predict(self, X):
        raise RuntimeError(self.MESSAGE)


def family(name, log, estimator=WrapEstimator, grid=None, **extra):
    """A family mapping for `extensions.use`, building a fresh fake per call."""
    def make(task, params, seed, n_jobs):
        return estimator(log, name, task, params, seed, n_jobs, **extra)
    return {"name": name, "grid": list(grid) if grid is not None else [{}], "make": make}


class RecordingStep:
    """A preprocessing step that changes nothing and records every call."""
    kind = "step"

    def __init__(self, log):
        self.log = log
        self.oid = log.new_id()

    def fit(self, X, y=None):
        self.log.add(self, "fit", X, y)
        return self

    def transform(self, X):
        self.log.add(self, "transform", X)
        return X.copy()


def step_factory(log):
    return lambda: RecordingStep(log)


# ------------------------------------------------------------- command line

def run_cli(args, timeout=900, as_given=False):
    """Run `python -m mlprofessor` in a new process from the repo root.

    D59, D63: when the run stops only because nothing was said about time order or
    about entities, it is run once more with `--no-time-order` and `--no-entities`
    added for whichever was not said, and that run is returned. `as_given=True`
    returns the first run whatever it is.
    """
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    def once(arguments):
        return subprocess.run(
            [PYTHON, "-m", "mlprofessor", *[str(a) for a in arguments]],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout, env=env,
        )

    proc = once(args)
    more = [] if as_given else missing_flags(proc, args)
    return once([*args, *more]) if more else proc


# ------------------------------------------------- the two statements (D63)

def missing_statements(code, said):
    """The statements a call left unsaid when it stopped with `code`, as arguments.

    `said` maps an argument's name to what the call passed. Empty unless the run
    stopped because nothing was said about time order or about entities (D63).
    """
    if code == "time_order_undeclared" and said.get("time_col") is None:
        more = {"time_col": False}
        if said.get("group_col") is None:
            more["group_col"] = False
        return more
    if code == "entities_undeclared" and said.get("group_col") is None:
        return {"group_col": False}
    return {}


def missing_flags(proc, args):
    """The same for a run of the command line: the flags to add, or []."""
    if proc.returncode != 3:
        return []
    words = [str(a) for a in args]

    def said(*flags):
        return any(w in flags or w.startswith(flags[0] + "=") for w in words) or None

    code = proc.stderr.strip().partition("error: ")[2].partition(":")[0]
    more = missing_statements(code, {"time_col": said("--time-col", "--no-time-order"),
                                     "group_col": said("--group-col", "--no-entities")})
    return [flag for name, flag in (("time_col", "--no-time-order"),
                                    ("group_col", "--no-entities")) if name in more]


def stated_profile(*args, **kw):
    """`mlprofessor.profile(*args, **kw)`, for a test written before every call had to
    say whether the rows are ordered in time and whether rows belong together (D59, D63).

    The call is made as the test wrote it. Only when it stops because a statement
    was not made is it made again, with each missing statement made as "no":
    `time_col=False`, `group_col=False`. Nothing else is changed, and any other
    error is raised as it is.
    """
    import mlprofessor
    from mlprofessor import errors

    try:
        return mlprofessor.profile(*args, **kw)
    except errors.ProfileError as exc:
        more = missing_statements(exc.code, kw)
        if not more:
            raise
    return mlprofessor.profile(*args, **{**kw, **more})


# D66: `validation.n_rows_with_copies` became `n_rows_same_features`, and the warning
# `copies_grouped` became `same_features_grouped`. A test written before that reads
# whichever of the two the application gives (D59).
SAME_FEATURES_CODES = ("same_features_grouped", "copies_grouped")


def same_features_rows(validation):
    """`validation.n_rows_same_features`, or `n_rows_with_copies` where it is still named so."""
    if "n_rows_same_features" in validation:
        return validation["n_rows_same_features"]
    return validation["n_rows_with_copies"]


def count_same_features_warnings(result):
    """How many warnings say that rows with the same features were kept together."""
    return sum(code in SAME_FEATURES_CODES for code, _ in warnings_of(result))


def files_in(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p.name for p in folder.rglob("*") if p.is_file())
