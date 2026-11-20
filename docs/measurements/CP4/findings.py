"""CP4: what one implementation does on the inputs of the second review's findings.

    python -B findings.py app|ref|<folder> [case ...]

`app` is the application, the folder `mlprofessor` at the repo root; `ref` is
the reference implementation under verify/reference; `<folder>` is any folder
that holds a package `mlprofessor`, such as a copy of verify/reference taken
before a change. Each line names a case
and prints what happened: the named error, a raw exception, or a short account
of the result (the headline, the baselines, the models, the warnings, whether
every prediction is finite, the classes). A case is selected by its first
word: A1 to A9 (numbers that are not finite), B (the seed), C (the summary),
D (white space round the target), E (the fast profile's error), G and G3
(lines before the header).

Files are written to a new folder in the system's temporary folder; nothing
is written inside the repo.
"""
import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if len(sys.argv) < 2:
    raise SystemExit(__doc__)
WHICH = sys.argv[1]
ROOT = {"app": REPO, "ref": REPO / "verify" / "reference"}.get(WHICH) or Path(WHICH).resolve()
WHICH = WHICH if WHICH in ("app", "ref") else "copy"
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mlprofessor  # noqa: E402
from mlprofessor import errors, extensions, stages  # noqa: E402

assert Path(mlprofessor.__file__).resolve().parent == ROOT / "mlprofessor", mlprofessor.__file__
WORK = Path(tempfile.mkdtemp(prefix=f"mlprof-cp4-findings-{WHICH}-"))
FAST = dict(profile="fast", n_jobs=1)
WANT = set(sys.argv[2:])


def brief(result):
    h = result.headline
    doc = result.to_dict()
    keys = ["score", "baseline", "baseline_score", "skill", "interval_method", "interval_low",
            "interval_high", "skill_low", "skill_high", "p_value", "model"]
    pred = result.predictions["y_pred"].to_numpy(dtype=float)
    return ("OK " + " ".join(f"{k}={h[k]!r}" for k in keys)
            + f" | baselines={[(b['name'], b['score']) for b in doc['baselines']]}"
            + f" | models={[(m['config'], m['status'], m['cv_mean']) for m in doc['models']]}"
            + f" | warnings={[w['code'] for w in doc['warnings']]}"
            + f" | y_pred finite={bool(np.isfinite(pred).all())}"
            + f" | classes={doc['task']['classes']} positive={doc['task']['positive_class']!r}"
            + f" n_distinct={doc['task']['n_distinct']}"
            + f" | rows={ {r['item']: r['count'] for r in doc['rows']} }")


def attempt(label, fn):
    if WANT and label.split(" ")[0] not in WANT:
        return
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001  (the point is to see what is raised)
        code = getattr(exc, "code", None)
        print(f"{label}: RAISES {type(exc).__name__} code={code}: {' '.join(str(exc).split())[:260]}")
    else:
        print(f"{label}: {out}")
    sys.stdout.flush()


def reg(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"t": np.arange(n, dtype=float), "x1": x1, "x2": x2,
                         "y": 20 + 2 * x1 + x2 + rng.normal(size=n)})


def clf(n=400, seed=0, labels=("no", "yes")):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-2.5 * x1))).astype(int)
    return pd.DataFrame({"t": np.arange(n, dtype=float), "x1": x1, "z1": rng.normal(size=n),
                         "y": [labels[v] for v in y]})


def run(frame, **kw):
    args = dict(FAST)
    args.update(kw)
    return brief(mlprofessor.profile(frame, "y", **args))


# ---------------------------------------------- A: numbers that are not finite
for scheme in ("random", "time"):
    kw = {} if scheme == "random" else {"time_col": "t"}
    for where in (5, 395):          # an early row, a late one (the holdout under time order)
        f = reg()
        f.loc[where, "y"] = 1.7976931348623157e308
        attempt(f"A1 one cell 1.797e308 at row {where}, {scheme}", lambda f=f, kw=kw: run(f, **kw))
    f = reg()
    f["y"] = f["y"] * 1e160
    attempt(f"A2 target x 1e160, {scheme}", lambda f=f, kw=kw: run(f, **kw))
    for where in (5, 120, 395):
        f = reg()
        f.loc[where, "y"] = 1e155
        attempt(f"A3 one cell 1e155 at row {where}, {scheme}", lambda f=f, kw=kw: run(f, **kw))
f = reg()
f["y"] = (f["y"] - 20) * 1e80
attempt("A3b target on the scale of 1e80, random", lambda: run(f))

f = reg()
f["x1"] = f["x1"] * 1e200
attempt("A4 feature x1 x 1e200 in every row, regression", lambda: run(f))
f = reg()
f.loc[7, "x1"] = 1e39
attempt("A4 one feature cell 1e39 (just above the 32-bit range), regression", lambda: run(f))
f = reg()
f.loc[7, "x1"] = 1e38
attempt("A4 one feature cell 1e38 (inside the 32-bit range), regression", lambda: run(f))
for where in (5, 395):
    f = reg()
    f.loc[where, "x1"] = 1e200
    attempt(f"A5 one feature cell 1e200 at row {where}, time order", lambda f=f: run(f, time_col="t"))


def prep_stats():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 1e200, np.inf, -1e39]})
    out = stages.Preprocessor({"a": "numeric"}).fit(frame).transform(frame)
    return out["a"].tolist()


attempt("A6 stages.Preprocessor on 1, 2, 3, 4, 1e200, inf, -1e39", prep_stats)


def roles_count():
    frame = clf()
    frame.loc[[3, 9], "x1"] = [1e39, np.inf]
    roles = stages.column_roles(frame, "y").set_index("name")
    return f"n_infinite of x1 (one cell 1e39, one inf): {int(roles.loc['x1', 'n_infinite'])}"


attempt("A6 stages.column_roles", roles_count)


class Fixed:
    """Predicts one value everywhere, with a little noise; `bad` on the rows named."""

    def __init__(self, task, value, bad_rows=(), bad=float("nan")):
        self.task, self.value, self.bad_rows, self.bad = task, value, set(bad_rows), bad

    def fit(self, X, y):
        return self

    def _v(self, X):
        out = np.full(len(X), self.value, dtype=float)
        out = out + np.array([((int(i) * 7919) % 97) / 970.0 for i in X.index])
        out[[int(i) in self.bad_rows for i in X.index]] = self.bad
        return out

    def predict(self, X):
        return self._v(X)

    def predict_proba(self, X):
        v = self._v(X)
        return np.column_stack([1 - v, v])


def fam(name, **kw):
    return {"name": name, "grid": [{}],
            "make": lambda task, params, seed, n_jobs: Fixed(task, **kw)}


def with_families(frame, families, **kw):
    with extensions.use(families=families):
        return run(frame, **kw)


first = mlprofessor.profile(clf(), "y", **FAST)
holdout = [int(r) for r in first.predictions["row"]]
train_some = [r for r in range(400) if r not in set(holdout)][:3]
attempt("A7 clf, a family predicting NaN on three holdout rows",
        lambda: with_families(clf(), [fam("nanny", value=0.4, bad_rows=holdout[:3])]))
attempt("A7 clf, a family predicting NaN on three training rows (folds)",
        lambda: with_families(clf(), [fam("nanny", value=0.4, bad_rows=train_some)]))
attempt("A7 clf, NaN-on-holdout family beside a sound one",
        lambda: with_families(clf(), [fam("nanny", value=0.4, bad_rows=holdout[:3]),
                                      fam("sound", value=0.3)]))
first = mlprofessor.profile(reg(), "y", **FAST)
holdout = [int(r) for r in first.predictions["row"]]
train_some = [r for r in range(400) if r not in set(holdout)][:3]
attempt("A8 reg, a family predicting NaN on three holdout rows",
        lambda: with_families(reg(), [fam("nanny", value=20.0, bad_rows=holdout[:3])]))
attempt("A8 reg, a family predicting NaN on three training rows (folds), beside a sound one",
        lambda: with_families(reg(), [fam("nanny", value=20.0, bad_rows=train_some),
                                      fam("sound", value=19.0)]))
attempt("A8 reg, a family predicting 1e200 everywhere (finite, far off)",
        lambda: with_families(reg(), [fam("far", value=1e200)]))
attempt("A8 reg, far-off family beside a sound one",
        lambda: with_families(reg(), [fam("far", value=1e200), fam("sound", value=19.0)]))

f = reg()
f["y"] = f["y"].astype(complex) + 2j
attempt("A9 complex target, task=regression", lambda: run(f, task="regression"))
attempt("A9 complex target, task=auto", lambda: run(f))
attempt("A9 stages.decide_task on complex values, regression",
        lambda: stages.decide_task(pd.Series(np.arange(40) + 1j), task="regression"))

# ------------------------------------------------------------- B: the seed
for seed in (2 ** 32 - 1, 2 ** 32, 2 ** 63, 20261119123456, np.int64(7), True, 3.0):
    attempt(f"B seed={seed!r} ({type(seed).__name__})", lambda seed=seed: run(clf(), seed=seed))
for seed in (2 ** 32, 2 ** 32 - 1, -1):
    attempt(f"B2 stages.split_random seed={seed}",
            lambda seed=seed: stages.split_random(120, None, seed)["holdout"][:3])


def cli(*args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-B", "-m", "mlprofessor", *[str(a) for a in args]],
                          cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8")


def cli_seed():
    path = WORK / "clf.csv"
    clf().to_csv(path, index=False)
    out = WORK / "seed_out"
    proc = cli("profile", "--data", path, "--target", "y", "--out", out, "--profile", "fast",
               "--n-jobs", "1", "--seed", 2 ** 32)
    lines = [ln for ln in proc.stderr.splitlines() if ln.strip()]
    return (f"exit {proc.returncode}; {len(lines)} line(s) on standard error; first "
            f"{lines[:1]}; last {lines[-1:]}; files "
            f"{sorted(p.name for p in out.glob('*')) if out.exists() else []}")


attempt("B3 command line --seed 2**32", cli_seed)


# ---------------------------------------------------------- C: the summary
def small_scale():
    f = reg()
    f["y"] = (f["y"] - 20) * 1e-4
    path = WORK / "small.csv"
    f.to_csv(path, index=False)
    out = WORK / "small_out"
    proc = cli("profile", "--data", path, "--target", "y", "--out", out, "--profile", "fast",
               "--n-jobs", "1")
    head = json.loads((out / "profile.json").read_text(encoding="utf-8"))["headline"]
    return (f"exit {proc.returncode}\n--- stdout\n{proc.stdout}--- profile.json: "
            + ", ".join(f"{k}={head[k]!r}" for k in ("score", "baseline_score", "skill",
                                                     "interval_low", "interval_high", "skill_low",
                                                     "skill_high", "p_value")))


attempt("C small-scale target through the command line", small_scale)

# ------------------------------------------ D: white space round the target
f = clf()
f.loc[[3, 50, 120, 300], "y"] = f.loc[[3, 50, 120, 300], "y"] + " "
attempt("D target with four cells written with a trailing space, auto", lambda: run(f))
attempt("D the same, task=classification", lambda: run(f, task="classification"))
f2 = clf()
f2["y"] = np.where(f2["y"] == "yes", "yes ", "no")
attempt("D every yes written 'yes '", lambda: run(f2))
attempt("D stages.decide_task(['yes', 'yes ', 'no', ' no'])",
        lambda: stages.decide_task(pd.Series(["yes", "yes ", "no", " no"] * 30)))
f3 = clf()
f3.loc[[3, 50], "y"] = "   "
attempt("D two target cells of only spaces (DataFrame)", lambda: run(f3))
attempt("D stages.usable_rows on the same",
        lambda: {k: v for k, v in stages.usable_rows(f3, "y").items() if k != "usable"})


# ------------------------------------------- E: the fast profile's error
def panel(n_times, per_time, seed=0):
    rng = np.random.default_rng(seed)
    n = n_times * per_time
    f = pd.DataFrame({"t": np.repeat(np.arange(n_times, dtype=float), per_time),
                      "x1": rng.normal(size=n)})
    f["y"] = f["x1"] + rng.normal(size=n)
    return f


light = [fam("sound", value=0.0)]
for n_times, per_time, gap in ((5, 600, 0), (7, 450, 0), (8, 450, 0), (30, 300, 0), (3, 2500, 0),
                               (12, 2500, 0), (9, 450, 2), (30, 300, 3)):
    f = panel(n_times, per_time)
    for profile in ("fast", "full"):
        attempt(f"E {n_times} timestamps x {per_time} rows, gap {gap}, {profile}",
                lambda f=f, gap=gap, profile=profile:
                with_families(f, light, time_col="t", gap=gap, profile=profile)[:60])


# -------------------------------------------- G: lines before the header
def lead(first_line, columns=2):
    def go():
        path = WORK / "lead.csv"
        if columns == 2:
            body = "a,y\n" + "".join(f"{(i * 7) % 13 + 0.5},{i % 2}\n" for i in range(160))
        else:
            body = "a,b,y\n" + "".join(f"{(i * 7) % 13 + 0.5},{i % 5},{i % 2}\n" for i in range(160))
        path.write_bytes(first_line + body.encode("utf-8"))
        seen = pd.read_csv(path, low_memory=False)
        pandas_says = (f"pandas: columns {list(seen.columns)!r}, {len(seen)} rows, index "
                       f"{type(seen.index).__name__}")
        try:
            frame, _ = stages.load_table(str(path))
            loaded = f"load_table: columns {list(frame.columns)!r}, {len(frame)} rows"
        except errors.ProfileError as exc:
            loaded = f"load_table: {exc.code}"
        try:
            mlprofessor.profile(str(path), "y", **FAST)
            ran = "profile: runs"
        except errors.ProfileError as exc:
            ran = f"profile: {exc.code}"
        return f"{pandas_says} | {loaded} | {ran}"
    return go


for name, line in (("form feed", b"\f\n"), ("no-break space", " \n".encode("utf-8")),
                   ("two quotes round a space", b'" "\n'), ("two quotes", b'""\n'),
                   ("vertical tab", b"\v\n"), ("em space", " \n".encode("utf-8"))):
    attempt(f"G a line of {name} before a two-column header", lead(line))
    attempt(f"G3 a line of {name} before a three-column header", lead(line, columns=3))
