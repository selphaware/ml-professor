"""CP9 measurement lab: runs of a package on the gate's tables, clean or with one
change, at many data seeds, side by side.

Nothing here writes inside the repo. A job is a dict:

    run      a run of the gate (verify/smoke_test.py RUNS): its dataset and its
             statements are the gate's own
    k        the data seed is the gate's base seed + 100 * k (k = 0 is the gate's)
    change   the name of a change in CHANGES, or None
    args     overrides of the call: time_col, group_col, gap, duplicates, profile
    seed     the tool's seed (default 0)

`execute(package, jobs, env)` runs them in worker processes that import `package`
(a folder named mlprofessor) from its parent, with `env` set before the import,
and returns one record for each job, in order.
"""
import io
import json
import math
import os
import random
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
# the repo: three folders up from docs/measurements/CP9; MLPROF_REPO names another tree
REPO = Path(os.environ.get("MLPROF_REPO") or Path(__file__).resolve().parents[3])
GATE_FILE = REPO / "verify" / "smoke_test.py"
WORK = Path(os.environ.get("MLPROF_LAB_WORK") or Path(tempfile.gettempdir()) / "mlprof-lab") / "cp9"

# The runs this pass added to the gate, each on a table with something planted in it.
# The measurements here are of the runs the gate had before.
ADDED = ("walk_repeated", "walk_copy", "walk_copy_cell", "walk_late_cell", "messy_shuffled")

_gate = None


def gate():
    global _gate
    if _gate is None:
        import importlib.util
        sys.path.insert(0, str(GATE_FILE.parent))
        spec = importlib.util.spec_from_file_location("mlprof_gate", str(GATE_FILE))
        module = importlib.util.module_from_spec(spec)
        saved = sys.argv
        sys.argv = [str(GATE_FILE)]
        try:
            spec.loader.exec_module(module)
        finally:
            sys.argv = saved
        _gate = module
    return _gate


def statements(extra):
    out = {"time_col": None, "group_col": None, "gap": 0, "duplicates": "error"}
    words = iter(extra)
    for word in words:
        if word == "--time-col":
            out["time_col"] = next(words)
        elif word == "--no-time-order":
            out["time_col"] = False
        elif word == "--group-col":
            out["group_col"] = next(words)
        elif word == "--no-entities":
            out["group_col"] = False
        elif word == "--gap":
            out["gap"] = int(next(words))
        elif word == "--duplicates":
            out["duplicates"] = next(words)
    return out


# ------------------------------------------------------------- the changes --
# Each takes (header, rows, target name, rng, info) and changes the table in place.
# `info` is a dict the change may write to: what it did, for the record.

def _usable(header, rows, target):
    t = header.index(target)
    return [i for i, row in enumerate(rows) if str(row[t]).strip() not in ("", "n/a")]


def _num(cell):
    return float(cell)


def ch_rows_repeated(share):
    def change(header, rows, target, rng, info):
        usable = _usable(header, rows, target)
        n = 1 if share == 0 else max(1, round(share * len(usable)))
        picks = rng.sample(usable, n)
        for i in picks:
            rows.insert(rng.randrange(len(rows) + 1), list(rows[i]))
        info["repeated"] = n
    return change


def ch_rows_shuffled(header, rows, target, rng, info):
    rng.shuffle(rows)


def _copy(make, cells=0, sentinel=None, name="target_copy"):
    def change(header, rows, target, rng, info):
        t = header.index(target)
        usable = _usable(header, rows, target)
        values = {i: make(_num(rows[i][t])) for i in usable}
        if cells:
            picks = rng.sample(usable, cells)
            moved = [values[i] for i in picks]
            moved = moved[1:] + moved[:1]
            for i, v in zip(picks, moved):
                values[i] = v
            info["differ"] = picks
        if sentinel is not None:
            i = rng.choice(usable)
            values[i] = sentinel
            info["sentinel_row"] = i
        header.append(name)
        for i, row in enumerate(rows):
            row.append(repr(values[i]) if i in values else "")
    return change


def _noisy_copy(sd_share):
    def change(header, rows, target, rng, info):
        t = header.index(target)
        usable = _usable(header, rows, target)
        ys = [_num(rows[i][t]) for i in usable]
        mean = sum(ys) / len(ys)
        sd = math.sqrt(sum((v - mean) ** 2 for v in ys) / len(ys))
        values = {i: 1.07 * _num(rows[i][t]) + 3 + rng.gauss(0, sd_share * sd) for i in usable}
        header.append("target_copy")
        for i, row in enumerate(rows):
            row.append(repr(values[i]) if i in values else "")
    return change


def _sentinel(column, value, which="any"):
    """One cell of `column` (a name, or the target when None) set to `value`."""
    def change(header, rows, target, rng, info):
        c = header.index(target if column is None else column)
        i = rng.choice(_usable(header, rows, target))
        rows[i][c] = value
        info["sentinel_row"] = i
    return change


def _empty(column):
    def change(header, rows, target, rng, info):
        c = header.index(column)
        i = rng.choice(_usable(header, rows, target))
        rows[i][c] = ""
        info["sentinel_row"] = i
    return change


def _both(*changes):
    def change(header, rows, target, rng, info):
        for one in changes:
            one(header, rows, target, rng, info)
    return change


CHANGES = {
    "row_repeated": ch_rows_repeated(0),
    "rows_repeated_1pct": ch_rows_repeated(0.01),
    "rows_repeated_10pct": ch_rows_repeated(0.10),
    "rows_shuffled": ch_rows_shuffled,
    "copy_exact": _copy(lambda v: v),
    "copy_rescaled": _copy(lambda v: 1.07 * v + 3),
    "copy_negative": _copy(lambda v: -250.0 * v + 1e6),
    "copy_tiny_scale": _copy(lambda v: 1e-9 * v),
    "copy_5_cells_differ": _copy(lambda v: 1.07 * v + 3, cells=5),
    "copy_15_cells_differ": _copy(lambda v: 1.07 * v + 3, cells=15),
    "copy_2pct_cells_differ": _copy(lambda v: 1.07 * v + 3, cells=30),
    "copy_sentinel_999999": _copy(lambda v: 1.07 * v + 3, sentinel=999999.0),
    "copy_sentinel_-999": _copy(lambda v: 1.07 * v + 3, sentinel=-999.0),
    "copy_noise_5pct": _noisy_copy(0.05),
    "copy_noise_10pct": _noisy_copy(0.10),
    "copy_noise_20pct": _noisy_copy(0.20),
    "copy_squared": _copy(lambda v: (v + 100.0) ** 2),
    "copy_exp": _copy(lambda v: math.exp(v / 20.0)),
    "target_-999": _sentinel(None, "-999"),
    "target_999999": _sentinel(None, "999999"),
    "target_-999_and_copy": _both(_copy(lambda v: 1.07 * v + 3), _sentinel(None, "-999")),
}


def feature_sentinel(column, value):
    return _sentinel(column, value)


def feature_empty(column):
    return _empty(column)


# --------------------------------------------------------------- one job --

def _init(package, env):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(env)
    os.environ["MLP_DEBUG"] = "1"
    sys.path.insert(0, str(Path(package).resolve().parent))
    import mlprofessor      # noqa: F401
    gate()


def table_of(job):
    g = gate()
    dataset, extra, prof, _ = g.RUNS[job["run"]]
    target = g.DATASETS[dataset][1]
    header, rows, blank_after = g.DATASETS[dataset][0](g.dataset_seed(dataset, g.SEED + 100 * job["k"]))
    header, rows = [str(h) for h in header], [[str(c) for c in row] for row in rows]
    info = {}
    change = job.get("change")
    if change:
        rng = random.Random(f"cp9.{job['run']}.{change}.{job['k']}.{job.get('draw', 0)}")
        if change.startswith("feature:"):
            _, column, value = change.split(":")
            fn = feature_empty(column) if value == "empty" else feature_sentinel(column, value)
        else:
            fn = CHANGES[change]
        fn(header, rows, target, rng, info)
        blank_after = ()
    lines = [",".join(g._quote(h) for h in header)]
    for i, row in enumerate(rows):
        lines.append(",".join(g._quote(c) for c in row))
        if i in blank_after:
            lines.append("")
    call = statements(extra if isinstance(extra, list) else g.NEITHER)
    call["profile"] = prof
    call.update(job.get("args") or {})
    return "\n".join(lines) + "\n", target, call, info


def _job(job):
    import mlprofessor
    from mlprofessor import errors
    text, target, call, info = table_of(job)
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"t{os.getpid()}.csv"
    path.write_bytes(text.encode("utf-8"))
    out = {"run": job["run"], "k": job["k"], "change": job.get("change"), "info": info}
    core = sys.modules.get("mlprofessor.core")
    proto = sys.modules.get("mlprofessor._proto")
    if proto is not None:
        proto.LOG = {}
    try:
        d = mlprofessor.profile(str(path), target, n_jobs=1, seed=job.get("seed", 0),
                                **call).to_dict()
    except errors.ProfileError as exc:
        out["error"] = exc.code
        out["message"] = str(exc)[:300]
        return out
    except Exception as exc:                                    # a raw error is a finding
        out["error"] = "RAW:" + type(exc).__name__
        out["message"] = str(exc)[:300]
        return out
    h = d["headline"]
    out.update(
        skill=h["skill"], score=h["score"], model=h["model"], baseline=h["baseline"],
        baseline_score=h["baseline_score"], scheme=d["validation"]["scheme"],
        baselines={b["name"]: b["score"] for b in d["baselines"]},
        roles={c["name"]: (c["role"], c["reason"], c["leak_rule"]) for c in d["columns"]},
        warnings=[(w["code"], w["column"]) for w in d["warnings"]],
        rows={r["item"]: r["count"] for r in d["rows"]},
        folds=[(f["training_rows"], f["validation_rows"], f["scored"])
               for f in d["validation"]["folds"]],
        cv={m["config"]: m["cv_mean"] for m in d["models"]},
        unique=d["validation"].get("timestamps_unique"),
        share=h.get("largest_row_share"), share_row=h.get("largest_row"),
        base_share=h.get("baseline_largest_row_share"), base_share_row=h.get("baseline_largest_row"),
    )
    if proto is not None:
        out["log"] = getattr(proto, "LOG", {})
    del core
    return out


def execute(package, jobs, env=None, workers=28):
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(str(package), dict(env or {}))) as pool:
        return list(pool.map(_job, jobs, chunksize=1))


def save(path, records):
    Path(path).write_bytes(json.dumps(records, indent=1, default=str).encode("utf-8"))


def span(values, places=3):
    values = [v for v in values if v is not None]
    if not values:
        return "none"
    mean = sum(values) / len(values)
    return f"{min(values):+.{places}f} to {max(values):+.{places}f} (mean {mean:+.{places}f})"
