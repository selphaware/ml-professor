"""CP10 measurement lab: runs of a package on the gate's tables, clean or with one
change, and on tables made here, at many data seeds, side by side.

Nothing here writes inside the repo. A job is a dict:

    run      a run of the gate (verify/smoke_test.py RUNS): its dataset and statements
    k        the data seed is the gate's base seed + 100 * k (k = 0 is the gate's)
    change   the name of a change (CHANGES here, or lab9's), or None
    args     overrides of the call: time_col, group_col, gap, duplicates, profile
    seed     the tool's seed (default 0)

or

    make     the name of a table maker in tables10.py
    params   its keyword arguments
    k        its seed

`execute(package, jobs, env)` runs them in worker processes that import `package`
(a folder named mlprofessor) from its parent, with `env` set before the import.
"""
import json
import math
import os
import random
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("MLPROF_REPO") or r"E:\CODE\ml-professor")
sys.path.insert(0, str(REPO / "docs" / "measurements" / "CP9"))
sys.path.insert(0, str(HERE))
os.environ.setdefault("MLPROF_REPO", str(REPO))
import lab9                                             # noqa: E402

WORK = Path(os.environ.get("MLPROF_LAB_WORK") or Path(tempfile.gettempdir()) / "mlprof-lab") / "cp10"
SHOW = ("strong_single_column", "suspected_leak", "near_perfect_headline", "large_holdout",
        "holdout_far_above_quarter", "shuffled_score_not_computed", "extreme_value",
        "largest_row_dominates")


# ------------------------------------------------------------- the changes --

def _add(header, rows, name, values):
    header.append(name)
    for i, row in enumerate(rows):
        row.append(values.get(i, ""))


def _ys(header, rows, target):
    t = header.index(target)
    usable = lab9._usable(header, rows, target)
    return usable, {i: float(rows[i][t]) for i in usable}


def _unit(header, rows, target, info):
    """The target's typical size: its median absolute step in time order when the run
    has a time column (within each entity when it has an entity column too), or its
    standard deviation."""
    usable, ys = _ys(header, rows, target)
    call = info.get("call", {})
    time_col, group_col = call.get("time_col"), call.get("group_col")
    if isinstance(time_col, str):
        c = header.index(time_col)
        g = header.index(group_col) if isinstance(group_col, str) else None
        series = {}
        for i in usable:
            series.setdefault(rows[i][g] if g is not None else "", []).append((rows[i][c], ys[i]))
        steps = []
        for points in series.values():
            try:
                points.sort(key=lambda p: float(p[0]))
            except ValueError:
                points.sort(key=lambda p: p[0])
            steps += [abs(b[1] - a[1]) for a, b in zip(points, points[1:])]
        steps.sort()
        info["unit"] = "step"
        return steps[len(steps) // 2]
    mean = sum(ys.values()) / len(ys)
    info["unit"] = "spread"
    return math.sqrt(sum((v - mean) ** 2 for v in ys.values()) / len(ys))


def bands(q, kind):
    """Equal-count bands of a regression target, as a text column whose labels are in
    random order or in the target's order, or as a numeric code in random order."""
    def change(header, rows, target, rng, info):
        usable, ys = _ys(header, rows, target)
        order = sorted(usable, key=lambda i: ys[i])
        n = len(order)
        names = list(range(q))
        rng.shuffle(names)
        values = {}
        for pos, i in enumerate(order):
            band = min(q - 1, pos * q // n)
            values[i] = (f"tier_{names[band]:03d}" if kind == "text_random"
                         else f"tier_{band:03d}" if kind == "text_ordered"
                         else str(100 + names[band]))
        _add(header, rows, "target_band", values)
    return change


def codes(levels, kind):
    """A classification: a code of `levels` levels, each of which holds one class, as text
    or as numbers, the labels in random order."""
    def change(header, rows, target, rng, info):
        t = header.index(target)
        usable = lab9._usable(header, rows, target)
        classes = sorted({str(rows[i][t]).strip() for i in usable})
        names = list(range(levels))
        rng.shuffle(names)
        half = levels // 2
        of_class = {classes[0]: names[:half], classes[1]: names[half:]}
        values = {}
        for i in usable:
            name = rng.choice(of_class[str(rows[i][t]).strip()])
            values[i] = f"code_{name:03d}" if kind == "text" else str(100 + name)
        _add(header, rows, "target_code", values)
    return change


def tied_noise(kind, amount):
    """A numeric column of noise with repeated values: `levels` distinct whole numbers,
    or normal draws rounded to `places` decimal places."""
    def change(header, rows, target, rng, info):
        if kind == "levels":
            values = {i: str(rng.randint(1, amount)) for i in range(len(rows))}
        else:
            values = {i: f"{rng.gauss(0, 1):.{amount}f}" for i in range(len(rows))}
        _add(header, rows, "tied_noise", values)
    return change


def copy_cells(cells, size, scale=1.07, shift=3.0):
    """A rescaled copy of the target with `cells` cells changed: by `size` times the
    target's typical step or spread, by a factor (`x1.5`), or to another row's value
    (`other`)."""
    def change(header, rows, target, rng, info):
        usable, ys = _ys(header, rows, target)
        unit = _unit(header, rows, target, info)
        picks = rng.sample(usable, cells)
        changed = dict(ys)
        for i in picks:
            if size == "other":
                changed[i] = ys[rng.choice([j for j in usable if j != i])]
            elif isinstance(size, str) and size.startswith("x"):
                changed[i] = ys[i] * float(size[1:])
            else:
                changed[i] = ys[i] + float(size) * unit
        info["differ"] = picks
        _add(header, rows, "target_copy", {i: repr(scale * changed[i] + shift) for i in usable})
    return change


def copy_rounded(base, distinct):
    """One of CP9's copies (2% of the cells rotated, the square, the exponential), put
    into `distinct` bands of equal width: a column of that many distinct values."""
    def change(header, rows, target, rng, info):
        usable, ys = _ys(header, rows, target)
        if base == "2pct":
            values = {i: 1.07 * ys[i] + 3 for i in usable}
            picks = rng.sample(usable, round(0.02 * len(usable)))
            moved = [values[i] for i in picks]
            for i, v in zip(picks, moved[1:] + moved[:1]):
                values[i] = v
        elif base == "squared":
            values = {i: (ys[i] + 100.0) ** 2 for i in usable}
        elif base == "exp":
            values = {i: math.exp(ys[i] / 20.0) for i in usable}
        else:
            values = {i: 1.07 * ys[i] + 3 for i in usable}
        low, high = min(values.values()), max(values.values())
        width = (high - low) / distinct or 1.0
        out = {i: repr(low + (min(distinct - 1, int((v - low) / width)) + 0.5) * width)
               for i, v in values.items()}
        _add(header, rows, "target_copy", out)
    return change


def two_together(header, rows, target, rng, info):
    """`pair_a` = the target + z and `pair_b` = z: either says little, both give the target."""
    t = header.index(target)
    usable = lab9._usable(header, rows, target)
    try:
        ys = {i: float(rows[i][t]) for i in usable}
    except ValueError:
        classes = sorted({str(rows[i][t]).strip() for i in usable})
        ys = {i: float(classes.index(str(rows[i][t]).strip())) for i in usable}
    if len(set(ys.values())) == 2:
        low = min(ys.values())
        ys = {i: float(v != low) for i, v in ys.items()}
    mean = sum(ys.values()) / len(ys)
    sd = math.sqrt(sum((v - mean) ** 2 for v in ys.values()) / len(ys)) or 1.0
    z = {i: rng.gauss(0, sd) for i in usable}
    _add(header, rows, "pair_a", {i: repr(ys[i] + z[i]) for i in usable})
    _add(header, rows, "pair_b", {i: repr(z[i]) for i in usable})


def duplicated(column, copies=1):
    def change(header, rows, target, rng, info):
        c = header.index(column)
        for n in range(copies):
            _add(header, rows, f"{column}_again{n + 1 if copies > 1 else ''}",
                 {i: row[c] for i, row in enumerate(rows)})
    return change


def duplicated_scaled(column):
    """A numeric column written again on another scale: 2 x the column + 3."""
    def change(header, rows, target, rng, info):
        c = header.index(column)
        values = {}
        for i, row in enumerate(rows):
            try:
                values[i] = repr(2.0 * float(row[c]) + 3.0)
            except ValueError:
                values[i] = row[c]
        _add(header, rows, f"{column}_scaled", values)
    return change


def duplicated_noisy(column, share):
    """A numeric column written again with a noise of `share` of its spread added: the
    same information, and not a copy as the models receive it."""
    def change(header, rows, target, rng, info):
        c = header.index(column)
        vals = []
        for row in rows:
            try:
                vals.append(float(row[c]))
            except ValueError:
                vals.append(None)
        got = [v for v in vals if v is not None]
        m = sum(got) / len(got)
        sd = (sum((v - m) ** 2 for v in got) / len(got)) ** 0.5 or 1.0
        _add(header, rows, f"{column}_near",
             {i: ("" if v is None else repr(v + rng.gauss(0, share * sd)))
              for i, v in enumerate(vals)})
    return change


def duplicated_negated(column):
    """A numeric column written again with its sign changed."""
    def change(header, rows, target, rng, info):
        c = header.index(column)
        values = {}
        for i, row in enumerate(rows):
            try:
                values[i] = repr(-float(row[c]))
            except ValueError:
                values[i] = row[c]
        _add(header, rows, f"{column}_negated", values)
    return change


def duplicated_relabelled(column):
    """A text column written again with every level renamed."""
    def change(header, rows, target, rng, info):
        c = header.index(column)
        _add(header, rows, f"{column}_renamed",
             {i: ("" if row[c].strip() == "" else "L_" + row[c].strip().upper())
              for i, row in enumerate(rows)})
    return change


def rounded_feature(column, distinct):
    def change(header, rows, target, rng, info):
        c = header.index(column)
        held = {}
        for i, row in enumerate(rows):
            try:
                held[i] = float(row[c])
            except ValueError:
                pass
        low, high = min(held.values()), max(held.values())
        width = (high - low) / distinct or 1.0
        for i, v in held.items():
            rows[i][c] = repr(low + (min(distinct - 1, int((v - low) / width)) + 0.5) * width)
    return change


def _parse(name):
    """`kind:arg:arg` -> a change."""
    if name in lab9.CHANGES:
        return lab9.CHANGES[name]
    kind, *rest = name.split(":")
    if kind == "bands":                 # bands:24:text_random
        return bands(int(rest[0]), rest[1])
    if kind == "codes":                 # codes:40:num
        return codes(int(rest[0]), rest[1])
    if kind == "tied":                  # tied:levels:40, tied:round:1
        return tied_noise(rest[0], int(rest[1]))
    if kind == "cells":                 # cells:5:20, cells:5:other, cells:5:x1.5
        size = rest[1] if rest[1] == "other" or rest[1].startswith("x") else float(rest[1])
        return copy_cells(int(rest[0]), size)
    if kind == "round":                 # round:2pct:40
        return copy_rounded(rest[0], int(rest[1]))
    if kind == "two":
        return two_together
    if kind == "dup":                   # dup:f1, dup:f1:2
        return duplicated(rest[0], int(rest[1]) if len(rest) > 1 else 1)
    if kind == "dupscaled":             # dupscaled:f1
        return duplicated_scaled(rest[0])
    if kind == "dupnoise":              # dupnoise:f1:0.05
        return duplicated_noisy(rest[0], float(rest[1]))
    if kind == "dupneg":                # dupneg:f1
        return duplicated_negated(rest[0])
    if kind == "duprelabel":            # duprelabel:region
        return duplicated_relabelled(rest[0])
    if kind == "roundf":                # roundf:f1:40
        return rounded_feature(rest[0], int(rest[1]))
    if kind == "feature":
        _, column, value = name.split(":")
        return lab9.feature_empty(column) if value == "empty" else lab9.feature_sentinel(column, value)
    raise KeyError(name)


def table_of(job):
    g = lab9.gate()
    dataset, extra, prof, _ = g.RUNS[job["run"]]
    target = g.DATASETS[dataset][1]
    header, rows, blank_after = g.DATASETS[dataset][0](g.dataset_seed(dataset, g.SEED + 100 * job["k"]))
    header, rows = [str(h) for h in header], [[str(c) for c in row] for row in rows]
    call = lab9.statements(extra if isinstance(extra, list) else g.NEITHER)
    call["profile"] = prof
    call.update(job.get("args") or {})
    info = {"call": call}
    changes = job.get("change")
    if changes:
        for change in changes.split("+"):
            rng = random.Random(f"cp10.{job['run']}.{change}.{job['k']}.{job.get('draw', 0)}")
            _parse(change)(header, rows, target, rng, info)
        blank_after = ()
    lines = [",".join(g._quote(h) for h in header)]
    for i, row in enumerate(rows):
        lines.append(",".join(g._quote(c) for c in row))
        if i in blank_after:
            lines.append("")
    info.pop("call")
    return "\n".join(lines) + "\n", target, call, info


# --------------------------------------------------------------- one job --

def _init(package, env):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(env)
    os.environ.setdefault("MLP_DEBUG", "1")
    sys.path.insert(0, str(Path(package).resolve().parent))
    import mlprofessor      # noqa: F401
    lab9.gate()


def _job(job):
    import warnings as pywarnings
    import mlprofessor
    from mlprofessor import errors
    out = {key: job.get(key) for key in ("run", "make", "params", "k", "change", "tag")}
    if job.get("make"):
        import tables10
        data, target, call, info = getattr(tables10, job["make"])(seed=job["k"], **(job.get("params") or {}))
        call = dict(call)
        call.setdefault("profile", "fast")
        call.update(job.get("args") or {})
    else:
        text, target, call, info = table_of(job)
        WORK.mkdir(parents=True, exist_ok=True)
        data = WORK / f"t{os.getpid()}.csv"
        data.write_bytes(text.encode("utf-8"))
        data = str(data)
    out["info"] = info
    proto = sys.modules.get("mlprofessor._proto")
    if proto is not None:
        proto.LOG = {}
    try:
        with pywarnings.catch_warnings():
            pywarnings.simplefilter("ignore")
            d = mlprofessor.profile(data, target, n_jobs=job.get("n_jobs", 1), seed=job.get("seed", 0),
                                    **call).to_dict()
    except errors.ProfileError as exc:
        out["error"] = exc.code
        out["message"] = str(exc)[:400]
        return out
    except Exception as exc:                                    # a raw error is a finding
        out["error"] = "RAW:" + type(exc).__name__
        out["message"] = str(exc)[:400]
        return out
    h = d["headline"]
    out.update(
        skill=h["skill"], score=h["score"], model=h["model"], baseline=h["baseline"],
        baseline_score=h["baseline_score"], scheme=d["validation"]["scheme"],
        holdout_share=d["validation"]["holdout_share"],
        low=h.get("skill_low"), high=h.get("skill_high"),
        baselines={b["name"]: b["score"] for b in d["baselines"]},
        roles={c["name"]: (c["role"], c["reason"], c["leak_rule"]) for c in d["columns"]},
        rests={c["name"]: (c.get("shuffled_score"), c.get("shuffled_skill"), c.get("skill_fall"))
               for c in d["columns"] if c["role"] == "feature"},
        warnings=[(w["code"], w["column"]) for w in d["warnings"]],
        texts=[(w["code"], w["column"], w["message"]) for w in d["warnings"] if w["code"] in SHOW],
        rows={r["item"]: r["count"] for r in d["rows"]},
        folds=[(f["training_rows"], f["validation_rows"], f["scored"])
               for f in d["validation"]["folds"]],
        cv={m["config"]: m["cv_mean"] for m in d["models"]},
        seconds=d["meta"]["timing"]["seconds"],
        rests_on=h.get("rests_on"),
        floor=h.get("rests_on_floor"),
    )
    if proto is not None:
        out["log"] = getattr(proto, "LOG", {})
    return out


def execute(package, jobs, env=None, workers=28):
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(str(package), dict(env or {}))) as pool:
        return list(pool.map(_job, jobs, chunksize=1))


def save(path, records):
    Path(path).write_bytes(json.dumps(records, indent=1, default=str).encode("utf-8"))


def load(path):
    return json.loads(Path(path).read_bytes().decode("utf-8"))


def span(values, places=3):
    values = [v for v in values if v is not None]
    if not values:
        return "none"
    mean = sum(values) / len(values)
    return f"{min(values):+.{places}f} to {max(values):+.{places}f} (mean {mean:+.{places}f})"


def fate(record, column):
    """What became of an added column: excluded, warned, named (by the near-perfect
    warning), silent, or the run's error."""
    if "error" in record:
        return "error:" + record["error"]
    role = record["roles"].get(column)
    if role is None:
        return "absent"
    if role[1] == "suspected_leak":
        return "excluded"
    if role[0] != "feature":
        return "role:" + str(role[1])
    if any(c == "strong_single_column" and col == column for c, col in record["warnings"]):
        return "warned"
    for code, _, message in record.get("texts", []):
        if code == "near_perfect_headline" and repr(column) in message:
            return "named"
    if any(entry["column"] == column for entry in record.get("rests_on") or []):
        return "listed"
    return "silent"
