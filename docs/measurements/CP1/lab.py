"""Measurements behind the bands and cut-offs of phase 1 (CP1 session; D35).

Run from the repo root with the project interpreter, one process at a time:

    python docs/measurements/CP1/lab.py <command> ...

    headlines <tag> <ref_parent> <gate_file> <out.json> <bases> [runs]
        Run the reference on the gate's datasets and keep each headline.
    screen    <tag> <ref_parent> <gate_file> <out.json> <bases>
        Every single-column skill the leak screen computes (stops before any model).
    nearleak  <ref_parent> <gate_file> <out.json>
        A copy of the target with a share of labels flipped; a rescaled noisy copy.
    scheme    <ref_parent> <gate_file>
        Try to recover the first session's 30 seeds from its recorded range on messy.
    messycols <gate_before> <gate_after>
        Which cells of the datasets differ between two versions of the gate.
    compare   <a.json> <b.json>     two `headlines` files, run by run
    tally     <file.json>           warnings and roles of interest in a `headlines` file
    table     <out.csv>             the per-seed table, from the JSON files in WORK/out

<ref_parent> is the folder holding the `mlprofessor` package to measure and
<gate_file> the gate script to take the datasets from; `-` means the repo's own
(verify/reference and the gate script beside it). <bases> is `gate` (the gate's
own seed) or `k30` (the base seeds SEED + 100*k, k = 0..29). <tag> names the
folder the generated CSV files go to, so two versions of a dataset never mix.

Nothing is written inside the repo: datasets and results go to WORK, which is
the folder named by the environment variable MLPROF_LAB_WORK, or `mlprof-lab`
in the system's temporary folder. See README.md beside this file.
"""
import sys
import os

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import hashlib
import importlib.util
import io
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JUDGE = ROOT / "verify"
GATE_FILE = JUDGE / "smoke_test.py"
REFERENCE_PARENT = JUDGE / "reference"
WORK = Path(os.environ.get("MLPROF_LAB_WORK") or Path(tempfile.gettempdir()) / "mlprof-lab")
OUT = WORK / "out"


def _path(value, default):
    return default if value == "-" else Path(value)


def load_gate(path, alias="gate_mod"):
    """Import one version of the gate script under its own module name."""
    path = _path(path, GATE_FILE)
    if str(JUDGE) not in sys.path:
        sys.path.insert(0, str(JUDGE))          # the gate imports its kit from here
    spec = importlib.util.spec_from_file_location(alias, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def load_ref(ref_parent):
    """Import the `mlprofessor` package found in `ref_parent`, and only that one."""
    ref_parent = _path(ref_parent, REFERENCE_PARENT)
    sys.path.insert(0, str(ref_parent))
    import mlprofessor
    from mlprofessor import core
    assert Path(mlprofessor.__file__).resolve().parent.parent == Path(ref_parent).resolve(), \
        mlprofessor.__file__
    return mlprofessor, core


def table_text(G, header, rows, blank_after=()):
    lines = [",".join(header)]
    for i, row in enumerate(rows):
        lines.append(",".join(G._quote(cell) for cell in row))
        if i in blank_after:
            lines.append("")
    return "\n".join(lines) + "\n"


def write_table(tag, label, text):
    path = WORK / "data" / tag / f"{label}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    if not path.is_file() or path.read_bytes() != data:
        path.write_bytes(data)
    return path


def dataset_file(G, tag, name, seed):
    header, rows, blank_after = G.DATASETS[name][0](seed)
    return write_table(tag, f"{name}-{seed}", table_text(G, header, rows, blank_after))


def bases_of(G, spec):
    if spec == "gate":
        return [G.SEED]
    if spec == "k30":
        return [G.SEED + 100 * k for k in range(30)]
    raise SystemExit(f"unknown bases {spec}")


def kwargs_of(extra):
    kw = {}
    if extra == "api":
        return kw
    it = iter(extra)
    for flag in it:
        value = next(it)
        if flag == "--time-col":
            kw["time_col"] = value
        elif flag == "--gap":
            kw["gap"] = int(value)
        else:
            raise SystemExit(f"unknown flag {flag}")
    return kw


def stats(values):
    values = list(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"n": len(values), "min": min(values), "max": max(values),
            "mean": statistics.fmean(values), "sd": sd}


def show(label, values):
    s = stats(values)
    print(f"{label:<44} n={s['n']:<3} {s['min']:+.4f} .. {s['max']:+.4f}  "
          f"mean {s['mean']:+.4f} sd {s['sd']:.4f}", flush=True)
    return s


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


# ------------------------------------------------------------- headlines --

def profile_once(mlprofessor, path, target, extra, profile):
    try:
        result = mlprofessor.profile(str(path), target, profile=profile, seed=0, n_jobs=1,
                                     **kwargs_of(extra))
    except mlprofessor.errors.ProfileError as exc:
        return {"error": exc.code}
    d = result.to_dict()
    body = {k: d[k] for k in ("rows", "task", "validation", "headline", "baselines", "cv",
                              "models")}
    return {
        "skill": d["headline"]["skill"],
        "headline": d["headline"],
        "body": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
        "predictions": hashlib.sha256(
            result.predictions.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
        "warnings": sorted([w["code"], w["column"] or ""] for w in d["warnings"]),
        "columns": {c["name"]: [c["role"], c["reason"], c["leak_rule"], c["kind"],
                                c["n_missing"], c["n_coerced"], c["n_infinite"], c["n_imputed"]]
                    for c in d["columns"]},
        "limitations": len(d["limitations"]),
    }


def cmd_headlines(tag, ref_parent, gate_file, out, bases, runs=None):
    G = load_gate(gate_file)
    mlprofessor, _ = load_ref(ref_parent)
    names = runs.split(",") if runs else [n for n in G.RUNS if not n.endswith("_j4")]
    results = {}
    for name in names:
        dataset, extra, profile, _ = G.RUNS[name]
        target = G.DATASETS[dataset][1]
        started = time.perf_counter()
        for base in bases_of(G, bases):
            path = dataset_file(G, tag, dataset, G.dataset_seed(dataset, base))
            results[f"{name}@{base}"] = profile_once(mlprofessor, path, target, extra, profile)
        got = [results[f"{name}@{b}"] for b in bases_of(G, bases)]
        if all("skill" in g for g in got):
            show(name, [g["skill"] for g in got])
        else:
            print(f"{name:<44} stops with {sorted({g.get('error') for g in got})}", flush=True)
        print(f"    ({time.perf_counter() - started:.1f}s)", flush=True)
    save(out, results)


# ---------------------------------------------------------------- screen --

class Stop(Exception):
    pass


def instrument(core):
    """Record every single-column skill the screen computes; stop before any model."""
    seen = {}
    original = core._screen_column

    def wrapper(values, numeric, y, folds, classification, seed, prev):
        skill = original(values, numeric, y, folds, classification, seed, prev)
        seen[sys._getframe(1).f_locals.get("name")] = float(skill)
        return skill

    def stop(*args, **kwargs):
        raise Stop()

    core._screen_column = wrapper
    core._families = stop
    return seen


def screen_once(mlprofessor, seen, path, target, extra):
    seen.clear()
    try:
        mlprofessor.profile(str(path), target, profile="fast", seed=0, n_jobs=1,
                            **kwargs_of(extra))
    except Stop:
        return dict(seen)
    except mlprofessor.errors.ProfileError as exc:
        return {"__error__": exc.code}
    raise SystemExit("the run did not stop after the screen")


# Columns planted as leaks or near-leaks; every other column is legitimate.
PLANTED = {"outcome_copy", "settlement_amount", "amount_scaled", "outcome_noisy"}


def cmd_screen(tag, ref_parent, gate_file, out, bases):
    G = load_gate(gate_file)
    mlprofessor, core = load_ref(ref_parent)
    seen = instrument(core)
    configs = {}
    for name, (dataset, extra, profile, _) in G.RUNS.items():
        configs.setdefault((dataset, tuple(extra) if extra != "api" else ()), name)
    results, top = {}, {}
    for (dataset, extra), name in configs.items():
        target = G.DATASETS[dataset][1]
        per_column = {}
        for base in bases_of(G, bases):
            path = dataset_file(G, tag, dataset, G.dataset_seed(dataset, base))
            got = screen_once(mlprofessor, seen, path, target, list(extra))
            results[f"{name}@{base}"] = got
            for column, skill in got.items():
                per_column.setdefault(column, []).append(skill)
        if "__error__" in per_column:
            print(f"{name}: stops with {set(per_column['__error__'])}", flush=True)
            continue
        legit = {c: v for c, v in per_column.items() if c not in PLANTED}
        best = max(legit, key=lambda c: max(legit[c]))
        top[name] = (best, max(legit[best]))
        show(f"{name}: top legit {best}", legit[best])
        show(f"{name}: highest legit column per seed",
             [max(v for c, v in results[f"{name}@{b}"].items() if c not in PLANTED)
              for b in bases_of(G, bases)])
        for column in sorted(PLANTED & set(per_column)):
            show(f"{name}: planted {column}", per_column[column])
    print("\nhighest legit single-column skill overall:",
          max(top.items(), key=lambda kv: kv[1][1]), flush=True)
    save(out, results)


# -------------------------------------------------------------- nearleak --

FLIP_SHARES = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.15)
NOISE_SHARES = (0.05, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)


def near_seeds(G):
    """The seeds of `near_leak` under the 30 bases (it is the gate's 17th dataset)."""
    position = list(G.DATASETS).index("near_leak") if "near_leak" in G.DATASETS \
        else len(G.DATASETS)
    return [G.SEED + 100 * k + position for k in range(30)]


def flipped_copy(G, seed, n, share, exact, time_col=False):
    """signal_clf plus `outcome_noisy`: the target with a share of its labels flipped."""
    header, rows, _ = G.make_signal_clf(seed, n=n)
    stream = random.Random(f"near_leak.outcome_noisy.{seed}")
    if exact:
        flips = set(stream.sample(range(n), round(share * n)))
    else:
        flips = {i for i in range(n) if stream.random() < share}
    other = {"yes": "no", "no": "yes"}
    rows = [([i] if time_col else []) + row[:-1]
            + [other[row[-1]] if i in flips else row[-1], row[-1]] for i, row in enumerate(rows)]
    return (["t"] if time_col else []) + header[:-1] + ["outcome_noisy", "outcome"], rows


def cmd_nearleak(ref_parent, gate_file, out):
    G = load_gate(gate_file)
    mlprofessor, core = load_ref(ref_parent)
    seen = instrument(core)
    seeds, results = near_seeds(G), {}

    def keep(label, values):
        s = show(label, values)
        s.update(at_or_above_098=sum(v >= 0.98 for v in values),
                 below_090=sum(v < 0.90 for v in values), values=values)
        results[label] = s
        print(f"    at or above 0.98: {s['at_or_above_098']} of 30; below 0.90: "
              f"{s['below_090']} of 30", flush=True)

    for n in (1500, 400):
        for exact in (False, True):
            for share in FLIP_SHARES:
                values = []
                for seed in seeds:
                    header, rows = flipped_copy(G, seed, n, share, exact)
                    path = write_table("custom", "near", table_text(G, header, rows))
                    values.append(screen_once(mlprofessor, seen, path, "outcome", [])
                                  ["outcome_noisy"])
                keep(f"flip.random.n{n}.{'exact' if exact else 'chance'}.{share:.3f}", values)
    # regression: the target rescaled, with noise as a share of the target's deviation
    sd_y = (2.0 ** 2 + 1.2 ** 2 + 1.0) ** 0.5
    for n in (1500, 400):
        for share in NOISE_SHARES:
            values = []
            for seed in seeds:
                header, rows, _ = G.make_signal_reg(seed, n=n)
                stream = random.Random(f"near_leak.amount_noisy.{seed}")
                rows = [row[:-1] + [G._f(3.7 * float(row[-1]) + 12
                                         + stream.gauss(0, share * 3.7 * sd_y)), row[-1]]
                        for row in rows]
                header = header[:-1] + ["amount_noisy", "amount"]
                path = write_table("custom", "near_reg", table_text(G, header, rows))
                values.append(screen_once(mlprofessor, seen, path, "amount", [])["amount_noisy"])
            keep(f"regnoise.random.n{n}.{share:.2f}", values)
    save(out, results)


# ---------------------------------------------------------------- scheme --

def cmd_scheme(ref_parent, gate_file):
    """The first session recorded 0.630 to 0.824 (mean 0.731, sd 0.039) on messy and
    not its 30 seeds. None of these schemes reproduces it (run against the gate and
    the reference as they were at commit 070e5b8)."""
    G = load_gate(gate_file)
    mlprofessor, _ = load_ref(ref_parent)
    S = G.SEED
    idx = list(G.DATASETS).index("messy")
    schemes = {
        "builder seed k": [k for k in range(30)],
        "builder seed 1+k": [1 + k for k in range(30)],
        "base k (seed k+idx)": [k + idx for k in range(30)],
        "base 1+k": [1 + k + idx for k in range(30)],
        "base SEED+k": [S + k + idx for k in range(30)],
        "builder SEED+k": [S + k for k in range(30)],
        "base SEED+1+k": [S + 1 + k + idx for k in range(30)],
        "base SEED+100k": [S + 100 * k + idx for k in range(30)],
        "base SEED+1000k": [S + 1000 * k + idx for k in range(30)],
        "base 100k": [100 * k + idx for k in range(30)],
        "base 1000k": [1000 * k + idx for k in range(30)],
        "base 1000+k": [1000 + k + idx for k in range(30)],
        "builder 1000+k": [1000 + k for k in range(30)],
        "base 100+k": [100 + k + idx for k in range(30)],
        "builder 100+k": [100 + k for k in range(30)],
        "base 2000+k": [2000 + k + idx for k in range(30)],
        "base SEED+16k": [S + 16 * k + idx for k in range(30)],
        "base SEED+20k": [S + 20 * k + idx for k in range(30)],
        "base SEED+50k": [S + 50 * k + idx for k in range(30)],
        "base SEED+10k": [S + 10 * k + idx for k in range(30)],
        "base SEED*(k+1)": [S * (k + 1) + idx for k in range(30)],
        "base 42+k": [42 + k + idx for k in range(30)],
    }
    cache = {}

    def skill(seed):
        if seed not in cache:
            path = dataset_file(G, "scheme", "messy", seed)
            cache[seed] = profile_once(mlprofessor, path, "outcome", [], "fast")["skill"]
        return cache[seed]

    for label, seeds in schemes.items():
        s = show(label, [skill(x) for x in seeds])
        if (round(s["min"], 3), round(s["max"], 3), round(s["mean"], 3), round(s["sd"], 3)) \
                == (0.630, 0.824, 0.731, 0.039):
            print("    ^^^ reproduces the recorded range on messy", flush=True)
    for start in list(range(0, 61)) + list(range(S - 20, S + 61)):
        s = stats([skill(start + k) for k in range(30)])
        if (round(s["min"], 3), round(s["max"], 3)) == (0.630, 0.824):
            print(f"window of 30 builder seeds from {start}: min/max match; "
                  f"mean {s['mean']:.4f} sd {s['sd']:.4f}", flush=True)
    print("scheme search done", flush=True)


# ------------------------------------------------------------- messycols --

def cell_state(cell):
    """How a cell of a numeric column reads: missing, a number, or text that is coerced."""
    cell = cell.strip()
    if cell in ("", "n/a"):
        return ("missing",)
    try:
        return ("number", float(cell))
    except ValueError:
        return ("text",)


def cmd_messycols(gate_before, gate_after):
    """Every dataset the two gates share, cell by cell, over the 30 bases for messy."""
    A, B = load_gate(gate_before, "gate_a"), load_gate(gate_after, "gate_b")
    exact = retyped = 0
    names, width = None, 0
    for k in range(30):
        base = A.SEED + 100 * k
        ta = list(csv.reader(io.StringIO(A.dataset_text("messy", base))))
        tb = list(csv.reader(io.StringIO(B.dataset_text("messy", base))))
        assert len(ta) == len(tb), (base, len(ta), len(tb))
        names, width = ta[0], len(ta[0])
        assert tb[0][:width] == names, (tb[0], names)
        for ra, rb in zip(ta[1:], tb[1:]):
            assert bool(ra) == bool(rb), (base, ra, rb)
            for name, a, b in zip(names, ra, rb):
                if a == b:
                    exact += 1
                else:
                    assert cell_state(a) == cell_state(b) == ("text",), (base, name, a, b)
                    retyped += 1
    print(f"messy, 30 bases, the {width} columns of the first gate: {exact} cells identical; "
          f"{retyped} cells differ as text and read the same (text that is coerced to "
          f"missing, before and after)")
    print("columns of the second gate:", tb[0])
    for name in A.DATASETS:
        if name == "messy":
            continue
        assert A.dataset_text(name) == B.dataset_text(name), name
        assert A.dataset_seed(name) == B.dataset_seed(name), name
    print(f"the other {len(A.DATASETS) - 1} datasets of the first gate are byte-identical at "
          "the gate seed, and their seeds are unchanged")
    print("datasets only in the second gate:", [n for n in B.DATASETS if n not in A.DATASETS])


# -------------------------------------------------------- compare, tally --

def cmd_compare(a_file, b_file):
    a = json.loads(Path(a_file).read_text(encoding="utf-8"))
    b = json.loads(Path(b_file).read_text(encoding="utf-8"))
    keys = [k for k in a if k in b]
    differ = []
    for key in keys:
        x, y = a[key], b[key]
        if "error" in x or "error" in y:
            if x != y:
                differ.append((key, "error", x, y))
            continue
        for part in ("skill", "headline", "body", "predictions"):
            if x[part] != y[part]:
                differ.append((key, part))
    print(f"{len(keys)} runs in both files; {len(differ)} difference(s) in skill, headline, "
          "rows, task, validation, baselines, cv, models or predictions")
    for d in differ[:20]:
        print("  DIFF", d)
    only = [k for k in b if k not in a]
    if only:
        print("only in the second file:", sorted({k.split('@')[0] for k in only}))
    changed = {}
    for key in keys:
        if "error" in a[key] or "error" in b[key]:
            continue
        for w in b[key]["warnings"]:
            if w not in a[key]["warnings"]:
                changed.setdefault(("+",) + tuple(w), []).append(key)
        for w in a[key]["warnings"]:
            if w not in b[key]["warnings"]:
                changed.setdefault(("-",) + tuple(w), []).append(key)
    for w, where in sorted(changed.items()):
        print(f"  warning {w}: {len(where)} run(s), e.g. {where[0]}")


CODES = ("numeric_with_separators", "strong_single_column", "interval_approximate",
         "values_coerced", "suspected_leak", "no_interval")
WATCHED = ("revenue", "fee", "spend", "outcome_noisy", "strong_feature", "outcome_copy",
           "settlement_amount", "amount_scaled")


def cmd_tally(file):
    """Per run: how many seeds carry each warning of interest, and the roles of some columns."""
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    runs = {}
    for key, value in data.items():
        runs.setdefault(key.split("@")[0], []).append(value)
    for name, items in runs.items():
        items = [i for i in items if "error" not in i]
        if not items:
            continue
        counts, roles, methods = {}, {}, {}
        for item in items:
            for code, column in {tuple(w) for w in item["warnings"]}:
                if code in CODES:
                    counts[(code, column)] = counts.get((code, column), 0) + 1
            for w in item["warnings"]:
                if w[0] in CODES and item["warnings"].count(w) > 1:
                    counts[("REPEATED " + w[0], w[1])] = counts.get(("REPEATED " + w[0], w[1]), 0) + 1
            for column in WATCHED:
                if column in item["columns"]:
                    role = "/".join(str(x) for x in item["columns"][column][:3])
                    roles.setdefault(column, {})
                    roles[column][role] = roles[column].get(role, 0) + 1
            m = str(item["headline"]["interval_method"])
            methods[m] = methods.get(m, 0) + 1
        print(f"{name} ({len(items)} seeds): limitations "
              f"{sorted({i['limitations'] for i in items})}; interval methods {methods}")
        for (code, column), count in sorted(counts.items()):
            print(f"    warning {code:<24} {column or '(null)':<16} on {count} of {len(items)}")
        for column, found in roles.items():
            print(f"    column {column:<18} {found}")


# ----------------------------------------------------------------- table --

def cmd_table(out):
    """One row per base seed, one column per measured value, from the files in WORK/out."""
    def load(name):
        path = OUT / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    columns = {}                                    # column name -> {k: value}

    def put(column, k, value):
        columns.setdefault(column, {})[k] = value

    base_of = {1115 + 100 * k: k for k in range(30)}
    for key, value in load("head_k30.json").items():
        name, base = key.split("@")
        if "skill" in value:
            put(f"skill.{name}", base_of[int(base)], value["skill"])
    for key, value in load("screen_k30.json").items():
        name, base = key.split("@")
        if "__error__" in value:
            continue
        k = base_of[int(base)]
        legit = {c: v for c, v in value.items() if c not in PLANTED}
        put(f"single_top.{name}", k, max(legit.values()))
        for column in sorted(set(value) & (PLANTED | {"strong_feature"})):
            put(f"single.{name}.{column}", k, value[column])
    for name in ("nearleak.json", "neartime.json", "shares.json"):
        for label, entry in load(name).items():
            for k, value in enumerate(entry["values"]):
                put(label, k, value)
    names = list(columns)
    out = Path(out)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["k", "base_seed"] + names)
        for k in range(30):
            writer.writerow([k, 1115 + 100 * k] + [
                "" if columns[n].get(k) is None else f"{columns[n][k]:.6f}".rstrip("0").rstrip(".")
                for n in names])
    print(f"{out}: 30 rows, {len(names)} measured columns")


if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    {"headlines": cmd_headlines, "screen": cmd_screen, "scheme": cmd_scheme,
     "nearleak": cmd_nearleak, "messycols": cmd_messycols, "compare": cmd_compare,
     "tally": cmd_tally, "table": cmd_table}[command](*args)
