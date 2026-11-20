"""CP5, section A (D49, D53): how many holdout rows have a copy among the training rows.

    python -B copies.py measure <data seeds> <out.jsonl>
    python -B copies.py table <out.jsonl>

`measure` runs the reference implementation (verify/reference) with
`duplicates="keep"`, so that no run stops for copies, and writes one line per
run: the count and the share the tool reports (`validation.n_holdout_copies`,
`validation.holdout_copy_share`), and a count of this script's own.

For each data seed (the gate's base seed 1115, then 1215, 1315, ...):
  - every run of the gate, with the gate's own arguments;
  - every gate dataset with a time column, once more with no time column
    declared (the random scheme);
  - duplicated data: the gate's `dup_noise`, and `signal_clf` and `signal_reg`
    with 0.5% to 100% of their rows written a second time.
For each tool seed 0 to (data seeds - 1): the README's example table and the
suite's copy of it, under both profiles.

The count of this script's own is independent of the tool's. A step supplied
through the seam (spec.md 2.6) is handed the feature columns of every fit and
of every row it transforms, indexed by source row. From the largest fit (the
training rows) and the frame whose rows are those of predictions.csv (the
holdout), a holdout row is counted when a training row has the same value in
every one of those columns and the same target cell in the source, compared as
Python tuples, with a missing cell equal to a missing cell and -0.0 to 0.0.

The copies are counted before any model is fitted, so the model families do not
matter: one family that predicts a constant is supplied through the seam, which
makes a run quick. Nothing is written inside the repo.
"""
import importlib.util
import json
import random
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "verify" / "reference"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mlprofessor  # noqa: E402
from mlprofessor import errors, extensions  # noqa: E402

assert Path(mlprofessor.__file__).resolve().parent == REPO / "verify" / "reference" / "mlprofessor"
SHARES = (0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00)
COPY_SHARE, COPY_ROWS = 0.01, 10            # the rule of D53, to say which runs would stop


def gate_module():
    """The gate script as a module, for its datasets and its runs."""
    sys.path.insert(0, str(REPO / "verify"))
    spec = importlib.util.spec_from_file_location("mlprof_gate", str(REPO / "verify" / "smoke_test.py"))
    module = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, [str(REPO / "verify" / "smoke_test.py")]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


class Flat:
    """Predicts one value for every row."""

    def fit(self, X, y):
        self.value = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value)

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


FAMILY = [{"name": "flat", "grid": [{}], "make": lambda task, params, seed, n_jobs: Flat()}]


class Recorder:
    """A step that changes nothing and keeps every frame it is handed."""

    fits, transforms = [], []

    def fit(self, X, y):
        Recorder.fits.append(X)
        return self

    def transform(self, X):
        Recorder.transforms.append(X)
        return X


def _cell(value):
    if isinstance(value, float):
        return None if value != value else value + 0.0      # NaN; and -0.0 becomes 0.0
    return value


def own_count(result, source, target):
    """This script's count of the holdout rows with a copy among the training rows."""
    held = set(result.predictions["row"])
    train = max(Recorder.fits, key=len)
    holdout = next(f for f in Recorder.transforms if set(f.index) == held)
    truth = source[target]

    def rows(frame):
        cells = [[_cell(v) for v in frame[name].tolist()] for name in frame.columns]
        ys = [str(truth.iloc[i]).strip() for i in frame.index]
        return list(zip(*cells, ys)) if cells else [(y,) for y in ys]

    seen = set(rows(train))
    return sum(row in seen for row in rows(holdout))


def measure_one(data, target, profile="fast", seed=0, time_col=None, gap=0):
    """One run. Returns a dict of what was found."""
    Recorder.fits, Recorder.transforms = [], []
    source = pd.read_csv(data, low_memory=False) if isinstance(data, (str, Path)) else data
    try:
        with warnings.catch_warnings(), extensions.use(families=FAMILY, steps=[Recorder]):
            warnings.simplefilter("ignore")
            result = mlprofessor.profile(str(data), target, profile=profile, seed=seed, n_jobs=1,
                                         time_col=time_col, gap=gap, duplicates="keep")
    except errors.ProfileError as exc:
        return {"outcome": exc.code}
    doc = result.to_dict()
    found = {"outcome": "result", "scheme": doc["validation"]["scheme"],
             "holdout": int(len(result.predictions)),
             "copies": doc["validation"]["n_holdout_copies"],
             "share": doc["validation"]["holdout_copy_share"],
             "warned": any(w["code"] == "holdout_copies_kept" for w in doc["warnings"])}
    if doc["validation"]["scheme"] == "random":
        found["own_count"] = own_count(result, source, target)
        found["would_stop"] = bool(found["copies"] >= COPY_ROWS and found["share"] > COPY_SHARE)
    return found


def duplicated(text, share, rng):
    """The CSV with `share` of its data rows written a second time, in shuffled order."""
    lines = text.rstrip("\n").split("\n")
    header, rows = lines[0], [line for line in lines[1:] if line]
    rows = rows + rng.sample(rows, round(share * len(rows)))
    rng.shuffle(rows)
    return header + "\n" + "\n".join(rows) + "\n"


def measure(n_seeds, out):
    gate = gate_module()
    folder = Path(tempfile.mkdtemp(prefix="cp5-copies-"))
    started = time.perf_counter()
    with open(out, "w", encoding="utf-8") as fh:
        def record(**fields):
            fh.write(json.dumps(fields) + "\n")
            fh.flush()

        def args_of(extra):
            found = {}
            for i, arg in enumerate(extra):
                if arg == "--time-col":
                    found["time_col"] = extra[i + 1]
                if arg == "--gap":
                    found["gap"] = int(extra[i + 1])
            return found

        for k in range(n_seeds):
            base = gate.SEED + 100 * k
            paths = {}
            for dataset in gate.DATASETS:
                paths[dataset] = folder / f"{dataset}.csv"
                paths[dataset].write_text(gate.dataset_text(dataset, base), encoding="utf-8",
                                          newline="")
            for run, (dataset, extra, profile, n_jobs) in gate.RUNS.items():
                if extra == "api" or n_jobs != 1:       # the same table and arguments again
                    continue
                target = gate.DATASETS[dataset][1]
                record(kind="gate run", k=k, run=run,
                       **measure_one(paths[dataset], target, profile, 0, **args_of(extra)))
                if "--time-col" in extra and "--gap" not in extra:
                    record(kind="gate dataset, time column not declared", k=k, run=dataset,
                           **measure_one(paths[dataset], target, profile, 0))
            for dataset in ("signal_clf", "signal_reg"):
                text = gate.dataset_text(dataset, base)
                for share in SHARES:
                    path = folder / f"dup_{dataset}.csv"
                    path.write_text(duplicated(text, share, random.Random(f"dup.{base}.{share}")),
                                    encoding="utf-8", newline="")
                    record(kind="duplicated", k=k, run=f"{dataset} + {100 * share:g}% of its rows",
                           **measure_one(path, gate.DATASETS[dataset][1]))
            print(f"data seed {base}: done, {time.perf_counter() - started:.0f} s", flush=True)
        for seed in range(n_seeds):
            for name in ("examples/customers.csv", "tests/fixtures/example.csv"):
                for profile in ("fast", "full"):
                    record(kind="example table", k=seed, run=f"{name} ({profile})",
                           **measure_one(REPO / name, "churned", profile, seed))
    print(f"wrote {out} in {time.perf_counter() - started:.0f} s")


def table(out):
    lines = [json.loads(line) for line in Path(out).read_text(encoding="utf-8").splitlines()]
    groups = {}
    for line in lines:
        groups.setdefault((line["kind"], line["run"]), []).append(line)
    disagree = [l for l in lines if "own_count" in l and l["own_count"] != l["copies"]]
    print(f"{len(lines)} runs; the tool's count and this script's own differ in {len(disagree)}")
    print("| Kind | Run | Runs | Outcome | Holdout rows | Rows with a copy | Share | Would stop |")
    print("|---|---|---|---|---|---|---|---|")
    for (kind, run), found in groups.items():
        outcomes = sorted({f["outcome"] if f["outcome"] != "result" else f["scheme"] for f in found})
        counted = [f for f in found if f.get("copies") is not None]
        if counted:
            holdout = f"{min(f['holdout'] for f in counted)} to {max(f['holdout'] for f in counted)}"
            copies = f"{min(f['copies'] for f in counted)} to {max(f['copies'] for f in counted)}"
            share = (f"{100 * min(f['share'] for f in counted):.2f}% to "
                     f"{100 * max(f['share'] for f in counted):.2f}%")
            stop = f"{sum(f['would_stop'] for f in counted)} of {len(counted)}"
        else:
            holdout = copies = share = stop = "-"
        print(f"| {kind} | {run} | {len(found)} | {', '.join(outcomes)} | {holdout} | {copies} | "
              f"{share} | {stop} |")


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        measure(int(sys.argv[2]), sys.argv[3])
    else:
        table(sys.argv[2])
