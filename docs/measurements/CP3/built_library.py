"""CP3, part G: one measurement of the built library (D39).

Runs the package at the repo root the way a user would, `python -m mlprofessor
profile ...` in a new process, on synthetic tables written outside the repo.

    python built_library.py time      100,000 rows x 100 columns, full profile, every core:
                                      seconds for classification and for regression
    python built_library.py threads   5,000 rows x 30 columns, full profile, n_jobs 1 and 64,
                                      both tasks: are the two output files identical?

Files go under MLPROF_LAB_WORK, or `mlprof-lab` in the system's temporary
folder. Nothing is written inside the repo.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
WORK = Path(os.environ.get("MLPROF_LAB_WORK") or Path(tempfile.gettempdir()) / "mlprof-lab") / "cp3"
SEED = 3917                      # of both tables
BUDGET_SECONDS = 600             # D16
TARGETS = {"classification": "outcome", "regression": "amount"}


def make_table(n_rows, n_columns, task):
    """A table with a known signal: `n_columns` counts the target.

    One column in ten is text with six levels; the others are numbers, the first
    three of them with 2% of their cells blank. The target depends on four of
    the numeric columns and on the first text column.
    """
    rng = np.random.default_rng([SEED, n_rows, n_columns])
    n_text = (n_columns - 1) // 10
    n_numeric = n_columns - 1 - n_text
    numbers = rng.normal(size=(n_rows, n_numeric))
    frame = pd.DataFrame(numbers, columns=[f"x{i + 1:03d}" for i in range(n_numeric)])
    levels = np.array(list("abcdef"))
    for i in range(n_text):
        frame[f"c{i + 1:02d}"] = levels[rng.integers(0, 6, size=n_rows)]
    effect = dict(zip(levels, [-1.0, -0.4, 0.0, 0.3, 0.7, 1.1]))
    z = (1.2 * numbers[:, 0] - 0.8 * numbers[:, 1] + 0.6 * numbers[:, 2] * numbers[:, 3]
         + frame["c01"].map(effect).to_numpy())
    noise = rng.random(n_rows)
    spread = rng.normal(size=n_rows)
    for name in ("x001", "x005", "x006"):
        frame.loc[rng.random(n_rows) < 0.02, name] = np.nan
    if task == "classification":
        frame["outcome"] = np.where(noise < 1.0 / (1.0 + np.exp(-z)), "yes", "no")
    else:
        frame["amount"] = 2.0 * z + spread
    return frame


def table_file(n_rows, n_columns, task):
    path = WORK / f"{task}_{n_rows}x{n_columns}.csv"
    if not path.is_file():
        WORK.mkdir(parents=True, exist_ok=True)
        make_table(n_rows, n_columns, task).to_csv(path, index=False, float_format="%.5f")
    return path


def run(data, target, out, n_jobs=None):
    """One run of the command line from the repo root. Returns (seconds, profile.json)."""
    cmd = [sys.executable, "-m", "mlprofessor", "profile", "--data", str(data),
           "--target", target, "--out", str(out)]
    if n_jobs is not None:
        cmd += ["--n-jobs", str(n_jobs)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH"):
        env.pop(name, None)             # the tool governs its own threads
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    seconds = time.perf_counter() - started
    if proc.returncode != 0:
        raise SystemExit(f"exit {proc.returncode} from {' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return seconds, json.loads((Path(out) / "profile.json").read_text(encoding="utf-8"))


def cmd_time():
    print(f"cores: {os.cpu_count()}; budget {BUDGET_SECONDS} s (D16); work folder {WORK}")
    for task, target in TARGETS.items():
        data = table_file(100_000, 100, task)
        seconds, doc = run(data, target, WORK / f"time_{task}")
        rows = {r["item"]: r["count"] for r in doc["rows"]}
        h = doc["headline"]
        print(f"{task}: {seconds:.1f} s in all, {doc['meta']['timing']['seconds']:.1f} s inside "
              f"profile(); n_jobs {doc['meta']['n_jobs']}; {doc['input']['n_rows']} rows x "
              f"{doc['input']['n_columns']} columns ({data.stat().st_size / 1e6:.0f} MB); "
              f"training {rows['training']}, holdout {rows['holdout']}; "
              f"{doc['cv']['n_configurations']} configurations, chosen {h['model']}, "
              f"skill {h['skill']:.4f}; "
              f"{'inside' if seconds <= BUDGET_SECONDS else 'OVER'} the budget")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def cmd_threads():
    for task, target in TARGETS.items():
        data = table_file(5_000, 30, task)
        seen = {}
        for n_jobs in (1, 64):
            out = WORK / f"threads_{task}_j{n_jobs}"
            seconds, doc = run(data, target, out, n_jobs=n_jobs)
            recorded = doc["meta"].pop("n_jobs")
            doc["meta"].pop("timing")
            seen[n_jobs] = (doc, _digest(out / "predictions.csv"))
            print(f"{task}, n_jobs {n_jobs}: {seconds:.1f} s; meta.n_jobs {recorded}; chosen "
                  f"{doc['headline']['model']}, skill {doc['headline']['skill']!r}; "
                  f"predictions.csv sha256 {seen[n_jobs][1]}")
        same_json = seen[1][0] == seen[64][0]
        same_csv = seen[1][1] == seen[64][1]
        print(f"{task}: profile.json apart from meta.timing and meta.n_jobs "
              f"{'IDENTICAL' if same_json else 'DIFFERENT'}; predictions.csv "
              f"{'IDENTICAL' if same_csv else 'DIFFERENT'}")
        if not same_json:
            for key in seen[1][0]:
                if seen[1][0][key] != seen[64][0][key]:
                    print(f"   differs in `{key}`")


if __name__ == "__main__":
    commands = {"time": cmd_time, "threads": cmd_threads}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit(__doc__)
    commands[sys.argv[1]]()
