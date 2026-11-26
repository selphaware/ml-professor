"""CP10, item 6: what this pass adds to a run of the full profile at 100,000 rows by 100
columns (the 10-minute budget of D16), on a prototype of the changed reference.

    python -B timing10.py <package folder> [--rows 100000] [--columns 100] [--tasks classification,regression]
                          [--modes new,old] [--n-jobs -1]

The tables are those of docs/measurements/CP3/built_library.py (the same generator and
seed). Each run is a process of its own, one after another; run it with nothing else on
the machine. `old` switches every proposed rule off. Printed for each: seconds inside
profile(), the seconds of the leak screen and of the statement of section A, the
chosen configuration.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(r"E:\CODE\ml-professor")
WORK = Path(os.environ.get("MLPROF_LAB_WORK") or Path(tempfile.gettempdir()) / "mlprof-lab") / "cp10"
OLD = {"MLP_A": "0", "MLP_B": "none", "MLP_C": "old", "MLP_D": "old", "MLP_E": "0"}
TARGETS = {"classification": "outcome", "regression": "amount"}
CHILD = r'''
import json, sys, time
sys.path.insert(0, sys.argv[1])
import mlprofessor
from mlprofessor import _proto as P
started = time.perf_counter()
doc = mlprofessor.profile(sys.argv[2], sys.argv[3], time_col=False, group_col=False,
                          profile=sys.argv[5], n_jobs=int(sys.argv[4])).to_dict()
seconds = time.perf_counter() - started
h = doc["headline"]
falls = sorted(((c["skill_fall"], c["name"]) for c in doc["columns"] if c.get("skill_fall") is not None),
               reverse=True)[:5]
print("@@" + json.dumps({"seconds": seconds, "screen": P.LOG.get("screen_seconds"),
                         "a": P.LOG.get("a_seconds"), "model": h["model"], "skill": h["skill"],
                         "n_jobs": doc["meta"]["n_jobs"], "top": falls,
                         "rows": {r["item"]: r["count"] for r in doc["rows"]}}))
'''


def table_file(n_rows, n_columns, task):
    sys.path.insert(0, str(REPO / "docs" / "measurements" / "CP3"))
    import built_library
    path = WORK / f"{task}_{n_rows}x{n_columns}.csv"
    if not path.is_file():
        WORK.mkdir(parents=True, exist_ok=True)
        built_library.make_table(n_rows, n_columns, task).to_csv(path, index=False,
                                                                 float_format="%.5f")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--columns", type=int, default=100)
    parser.add_argument("--tasks", default="classification,regression")
    parser.add_argument("--modes", default="new,old")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args()
    parent = str(Path(args.package).resolve().parent)
    print(f"cores: {os.cpu_count()}; {args.rows} rows x {args.columns} columns; {args.profile} "
          f"profile; n_jobs {args.n_jobs}", flush=True)
    for task in args.tasks.split(","):
        data = table_file(args.rows, args.columns, task)
        for mode in args.modes.split(","):
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MLP_DEBUG="1")
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH"):
                env.pop(name, None)
            if mode == "old":
                env.update(OLD)
            for pair in args.env:
                name, _, value = pair.partition("=")
                env[name] = value
            proc = subprocess.run([sys.executable, "-B", "-c", CHILD, parent, str(data),
                                   TARGETS[task], str(args.n_jobs), args.profile],
                                  env=env, capture_output=True, text=True)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("@@")), None)
            if line is None:
                print(f"{task} {mode}: FAILED\n{proc.stderr[-1500:]}", flush=True)
                continue
            r = json.loads(line[2:])
            print(f"{task} {mode}: {r['seconds']:.1f} s inside profile(); leak screen "
                  f"{r['screen']:.1f} s; the statement of A "
                  f"{'-' if r['a'] is None else format(r['a'], '.1f') + ' s'}; chosen {r['model']}, "
                  f"skill {r['skill']:.4f}; n_jobs {r['n_jobs']}; training {r['rows']['training']}, "
                  f"holdout {r['rows']['holdout']}; largest falls {r['top'][:3]}", flush=True)
