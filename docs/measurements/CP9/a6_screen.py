"""CP9 A6: what the leak screen rests on beside a dominant entity.

    python -B a6_screen.py <package folder> [--seeds 30]

The gate's `dominant_entity` shape (one entity of 1,490 rows and 255 of two;
2,000 rows) with 100 columns of noise and a target of noise: nothing predicts
anything. For each rule on the fold that trains on ten rows: the largest
single-column skill among the 100 columns, on how many seeds a column is warned
about (0.90) or excluded (0.98), and the same with a rescaled copy of the target
added (which must be excluded).

    scored      the fold is scored, as today (MLP_A6=old)
    not scored  the fold is not scored (MLP_A6=floor)
"""
import argparse
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
COLUMNS = 100


def table(seed, copy):
    rng = random.Random(seed)
    names = ["C0000"] * 1490
    for e in range(255):
        names += [f"C{e + 1:04d}"] * 2
    rows = []
    for name in names:
        y = rng.gauss(0, 1)
        row = [name] + [f"{rng.gauss(0, 1):.4f}" for _ in range(COLUMNS)] + [f"{y:.4f}"]
        if copy:
            row.append(repr(1.07 * float(f"{y:.4f}") + 3))
        rows.append(row)
    rng.shuffle(rows)
    header = ["customer"] + [f"n{i + 1:03d}" for i in range(COLUMNS)] + ["amount"] \
        + (["target_copy"] if copy else [])
    return "\n".join([",".join(header)] + [",".join(r) for r in rows]) + "\n"


def _init(package, env):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(env)
    os.environ["MLP_DEBUG"] = "1"
    sys.path.insert(0, str(Path(package).resolve().parent))
    import mlprofessor      # noqa: F401


def _job(job):
    import io
    import pandas as pd
    import mlprofessor
    from mlprofessor import _proto as P
    copy, k = job
    P.LOG = {}
    frame = pd.read_csv(io.StringIO(table(1115 + 100 * k, copy)))
    try:
        d = mlprofessor.profile(frame, "amount", time_col=False, group_col="customer",
                                profile="fast", n_jobs=1, seed=0).to_dict()
    except Exception as exc:
        return copy, k, "error:" + getattr(exc, "code", type(exc).__name__), None, 0, 0, None
    screen = P.LOG.get("screen", {})
    noise = [v["alone"] for name, v in screen.items() if name != "target_copy"
             and v.get("alone") is not None]
    roles = {c["name"]: c["reason"] for c in d["columns"]}
    warned = sum(w["code"] == "strong_single_column" and w["column"] != "target_copy"
                 for w in d["warnings"])
    excluded = sum(reason == "suspected_leak" and name != "target_copy"
                   for name, reason in roles.items())
    return (copy, k, d["headline"]["model"], max(noise), warned, excluded,
            roles.get("target_copy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    jobs = [(copy, k) for copy in (False, True) for k in range(args.seeds)]
    print("| the fold that trains on ten rows | copy added | largest skill of a noise column, "
          "over the seeds | seeds with a noise column warned | excluded | the copy excluded | errors |")
    print("|---|---|---|---|---|---|---|")
    for label, env in (("scored", {"MLP_A6": "old"}), ("not scored", {"MLP_A6": "floor"})):
        with ProcessPoolExecutor(max_workers=10, initializer=_init,
                                 initargs=(args.package, env)) as pool:
            found = list(pool.map(_job, jobs))
        for copy in (False, True):
            rows = [f for f in found if f[0] == copy]
            ok = [f for f in rows if f[3] is not None]
            tops = [f[3] for f in ok]
            print(f"| {label} | {'yes' if copy else 'no'} | {min(tops):+.3f} to {max(tops):+.3f} | "
                  f"{sum(f[4] > 0 for f in ok)} of {len(ok)} | {sum(f[5] > 0 for f in ok)} of {len(ok)} | "
                  f"{sum(f[6] == 'suspected_leak' for f in ok) if copy else ''} | "
                  f"{[f[2] for f in rows if f[3] is None] or ''} |")
