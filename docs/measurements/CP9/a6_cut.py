"""CP9 A6: on how few validation rows does the choice of model stop being reliable?

    python -B a6_cut.py <package folder> [--seeds 30]

The gate's `dominant_entity` (one entity and many of two rows, 2,000 rows, noise
throughout) with the number of small entities varied, so that after the holdout
has taken its 500 rows the training rows hold the large entity and 10, 20, 30,
50, 100 or 200 rows of small ones. Data seeds 1115 + 100 k. For each: how often
the unlimited tree is chosen over the linear model (on noise the tree scores
about -1 and the linear model about 0), with every fold scored (`old`) and with
the fold that trains on those few rows not scored (`units`: it trains on fewer
rows than it validates on), to see the other four folds alone.
"""
import argparse
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
LEFT = (10, 20, 30, 50, 100, 200)


def table(seed, left):
    small = (500 + left) // 2
    dominant = 2000 - 2 * small
    rng = random.Random(seed)
    names = ["C0000"] * dominant
    for e in range(small):
        names += [f"C{e + 1:04d}"] * 2
    rows = [[name] + [f"{rng.gauss(0, 1):.4f}" for _ in range(4)] + [f"{rng.gauss(0, 1):.4f}"]
            for name in names]
    rng.shuffle(rows)
    lines = ["customer,n1,n2,n3,n4,amount"] + [",".join(r) for r in rows]
    return "\n".join(lines) + "\n"


def _init(package, env):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(env)
    sys.path.insert(0, str(Path(package).resolve().parent))
    import mlprofessor      # noqa: F401


def _job(job):
    import io
    import pandas as pd
    import mlprofessor
    left, k = job
    frame = pd.read_csv(io.StringIO(table(1115 + 100 * k, left)))
    d = mlprofessor.profile(frame, "amount", time_col=False, group_col="customer",
                            profile="fast", n_jobs=1, seed=0).to_dict()
    folds = d["validation"]["folds"]
    return (left, k, d["headline"]["model"], d["headline"]["skill"],
            sum(f["validation_rows"] for f in folds if f["scored"]),
            min(f["training_rows"] for f in folds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    jobs = [(left, k) for left in LEFT for k in range(args.seeds)]
    print("| small-entity rows left among the training rows | folds | tree chosen, of "
          f"{args.seeds} | skill when the linear model is chosen | when the tree is |")
    print("|---|---|---|---|---|")
    for label, env in (("every fold scored", {"MLP_A6": "old"}),
                       ("the fold that trains on them not scored",
                        {"MLP_A6": "units"})):
        with ProcessPoolExecutor(max_workers=12, initializer=_init,
                                 initargs=(args.package, env)) as pool:
            found = list(pool.map(_job, jobs))
        for left in LEFT:
            rows = [f for f in found if f[0] == left]
            tree = [f[3] for f in rows if f[2].startswith("tree")]
            lin = [f[3] for f in rows if f[2].startswith("linear")]
            print(f"| {left} | {label} | {len(tree)} | "
                  f"{min(lin):+.3f} to {max(lin):+.3f} | "
                  + (f"{min(tree):+.3f} to {max(tree):+.3f}" if tree else "") + " |")
