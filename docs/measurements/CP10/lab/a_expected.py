"""CP10, section A: can the gate check one column's shuffled score for itself?

    python -B a_expected.py <package folder> [--seeds 30]

On a run with ONE feature column, moving that column's holdout values to other holdout
rows moves the predictions with them, so the shuffled score is the score of the
holdout targets against a shuffle of the predictions in predictions.csv. Over all
shuffles its mean and its variance are exact, in plain arithmetic:

    mean      = mean(y^2) - 2 mean(y) mean(p) + mean(p^2)
    variance  = 4 / n^2  x  S_yy x S_pp / (n - 1),   S the sum of squared deviations

and the mean of K shuffles has that variance divided by K. Printed for each run and
seed: z = (the reported shuffled_score - mean) / sqrt(variance / K), for the statement
as proposed, and for three faults: the column not shuffled at all (the headline
score), the shuffle made over the training rows, and over training and holdout rows
together (computed here from the same fitted model's own account: see below).
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

RUNS = ("random_walk", "int_stamps", "panel", "walk_repeated", "walk_copy")


def one(job):
    import mlprofessor
    text, target, call, _ = lab10.table_of(job)
    lab10.WORK.mkdir(parents=True, exist_ok=True)
    path = lab10.WORK / f"e{os.getpid()}.csv"
    path.write_bytes(text.encode("utf-8"))
    result = mlprofessor.profile(str(path), target, n_jobs=1, seed=0, **call)
    doc = result.to_dict()
    feats = [c for c in doc["columns"] if c["role"] == "feature"]
    y = result.predictions["y_true"].to_numpy(dtype=float)
    p = result.predictions["y_pred"].to_numpy(dtype=float)
    n = len(y)
    mean = float(np.mean(y ** 2) - 2 * y.mean() * p.mean() + np.mean(p ** 2))
    var = 4 / n ** 2 * float(((y - y.mean()) ** 2).sum() * ((p - p.mean()) ** 2).sum()) / (n - 1)
    k = int(os.environ.get("MLP_A_K", "5"))
    sd = math.sqrt(var / k)
    return dict(run=job["run"], k=job["k"], features=len(feats), n=n,
                reported=feats[0]["shuffled_score"], mean=mean, sd=sd,
                z=(feats[0]["shuffled_score"] - mean) / sd,
                z_unshuffled=(doc["headline"]["score"] - mean) / sd,
                rel_sd=sd / mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    jobs = [dict(run=run, k=k) for run in RUNS for k in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=10, initializer=lab10._init,
                             initargs=(args.package, {"MLP_DEBUG": "0"})) as pool:
        out = list(pool.map(one, jobs))
    print("| run | features | holdout rows | z of the reported score: range | largest abs z | "
          "sd of the mean of 5 shuffles, as a share of the mean | z of the score with nothing "
          "shuffled |")
    print("|---|---|---|---|---|---|---|")
    for run in RUNS:
        recs = [r for r in out if r["run"] == run]
        zs = [r["z"] for r in recs]
        print(f"| {run} | {sorted({r['features'] for r in recs})} | "
              f"{min(r['n'] for r in recs)} to {max(r['n'] for r in recs)} | "
              f"{min(zs):+.2f} to {max(zs):+.2f} | {max(abs(z) for z in zs):.2f} | "
              f"{min(r['rel_sd'] for r in recs):.3f} to {max(r['rel_sd'] for r in recs):.3f} | "
              f"{min(r['z_unshuffled'] for r in recs):+.0f} to "
              f"{max(r['z_unshuffled'] for r in recs):+.0f} |")
    allz = sorted(abs(r["z"]) for r in out)
    print(f"\nall {len(allz)} runs: abs z at most {allz[-1]:.2f}; above 2 on "
          f"{sum(z > 2 for z in allz)}, above 3 on {sum(z > 3 for z in allz)}")
