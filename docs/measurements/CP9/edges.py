"""CP9: where the edges of the two leak cut-offs are on the screen of D74 and D76
(spec 10.2), measured again on the tables of the first pass.

    python -B edges.py <package folder> [--workers 12] [--per-seed]

The tables are those of docs/measurements/CP1/lab.py (`nearleak`), made the same
way: 1,500 rows, the 30 seeds of the gate's `near_leak` (its base seed + 100 k +
the dataset's position, k = 0 to 29), random scheme, fast profile, tool seed 0.

    regression      `signal_reg` with a column `amount_noisy`: the target x 3.7 + 12
                    plus normal noise whose deviation is a share of the target's
                    own, sqrt(2.0**2 + 1.2**2 + 1.0), from
                    random.Random("near_leak.amount_noisy.<seed>")
    classification  `signal_clf` with a column `outcome_noisy`: the target with an
                    exact share of its labels flipped, the rows drawn from
                    random.Random("near_leak.outcome_noisy.<seed>")

For each share, over the 30 seeds: the screen's two readings of the added column
(the one rule 2 reads, with the worst 1% unscored; the one rule 3 reads, with every
row scored: step 8), each as range, mean and sample deviation; and what the run
did with the column: excluded as a suspected leak, named by
`strong_single_column`, or kept in silence. The readings are caught where the
package computes them (`core._screen_column`); the run is not stopped.
"""
import argparse
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab9                                             # noqa: E402

NOISE_SHARES = (0.05, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
FLIP_SHARES = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)
ROWS = 1500
SD_Y = (2.0 ** 2 + 1.2 ** 2 + 1.0) ** 0.5
_seen = {}


def _init(package):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(Path(package).resolve().parent))
    import mlprofessor                                  # noqa: F401
    from mlprofessor import core
    original = core._screen_column

    def wrapper(*args, **kwargs):
        found = original(*args, **kwargs)
        _seen[sys._getframe(1).f_locals.get("name")] = found
        return found

    core._screen_column = wrapper
    lab9.gate()


def table(kind, seed, share):
    g = lab9.gate()
    if kind == "regression":
        header, rows, _ = g.make_signal_reg(seed, n=ROWS)
        stream = random.Random(f"near_leak.amount_noisy.{seed}")
        rows = [row[:-1] + [g._f(3.7 * float(row[-1]) + 12
                                 + stream.gauss(0, share * 3.7 * SD_Y)), row[-1]] for row in rows]
        return header[:-1] + ["amount_noisy", "amount"], rows, "amount_noisy", "amount"
    header, rows, _ = g.make_signal_clf(seed, n=ROWS)
    stream = random.Random(f"near_leak.outcome_noisy.{seed}")
    flips = set(stream.sample(range(ROWS), round(share * ROWS)))
    other = {"yes": "no", "no": "yes"}
    rows = [row[:-1] + [other[row[-1]] if i in flips else row[-1], row[-1]]
            for i, row in enumerate(rows)]
    return header[:-1] + ["outcome_noisy", "outcome"], rows, "outcome_noisy", "outcome"


def _job(job):
    import mlprofessor
    kind, seed, share = job
    g = lab9.gate()
    header, rows, column, target = table(kind, seed, share)
    lines = [",".join(g._quote(str(h)) for h in header)]
    lines += [",".join(g._quote(str(c)) for c in row) for row in rows]
    lab9.WORK.mkdir(parents=True, exist_ok=True)
    path = lab9.WORK / f"edges{os.getpid()}.csv"
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    _seen.clear()
    d = mlprofessor.profile(str(path), target, time_col=False, group_col=False, profile="fast",
                            seed=0, n_jobs=1).to_dict()
    record = next(c for c in d["columns"] if c["name"] == column)
    warned = any(w["code"] == "strong_single_column" and w["column"] == column
                 for w in d["warnings"])
    what = "excluded" if record["reason"] == "suspected_leak" else "warned" if warned else "silent"
    found = _seen.get(column)
    decided, told = found if isinstance(found, tuple) else (found, found)
    return kind, share, seed, decided, told, what


def span(values):
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return f"{min(values):.3f} to {max(values):.3f} ({mean:.3f}, {sd:.3f})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--per-seed", action="store_true")
    args = parser.parse_args()
    g = lab9.gate()
    position = list(g.DATASETS).index("near_leak")
    seeds = [g.SEED + 100 * k + position for k in range(30)]
    jobs = [("regression", seed, share) for share in NOISE_SHARES for seed in seeds]
    jobs += [("classification", seed, share) for share in FLIP_SHARES for seed in seeds]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.package,)) as pool:
        found = list(pool.map(_job, jobs, chunksize=1))
    for kind, shares, word in (("regression", NOISE_SHARES, "noise, share of the target's deviation"),
                               ("classification", FLIP_SHARES, "labels flipped")):
        print(f"\n## {kind}, {ROWS:,} rows, 30 seeds\n")
        print(f"| {word} | the reading rule 2 takes: range (mean, sd) | the reading rule 3 "
              "takes: range (mean, sd) | excluded | warned | kept in silence |")
        print("|---|---|---|---|---|---|")
        for share in shares:
            rows = [r for r in found if r[0] == kind and r[1] == share]
            count = {w: sum(r[5] == w for r in rows) for w in ("excluded", "warned", "silent")}
            print(f"| {100 * share:g}% | {span([r[3] for r in rows])} | "
                  f"{span([r[4] for r in rows])} | {count['excluded']} | {count['warned']} | "
                  f"{count['silent']} |")
        if args.per_seed:
            print("\nseed by seed (k = 0 to 29), the reading rule 3 takes:")
            for share in shares:
                rows = sorted((r for r in found if r[0] == kind and r[1] == share),
                              key=lambda r: r[2])
                print(f"  {100 * share:g}%: " + " ".join(f"{r[4]:.3f}" for r in rows))
