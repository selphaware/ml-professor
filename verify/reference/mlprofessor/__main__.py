"""python -m mlprofessor profile --data file.csv --target col --out dir"""
import argparse
import sys

from .core import profile, summary
from .errors import ProfileError


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mlprofessor",
                                     description="Measure how predictable a target column is.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("profile", help="profile one table")
    p.add_argument("--data", required=True, help="path to a CSV file")
    p.add_argument("--target", required=True, help="name of the target column")
    p.add_argument("--out", required=True, help="folder for profile.json and predictions.csv")
    p.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    p.add_argument("--time-col", default=None)
    p.add_argument("--gap", type=int, default=0)
    p.add_argument("--profile", default="full", choices=["full", "fast"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--duplicates", default="error", choices=["error", "keep"],
                   help="what to do when holdout rows have copies among the training rows "
                        "under the random scheme: stop (error, the default) or run (keep)")
    args = parser.parse_args(argv)
    try:
        result = profile(args.data, args.target, task=args.task, time_col=args.time_col,
                         gap=args.gap, profile=args.profile, seed=args.seed,
                         out_dir=args.out, n_jobs=args.n_jobs, duplicates=args.duplicates)
    except ProfileError as exc:
        print(f"error: {exc.code}: {exc}", file=sys.stderr)
        return 3
    print(summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
