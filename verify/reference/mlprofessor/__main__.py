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
    # D63: one of each pair is needed. Neither group is `required` for argparse: a
    # missing pair is the function's own named error, exit 3, with its message.
    when = p.add_mutually_exclusive_group()
    when.add_argument("--time-col", default=None,
                      help="the time column: the holdout is then the latest rows. This or "
                           "--no-time-order is needed")
    when.add_argument("--no-time-order", action="store_true",
                      help="state that the order of the rows in time does not matter "
                           "(time_col=False): the rows are split at random")
    p.add_argument("--gap", type=int, default=0)
    p.add_argument("--profile", default="full", choices=["full", "fast"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--duplicates", default="error", choices=["error", "keep", "group"],
                   help="what to do when holdout rows have copies among the training rows "
                        "and there is no time order: stop (error, the default), run (keep), "
                        "or keep the rows that agree in every feature column on one side of "
                        "the split (group)")
    who = p.add_mutually_exclusive_group()
    who.add_argument("--group-col", default=None,
                     help="the entity column: the rows of one entity stay on one side of the "
                          "split, or under time order have their own previous value. This or "
                          "--no-entities is needed")
    who.add_argument("--no-entities", action="store_true",
                     help="state that no rows belong together (group_col=False)")
    args = parser.parse_args(argv)
    try:
        result = profile(args.data, args.target, task=args.task,
                         time_col=False if args.no_time_order else args.time_col,
                         gap=args.gap, profile=args.profile, seed=args.seed,
                         out_dir=args.out, n_jobs=args.n_jobs, duplicates=args.duplicates,
                         group_col=False if args.no_entities else args.group_col)
    except ProfileError as exc:
        print(f"error: {exc.code}: {exc}", file=sys.stderr)
        return 3
    print(summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
