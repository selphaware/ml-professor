"""Command line (spec 2.2): a thin wrapper over `profile()`.

Exit codes: 0 finished, 2 unparsed command line (argparse), 3 a named error.
"""
import argparse
import sys

from mlprofessor import errors


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m mlprofessor",
        description="Measure how predictable one target column of one table is.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("profile", help="profile one target column")
    run.add_argument("--data", required=True, metavar="FILE.csv",
                     help="the table, as a CSV file")
    run.add_argument("--target", required=True, metavar="COLUMN",
                     help="the target column")
    run.add_argument("--out", required=True, metavar="FOLDER",
                     help="folder for profile.json and predictions.csv")
    run.add_argument("--task", default="auto",
                     choices=["auto", "classification", "regression"],
                     help="the task (default: auto)")
    run.add_argument("--time-col", default=None, metavar="COLUMN",
                     help="the time column; enables the time-ordered scheme")
    run.add_argument("--gap", type=int, default=0, metavar="N",
                     help="whole timestamps between training and holdout (default: 0)")
    run.add_argument("--profile", default="full", choices=["full", "fast"],
                     help="full or fast (default: full)")
    run.add_argument("--seed", type=int, default=0, metavar="N",
                     help="the random seed (default: 0)")
    run.add_argument("--n-jobs", type=int, default=-1, metavar="N",
                     help="threads; -1 for every core (default: -1)")
    return parser


def main(argv=None):
    """Run the command line and return its exit code."""
    from mlprofessor import profile

    args = _build_parser().parse_args(argv)
    try:
        result = profile(
            args.data, args.target, task=args.task, time_col=args.time_col,
            gap=args.gap, profile=args.profile, seed=args.seed,
            out_dir=args.out, n_jobs=args.n_jobs,
        )
    except errors.ProfileError as exc:
        print(f"error: {exc.code}: {exc}", file=sys.stderr)
        return 3
    print(_summary(result.to_dict(), args.out))
    return 0


def _number(value):
    return "n/a" if value is None else f"{value:.4f}"


def _summary(document, out_dir):
    """A few plain lines: the task, the rows, the headline and the files written."""
    headline = document["headline"]
    rows = {record["item"]: record["count"] for record in document["rows"]}
    lines = [
        f"task: {document['task']['type']} ({document['task']['decided_by']}), "
        f"scheme: {document['validation']['scheme']}",
        f"rows: {rows['loaded']} loaded, {rows['training']} training, "
        f"{rows['holdout']} holdout",
        f"headline: {headline['metric']} {_number(headline['score'])} with "
        f"{headline['model']}; baseline {headline['baseline']} "
        f"{_number(headline['baseline_score'])}; skill {_number(headline['skill'])}",
        f"warnings: {len(document['warnings'])}",
        f"wrote profile.json and predictions.csv to {out_dir}",
    ]
    return "\n".join(lines)
