"""Command line (spec 2.2): a thin wrapper over `profile()`.

Exit codes: 0 finished, 2 unparsed command line (argparse), 3 a named error.
Each call says whether the rows are ordered in time (--time-col or
--no-time-order) and whether rows belong together (--group-col or
--no-entities); a pair left out passes None, and the run stops as profile() does.
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
    time_order = run.add_mutually_exclusive_group()
    time_order.add_argument("--time-col", default=None, metavar="COLUMN",
                            help="the time column; enables the time-ordered scheme. One "
                                 "of --time-col and --no-time-order is needed")
    time_order.add_argument("--no-time-order", action="store_true",
                            help="state that the order of the rows in time does not "
                                 "matter: split at random even when a column holds dates. "
                                 "One of --time-col and --no-time-order is needed")
    run.add_argument("--gap", type=int, default=0, metavar="N",
                     help="whole timestamps between training and holdout (default: 0)")
    run.add_argument("--profile", default="full", choices=["full", "fast"],
                     help="full or fast (default: full)")
    run.add_argument("--seed", type=int, default=0, metavar="N",
                     help="the random seed (default: 0)")
    run.add_argument("--n-jobs", type=int, default=-1, metavar="N",
                     help="threads; -1 for every core (default: -1)")
    run.add_argument("--duplicates", default="error", choices=["error", "keep", "group"],
                     help="rows that are copies of one another (no time column): stop when "
                          "holdout rows have copies among the training rows (error), go "
                          "ahead with a warning (keep), or keep together the rows that agree "
                          "in every feature column, whatever their targets, so that no "
                          "combination of feature values is on both sides of the split or "
                          "in two folds (group) (default: error)")
    entities = run.add_mutually_exclusive_group()
    entities.add_argument("--group-col", default=None, metavar="COLUMN",
                          help="the entity column: the rows of one entity (a customer, a "
                               "patient, a device) are kept together. One of --group-col "
                               "and --no-entities is needed")
    entities.add_argument("--no-entities", action="store_true",
                          help="state that no rows belong to one entity. One of "
                               "--group-col and --no-entities is needed")
    return parser


def main(argv=None):
    """Run the command line and return its exit code."""
    from mlprofessor import profile

    args = _build_parser().parse_args(argv)
    time_col = False if args.no_time_order else args.time_col
    group_col = False if args.no_entities else args.group_col
    try:
        result = profile(
            args.data, args.target, task=args.task, time_col=time_col,
            gap=args.gap, profile=args.profile, seed=args.seed,
            out_dir=args.out, n_jobs=args.n_jobs, duplicates=args.duplicates,
            group_col=group_col,
        )
    except errors.ProfileError as exc:
        print(f"error: {exc.code}: {exc}", file=sys.stderr)
        return 3
    print(_summary(result.to_dict(), args.out))
    return 0


def _number(value):
    return "n/a" if value is None else f"{value:.6g}"


def _summary(document, out_dir):
    """Plain lines: the task, the rows, the headline, its spread, each warning, the files."""
    headline = document["headline"]
    rows = {record["item"]: record["count"] for record in document["rows"]}
    if headline["skill"] is None:
        skill_text = "none (see skill_undefined)"
    else:
        skill_text = _number(headline["skill"])
    lines = [
        f"task: {document['task']['type']} ({document['task']['decided_by']}), "
        f"scheme: {document['validation']['scheme']}",
        f"rows: {rows['loaded']} loaded, {rows['training']} training, "
        f"{rows['holdout']} holdout",
        f"headline: {headline['metric']} {_number(headline['score'])} with "
        f"{headline['model']}; baseline {headline['baseline']} "
        f"{_number(headline['baseline_score'])}; skill {skill_text}",
    ]
    if headline["interval_method"] is None:
        lines.append(f"interval: none: {headline['interval_reason']}")
    else:
        level = f"{100 * headline['interval_level']:g}%"
        lines.append(
            f"interval ({level}, {headline['interval_method']}): score "
            f"{_number(headline['interval_low'])} to {_number(headline['interval_high'])}")
        if headline["skill_low"] is None:
            lines.append("interval on skill: none (see the warnings)")
        else:
            lines.append(f"interval on skill: {_number(headline['skill_low'])} to "
                         f"{_number(headline['skill_high'])}")
    if headline["p_value"] is None:
        lines.append(f"p-value: none: {headline['p_value_reason']}")
    else:
        lines.append(f"p-value: {_number(headline['p_value'])} "
                     f"({headline['n_permutations']} shuffles)")
    lines.append(f"warnings: {len(document['warnings'])}")
    for warning in document["warnings"]:
        where = f" [{warning['column']}]" if warning["column"] is not None else ""
        message = " ".join(str(warning["message"]).split())
        lines.append(f"  {warning['code']}{where}: {message}")
    lines.append(f"wrote profile.json and predictions.csv to {out_dir}")
    return "\n".join(lines)
