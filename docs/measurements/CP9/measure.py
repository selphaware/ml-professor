"""CP9 measurements on the prototype of the changed reference: 30 data seeds each.

    python -B measure.py <what> <package folder> <out.json> [--seeds 30] [--workers 28]

    gate_old   every run of the gate, every rule of section A off (the reference as it is)
    gate_new   the same runs with the proposed rules
    a1         repeated rows under time order with no entities; panels under --no-entities
    a2         copies of a regression target under every scheme, every variant of the screen
    a3         one extreme cell: where the row falls, what the result says
    a6         the dominant entity: old folds and new
"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab9                                             # noqa: E402

OLD = {"MLP_A1": "0", "MLP_LINE": "none", "MLP_TRIM": "none", "MLP_A3": "0", "MLP_A5": "old", "MLP_DIRECT": "0", "MLP_A3CV": "0", "MLP_EXTREME": "0", "MLP_CHOSEN": "0", "MLP_ORDER": "0",
       "MLP_A6": "old"}
NEW = {"MLP_VARIANTS": "1"}
COPIES = ["copy_exact", "copy_rescaled", "copy_negative", "copy_tiny_scale", "copy_5_cells_differ",
          "copy_15_cells_differ", "copy_2pct_cells_differ", "copy_sentinel_999999",
          "copy_sentinel_-999", "copy_noise_5pct", "copy_noise_10pct", "copy_noise_20pct",
          "copy_squared", "copy_exp", "target_-999_and_copy"]
# run -> (a numeric feature, extra arguments)
REGRESSIONS = {
    "signal_reg": ("f1", {}), "noise_reg": ("n001", {}), "leak_reg": ("f1", {}),
    "random_walk": ("level_prev", {}), "int_stamps": ("level_prev", {}),
    "flat_tail": ("footfall", {}), "panel": ("level_prev", {}),
    "dominant_entity": ("n1", {}), "quarter_entity": ("n1", {}),
}


def jobs_for(what, seeds):
    g = lab9.gate()
    ks = range(seeds)
    if what in ("gate_old", "gate_new"):
        return [dict(run=run, k=k) for run in g.RUNS
                if run != "api_signal_clf" and run not in lab9.ADDED for k in ks]
    if what == "a1":
        out = []
        for run in ("random_walk", "int_stamps", "flat_tail"):
            for change in (None, "row_repeated", "rows_repeated_1pct", "rows_repeated_10pct"):
                out += [dict(run=run, k=k, change=change) for k in ks]
        for change in (None, "row_repeated", "copy_rescaled"):
            out += [dict(run="panel", k=k, change=change, args={"group_col": False}, tag="no_entities")
                    for k in ks]
        out += [dict(run="time_blocks", k=k, change=change) for k in ks
                for change in (None, "row_repeated")]
        return out
    if what == "a2":
        out = []
        for run in REGRESSIONS:
            out += [dict(run=run, k=k, change=change) for change in [None] + COPIES for k in ks]
        out += [dict(run="panel", k=k, change=change, args={"group_col": False}, tag="no_entities")
                for change in ("copy_rescaled", "copy_5_cells_differ") for k in ks]
        return out
    if what == "a3":
        out = []
        for run, (feature, _) in REGRESSIONS.items():
            changes = [None, f"feature:{feature}:-999", f"feature:{feature}:999999",
                       "target_-999", "target_999999"]
            if run == "panel":
                changes.append("feature:level_prev:empty")
            for change in changes:
                draws = (0,) if change is None else (0, 1, 2)
                out += [dict(run=run, k=k, change=change, draw=d) for k in ks for d in draws]
        for run, feature in (("signal_clf", "f1"), ("messy", "f1"), ("binary_numeric", "f1")):
            for change in (None, f"feature:{feature}:-999", f"feature:{feature}:999999"):
                draws = (0,) if change is None else (0, 1, 2)
                out += [dict(run=run, k=k, change=change, draw=d) for k in ks for d in draws]
        return out
    if what == "a6":
        out = []
        for run in ("dominant_entity", "quarter_entity", "entities", "few_valued"):
            out += [dict(run=run, k=k) for k in ks]
        out += [dict(run="dominant_entity", k=k, change="copy_rescaled") for k in ks]
        return out
    raise SystemExit(f"unknown: {what}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what")
    parser.add_argument("package")
    parser.add_argument("out")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args()
    env = dict(OLD if args.what == "gate_old" else NEW)
    if args.what == "a3":
        env["MLP_A3CV"] = "1"
    for pair in args.env:
        name, _, value = pair.partition("=")
        env[name] = value
    jobs = jobs_for(args.what, args.seeds)
    print(f"{len(jobs)} jobs on {args.package} with {env}", flush=True)
    records = lab9.execute(args.package, jobs, env, workers=args.workers)
    for job, record in zip(jobs, records):
        record["tag"] = job.get("tag")
        record["draw"] = job.get("draw", 0)
    lab9.save(args.out, records)
    print(f"wrote {args.out}: {sum('error' in r for r in records)} of {len(records)} ended in an error")
