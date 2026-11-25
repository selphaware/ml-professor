"""CP9: the summary of gate9_seeds.py outputs.

    python -B gate9_summary.py <out folder> [--per-seed]

Reads seeds_<label>.json for the labels good (the staged reference), before (the
reference as it was), app (the application as it stands) and F23 to F28 (the
broken copies, each on the check that names it). Prints, for each check and
label: on how many seeds it passed, and for each run it reads the range of
headline.skill, the models used, the named errors, and the facts the check
asserts (roles, baselines, warnings, the largest row's share).
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

folder = Path(sys.argv[1])
per_seed = "--per-seed" in sys.argv
OWN = {"time.previous_value_at_repeats": "F23", "leak.copy_excluded_under_time": "F24",
       "leak.excluded_despite_extreme_target": "F25", "result.largest_row_reported": "F26",
       "data.identifier_survives_shuffle": "F27", "groups.choice_not_made": "F28"}
RUN = {"time.previous_value_at_repeats": "walk_repeated",
       "leak.copy_excluded_under_time": "walk_copy",
       "leak.excluded_despite_extreme_target": "walk_copy_cell",
       "result.largest_row_reported": "walk_late_cell",
       "data.identifier_survives_shuffle": "messy_shuffled",
       "groups.choice_not_made": "dominant_entity"}
WATCH = {"walk_repeated": ["level_prev"], "walk_copy": ["level_scaled", "level_prev"],
         "walk_copy_cell": ["level_scaled", "level_prev"], "walk_late_cell": ["level_prev"],
         "messy_shuffled": ["record_id", "visits"], "dominant_entity": []}
CODES = ("extreme_value", "largest_row_dominates", "selection_not_made", "suspected_leak",
         "strong_single_column", "autocorrelated_panel_target", "fold_not_scored")


def load(label):
    path = folder / f"seeds_{label}.json"
    return json.loads(path.read_text(encoding="utf-8"))["records"] if path.is_file() else None


def span(values):
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return "no value"
    m = sum(values) / len(values)
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / max(1, len(values) - 1))
    return (f"{min(values):+.4f} to {max(values):+.4f} (mean {m:+.4f}, sd {sd:.4f}, "
            f"n {len(values)})")


def describe(records, check, run):
    n = len(records)
    passed = sum(r["checks"][check][0] for r in records)
    runs = [r["runs"][run] for r in records]
    done = [x for x in runs if "error" not in x]
    print(f"  passes on {passed} of {n} seeds; the run finishes on {len(done)}")
    errors = Counter(x["error"] for x in runs if "error" in x)
    if errors:
        print(f"  runs that stopped: {dict(errors)}")
    print(f"  skill: {span([x['skill'] for x in done])}")
    print(f"  model: {dict(Counter(x['model'] for x in done))}; baseline: "
          f"{dict(Counter(x['baseline'] for x in done))}; baselines listed: "
          f"{dict(Counter(','.join(sorted(x['baselines'])) for x in done))}")
    for name in WATCH[run]:
        roles = Counter("/".join(str(v) for v in x["roles"].get(name, ["absent"])) for x in done)
        print(f"  {name}: {dict(roles)}")
    for code in CODES:
        seen = Counter(tuple(sorted(str(w[1]) for w in x["warnings"] if w[0] == code))
                       for x in done)
        if set(seen) != {()}:
            print(f"  {code}, columns named -> seeds: "
                  + "; ".join(f"{list(k) or 'none'}: {v}" for k, v in sorted(seen.items())))
    if run == "walk_late_cell":
        for key in ("largest_row_share", "baseline_largest_row_share"):
            print(f"  headline.{key}: {span([x['largest'].get(key) for x in done])}")
            print(f"  cv.{key}: {span([x['cv_largest'].get(key) for x in done])}")
    if run == "dominant_entity":
        folds = Counter(str(sorted(f[0] for f in x["folds"])[0]) + " training rows, scored "
                        + str(sum(1 for f in x["folds"] if f[2])) for x in done)
        print(f"  the smallest fold, and the folds scored: {dict(folds)}")
    reasons = Counter(r["checks"][check][1][:150] for r in records if not r["checks"][check][0])
    for text, count in reasons.most_common(4):
        print(f"  fails ({count}): {text}")
    if per_seed:
        print("  per seed (k: pass, skill, model): " + "; ".join(
            f"{r['k']}: {'P' if r['checks'][check][0] else 'F'} "
            + (x["error"] if "error" in x else f"{x['skill']:+.4f} {x['model']}")
            for r, x in zip(records, runs)))


for check, mode in OWN.items():
    run = RUN[check]
    print(f"\n## {check} (run `{run}`, names {mode})")
    for label in ("good", mode, "before", "app"):
        records = load(label)
        if records is None or check not in records[0]["checks"]:
            print(f" {label}: not measured")
            continue
        print(f" {label}:")
        describe(records, check, run)
