"""The battery's failing pairs on a package, grouped by the rule of pass 8 that the
build must change (CP9 section A).

    python -B by_rule_battery.py <the battery's --json file>

A pair may fail several claims, and so belong to several rules. Each failing claim
is given one rule: by the claim where the claim says it, else by the change of the
pair. A claim that fits no rule is listed in full under `none`.
"""
import json
import re
import sys
from collections import Counter, defaultdict

RULES = ("A1 baseline", "A2/A3b screen and order reading", "A3 statements",
         "A3 extreme value", "A5 identifier", "A6 no choice")
CHANGE_RULE = {}
for rule, names in {
    "A1 baseline": ("row_repeated", "rows_repeated_1pct", "rows_repeated_10pct",
                    "row_repeated_other_target", "stamp_shared", "gap_2", "gap_2_row_repeated"),
    "A2/A3b screen and order reading": (
        "copy_distinct", "copy_two_valued", "copy_exact", "copy_negative_scale",
        "copy_5_cells_differ", "copy_one_cell_999999", "copy_and_target_sentinel",
        "copy_2pct_cells_altered", "copy_curve", "copy_noise_1pct", "copy_noise_5pct"),
    "A3 extreme value": (
        "number_sentinel_-999", "number_sentinel_999999", "target_sentinel_-999",
        "target_sentinel_999999", "target_two_sentinels", "number_two_sentinels",
        "feature_999999_latest_row", "feature_999999_early_row", "target_-999_latest_row",
        "target_-999_early_row", "missing_cell_feature", "measure_sentinel_-999"),
    "A5 identifier": ("rows_shuffled", "id_shuffled_and_repeated", "rows_deleted_shuffled",
                      "id_floats_shuffled", "id_steps_by_7", "id_added_gaps",
                      "sorted_by_measure"),
    "A6 no choice": ("entities_merged_45pct",),
}.items():
    for name in names:
        CHANGE_RULE[name] = rule


RUN_RULE = {"walk_repeated": "A1 baseline", "walk_copy": "A2/A3b screen and order reading",
            "walk_copy_cell": "A2/A3b screen and order reading",
            "walk_late_cell": "A3 extreme value", "messy_shuffled": "A5 identifier",
            "dominant_entity": "A6 no choice"}


def rule_of(run, change, claim, message):
    head = claim.split(":")[0]
    if head == "walk_copy_roles":
        return "A2/A3b screen and order reading"
    if head == "named":
        return "A3 extreme value"
    if head == "u_largest_row":
        return "A3 statements"
    if head == "band_or_named":
        return "A3 extreme value"
    if head == "fold_rule" or claim == "band:dominant_entity":
        return "A6 no choice"
    if head in ("copy_told", "leak_roles", "leak_reg_roles", "near_leak_roles", "leak_status")             or (head == "added" and "target_copy" in message):
        return "A2/A3b screen and order reading"
    if head == "added" and "row_id_gaps" in message:
        return "A5 identifier"
    if head in ("walk_baselines", "panel_baselines", "flat"):
        return "A1 baseline"
    if head == "messy_roles" and re.search(r"record_id|visits", message):
        return "A5 identifier"
    if head == "roles_as_base" and "suspected_leak" in message:
        return "A2/A3b screen and order reading"    # one extreme target cell and the screen
    if run == "messy" and change == "row_repeated":
        return "A5 identifier"
    if head in ("band", "same_as_base", "roles_as_base", "stop", "stopped", "duplicate_rows",
                "time_split", "int_keys", "messy_roles", "messy_coercion", "u_columns"):
        return CHANGE_RULE.get(change) or RUN_RULE.get(run, "none")
    return "none"


def main(path):
    doc = json.loads(open(path, encoding="utf-8").read())
    pairs = doc["pairs"]
    print(f"package {doc['package']}")
    print(f"{len(pairs)} runs; counts {doc['counts']}; wall {doc['wall_seconds']} s at "
          f"{doc['jobs']} side by side, {doc['tool_seconds']} s of tool time")
    failing = [p for p in pairs if p["class"] == "fails"]
    by_rule, sets, claims = defaultdict(list), Counter(), Counter()
    none = []
    for p in failing:
        rules = set()
        for claim, message in p["fails"]:
            rule = rule_of(p["run"], p["change"], claim, message)
            rules.add(rule)
            claims[(rule, claim.split(":")[0])] += 1
            if rule == "none":
                none.append((p, claim, message))
        for rule in rules:
            by_rule[rule].append(f"{p['run']}:{p['change']}")
        sets[" + ".join(sorted(rules))] += 1
    finished = sum(1 for p in pairs if p["observed"].get("rc") == 0)
    print(f"\n{len(failing)} pairs fail and are on neither list ({finished} of the "
          f"{len(pairs)} runs finished)")
    print("\npairs by rule (a pair is counted under every rule one of its failing claims has):")
    for rule in RULES + ("none",):
        names = by_rule.get(rule, [])
        print(f"  {rule}: {len(names)}")
    print("\npairs by the set of rules they fail for:")
    for key, count in sorted(sets.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>5}  {key}")
    print("\nfailing claims by rule and claim:")
    for (rule, claim), count in sorted(claims.items()):
        print(f"  {rule} | {claim}: {count}")
    print("\npairs by rule, in full (the statements of A3 aside: every finished run lacks them):")
    for rule in RULES + ("none",):
        if rule == "A3 statements":
            continue
        names = by_rule.get(rule, [])
        runs = defaultdict(list)
        for name in names:
            run, change = name.split(":", 1)
            runs[change].append(run)
        print(f"\n  {rule}: {len(names)} pairs")
        for change, where in sorted(runs.items()):
            print(f"    {change} ({len(where)}): {', '.join(where)}")
    print(f"\nfailing claims that fit no rule, in full: {len(none)}")
    for p, claim, message in none:
        print(f"  {p['run']}:{p['change']} [{p['kind']}] {claim}: {message[:700]}")
        print(f"      seen: {p['observed']}")
    for label in ("stated limit", "open"):
        names = [f"{p['run']}:{p['change']} ({p['listed']})" for p in pairs if p["class"] == label]
        print(f"\n{label}: {len(names)}: " + "; ".join(names))
    moved = [f"{p['run']}:{p['change']}" for p in pairs if p["class"] == "holds" and p["listed"]]
    print(f"\nlisted and holds here: {len(moved)}: " + "; ".join(moved))


if __name__ == "__main__":
    main(sys.argv[1])
