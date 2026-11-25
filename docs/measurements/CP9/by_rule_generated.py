"""The generated tables on which a package and the reference differ, grouped by the
rule of pass 8 that the build must change (CP9 section A).

    python -B by_rule_generated.py <the tool's --json file> <the kept working folder>

Each failing finding of a table is given a rule by what it is about. A finding that
only follows from another (a score, a chosen configuration, the predictions, a
count of imputed cells) is given no rule of its own: it must be explained by a rule
the table already has, which can move it. Under time order the two documents are
also compared here, record by record, to find the tables where only the forests'
fold scores differ (A4): the tool's own finding on `models` shows the first
difference only. A table with a finding that no rule explains is listed in full.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RULES = ("A1 baseline", "A2/A3b screen and order reading", "A3 statements",
         "A3 warnings", "A4 order of rows", "A5 identifier", "A6 no choice")
FAILS = ("raw", "outcome", "exact", "fit", "property")
FOLLOWS = ("headline.score", "headline.skill", "headline.model", "cv.chosen", "cv.chosen_mean",
           "cv.chosen_std", "predictions.csv (y_pred)", "headline.interval_method",
           "headline.p_value_reason", "headline.interval_reason")


def close(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def rule_of(finding):
    """(rule, or `follows`, or `none`)."""
    field = finding["field"]
    both = f"{finding['application']!r} {finding['reference']!r}"
    if field.startswith("warnings[largest_row_dominates") or field.startswith(
            "warnings[extreme_value"):
        return "A3 warnings"
    if "largest row" in field or "largest_row" in field:
        return "A3 statements"
    if field == "models" and "largest_row" in both:
        return "A3 statements"
    if field.startswith("warnings[selection_not_made") or "selection_not_made" in field:
        return "A6 no choice"
    if "id-like" in field:
        return "A5 identifier"
    if field.startswith("columns[") and "role, reason" in field:
        if "id_like" in both and "suspected_leak" not in both:
            return "A5 identifier"
        if "suspected_leak" in both:
            return "A2/A3b screen and order reading"
        return "none"
    if "exact copy of the target" in field or field.startswith(
            "warnings[strong_single_column") or field.startswith("warnings[suspected_leak"):
        return "A2/A3b screen and order reading"
    if field in ("baselines (names)", "baselines", "headline.baseline",
                 "headline.baseline_score") or "baseline" in field \
            or field.startswith("warnings[no_previous_value") \
            or field.startswith("warnings[autocorrelated_panel_target"):
        return "A1 baseline"
    if field in FOLLOWS or field == "models" or (field.startswith("columns[") and (
            field.endswith(".n_imputed") or field.endswith(".kind"))) \
            or field.startswith("headline.p_value_reason") \
            or field.startswith("headline.interval_reason"):
        return "follows"
    return "none"


def forests_only(a, b):
    """Under time order with the same features on both sides: do the fold scores of the
    forests differ while those of every other family agree?"""
    if a["validation"]["scheme"] != "time_ordered":
        return False, False
    roles_a = [(c["name"], c["role"], c["reason"]) for c in a["columns"]]
    roles_b = [(c["name"], c["role"], c["reason"]) for c in b["columns"]]
    if roles_a != roles_b:
        return False, False
    forest = other = False
    for x, y in zip(a["models"], b["models"]):
        fa, fb = x.get("fold_scores") or [], y.get("fold_scores") or []
        differs = len(fa) != len(fb) or any(not close(p, q) for p, q in zip(fa, fb)) \
            or x.get("status") != y.get("status")
        if differs and x.get("family") == "forest":
            forest = True
        elif differs:
            other = True
    return forest, other


def main(path, work):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    work = Path(work)
    print(f"package {doc['package']}\nreference {doc['reference']}")
    print(f"{doc['tables']} tables of seed {doc['seed']}: {doc['tables_by_class']}; "
          f"{doc['wall_seconds']} s at {doc['jobs']} workers; in the tool {doc['tool_seconds']}")
    print(f"outcomes {doc['outcomes']}")
    print(f"failing findings by class {doc['failing_findings_by_class']}")
    by_rule, sets, fields = defaultdict(list), Counter(), Counter()
    unexplained = []
    failing = [e for e in doc["tables_with_findings"] if e["class"] == "fails"]
    for entry in failing:
        number = entry["number"]
        found = [f for f in entry["findings"] if f["class"] in FAILS]
        rules, follows, none = set(), [], []
        for f in found:
            rule = "none" if f["class"] in ("raw", "outcome") else rule_of(f)
            fields[(rule, f["class"], f["field"].split("[")[0][:70])] += 1
            if rule == "follows":
                follows.append(f)
            elif rule == "none":
                none.append(f)
            else:
                rules.add(rule)
        outs = {}
        for side in ("application", "reference"):
            file = work / side / f"{number:05d}.json"
            outs[side] = json.loads(file.read_text(encoding="utf-8")) if file.is_file() else None
        a, b = outs["application"], outs["reference"]
        if a and b and a["kind"] == "result" and b["kind"] == "result":
            forest, other = forests_only(a["doc"], b["doc"])
            if forest and not other:
                rules.add("A4 order of rows")
        # what can move what follows: another set of features (A2, A5), the order of the
        # rows of a fit (A4); the baseline (A1) moves skill alone
        movers = rules & {"A2/A3b screen and order reading", "A5 identifier", "A4 order of rows"}
        loose = [f for f in follows if not movers and not (
            f["field"] == "headline.skill" and "A1 baseline" in rules)]
        for rule in rules:
            by_rule[rule].append(number)
        sets[" + ".join(sorted(rules)) or "(none)"] += 1
        if none or loose:
            unexplained.append((entry, none + loose))
    print(f"\n{len(failing)} tables fail")
    print("\ntables by rule (a table is counted under every rule one of its findings has):")
    for rule in RULES:
        numbers = by_rule.get(rule, [])
        shown = ", ".join(map(str, numbers)) if len(numbers) <= 60 else \
            ", ".join(map(str, numbers[:20])) + ", ..."
        print(f"  {rule}: {len(numbers)}" + (f": {shown}" if numbers else ""))
    print("\ntables by the set of rules they fail for:")
    for key, count in sorted(sets.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>5}  {key}")
    print("\nfailing findings by rule, class and field:")
    for (rule, kind, field), count in sorted(fields.items()):
        print(f"  {rule} | {kind} | {field}: {count}")
    print(f"\ntables with a finding that no rule explains, in full: {len(unexplained)}")
    for entry, items in unexplained:
        print(f"\n  {entry['description']}")
        print(f"    application: {entry['application']}\n    reference:   {entry['reference']}")
        for f in items:
            print(f"    [{f['class']}] {f['field']}: {str(f['application'])[:300]!r} | "
                  f"{str(f['reference'])[:300]!r}" + (f" ({str(f['note'])[:300]})" if f["note"] else ""))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
