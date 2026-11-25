"""What the two tools find on the application as it stands, each failure with the rule
of pass 8 that owns it: one JSON file a test can read.

    python -B owned.py <battery --json> <generated tables --json> <its kept folder> <out.json>
           [--note TEXT]

    battery.pairs            "run:change" -> the rules its failing claims belong to
    battery.by_rule          rule -> the pairs
    generated_tables.tables  table number -> the rules its failing findings belong to
    generated_tables.by_rule rule -> the table numbers

`none` as a rule means a failure that no rule of pass 8 explains: a finding.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import by_rule_battery as B          # noqa: E402
import by_rule_generated as G        # noqa: E402


def sha(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def battery(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = {}
    for p in doc["pairs"]:
        if p["class"] == "fails":
            pairs[f"{p['run']}:{p['change']}"] = sorted(
                {B.rule_of(p["run"], p["change"], claim, message) for claim, message in p["fails"]})
    return doc, pairs


def generated(path, work):
    doc, work = json.loads(Path(path).read_text(encoding="utf-8")), Path(work)
    tables = {}
    for entry in doc["tables_with_findings"]:
        if entry["class"] != "fails":
            continue
        rules, follows = set(), []
        for f in entry["findings"]:
            if f["class"] not in G.FAILS:
                continue
            rule = "none" if f["class"] in ("raw", "outcome") else G.rule_of(f)
            (follows.append(f) if rule == "follows" else rules.add(rule))
        outs = []
        for side in ("application", "reference"):
            file = work / side / f"{entry['number']:05d}.json"
            outs.append(json.loads(file.read_text(encoding="utf-8")) if file.is_file() else None)
        if all(o and o["kind"] == "result" for o in outs):
            forest, other = G.forests_only(outs[0]["doc"], outs[1]["doc"])
            if forest and not other:
                rules.add("A4 order of rows")
        movers = rules & {"A2/A3b screen and order reading", "A5 identifier", "A4 order of rows"}
        if any(not movers and not (f["field"] == "headline.skill" and "A1 baseline" in rules)
               for f in follows):
            rules.add("none")
        tables[str(entry["number"])] = sorted(rules)
    return doc, tables


def by_rule(owned):
    out = {}
    for name, rules in owned.items():
        for rule in rules:
            out.setdefault(rule, []).append(name)
    return {rule: out[rule] for rule in sorted(out)}


if __name__ == "__main__":
    bat, gen, work, target = sys.argv[1:5]
    note = sys.argv[6] if len(sys.argv) > 6 and sys.argv[5] == "--note" else ""
    bdoc, pairs = battery(bat)
    gdoc, tables = generated(gen, work)
    stage = Path(bdoc["package"]).parent
    document = {
        "what": "the failures of the judge's two other tools on the application as it stands, "
                "each with the rule of pass 8 that owns it (CP9 section A)",
        "note": note,
        "sha256": {"verify/reference/mlprofessor/core.py":
                   sha(stage / "verify" / "reference" / "mlprofessor" / "core.py"),
                   "verify/battery.py": sha(stage / "verify" / "battery.py"),
                   "verify/generated_tables.py": sha(stage / "verify" / "generated_tables.py"),
                   "verify/smoke_test.py": sha(stage / "verify" / "smoke_test.py")},
        "battery": {
            "command": "python verify/battery.py mlprofessor --json FILE",
            "runs": len(bdoc["pairs"]), "counts": bdoc["counts"],
            "stated_limits": bdoc.get("stated_limits"),
            "stated_limit_pairs_that_fail": sorted(
                f"{p['run']}:{p['change']}" for p in bdoc["pairs"] if p["class"] == "stated limit"),
            "pairs": pairs, "by_rule": by_rule(pairs)},
        "generated_tables": {
            "command": "python verify/generated_tables.py mlprofessor --json FILE",
            "seed": gdoc["seed"], "tables_run": gdoc["tables"],
            "counts": gdoc["tables_by_class"],
            "tables": tables, "by_rule": by_rule(tables)},
    }
    Path(target).write_text(json.dumps(document, indent=1), encoding="utf-8")
    for tool in ("battery", "generated_tables"):
        print(tool, document[tool]["counts"],
              {rule: len(names) for rule, names in document[tool]["by_rule"].items()})
