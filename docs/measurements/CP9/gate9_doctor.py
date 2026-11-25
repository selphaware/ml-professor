"""CP9: the six new checks, run on outputs of the good reference with one value changed.

    python -B gate9_doctor.py <gate file> <verify/artifacts of a good gate run> <work folder>

Each case copies the output folder of one run, changes one thing in profile.json,
and calls the check on it through the gate's own code. A case holds when the check
fails with a message of its own (never an unexpected error) that contains the
words expected. The unchanged outputs must pass. Nothing is run: no tool process
is started.
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

gate_file, artifacts, work = (Path(a).resolve() for a in sys.argv[1:4])
sys.path.insert(0, str(gate_file.parent))
spec = importlib.util.spec_from_file_location("mlprof_gate", str(gate_file))
g = importlib.util.module_from_spec(spec)
sys.argv = [str(gate_file)]
spec.loader.exec_module(g)
from gatekit import CheckFailed, Context     # noqa: E402

DROP = object()


def warn(doc, code, keep):
    doc["warnings"] = [w for w in doc["warnings"] if w["code"] != code or keep(w)]


def column(doc, name):
    return next(c for c in doc["columns"] if c["name"] == name)


def baseline(doc, name):
    return next(b for b in doc["baselines"] if b["name"] == name)


def set_(path, value):
    def change(doc):
        node = doc
        parts = path.split(".")
        for part in parts[:-1]:
            node = node[int(part)] if isinstance(node, list) else node[part]
        if value is DROP:
            del node[parts[-1]]
        else:
            node[parts[-1]] = value(node[parts[-1]]) if callable(value) else value
    return change


def both(*changes):
    def change(doc):
        for one in changes:
            one(doc)
    return change


def as_feature(name):
    return lambda d: column(d, name).update(role="feature", reason=None, leak_rule=None)


def as_leak(name):
    return lambda d: column(d, name).update(role="excluded", reason="suspected_leak",
                                            leak_rule="predicts_target")


def scaled(name, factor):
    return lambda d: baseline(d, name).update(score=baseline(d, name)["score"] * factor)


def rename_warning(code, to):
    return lambda d: [w.update(column=to) for w in d["warnings"] if w["code"] == code]


A, B, C, D, E, F = ("time.previous_value_at_repeats", "leak.copy_excluded_under_time",
                    "leak.excluded_despite_extreme_target", "result.largest_row_reported",
                    "data.identifier_survives_shuffle", "groups.choice_not_made")
MORE = {"code": "largest_row_dominates", "column": None, "message": "one more"}
AGAIN = {"code": "selection_not_made", "column": None, "message": "again"}
CASES = [
    (A, "the previous value is not listed",
     lambda d: d["baselines"].remove(baseline(d, "previous_value")), "baselines on"),
    (A, "the previous value scores 1% more", scaled("previous_value", 1.01),
     "recomputed by the gate"),
    (A, "the training mean scores 1% less", scaled("training_mean", 0.99),
     "recomputed by the gate"),
    (A, "the headline baseline is the training mean", set_("headline.baseline", "training_mean"),
     "headline baseline"),
    (A, "skill is off by 0.001", set_("headline.skill", lambda v: v + 0.001), "headline.skill"),
    (A, "skill is null", set_("headline.skill", None), "not a finite number"),
    (B, "the copy is a feature", as_feature("level_scaled"), "must be excluded"),
    (B, "the copy is excluded by the other rule",
     lambda d: column(d, "level_scaled").update(leak_rule="missing_by_class"), "must be excluded"),
    (B, "no warning names the copy", lambda d: warn(d, "suspected_leak", lambda w: False),
     "no warning with code `suspected_leak`"),
    (B, "the lag is excluded", as_leak("level_prev"), "must stay a feature"),
    (B, "skill is 0.9", set_("headline.skill", 0.9), "skill on walk_copy"),
    (C, "no warning names the target", lambda d: warn(d, "extreme_value", lambda w: False),
     "`extreme_value` warnings"),
    (C, "the warning names another column", rename_warning("extreme_value", "level_prev"),
     "`extreme_value` warnings"),
    (C, "the copy is a feature", as_feature("level_scaled"), "must be excluded"),
    (C, "the lag is excluded", as_leak("level_prev"), "must stay a feature"),
    (D, "headline.largest_row_share is absent", set_("headline.largest_row_share", DROP),
     "has no key headline.largest_row_share"),
    (D, "a models record has no largest_row", set_("models.1.largest_row", DROP),
     "has no key models["),
    (D, "cv has none of the four keys",
     both(*[set_("cv." + k, DROP) for k in g.LARGEST_ROW_KEYS]), "has no key cv."),
    (D, "the share is of the absolute errors", set_("headline.largest_row_share", 0.9997),
     "of the absolute errors"),
    (D, "the share is null", set_("headline.largest_row_share", None), "not a finite number"),
    (D, "the row is another holdout row", "other_holdout_row", "headline.largest_row on"),
    (D, "the row is a training row", "training_row", "headline.largest_row on"),
    (D, "the row is null", set_("headline.largest_row", None), "expected int"),
    (D, "the baseline's share is halved",
     set_("headline.baseline_largest_row_share", lambda v: v / 2), "baseline_largest_row_share"),
    (D, "the baseline's row is the model's",
     lambda d: d["headline"].update(baseline_largest_row=d["headline"]["largest_row"]),
     "headline.baseline_largest_row on"),
    (D, "cv's share is not the chosen configuration's",
     set_("cv.largest_row_share", lambda v: v * 1.5), "those of the chosen configuration"),
    (D, "a configuration's row is a holdout row", "model_row_in_holdout",
     "over the selection folds"),
    (D, "a configuration's share is null", set_("models.1.largest_row_share", None),
     "over the selection folds"),
    (D, "the baseline's share over the folds is null",
     set_("cv.baseline_largest_row_share", None), "over the selection folds"),
    (D, "the baseline's share over the folds is 1.5",
     set_("cv.baseline_largest_row_share", 1.5), "over the selection folds"),
    (D, "no largest_row_dominates warning",
     lambda d: warn(d, "largest_row_dominates", lambda w: False), "statement(s) above one half"),
    (D, "one largest_row_dominates warning too many", lambda d: d["warnings"].append(dict(MORE)),
     "statement(s) above one half"),
    (D, "the baseline over the folds is above one half and no warning is added",
     set_("cv.baseline_largest_row_share", 0.7), "statement(s) above one half"),
    (D, "no warning names the feature", lambda d: warn(d, "extreme_value", lambda w: False),
     "`extreme_value` warnings"),
    (D, "the headline baseline is chance", set_("headline.baseline", "chance"),
     "headline baseline"),
    (E, "the identifier is a feature",
     lambda d: column(d, "record_id").update(role="feature", reason=None, kind="numeric"),
     "is an identifier"),
    (E, "the measure is excluded as id-like",
     lambda d: column(d, "visits").update(role="excluded", reason="id_like", kind=None),
     "a measure, not an identifier"),
    (E, "the measure is categorical", lambda d: column(d, "visits").update(kind="categorical"),
     "a measure, not an identifier"),
    (E, "skill is 0.3", set_("headline.skill", 0.3), "skill on messy_shuffled"),
    (F, "no selection_not_made warning", lambda d: warn(d, "selection_not_made", lambda w: False),
     "0 warning(s)"),
    (F, "two selection_not_made warnings", lambda d: d["warnings"].append(dict(AGAIN)),
     "2 warning(s)"),
    (F, "the tree is chosen",
     both(set_("cv.chosen", "tree-0"), set_("headline.model", "tree-0")),
     "first configuration that has"),
    (F, "the headline model is not the chosen one", set_("headline.model", "tree-0"),
     "first configuration that has"),
    (F, "a fold is not scored", set_("validation.folds.2.scored", False), "reports scored"),
    (F, "four folds scored", set_("cv.n_scored_folds", 4), "reports scored"),
    (F, "skill is -0.5", set_("headline.skill", -0.5), "skill on dominant_entity"),
    (F, "the first configuration has no mean", set_("models.0.cv_mean", None),
     "first configuration that has"),
]


def special(name, doc, held):
    in_holdout = set(held)
    training = next(i for i in range(10 ** 6) if i not in in_holdout)
    if name == "other_holdout_row":
        doc["headline"]["largest_row"] = next(p for p in held
                                              if p != doc["headline"]["largest_row"])
    elif name == "training_row":
        doc["headline"]["largest_row"] = training
    elif name == "model_row_in_holdout":
        doc["models"][0]["largest_row"] = held[0]
        doc["cv"]["largest_row"] = held[0]


def run_check(check, folder):
    run = g.CHECK_RUNS[check][0]
    dataset = g.RUNS[run][0]
    ctx = Context(Path("mlprofessor"), work)
    ctx.state.update(data=artifacts / "data", runs={run: {
        "name": run, "dataset": dataset, "data": artifacts / "data" / (dataset + ".csv"),
        "out": folder, "rc": 0, "stdout": "", "stderr": "", "cmd": "(doctored outputs)",
        "target": g.DATASETS[dataset][1]}})
    try:
        g.gate._checks[check]["fn"](ctx)
    except CheckFailed as exc:
        return "failed", str(exc)
    except Exception as exc:                        # what gatekit calls an unexpected error
        return "unexpected", type(exc).__name__ + ": " + str(exc)
    return "passed", ""


bad = 0
for check in (A, B, C, D, E, F):
    run = g.CHECK_RUNS[check][0]
    outcome, message = run_check(check, artifacts / "runs" / run)
    print(("ok  " if outcome == "passed" else "BAD ") + check + ": unchanged outputs " + outcome
          + (": " + message[:200] if message else ""))
    bad += outcome != "passed"
for number, (check, label, change, expect) in enumerate(CASES, 1):
    run = g.CHECK_RUNS[check][0]
    folder = work / ("case%02d" % number) / run
    if folder.parent.exists():
        shutil.rmtree(folder.parent)
    shutil.copytree(artifacts / "runs" / run, folder)
    path = folder / "profile.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(change, str):
        with open(folder / "predictions.csv", encoding="utf-8") as fh:
            held = [int(line.split(",")[0]) for line in fh.read().split("\n")[1:] if line]
        special(change, doc, held)
    else:
        change(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")
    outcome, message = run_check(check, folder)
    holds = outcome == "failed" and expect in message
    bad += not holds
    print(("ok  " if holds else "BAD ") + check + " | " + label + " -> " + outcome + ": "
          + message[:150].replace("\n", " "))
print()
print(f"{len(CASES)} changed outputs; problems (with the six unchanged ones): {bad}")
sys.exit(1 if bad else 0)
