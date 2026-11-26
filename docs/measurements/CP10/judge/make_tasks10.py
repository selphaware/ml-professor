"""Append the ninth pass's tasks, T69 to T76, to plan/tasks.json, in the file's own form
(one key a line, lists of files on one line).

    python -B make_tasks10.py <tasks.json in> <tasks.json out>
"""
import json
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
text = SRC.read_bytes().decode("utf-8")
assert json.loads(text)["tasks"][-1]["id"] == "T68"

ALL = ["mlprofessor/pipeline.py", "mlprofessor/columns.py", "mlprofessor/split.py",
       "mlprofessor/load.py", "mlprofessor/leakage.py", "mlprofessor/preprocess.py",
       "mlprofessor/metrics.py", "mlprofessor/uncertainty.py", "mlprofessor/result.py",
       "mlprofessor/cli.py", "mlprofessor/errors.py", "mlprofessor/stages.py",
       "mlprofessor/models.py"]

TASKS = [
    dict(
        id="T69",
        title="What the headline rests on: a shuffled score, its skill and its fall for every "
              "feature, copies shuffled together, the list with its floor, and the summary line",
        tier="complex",
        why_tier="Decides what the output says the headline rests on. Today a table with two "
                 "columns that give the target only together scores 1.000 with nothing said, and "
                 "the result names no column at all.",
        depends_on=["T68"],
        writes=["mlprofessor/pipeline.py", "mlprofessor/result.py", "mlprofessor/uncertainty.py",
                "mlprofessor/metrics.py", "mlprofessor/models.py", "mlprofessor/preprocess.py",
                "mlprofessor/cli.py", "mlprofessor/stages.py"],
        brief="D77 points 1 and 2, D78 points 1, 3 and 7, D80 points 1, 4 and 5, D81; spec 2.2, "
              "2.5 (`shuffled_score_not_computed`), 2.7 (`column_roles`), step 12a, 4.2, 4.3, 5.2, "
              "5.3 rules 17 to 19, section 7, 11. A NEW STEP AFTER STEP 12; NO NUMBER THAT EXISTS "
              "TODAY CHANGES, NOTHING IS REFITTED AND NOTHING IS CHOSEN BY IT. (1) THE SHUFFLE. Take "
              "the holdout matrix exactly as the chosen configuration was handed it in step 10 "
              "(imputed, scaled, one-hot encoded). For each feature, move that feature's part of "
              "it (its column, or its block of level columns) among the holdout rows by one "
              "permutation of the holdout rows, leave every other part where it is, predict with "
              "the chosen configuration's fitted model, and score with the headline metric "
              "against the holdout targets. Five shuffles, each its own permutation; "
              "`shuffled_score` is the mean of the five scores. The permutations come from a "
              "stream of their own drawn from the seed and the position of the feature (the "
              "first member's position for a set of copies), which no other step draws from: the "
              "files must be the same at any `n_jobs`. Which permutations are drawn is yours "
              "(spec 11). (2) COPIES, AS THE MODELS RECEIVE THEM, ARE SHUFFLED TOGETHER and share "
              "one statement: two numeric features whose columns of the final fit's training and "
              "holdout matrix agree, or one agrees with the other's NEGATION, within 1e-9 in "
              "every row; two categorical features whose blocks of level columns hold the same "
              "columns in any order. Never two features of different kinds. So 2 x f + 3 and -f "
              "are copies of f; a text column with its levels renamed is a copy. Compare the "
              "MATRIX, not the loaded values: a rescaled copy compared by its values is shuffled "
              "alone and shows a quarter of the fall. (3) THE THREE KEYS of every record of "
              "`columns`: `shuffled_score`; `shuffled_skill` = 2 x AUC - 1, or 1 - score / "
              "`headline.baseline_score`; `skill_fall` = `headline.skill` - `shuffled_skill`, "
              "which may be negative. All three null for a column that reached no model. With "
              "`headline.skill` null (D36): `shuffled_score` a number, the other two null. When the "
              "model raises, or predicts a value that is not finite, on a shuffled matrix: the "
              "feature's (and its copies') three keys are null, one `shuffled_score_not_computed` "
              "warning names each such feature, and the run FINISHES. `stages.column_roles` "
              "returns the three as null columns. (4) THE LIST: `headline.rests_on`, records "
              "`{column, skill_fall}`, the features whose fall is at least the floor, at most "
              "five, largest first, ties in file order (by `position`, never by name). THE "
              "FLOOR, `headline.rests_on_floor`: max(0.05, (skill_high - skill_low) / 2); where "
              "the skill has no interval, max(0.05, the half-width of the middle 95% of the skill "
              "over a bootstrap of the holdout rows taken as if independent, with the resamples "
              "of 4.2, on a stream of its own, resamples in which skill cannot be computed left "
              "out; none, and so 0.05, when fewer than half remain); null when skill is null, "
              "and the list then empty. (5) THE SUMMARY LINE: one line that begins `rests on`, "
              "holds the floor as a number, and names each listed column in quotes as `repr` "
              "writes it with its fall as a number; `none` when the list is empty. Printed on "
              "every finished run. (6) COST: at 100,000 rows by 100 columns under the full "
              "profile the statement took 22 seconds on the reference (spec 10.11); predict "
              "forests' shuffles side by side on the cores, the rest in turn. THE TESTS: "
              "`tests/test_rests_on.py`; the gate checks `result.rests_on_recomputed` (which "
              "recomputes the shuffled score of the one feature of `random_walk` from "
              "`predictions.csv` and asserts it within five standard deviations of a shuffle's "
              "mean) and `result.copies_share_a_fall`. Five existing cases were loosened under "
              "D59 because the statement predicts the holdout again with the chosen model "
              "(test plan, \"Ninth pass\", rule 3): they pass before and after.",
        gate_checks=["result.rests_on_recomputed", "result.copies_share_a_fall"],
        tests=["tests/test_rests_on.py"],
    ),
    dict(
        id="T70",
        title="The `near_perfect_headline` warning at a headline skill of 0.98 or more",
        tier="complex",
        why_tier="Decides what the output says a near-perfect headline rests on: a skill of 1.000 "
                 "on two columns that give the target only together is today reported with "
                 "nothing said.",
        depends_on=["T69"],
        writes=["mlprofessor/pipeline.py", "mlprofessor/result.py"],
        brief="D77 point 2, D78 point 1; spec 2.5 (`near_perfect_headline`), step 12a, 4.3, 5.3 "
              "rule 20, section 7, 11. When `headline.skill` is 0.98 OR MORE (at the cut-off, not "
              "only above it), in EITHER TASK and under EVERY SCHEME, exactly one warning with "
              "the code `near_perfect_headline` and a null column. Its message holds the skill "
              "as a number and names each column of `headline.rests_on` (T69) in quotes as "
              "`repr` writes it, with its shuffled skill and its fall, and asks whether each is "
              "known before the outcome; with an empty list it says that no column's fall "
              "reaches the floor. Never when skill is below 0.98 or null. It excludes nothing "
              "and changes no number. A legitimately easy table carries it too: the gate's "
              "`near_leak` and the undeclared `time_blocks` do (spec 11); that is the rule, "
              "not a fault. THE TESTS: `tests/test_near_perfect.py`; the gate check "
              "`rests_on.near_perfect_named` (`pair_together`: `signal_reg` with `pair_a`, the "
              "target plus a noise, and `pair_b`, that noise; skill 1.000, both named).",
        gate_checks=["rests_on.near_perfect_named"],
        tests=["tests/test_near_perfect.py"],
    ),
    dict(
        id="T71",
        title="The leak screen reads a column by its levels: text columns and numeric columns of "
              "few distinct values, unseen levels not scored, and the floor of the reading",
        tier="complex",
        why_tier="Decides which columns reach the models. Today the target cut into 30 bands "
                 "with labels in random order is kept, often in silence, with the headline at "
                 "0.95 to 1.0.",
        depends_on=["T70"],
        writes=["mlprofessor/leakage.py", "mlprofessor/pipeline.py", "mlprofessor/metrics.py"],
        brief="D77 point 3, D78 points 4, 5 and 6, D80 point 3; spec step 8 (the readers; \"A "
              "column of codes is judged like any other\"), 2.5 (`suspected_leak`), 4.2, 5.4, "
              "section 7, 10.11, 11. A NEW READER BESIDE THE TREE, in both tasks. (1) WHICH "
              "COLUMNS: every categorical column, and a numeric column whose training rows of "
              "step 7 in which it is not missing number at least FIVE FOR EACH DISTINCT VALUE "
              "among them (count over the training rows, not over every row). (2) THE READING: "
              "in each fold, each validation row is given the mean target of the fold's "
              "training rows that hold its level (a classification: the share of the positive "
              "class). A level is the stripped text, `(missing)` included, or the numeric value; "
              "a missing numeric cell is no level. A VALIDATION ROW WHOSE LEVEL THE FOLD'S "
              "TRAINING ROWS DO NOT HOLD IS NOT SCORED by this reader (never given the "
              "baseline's prediction and scored). Pool the folds; skill over the rows this reader "
              "scores, against every baseline of the screen, the lower counting; rule 2 leaves "
              "unscored the 1% / 1% of THOSE rows, rule 3 scores all of them. (3) THE FLOOR: the "
              "reading has a value only with at least 20 pooled rows scored and, in a "
              "classification, at least 10 of each class scored or every pooled validation row "
              "of that class; with fewer, or where a baseline's error over its rows is zero, it "
              "is no reader of that column and the others decide. (4) In a classification this "
              "reader takes the place of D74's direct reading of a categorical column (which "
              "gave an unseen level the fold's share); the numeric direct reader (one split, no "
              "minimum leaf) stays. (5) THE MESSAGE: a column read by its levels and excluded by "
              "rule 2 gets a `suspected_leak` message that holds, after the reason, the fixed "
              "sentence of spec 5.4 that names `group_col`. Rule 1, the cut-offs, the tree and "
              "the line are unchanged. EXPECTED OUTCOMES: 16 bands and more of a regression "
              "target, as text or a numeric code, any label order: excluded under the random "
              "and grouped schemes; 6 to 8 bands warned about; 5 sometimes; 4 or fewer neither "
              "(spec 11, D80 point 3: DO NOT add a rule for them); under time order no band "
              "count is excluded (a stated limit). An honest column that explains 98% or more "
              "is excluded like any column. THE TESTS: `tests/test_screen_levels.py`; the gate "
              "check `leak.coded_copy_excluded` (`leak_reg_coded`).",
        gate_checks=["leak.coded_copy_excluded"],
        tests=["tests/test_screen_levels.py"],
    ),
    dict(
        id="T72",
        title="The order reading of the warning tier without the tie fault: rows of "
              "neighbouring distinct values compared in pairs",
        tier="complex",
        why_tier="Decides which columns the output names: today a column of noise with repeated "
                 "values is warned about as one the target follows.",
        depends_on=["T71"],
        writes=["mlprofessor/leakage.py"],
        brief="D77 point 4, D78 point 7; spec step 8 (\"The second reading of the warning tier\"), "
              "2.5 (`strong_single_column`), 4.2, 4.3, section 7, 10.11. THE FAULT: the order "
              "reading sorts rows of equal column value by the target, so inside each run of "
              "equal values the target rises, and noise rounded to one decimal place, or a "
              "whole number from 1 to 40, reads as one the target follows. THE READING NOW: for "
              "each two neighbouring distinct values v < w of the column, compare every training "
              "row holding v with every training row holding w: the share of comparisons in "
              "which the target is higher in the row of w, and the share in which it is lower "
              "(equal targets count in neither). Average each over the neighbouring pairs of "
              "values, each pair counting once; the share against the order is the smaller "
              "average. It has a value only with at least 20 neighbouring pairs of values (21 "
              "distinct values). The second condition (the previous value in the column's "
              "place) is read the same way. Which columns are read, the cut-off of 0.10 and "
              "that the reading only warns are unchanged. ONE POINT PER DISTINCT VALUE (the mean "
              "target of its rows) IS NOT THE RULE: on large tables it warns about an honest "
              "weak column. A column whose values are all distinct reads as before. THE TESTS: "
              "`tests/test_order_ties.py`; the gate check `leak.tied_noise_quiet` "
              "(`noise_tied`).",
        gate_checks=["leak.tied_noise_quiet"],
        tests=["tests/test_order_ties.py"],
    ),
    dict(
        id="T73",
        title="Rule 2 when more training rows are marked than may be set aside: rows still left "
              "unscored, and a reading against the fold's training median",
        tier="complex",
        why_tier="Decides which rows rule 2 scores and which columns reach the models: today a "
                 "copy of a long-tailed target with five cells moved far off is warned about "
                 "and kept.",
        depends_on=["T72"],
        writes=["mlprofessor/leakage.py"],
        brief="D77 point 5, D78 point 7; spec step 8 (\"Rows whose target is wildly off are set "
              "aside\"), section 7, 10.11, 11. When the rows marked as far off (10 median "
              "deviations from the median, or 10 median steps from the previous value) are more "
              "than may be set aside (more than max(1, floor(1% of the training rows))), NONE IS "
              "SET ASIDE, as today, BUT: (1) rule 2 still leaves unscored the 1% of pooled rows "
              "the reader misses by most and the 1% the baseline misses by most (today every row "
              "is scored there); (2) rule 2's reading is also taken against each fold's "
              "TRAINING MEDIAN as one more baseline (every validation row given the median "
              "target of its fold's training rows), the lower skill counting as against every "
              "baseline. Rule 3 and the order reading score every row as before. Where the "
              "marked rows are at most the allowance, nothing changes. WHY THE MEDIAN: without "
              "it, noise columns are excluded beside a target with 1.2% to 1.5% of its cells set "
              "to -999. THE TESTS: `tests/test_screen_marked.py`; the gate check "
              "`leak.changed_cells_excluded` (`skewed_copy`). Three cases of "
              "`tests/test_screen_set_aside.py` that pinned \"every row scored\" were loosened "
              "under D59 (test plan, \"Ninth pass\", rule 3).",
        gate_checks=["leak.changed_cells_excluded"],
        tests=["tests/test_screen_marked.py"],
    ),
    dict(
        id="T74",
        title="Under time order a holdout of half of the rows or more is named: "
              "`holdout_far_above_quarter`",
        tier="routine",
        why_tier="A warning about the split, from counts the run already has; it changes no "
                 "number and no row.",
        depends_on=["T73"],
        writes=["mlprofessor/split.py", "mlprofessor/pipeline.py"],
        brief="D77 point 6, D78 point 7; spec step 7 (\"A time-ordered holdout far above a "
              "quarter is named\"), 2.5, 4.3, 5.3 rule 21, 5.4, section 7. Under time order, when "
              "2 x holdout rows >= training rows + gap rows + holdout rows (the rows split; a "
              "share of ONE HALF OR MORE, not only above it), exactly one warning "
              "`holdout_far_above_quarter` with a null column. Its message holds `H of the N "
              "rows` (H the holdout rows, N the rows split, whole numbers in digits), the share "
              "as a percentage, the rows of the holdout's earliest timestamp, and the fixed "
              "passage of spec 5.4. Never under the random or the grouped scheme. Nothing else "
              "changes. THE TESTS: `tests/test_large_holdout.py`; the gate check "
              "`time.large_holdout_named` (`last_stamp`, 600 of 1,000 rows on the last date).",
        gate_checks=["time.large_holdout_named"],
        tests=["tests/test_large_holdout.py"],
    ),
    dict(
        id="T75",
        title="The README: what the headline rests on, the near-perfect warning, a large "
              "time-ordered holdout, and what an easy table should make a caller check",
        tier="support",
        why_tier="Prose about behaviour T69 to T74 built; the spec says what it must hold.",
        depends_on=["T74"],
        writes=["README.md"],
        brief="D77 point 7 (F), D78, D80, D81; spec 2.2, step 12a, 2.5, 11. Bring README.md up to "
              "the pass: say what `shuffled_score`, `shuffled_skill`, `skill_fall`, "
              "`headline.rests_on` and `headline.rests_on_floor` mean and that the console "
              "summary prints the list; what `near_perfect_headline` means, and that a "
              "near-perfect headline on an easy table should make the caller check that each "
              "named column is known before the outcome; what `holdout_far_above_quarter` "
              "means; that copies of a column, as the models receive them, share one fall, and "
              "that two columns carrying the same information otherwise each show less than "
              "they share; and the new limits of spec 11 (coarse bands of the target are warned "
              "about or only named; under time order a coded copy is neither excluded nor "
              "warned about). UNTIL THIS TASK the README is unchanged and every existing README "
              "test stays green: T69 to T74 do not touch it, and only "
              "`tests/test_readme_pass9.py` is red. Keep the existing README tests green here "
              "too.",
        gate_checks=[],
        tests=["tests/test_readme_pass9.py"],
    ),
    dict(
        id="T76",
        title="The judge's two other tools are clean on the build, with nothing listed as open; "
              "and the guards of the ninth pass",
        tier="complex",
        why_tier="No code is expected if T69 to T74 did their work. But the two tools try over a "
                 "thousand tables no single test was written for, and a failure they find is a "
                 "wrong conclusion or a disagreement with the reference on one of them.",
        depends_on=["T75"],
        writes=ALL,
        brief="D77 points 8 and 9, D78 points 2 and 8, D80 point 2, D81; spec 8.1, 8.2, 11. NO "
              "NEW BEHAVIOUR IS BUILT HERE. (1) `tests/test_judge_tools_pass9.py` asserts that "
              "each of the judge's two other tools exits 0 on the application, from the repo "
              "root: `<interpreter> verify/battery.py mlprofessor` (several minutes; `--only "
              "run:change` runs one pair again; `--list` shows every pair) and `<interpreter> "
              "verify/generated_tables.py mlprofessor` (now 1,000 tables and 200 of the ninth "
              "pass's series, numbered from 1,000,000; `--only i` makes table i again). "
              "`tests/test_judge_tools.py` and `tests/test_judge_tools_pass8.py` accept, until "
              "then, exactly the failures T69 to T74 own (D78 point 8); with this task done they "
              "pass because both tools exit 0. (2) A PAIR THAT PLANTS A CARRIER OF THE TARGET "
              "holds when a planted column is excluded, warned about by name, named by the "
              "near-perfect warning or named in `headline.rests_on`; or when its headline skill "
              "is no more than the unperturbed run's `rests_on_floor` above the unperturbed "
              "run's skill (D81). The battery prints the six counts. THE STATED LIMITS are the "
              "four kinds of D76 and one pair of D36 (`flat_tail:two_columns_together`), fixed "
              "in the judge; NOTHING IS OPEN. DO NOT TRY TO FIX a stated limit, and never add a "
              "pair or a table to a list. FOR THE GENERATED TABLES A DISAGREEMENT IS A FINDING, "
              "NOT A VERDICT: read spec.md to see which side it says is right, and fix the "
              "application only where the spec says it is wrong; if the spec does not settle "
              "it, stop and say so. (3) `tests/test_guard_review8.py` holds a second test, by "
              "another route, for small faults of this pass that one test held.",
        gate_checks=[],
        tests=["tests/test_judge_tools_pass9.py", "tests/test_guard_review8.py"],
    ),
]


def block(task):
    lines = ["    {"]
    keys = ["id", "title", "tier", "why_tier", "depends_on", "writes", "brief", "gate_checks",
            "tests"]
    for i, key in enumerate(keys):
        value = json.dumps(task[key], ensure_ascii=False)
        lines.append(f'      "{key}": {value}' + ("," if i < len(keys) - 1 else ""))
    lines.append("    }")
    return "\n".join(lines)


end = text.rstrip().rfind("]")
head = text[:end].rstrip()
assert head.endswith("}")
out = head + ",\n" + ",\n".join(block(t) for t in TASKS) + "\n  ]\n}\n"
doc = json.loads(out)
assert [t["id"] for t in doc["tasks"]][-9:] == ["T68"] + [t["id"] for t in TASKS]
DST.write_bytes(out.encode("utf-8"))
print("wrote", DST, len(doc["tasks"]), "tasks")
