"""CP7, proof 5: does each broken copy fail the test that is named for it?

    python -B named_tests.py <mutation check output> [<folder of kept reports>]

<mutation check output> is what `verify/mutation_check.py` printed. For each
broken copy the test that CLAUDE.md and spec section 8.1 name for it must be
among the cases it fails (for a parametrised test, at least one case of it).
The mutation check prints the first five names of a longer list, so where a
named test is not among those shown the answer is read from the report that a
run with `--only good,<copy> --keep` leaves in its folder, `<copy>.junit.xml`:
the cases that fail on the copy and pass on the good reference.
The copies the gate catches (F1 to F10 and their other forms) have no one test
named for them in the tables, apart from those the sixth pass added; for them
the row shows the count only. Exits 1 when a named test is missing.
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NAMED = {
    "F11": ["test_statements.py::test_nothing_said_about_time_order_stops_whatever_the_table_holds"],
    "F12": ["test_duplicates_features.py::"
            "test_rows_that_agree_in_every_feature_stay_together_whatever_their_target"],
    "F13": ["test_grouped_holdout.py::"
            "test_a_large_entity_is_never_held_out_when_smaller_ones_can_fill_a_quarter"],
    "F14": ["test_date_numbers.py::"
            "test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date"],
    "F7d": ["test_time_order_stated.py::test_a_date_column_and_no_word_about_time_stops_the_run"],
    "M1": ["test_guard_final_fit.py::test_m1_final_fit_statistics_ignore_the_holdout"],
    "M2": ["test_guard_final_fit.py::test_m2_levels_come_from_the_rows_of_each_fit"],
    "M3": ["test_guard_leak_screen.py::"
           "test_m3_single_column_skill_is_fitted_on_fold_training_rows_only",
           "test_guard_leak_screen.py::test_m3_missing_by_class_counts_training_rows_only"],
    "M4": ["test_guard_uncertainty.py::test_m4_block_bootstrap_draws_whole_blocks"],
    "M5": ["test_guard_uncertainty.py::test_m5_ties_count_in_the_p_value"],
    "M6": ["test_non_finite.py::test_a_prediction_that_is_not_finite_counts_as_raising"],
    "M7": ["test_fast_profile_error.py::test_fast_error_stands_when_the_full_profile_would_run",
           "test_time_pipeline.py::test_wide_panel_under_fast_raises"],
    "M8": ["test_guard_review3.py::test_m8_the_interval_is_the_middle_95_percent"],
    "M9": ["test_guard_review3.py::test_m9_the_leak_screen_uses_the_previous_value"],
    "M10": ["test_guard_review3.py::test_m10_the_fast_cap_is_a_sample_from_the_seed"],
    "M11": ["test_guard_review3.py::test_m11_cv_std_is_the_sample_deviation"],
    "M12": ["test_guard_review3.py::test_m12_tied_predictions_in_a_resample"],
    "M13": ["test_guard_review3.py::test_m13_n_imputed_counts_rows_that_reach_a_model"],
    "M14": ["test_read_once.py::"],
    "M15": ["test_groups_random.py::test_no_entity_on_both_sides_or_in_two_folds"],
    "M16": ["test_groups_time.py::"
            "test_the_leak_screen_measures_a_lag_against_the_entity_baseline"],
    "M17": ["test_groups_random.py::test_the_fast_cap_keeps_whole_entities"],
    "M18": ["test_review4_small.py::"
            "test_no_part_of_the_output_can_hold_a_number_that_is_not_finite"],
    "M19": ["test_panel_stop.py::test_a_persistent_panel_with_no_entity_column_stops",
            "test_statements.py::"
            "test_nothing_said_about_entities_stops_whatever_the_table_holds"],
    "M20": ["test_guard_review4.py::"
            "test_m20_infinite_cells_of_the_target_and_the_time_column_are_counted"],
    "M21": ["test_groups_random.py::"
            "test_the_interval_resamples_whole_entities_and_there_is_no_p_value"],
    "M22": ["test_duplicates_group.py::"
            "test_each_set_of_copies_stays_on_one_side_and_in_one_fold"],
    "M23": ["test_guard_review5.py::test_m23_the_panel_warning_starts_just_above_one_half"],
    "M24": ["test_statements.py::test_one_message_asks_for_everything_that_is_missing"],
    "M25": ["test_duplicates_features.py::"
            "test_the_count_is_of_rows_that_agree_in_every_feature"],
    "M26": ["test_grouped_holdout.py::test_what_the_holdout_can_spare_goes_back"],
    "M27": ["test_grouped_holdout.py::test_a_large_unit_kept_in_training_is_named"],
    "M28": ["test_date_numbers.py::test_eight_digit_text_is_accepted_as_the_time_column"],
    "M29": ["test_date_numbers.py::test_the_date_rule_comes_before_the_id_like_rule"],
    "M30": ["test_guard_review5.py::test_m30_a_copy_at_step_8a_still_needs_the_same_target"],
}
LINE = re.compile(r"^(F\d+[a-z]?|M\d+)\s+fails (\d+)(?::\s*(.*?))?(?:\s+\[\d+ s\])?$")
CUT = re.compile(r" \(and \d+ more\)$")
KEPT = Path(sys.argv[2]) if len(sys.argv) > 2 else None


def failing_in(report):
    """The cases a kept report shows as failing, named as the mutation check names them."""
    failed = set()
    for item in ET.parse(report).iter("testcase"):
        if {child.tag for child in item} & {"failure", "error", "skipped"}:
            failed.add(f"{item.get('classname', '').split('.')[-1]}.py::{item.get('name', '?')}")
    return failed


def kept(copy):
    """Every case that fails on `copy` and passes on the good reference, or None."""
    if KEPT is None or not (KEPT / f"{copy}.junit.xml").is_file():
        return None
    return sorted(failing_in(KEPT / f"{copy}.junit.xml") - failing_in(KEPT / "good.junit.xml"))


missing = []
print("| Broken copy | cases failing | the test named for it | among them |")
print("|---|---|---|---|")
for line in Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    found = LINE.match(line.rstrip())
    if not found:
        continue
    copy, count, failing = found.group(1), int(found.group(2)), found.group(3) or ""
    cut = bool(CUT.search(failing))
    cases = [c.strip() for c in CUT.sub("", failing).split(", ") if c.strip()]
    names = NAMED.get(copy)
    if names is None:
        print(f"| {copy} | {count} | (caught by the gate; no one test named) | |")
        continue
    for name in names:
        def hits(listed):
            return sum(1 for c in listed if c.split("[")[0] == name or
                       (name.endswith("::") and c.startswith(name)))

        hit, where = hits(cases), ""
        if not hit and cut:
            full = kept(copy)
            if full is None:
                where = "not among the five shown, and no kept report"
            elif len(full) != count:
                where = f"the kept report has {len(full)} cases, the output {count}"
            else:
                hit, where = hits(full), " (from the kept report)"
        short = name.split("::")[0] + ("::" + name.split("::")[1] if name.split("::")[1] else
                                       " (its tests)")
        answer = f"yes, {hit} case(s){where}" if hit else f"NO {where}".strip()
        print(f"| {copy} | {count} | `{short}` | {answer} |")
        if not hit:
            missing.append((copy, name))
print()
print("every named test fails on its copy" if not missing else f"MISSING: {missing}")
sys.exit(1 if missing else 0)
