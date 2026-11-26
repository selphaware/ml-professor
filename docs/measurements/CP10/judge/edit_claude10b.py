"""CLAUDE.md for the ninth pass, part 2: the rows of the small-faults table, and the gate's
timings as measured by the doctor and the self-test of the pass.

    python -B edit_claude10b.py <CLAUDE.md from part 1> <CLAUDE.md out> <self-test minutes>
                                <per-edit s> <full s> <doctor per-edit s> <doctor full s>
"""
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
minutes, per_edit, full, d_edit, d_full = sys.argv[3:8]
t = SRC.read_bytes().decode("utf-8")


def sub(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:100])
    t = t.replace(old, new)


ROWS = [
    ("M52", "the shuffles of the statement are drawn from a stream that depends on the number of cores", "test_rests_on.py::test_the_statement_is_the_same_at_any_n_jobs"),
    ("M53", "a column that reached no model is given a statement as well", "test_rests_on.py::test_every_feature_has_a_statement_and_no_other_column_has_one"),
    ("M54", "the list takes the largest falls whatever their size: no floor", "test_rests_on.py::test_the_list_holds_the_columns_whose_fall_reaches_the_floor"),
    ("M55", "among equal falls the list goes by name, not by file order", "test_rests_on.py::test_equal_falls_go_in_file_order"),
    ("M56", "copies are found by their loaded values: a rescaled copy is shuffled alone", "test_rests_on.py::test_copies_are_shuffled_together_and_share_one_fall"),
    ("M57", "the near-perfect warning is raised in a classification only", "test_near_perfect.py::test_two_columns_that_give_the_target_together_are_named"),
    ("M58", "an unseen level is given the baseline's prediction and scored", "test_screen_levels.py::test_rows_whose_level_the_fold_has_not_seen_are_not_scored"),
    ("M59", "the reading by levels has a value on any number of scored rows", "test_screen_levels.py::test_a_reading_by_levels_on_too_few_rows_of_a_class_has_no_value"),
    ("M60", "\"few\" distinct values are counted over every row split, not the training rows", "test_screen_levels.py::test_few_values_are_counted_over_the_training_rows"),
    ("M61", "the `suspected_leak` message of a column read by its levels does not name `group_col`", "test_screen_levels.py::test_a_coded_copy_of_the_target_is_excluded"),
    ("M62", "the order reading takes one point for each distinct value, the mean target", "test_order_ties.py::test_a_column_whose_means_rise_with_the_target_is_not_named"),
    ("M63", "with too many rows marked, rule 2 leaves its rows unscored but is not read against the fold's median", "test_screen_marked.py::test_wild_target_cells_beyond_the_share_make_no_column_a_leak"),
    ("M64", "only a time-ordered holdout above one half is named, not one of one half", "test_large_holdout.py::test_a_holdout_of_half_the_rows_or_more_is_named"),
    ("M65", "the console summary prints nothing for an empty list", "test_rests_on.py::test_the_summary_prints_one_line_for_the_list"),
    ("M66", "a model that cannot predict a shuffled matrix stops the run", "test_rests_on.py::test_a_model_that_cannot_predict_a_shuffled_matrix_leaves_that_column_unstated"),
    ("M67", "the floor of the list is 0.05 alone, whatever the headline's uncertainty (D80)", "test_rests_on.py::test_the_floor_is_the_half_width_of_the_skill_interval"),
    ("M68", "with no skill interval the floor is 0.05 alone: no stand-in (D80)", "test_rests_on.py::test_where_the_headline_has_no_interval_the_floor_stands_in"),
    ("M69", "a negated copy is not a copy, and is shuffled alone (D80)", "test_rests_on.py::test_copies_are_shuffled_together_and_share_one_fall"),
]
anchor = ("| M51 | the panel warning names the training mean, whatever baseline the run used | "
          "`test_repeated_stamps.py::test_the_panel_warning_names_the_baseline_the_run_used` |\n")
sub(anchor, anchor + "".join(f"| {m} | {what} | `{test}` |\n" for m, what, test in ROWS))
sub("it takes about SELFTEST_MINUTES minutes)", f"it takes about {minutes} minutes)")
sub('''side (D72):** from the command line a per-edit run takes about 15 seconds and
a full one about 21 (three rounds on a quiet machine, 2026-11-25; 14 and 18
before the eighth pass). The loop's `doctor`, which times the gate inside the
self-test, reports 12.8 and 17.4.''', f'''side (D72):** from the command line a per-edit run takes about {per_edit} seconds and
a full one about {full} (on a quiet machine, 2026-11-26, with the six runs of the
ninth pass; 15 and 21 before it). The loop's `doctor`, which times the gate
inside the self-test, reports {d_edit} and {d_full}.''')
DST.write_bytes(t.encode("utf-8"))
print("wrote", DST)
