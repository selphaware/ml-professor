"""Shared helpers of the eighth pass (CP9). Builds on `mlp_helpers`, `cp6_helpers`,
`cp7_helpers` and `cp8_helpers`; holds no test.

Like `cp7_helpers`, nothing here makes a statement the call did not make (D63): the
tests of this pass call through `cp7_helpers.run`, `raised` and `run_cli`, and pass
`time_col` and `group_col` in every call that is meant to finish.

Every rule of this pass that is a function of a few columns is written here a second
time, in plain arithmetic (`previous_by_hand`, `against_order`, `far_outside`,
`largest_row`), so that a test compares the application with it.

Nothing here imports `mlprofessor` at module level.
"""
import math

import numpy as np
import pandas as pd

import cp6_helpers as g
import cp7_helpers as s
import cp8_helpers as e
import mlp_helpers as h

# --------------------------------------------------------- pinned values (4.2, 4.3)

SCREEN_FAR = 10
SCREEN_SET_ASIDE = 0.01
ORDER_AGAINST = 0.10
ORDER_PAIRS = 20
LARGEST_ROW_SHARE = 0.5
EXTREME_GAP = 30
EXTREME_SPREAD = 10
EXTREME_OUTERMOST = 3
SELECTION_FLOOR = 50

HEADLINE_LARGEST = ("largest_row_share", "largest_row",
                    "baseline_largest_row_share", "baseline_largest_row")
MODEL_LARGEST = HEADLINE_LARGEST[:2]

# The fixed passage of spec 5.4 that the panel warning holds (D74 point 1).
PANEL_BASELINE = (
    "With no entity column the previous value of a row is the mean target of the "
    "timestamp before, over every series at once: if the rows are several series, "
    "declare group_col, and each series is measured against its own previous value."
)


def panel_passage(baseline):
    """The whole passage, with the baseline's name in place of NAME."""
    return f"Skill is measured against the baseline {baseline}. " + PANEL_BASELINE


def holds_share(text, share):
    """Some number written in `text` is within 0.1 of 100 x `share`."""
    return s.holds_number(str(text), 100.0 * float(share))


def holds_row(text, row):
    """`row` is among the whole numbers written in `text`."""
    return int(row) in e.whole_numbers_in(text)


# ---------------------------------------------- what pass 8 owns (D74 point 7)
# The pairs of the battery and the tables of the generated-table test that fail on the
# application as the seventh pass left it, as the lead's runs wrote them in
# docs/measurements/CP9/owned_by_pass8.json (keys of `battery.pairs` and of
# `generated_tables.tables`). Not shortened, not added to.

BATTERY_OWNED = frozenset({
    'binary_numeric:base', 'binary_numeric:blank_line_added',
    'binary_numeric:constant_added', 'binary_numeric:copy_distinct',
    'binary_numeric:copy_two_valued', 'binary_numeric:feature_renamed',
    'binary_numeric:feature_renamed_awkward', 'binary_numeric:float_whole_numbers',
    'binary_numeric:id_added_gaps', 'binary_numeric:id_added_int',
    'binary_numeric:id_added_text', 'binary_numeric:missing_cell_feature',
    'binary_numeric:missing_cell_target', 'binary_numeric:noise_added',
    'binary_numeric:number_sentinel_-999', 'binary_numeric:number_sentinel_1e308',
    'binary_numeric:number_sentinel_999999', 'binary_numeric:number_two_sentinels',
    'binary_numeric:row_repeated', 'binary_numeric:rows_shuffled',
    'binary_numeric:second_date_added', 'binary_numeric:target_first',
    'binary_numeric:target_renamed', 'binary_numeric:target_respelt',
    'binary_numeric:target_respelt_flipped', 'binary_numeric:white_space_cells',
    'day_numbers:base', 'day_numbers:blank_line_added',
    'day_numbers:constant_added', 'day_numbers:copy_distinct',
    'day_numbers:copy_two_valued', 'day_numbers:feature_renamed',
    'day_numbers:feature_renamed_awkward', 'day_numbers:float_whole_numbers',
    'day_numbers:id_added_gaps', 'day_numbers:id_added_int',
    'day_numbers:id_added_text', 'day_numbers:missing_cell_date',
    'day_numbers:missing_cell_feature', 'day_numbers:missing_cell_target',
    'day_numbers:noise_added', 'day_numbers:number_sentinel_-999',
    'day_numbers:number_sentinel_1e308', 'day_numbers:number_sentinel_999999',
    'day_numbers:number_two_sentinels', 'day_numbers:role_column_renamed',
    'day_numbers:row_repeated', 'day_numbers:rows_shuffled',
    'day_numbers:second_date_added', 'day_numbers:target_first',
    'day_numbers:target_renamed', 'day_numbers:target_respelt',
    'day_numbers:target_respelt_flipped', 'day_numbers:white_space_cells',
    'day_numbers:ymd_sentinel_0', 'day_numbers:ymd_sentinel_20240230',
    'day_numbers:ymd_sentinel_99991231', 'day_numbers:ymd_sentinel_in_dropped_row',
    'day_sentinel:base', 'day_sentinel:blank_line_added',
    'day_sentinel:constant_added', 'day_sentinel:copy_distinct',
    'day_sentinel:copy_two_valued', 'day_sentinel:feature_renamed',
    'day_sentinel:feature_renamed_awkward', 'day_sentinel:float_whole_numbers',
    'day_sentinel:id_added_gaps', 'day_sentinel:id_added_int',
    'day_sentinel:id_added_text', 'day_sentinel:missing_cell_date',
    'day_sentinel:missing_cell_feature', 'day_sentinel:missing_cell_target',
    'day_sentinel:noise_added', 'day_sentinel:number_sentinel_-999',
    'day_sentinel:number_sentinel_1e308', 'day_sentinel:number_sentinel_999999',
    'day_sentinel:number_two_sentinels', 'day_sentinel:role_column_renamed',
    'day_sentinel:row_repeated', 'day_sentinel:rows_shuffled',
    'day_sentinel:second_date_added', 'day_sentinel:target_first',
    'day_sentinel:target_renamed', 'day_sentinel:target_respelt',
    'day_sentinel:target_respelt_flipped', 'day_sentinel:white_space_cells',
    'day_sentinel:ymd_sentinel_0', 'day_sentinel:ymd_sentinel_20240230',
    'day_sentinel:ymd_sentinel_99991231', 'day_sentinel:ymd_sentinel_in_dropped_row',
    'dominant_entity:base', 'dominant_entity:blank_line_added',
    'dominant_entity:constant_added', 'dominant_entity:copy_5_cells_differ',
    'dominant_entity:copy_and_target_sentinel', 'dominant_entity:copy_distinct',
    'dominant_entity:copy_exact', 'dominant_entity:copy_negative_scale',
    'dominant_entity:copy_one_cell_999999', 'dominant_entity:feature_renamed',
    'dominant_entity:feature_renamed_awkward', 'dominant_entity:id_added_gaps',
    'dominant_entity:id_added_int', 'dominant_entity:id_added_text',
    'dominant_entity:missing_cell_feature', 'dominant_entity:missing_cell_target',
    'dominant_entity:noise_added', 'dominant_entity:number_sentinel_-999',
    'dominant_entity:number_sentinel_1e308', 'dominant_entity:number_sentinel_999999',
    'dominant_entity:number_two_sentinels', 'dominant_entity:role_column_renamed',
    'dominant_entity:row_repeated', 'dominant_entity:rows_shuffled',
    'dominant_entity:second_date_added', 'dominant_entity:target_first',
    'dominant_entity:target_renamed', 'dominant_entity:target_sentinel_-999',
    'dominant_entity:target_sentinel_999999', 'dominant_entity:target_two_sentinels',
    'dominant_entity:white_space_cells', 'dup_noise:duplicates_group',
    'dup_noise:duplicates_keep', 'dup_noise:id_added_gaps',
    'entities:base', 'entities:blank_line_added',
    'entities:constant_added', 'entities:copy_distinct',
    'entities:copy_two_valued', 'entities:entities_merged_45pct',
    'entities:feature_renamed', 'entities:feature_renamed_awkward',
    'entities:id_added_gaps', 'entities:id_added_int',
    'entities:id_added_text', 'entities:missing_cell_feature',
    'entities:missing_cell_target', 'entities:noise_added',
    'entities:number_sentinel_-999', 'entities:number_sentinel_1e308',
    'entities:number_sentinel_999999', 'entities:number_two_sentinels',
    'entities:role_column_renamed', 'entities:row_repeated',
    'entities:rows_shuffled', 'entities:second_date_added',
    'entities:target_first', 'entities:target_renamed',
    'entities:target_respelt', 'entities:target_respelt_flipped',
    'entities:white_space_cells', 'few_valued:base',
    'few_valued:blank_line_added', 'few_valued:constant_added',
    'few_valued:duplicates_keep', 'few_valued:feature_renamed',
    'few_valued:feature_renamed_awkward', 'few_valued:id_added_gaps',
    'few_valued:id_added_int', 'few_valued:id_added_text',
    'few_valued:missing_cell_feature', 'few_valued:missing_cell_target',
    'few_valued:noise_added', 'few_valued:row_repeated',
    'few_valued:rows_shuffled', 'few_valued:second_date_added',
    'few_valued:target_first', 'few_valued:target_renamed',
    'few_valued:target_respelt', 'few_valued:target_respelt_flipped',
    'few_valued:white_space_cells', 'few_valued_leak:duplicates_keep',
    'few_valued_leak:id_added_gaps', 'flat_tail:base',
    'flat_tail:blank_line_added', 'flat_tail:constant_added',
    'flat_tail:copy_2pct_cells_altered', 'flat_tail:copy_5_cells_differ',
    'flat_tail:copy_and_target_sentinel', 'flat_tail:copy_curve',
    'flat_tail:copy_distinct', 'flat_tail:copy_exact',
    'flat_tail:copy_negative_scale', 'flat_tail:copy_noise_1pct',
    'flat_tail:copy_one_cell_999999', 'flat_tail:date_sentinel_1900',
    'flat_tail:feature_999999_early_row', 'flat_tail:feature_999999_latest_row',
    'flat_tail:feature_renamed', 'flat_tail:feature_renamed_awkward',
    'flat_tail:gap_2', 'flat_tail:gap_2_row_repeated',
    'flat_tail:id_added_gaps', 'flat_tail:id_added_int',
    'flat_tail:id_added_text', 'flat_tail:missing_cell_date',
    'flat_tail:missing_cell_feature', 'flat_tail:missing_cell_target',
    'flat_tail:noise_added', 'flat_tail:number_sentinel_-999',
    'flat_tail:number_sentinel_1e308', 'flat_tail:number_sentinel_999999',
    'flat_tail:number_two_sentinels', 'flat_tail:role_column_renamed',
    'flat_tail:row_repeated', 'flat_tail:row_repeated_other_target',
    'flat_tail:rows_repeated_10pct', 'flat_tail:rows_repeated_1pct',
    'flat_tail:rows_shuffled', 'flat_tail:second_date_added',
    'flat_tail:stamp_shared', 'flat_tail:target_-999_early_row',
    'flat_tail:target_-999_latest_row', 'flat_tail:target_first',
    'flat_tail:target_renamed', 'flat_tail:target_sentinel_-999',
    'flat_tail:target_sentinel_999999', 'flat_tail:target_two_sentinels',
    'imbalanced:base', 'imbalanced:blank_line_added',
    'imbalanced:constant_added', 'imbalanced:copy_distinct',
    'imbalanced:copy_two_valued', 'imbalanced:feature_renamed',
    'imbalanced:feature_renamed_awkward', 'imbalanced:id_added_gaps',
    'imbalanced:id_added_int', 'imbalanced:id_added_text',
    'imbalanced:missing_cell_feature', 'imbalanced:missing_cell_target',
    'imbalanced:noise_added', 'imbalanced:number_sentinel_-999',
    'imbalanced:number_sentinel_1e308', 'imbalanced:number_sentinel_999999',
    'imbalanced:number_two_sentinels', 'imbalanced:row_repeated',
    'imbalanced:rows_shuffled', 'imbalanced:second_date_added',
    'imbalanced:target_first', 'imbalanced:target_renamed',
    'imbalanced:target_respelt', 'imbalanced:target_respelt_flipped',
    'imbalanced:white_space_cells', 'int_stamps:base',
    'int_stamps:blank_line_added', 'int_stamps:constant_added',
    'int_stamps:copy_2pct_cells_altered', 'int_stamps:copy_5_cells_differ',
    'int_stamps:copy_and_target_sentinel', 'int_stamps:copy_curve',
    'int_stamps:copy_distinct', 'int_stamps:copy_exact',
    'int_stamps:copy_negative_scale', 'int_stamps:copy_noise_1pct',
    'int_stamps:copy_one_cell_999999', 'int_stamps:feature_999999_early_row',
    'int_stamps:feature_999999_latest_row', 'int_stamps:feature_renamed',
    'int_stamps:feature_renamed_awkward', 'int_stamps:gap_2',
    'int_stamps:gap_2_row_repeated', 'int_stamps:id_added_gaps',
    'int_stamps:id_added_int', 'int_stamps:id_added_text',
    'int_stamps:int_time_sentinel_0', 'int_stamps:int_time_sentinel_99991231',
    'int_stamps:missing_cell_feature', 'int_stamps:missing_cell_target',
    'int_stamps:noise_added', 'int_stamps:number_sentinel_-999',
    'int_stamps:number_sentinel_1e308', 'int_stamps:number_sentinel_999999',
    'int_stamps:number_two_sentinels', 'int_stamps:role_column_renamed',
    'int_stamps:row_repeated', 'int_stamps:row_repeated_other_target',
    'int_stamps:rows_repeated_10pct', 'int_stamps:rows_repeated_1pct',
    'int_stamps:rows_shuffled', 'int_stamps:second_date_added',
    'int_stamps:stamp_shared', 'int_stamps:target_-999_early_row',
    'int_stamps:target_-999_latest_row', 'int_stamps:target_first',
    'int_stamps:target_renamed', 'int_stamps:target_sentinel_-999',
    'int_stamps:target_sentinel_999999', 'int_stamps:target_two_sentinels',
    'leak:base', 'leak:blank_line_added',
    'leak:constant_added', 'leak:copy_distinct',
    'leak:copy_two_valued', 'leak:feature_renamed',
    'leak:feature_renamed_awkward', 'leak:id_added_gaps',
    'leak:id_added_int', 'leak:id_added_text',
    'leak:missing_cell_feature', 'leak:missing_cell_target',
    'leak:noise_added', 'leak:number_sentinel_-999',
    'leak:number_sentinel_1e308', 'leak:number_sentinel_999999',
    'leak:number_two_sentinels', 'leak:row_repeated',
    'leak:rows_shuffled', 'leak:second_date_added',
    'leak:target_first', 'leak:target_renamed',
    'leak:target_respelt', 'leak:target_respelt_flipped',
    'leak:white_space_cells', 'leak_reg:base',
    'leak_reg:blank_line_added', 'leak_reg:constant_added',
    'leak_reg:copy_5_cells_differ', 'leak_reg:copy_and_target_sentinel',
    'leak_reg:copy_distinct', 'leak_reg:copy_exact',
    'leak_reg:copy_negative_scale', 'leak_reg:copy_one_cell_999999',
    'leak_reg:feature_renamed', 'leak_reg:feature_renamed_awkward',
    'leak_reg:id_added_gaps', 'leak_reg:id_added_int',
    'leak_reg:id_added_text', 'leak_reg:missing_cell_feature',
    'leak_reg:missing_cell_target', 'leak_reg:noise_added',
    'leak_reg:number_sentinel_-999', 'leak_reg:number_sentinel_1e308',
    'leak_reg:number_sentinel_999999', 'leak_reg:number_two_sentinels',
    'leak_reg:row_repeated', 'leak_reg:rows_shuffled',
    'leak_reg:second_date_added', 'leak_reg:target_first',
    'leak_reg:target_renamed', 'leak_reg:target_sentinel_-999',
    'leak_reg:target_sentinel_999999', 'leak_reg:target_two_sentinels',
    'leak_reg:white_space_cells', 'messy:base',
    'messy:blank_line_added', 'messy:constant_added',
    'messy:copy_distinct', 'messy:copy_two_valued',
    'messy:date_sentinel_1900', 'messy:date_sentinel_9999',
    'messy:date_sentinel_zero', 'messy:feature_renamed',
    'messy:feature_renamed_awkward', 'messy:float_whole_numbers',
    'messy:id_added_gaps', 'messy:id_added_int',
    'messy:id_added_text', 'messy:id_floats_shuffled',
    'messy:id_shuffled_and_repeated', 'messy:id_steps_by_7',
    'messy:measure_sentinel_-999', 'messy:missing_cell_date',
    'messy:missing_cell_feature', 'messy:missing_cell_target',
    'messy:noise_added', 'messy:number_sentinel_-999',
    'messy:number_sentinel_1e308', 'messy:number_sentinel_999999',
    'messy:number_two_sentinels', 'messy:role_column_renamed',
    'messy:row_repeated', 'messy:rows_deleted_shuffled',
    'messy:rows_shuffled', 'messy:second_date_added',
    'messy:target_first', 'messy:target_renamed',
    'messy:target_respelt', 'messy:target_respelt_flipped',
    'messy:white_space_cells', 'messy_shuffled:base',
    'messy_shuffled:blank_line_added', 'messy_shuffled:constant_added',
    'messy_shuffled:copy_distinct', 'messy_shuffled:copy_two_valued',
    'messy_shuffled:date_sentinel_1900', 'messy_shuffled:date_sentinel_9999',
    'messy_shuffled:date_sentinel_zero', 'messy_shuffled:feature_renamed',
    'messy_shuffled:feature_renamed_awkward', 'messy_shuffled:id_added_gaps',
    'messy_shuffled:id_added_int', 'messy_shuffled:id_added_text',
    'messy_shuffled:missing_cell_date', 'messy_shuffled:missing_cell_feature',
    'messy_shuffled:missing_cell_target', 'messy_shuffled:noise_added',
    'messy_shuffled:number_sentinel_-999', 'messy_shuffled:number_sentinel_1e308',
    'messy_shuffled:number_sentinel_999999', 'messy_shuffled:number_two_sentinels',
    'messy_shuffled:role_column_renamed', 'messy_shuffled:row_repeated',
    'messy_shuffled:rows_shuffled', 'messy_shuffled:second_date_added',
    'messy_shuffled:target_first', 'messy_shuffled:target_renamed',
    'messy_shuffled:target_respelt', 'messy_shuffled:target_respelt_flipped',
    'messy_shuffled:white_space_cells', 'near_leak:base',
    'near_leak:blank_line_added', 'near_leak:constant_added',
    'near_leak:copy_distinct', 'near_leak:copy_two_valued',
    'near_leak:feature_renamed', 'near_leak:feature_renamed_awkward',
    'near_leak:id_added_gaps', 'near_leak:id_added_int',
    'near_leak:id_added_text', 'near_leak:missing_cell_feature',
    'near_leak:missing_cell_target', 'near_leak:noise_added',
    'near_leak:number_sentinel_-999', 'near_leak:number_sentinel_1e308',
    'near_leak:number_sentinel_999999', 'near_leak:number_two_sentinels',
    'near_leak:row_repeated', 'near_leak:rows_shuffled',
    'near_leak:second_date_added', 'near_leak:target_first',
    'near_leak:target_renamed', 'near_leak:target_respelt',
    'near_leak:target_respelt_flipped', 'near_leak:white_space_cells',
    'noise_clf:base', 'noise_clf:blank_line_added',
    'noise_clf:constant_added', 'noise_clf:copy_distinct',
    'noise_clf:copy_two_valued', 'noise_clf:feature_renamed',
    'noise_clf:feature_renamed_awkward', 'noise_clf:id_added_gaps',
    'noise_clf:id_added_int', 'noise_clf:id_added_text',
    'noise_clf:missing_cell_feature', 'noise_clf:missing_cell_target',
    'noise_clf:noise_added', 'noise_clf:number_sentinel_-999',
    'noise_clf:number_sentinel_1e308', 'noise_clf:number_sentinel_999999',
    'noise_clf:number_two_sentinels', 'noise_clf:row_repeated',
    'noise_clf:rows_shuffled', 'noise_clf:second_date_added',
    'noise_clf:target_first', 'noise_clf:target_renamed',
    'noise_clf:target_respelt', 'noise_clf:target_respelt_flipped',
    'noise_clf:white_space_cells', 'noise_reg:base',
    'noise_reg:blank_line_added', 'noise_reg:constant_added',
    'noise_reg:copy_5_cells_differ', 'noise_reg:copy_and_target_sentinel',
    'noise_reg:copy_distinct', 'noise_reg:copy_exact',
    'noise_reg:copy_negative_scale', 'noise_reg:copy_one_cell_999999',
    'noise_reg:feature_renamed', 'noise_reg:feature_renamed_awkward',
    'noise_reg:id_added_gaps', 'noise_reg:id_added_int',
    'noise_reg:id_added_text', 'noise_reg:missing_cell_feature',
    'noise_reg:missing_cell_target', 'noise_reg:noise_added',
    'noise_reg:number_sentinel_-999', 'noise_reg:number_sentinel_1e308',
    'noise_reg:number_sentinel_999999', 'noise_reg:number_two_sentinels',
    'noise_reg:row_repeated', 'noise_reg:rows_shuffled',
    'noise_reg:second_date_added', 'noise_reg:target_first',
    'noise_reg:target_renamed', 'noise_reg:target_sentinel_-999',
    'noise_reg:target_sentinel_999999', 'noise_reg:target_two_sentinels',
    'panel:base', 'panel:blank_line_added',
    'panel:constant_added', 'panel:copy_2pct_cells_altered',
    'panel:copy_5_cells_differ', 'panel:copy_and_target_sentinel',
    'panel:copy_curve', 'panel:copy_distinct',
    'panel:copy_exact', 'panel:copy_negative_scale',
    'panel:copy_noise_1pct', 'panel:copy_one_cell_999999',
    'panel:date_sentinel_1900', 'panel:feature_999999_early_row',
    'panel:feature_999999_latest_row', 'panel:feature_renamed',
    'panel:feature_renamed_awkward', 'panel:id_added_gaps',
    'panel:id_added_int', 'panel:id_added_text',
    'panel:missing_cell_date', 'panel:missing_cell_feature',
    'panel:missing_cell_target', 'panel:noise_added',
    'panel:number_sentinel_-999', 'panel:number_sentinel_1e308',
    'panel:number_sentinel_999999', 'panel:number_two_sentinels',
    'panel:role_column_renamed', 'panel:rows_shuffled',
    'panel:second_date_added', 'panel:target_-999_early_row',
    'panel:target_-999_latest_row', 'panel:target_first',
    'panel:target_renamed', 'panel:target_sentinel_-999',
    'panel:target_sentinel_999999', 'panel:target_two_sentinels',
    'panel:white_space_cells', 'quarter_entity:base',
    'quarter_entity:blank_line_added', 'quarter_entity:constant_added',
    'quarter_entity:copy_5_cells_differ', 'quarter_entity:copy_and_target_sentinel',
    'quarter_entity:copy_distinct', 'quarter_entity:copy_exact',
    'quarter_entity:copy_negative_scale', 'quarter_entity:copy_one_cell_999999',
    'quarter_entity:feature_renamed', 'quarter_entity:feature_renamed_awkward',
    'quarter_entity:id_added_gaps', 'quarter_entity:id_added_int',
    'quarter_entity:id_added_text', 'quarter_entity:missing_cell_feature',
    'quarter_entity:missing_cell_target', 'quarter_entity:noise_added',
    'quarter_entity:number_sentinel_-999', 'quarter_entity:number_sentinel_1e308',
    'quarter_entity:number_sentinel_999999', 'quarter_entity:number_two_sentinels',
    'quarter_entity:role_column_renamed', 'quarter_entity:row_repeated',
    'quarter_entity:rows_shuffled', 'quarter_entity:second_date_added',
    'quarter_entity:target_first', 'quarter_entity:target_renamed',
    'quarter_entity:target_sentinel_-999', 'quarter_entity:target_sentinel_999999',
    'quarter_entity:target_two_sentinels', 'quarter_entity:white_space_cells',
    'random_walk:base', 'random_walk:blank_line_added',
    'random_walk:constant_added', 'random_walk:copy_2pct_cells_altered',
    'random_walk:copy_5_cells_differ', 'random_walk:copy_and_target_sentinel',
    'random_walk:copy_curve', 'random_walk:copy_distinct',
    'random_walk:copy_exact', 'random_walk:copy_negative_scale',
    'random_walk:copy_noise_1pct', 'random_walk:copy_one_cell_999999',
    'random_walk:date_sentinel_1900', 'random_walk:feature_999999_early_row',
    'random_walk:feature_999999_latest_row', 'random_walk:feature_renamed',
    'random_walk:feature_renamed_awkward', 'random_walk:gap_2',
    'random_walk:gap_2_row_repeated', 'random_walk:id_added_gaps',
    'random_walk:id_added_int', 'random_walk:id_added_text',
    'random_walk:missing_cell_date', 'random_walk:missing_cell_feature',
    'random_walk:missing_cell_target', 'random_walk:noise_added',
    'random_walk:number_sentinel_-999', 'random_walk:number_sentinel_1e308',
    'random_walk:number_sentinel_999999', 'random_walk:number_two_sentinels',
    'random_walk:role_column_renamed', 'random_walk:row_repeated',
    'random_walk:row_repeated_other_target', 'random_walk:rows_repeated_10pct',
    'random_walk:rows_repeated_1pct', 'random_walk:rows_shuffled',
    'random_walk:second_date_added', 'random_walk:stamp_shared',
    'random_walk:target_-999_early_row', 'random_walk:target_-999_latest_row',
    'random_walk:target_first', 'random_walk:target_renamed',
    'random_walk:target_sentinel_-999', 'random_walk:target_sentinel_999999',
    'random_walk:target_two_sentinels', 'signal_clf:base',
    'signal_clf:blank_line_added', 'signal_clf:constant_added',
    'signal_clf:copy_distinct', 'signal_clf:copy_two_valued',
    'signal_clf:feature_renamed', 'signal_clf:feature_renamed_awkward',
    'signal_clf:id_added_gaps', 'signal_clf:id_added_int',
    'signal_clf:id_added_text', 'signal_clf:missing_cell_feature',
    'signal_clf:missing_cell_target', 'signal_clf:noise_added',
    'signal_clf:number_sentinel_-999', 'signal_clf:number_sentinel_1e308',
    'signal_clf:number_sentinel_999999', 'signal_clf:number_two_sentinels',
    'signal_clf:row_repeated', 'signal_clf:rows_shuffled',
    'signal_clf:second_date_added', 'signal_clf:target_first',
    'signal_clf:target_renamed', 'signal_clf:target_respelt',
    'signal_clf:target_respelt_flipped', 'signal_clf:white_space_cells',
    'signal_reg:base', 'signal_reg:blank_line_added',
    'signal_reg:constant_added', 'signal_reg:copy_5_cells_differ',
    'signal_reg:copy_and_target_sentinel', 'signal_reg:copy_distinct',
    'signal_reg:copy_exact', 'signal_reg:copy_negative_scale',
    'signal_reg:copy_one_cell_999999', 'signal_reg:feature_renamed',
    'signal_reg:feature_renamed_awkward', 'signal_reg:id_added_gaps',
    'signal_reg:id_added_int', 'signal_reg:id_added_text',
    'signal_reg:missing_cell_feature', 'signal_reg:missing_cell_target',
    'signal_reg:noise_added', 'signal_reg:number_sentinel_-999',
    'signal_reg:number_sentinel_1e308', 'signal_reg:number_sentinel_999999',
    'signal_reg:number_two_sentinels', 'signal_reg:row_repeated',
    'signal_reg:rows_shuffled', 'signal_reg:second_date_added',
    'signal_reg:target_first', 'signal_reg:target_renamed',
    'signal_reg:target_sentinel_-999', 'signal_reg:target_sentinel_999999',
    'signal_reg:target_two_sentinels', 'signal_reg:white_space_cells',
    'time_blocks:base', 'time_blocks:blank_line_added',
    'time_blocks:constant_added', 'time_blocks:copy_distinct',
    'time_blocks:copy_two_valued', 'time_blocks:date_sentinel_1900',
    'time_blocks:feature_renamed', 'time_blocks:feature_renamed_awkward',
    'time_blocks:id_added_gaps', 'time_blocks:id_added_int',
    'time_blocks:id_added_text', 'time_blocks:missing_cell_date',
    'time_blocks:missing_cell_feature', 'time_blocks:missing_cell_target',
    'time_blocks:noise_added', 'time_blocks:number_sentinel_-999',
    'time_blocks:number_sentinel_1e308', 'time_blocks:number_sentinel_999999',
    'time_blocks:number_two_sentinels', 'time_blocks:role_column_renamed',
    'time_blocks:row_repeated', 'time_blocks:row_repeated_other_target',
    'time_blocks:rows_repeated_10pct', 'time_blocks:rows_repeated_1pct',
    'time_blocks:rows_shuffled', 'time_blocks:second_date_added',
    'time_blocks:stamp_shared', 'time_blocks:target_first',
    'time_blocks:target_renamed', 'time_blocks:target_respelt',
    'time_blocks:target_respelt_flipped', 'time_blocks:white_space_cells',
    'time_blocks_gap:base', 'time_blocks_gap:blank_line_added',
    'time_blocks_gap:constant_added', 'time_blocks_gap:copy_distinct',
    'time_blocks_gap:copy_two_valued', 'time_blocks_gap:date_sentinel_1900',
    'time_blocks_gap:feature_renamed', 'time_blocks_gap:feature_renamed_awkward',
    'time_blocks_gap:id_added_gaps', 'time_blocks_gap:id_added_int',
    'time_blocks_gap:id_added_text', 'time_blocks_gap:missing_cell_date',
    'time_blocks_gap:missing_cell_feature', 'time_blocks_gap:missing_cell_target',
    'time_blocks_gap:noise_added', 'time_blocks_gap:number_sentinel_-999',
    'time_blocks_gap:number_sentinel_1e308', 'time_blocks_gap:number_sentinel_999999',
    'time_blocks_gap:number_two_sentinels', 'time_blocks_gap:role_column_renamed',
    'time_blocks_gap:row_repeated', 'time_blocks_gap:row_repeated_other_target',
    'time_blocks_gap:rows_repeated_10pct', 'time_blocks_gap:rows_repeated_1pct',
    'time_blocks_gap:rows_shuffled', 'time_blocks_gap:second_date_added',
    'time_blocks_gap:stamp_shared', 'time_blocks_gap:target_first',
    'time_blocks_gap:target_renamed', 'time_blocks_gap:target_respelt',
    'time_blocks_gap:target_respelt_flipped', 'time_blocks_gap:white_space_cells',
    'time_blocks_undeclared:base', 'time_blocks_undeclared:blank_line_added',
    'time_blocks_undeclared:constant_added', 'time_blocks_undeclared:copy_distinct',
    'time_blocks_undeclared:copy_two_valued', 'time_blocks_undeclared:date_sentinel_1900',
    'time_blocks_undeclared:date_sentinel_9999', 'time_blocks_undeclared:date_sentinel_zero',
    'time_blocks_undeclared:duplicates_group', 'time_blocks_undeclared:feature_renamed',
    'time_blocks_undeclared:feature_renamed_awkward', 'time_blocks_undeclared:id_added_gaps',
    'time_blocks_undeclared:id_added_int', 'time_blocks_undeclared:id_added_text',
    'time_blocks_undeclared:missing_cell_date', 'time_blocks_undeclared:missing_cell_feature',
    'time_blocks_undeclared:missing_cell_target', 'time_blocks_undeclared:noise_added',
    'time_blocks_undeclared:number_sentinel_-999', 'time_blocks_undeclared:number_sentinel_1e308',
    'time_blocks_undeclared:number_sentinel_999999', 'time_blocks_undeclared:number_two_sentinels',
    'time_blocks_undeclared:role_column_renamed', 'time_blocks_undeclared:row_repeated',
    'time_blocks_undeclared:rows_shuffled', 'time_blocks_undeclared:second_date_added',
    'time_blocks_undeclared:target_first', 'time_blocks_undeclared:target_renamed',
    'time_blocks_undeclared:target_respelt', 'time_blocks_undeclared:target_respelt_flipped',
    'time_blocks_undeclared:white_space_cells', 'walk_copy:base',
    'walk_copy:blank_line_added', 'walk_copy:constant_added',
    'walk_copy:copy_2pct_cells_altered', 'walk_copy:copy_5_cells_differ',
    'walk_copy:copy_and_target_sentinel', 'walk_copy:copy_curve',
    'walk_copy:copy_distinct', 'walk_copy:copy_exact',
    'walk_copy:copy_negative_scale', 'walk_copy:copy_noise_1pct',
    'walk_copy:copy_one_cell_999999', 'walk_copy:date_sentinel_1900',
    'walk_copy:feature_999999_early_row', 'walk_copy:feature_999999_latest_row',
    'walk_copy:feature_renamed', 'walk_copy:feature_renamed_awkward',
    'walk_copy:id_added_gaps', 'walk_copy:id_added_int',
    'walk_copy:id_added_text', 'walk_copy:missing_cell_date',
    'walk_copy:missing_cell_feature', 'walk_copy:missing_cell_target',
    'walk_copy:noise_added', 'walk_copy:number_sentinel_-999',
    'walk_copy:number_sentinel_1e308', 'walk_copy:number_sentinel_999999',
    'walk_copy:number_two_sentinels', 'walk_copy:role_column_renamed',
    'walk_copy:row_repeated', 'walk_copy:row_repeated_other_target',
    'walk_copy:rows_repeated_10pct', 'walk_copy:rows_repeated_1pct',
    'walk_copy:rows_shuffled', 'walk_copy:second_date_added',
    'walk_copy:stamp_shared', 'walk_copy:target_-999_early_row',
    'walk_copy:target_-999_latest_row', 'walk_copy:target_first',
    'walk_copy:target_renamed', 'walk_copy:target_sentinel_-999',
    'walk_copy:target_sentinel_999999', 'walk_copy:target_two_sentinels',
    'walk_copy_cell:base', 'walk_copy_cell:blank_line_added',
    'walk_copy_cell:constant_added', 'walk_copy_cell:copy_2pct_cells_altered',
    'walk_copy_cell:copy_5_cells_differ', 'walk_copy_cell:copy_and_target_sentinel',
    'walk_copy_cell:copy_curve', 'walk_copy_cell:copy_distinct',
    'walk_copy_cell:copy_exact', 'walk_copy_cell:copy_negative_scale',
    'walk_copy_cell:copy_one_cell_999999', 'walk_copy_cell:date_sentinel_1900',
    'walk_copy_cell:feature_999999_early_row', 'walk_copy_cell:feature_999999_latest_row',
    'walk_copy_cell:feature_renamed', 'walk_copy_cell:feature_renamed_awkward',
    'walk_copy_cell:id_added_gaps', 'walk_copy_cell:id_added_int',
    'walk_copy_cell:id_added_text', 'walk_copy_cell:missing_cell_date',
    'walk_copy_cell:missing_cell_feature', 'walk_copy_cell:missing_cell_target',
    'walk_copy_cell:noise_added', 'walk_copy_cell:number_sentinel_-999',
    'walk_copy_cell:number_sentinel_1e308', 'walk_copy_cell:number_sentinel_999999',
    'walk_copy_cell:number_two_sentinels', 'walk_copy_cell:role_column_renamed',
    'walk_copy_cell:row_repeated', 'walk_copy_cell:row_repeated_other_target',
    'walk_copy_cell:rows_repeated_10pct', 'walk_copy_cell:rows_repeated_1pct',
    'walk_copy_cell:rows_shuffled', 'walk_copy_cell:second_date_added',
    'walk_copy_cell:stamp_shared', 'walk_copy_cell:target_-999_early_row',
    'walk_copy_cell:target_-999_latest_row', 'walk_copy_cell:target_first',
    'walk_copy_cell:target_renamed', 'walk_copy_cell:target_sentinel_-999',
    'walk_copy_cell:target_sentinel_999999', 'walk_copy_cell:target_two_sentinels',
    'walk_late_cell:base', 'walk_late_cell:blank_line_added',
    'walk_late_cell:constant_added', 'walk_late_cell:copy_2pct_cells_altered',
    'walk_late_cell:copy_5_cells_differ', 'walk_late_cell:copy_and_target_sentinel',
    'walk_late_cell:copy_curve', 'walk_late_cell:copy_distinct',
    'walk_late_cell:copy_exact', 'walk_late_cell:copy_negative_scale',
    'walk_late_cell:copy_noise_1pct', 'walk_late_cell:copy_one_cell_999999',
    'walk_late_cell:date_sentinel_1900', 'walk_late_cell:feature_999999_early_row',
    'walk_late_cell:feature_999999_latest_row', 'walk_late_cell:feature_renamed',
    'walk_late_cell:feature_renamed_awkward', 'walk_late_cell:id_added_gaps',
    'walk_late_cell:id_added_int', 'walk_late_cell:id_added_text',
    'walk_late_cell:missing_cell_date', 'walk_late_cell:missing_cell_feature',
    'walk_late_cell:missing_cell_target', 'walk_late_cell:noise_added',
    'walk_late_cell:number_sentinel_-999', 'walk_late_cell:number_sentinel_1e308',
    'walk_late_cell:number_sentinel_999999', 'walk_late_cell:number_two_sentinels',
    'walk_late_cell:role_column_renamed', 'walk_late_cell:row_repeated',
    'walk_late_cell:row_repeated_other_target', 'walk_late_cell:rows_repeated_10pct',
    'walk_late_cell:rows_repeated_1pct', 'walk_late_cell:rows_shuffled',
    'walk_late_cell:second_date_added', 'walk_late_cell:stamp_shared',
    'walk_late_cell:target_-999_early_row', 'walk_late_cell:target_-999_latest_row',
    'walk_late_cell:target_first', 'walk_late_cell:target_renamed',
    'walk_late_cell:target_sentinel_-999', 'walk_late_cell:target_sentinel_999999',
    'walk_late_cell:target_two_sentinels', 'walk_repeated:base',
    'walk_repeated:blank_line_added', 'walk_repeated:constant_added',
    'walk_repeated:copy_2pct_cells_altered', 'walk_repeated:copy_5_cells_differ',
    'walk_repeated:copy_and_target_sentinel', 'walk_repeated:copy_curve',
    'walk_repeated:copy_distinct', 'walk_repeated:copy_exact',
    'walk_repeated:copy_negative_scale', 'walk_repeated:copy_noise_1pct',
    'walk_repeated:copy_one_cell_999999', 'walk_repeated:date_sentinel_1900',
    'walk_repeated:feature_999999_early_row', 'walk_repeated:feature_999999_latest_row',
    'walk_repeated:feature_renamed', 'walk_repeated:feature_renamed_awkward',
    'walk_repeated:id_added_gaps', 'walk_repeated:id_added_int',
    'walk_repeated:id_added_text', 'walk_repeated:missing_cell_date',
    'walk_repeated:missing_cell_feature', 'walk_repeated:missing_cell_target',
    'walk_repeated:noise_added', 'walk_repeated:number_sentinel_-999',
    'walk_repeated:number_sentinel_1e308', 'walk_repeated:number_sentinel_999999',
    'walk_repeated:number_two_sentinels', 'walk_repeated:role_column_renamed',
    'walk_repeated:row_repeated', 'walk_repeated:row_repeated_other_target',
    'walk_repeated:rows_repeated_10pct', 'walk_repeated:rows_repeated_1pct',
    'walk_repeated:rows_shuffled', 'walk_repeated:second_date_added',
    'walk_repeated:stamp_shared', 'walk_repeated:target_-999_early_row',
    'walk_repeated:target_-999_latest_row', 'walk_repeated:target_first',
    'walk_repeated:target_renamed', 'walk_repeated:target_sentinel_-999',
    'walk_repeated:target_sentinel_999999', 'walk_repeated:target_two_sentinels',
})
TABLES_OWNED = frozenset({
    0, 1, 2, 5, 6, 8, 11, 13, 14, 18, 21, 25,
    29, 31, 32, 33, 34, 36, 38, 39, 41, 42, 47, 49,
    50, 51, 52, 53, 54, 55, 56, 58, 63, 64, 66, 67,
    69, 70, 72, 74, 75, 76, 77, 78, 79, 80, 81, 83,
    84, 87, 91, 93, 95, 96, 97, 100, 101, 102, 103, 105,
    106, 108, 110, 111, 112, 113, 114, 116, 119, 120, 122, 128,
    130, 132, 134, 135, 136, 138, 141, 143, 151, 153, 155, 156,
    159, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174,
    175, 176, 179, 180, 182, 183, 185, 186, 187, 189, 190, 191,
    196, 197, 198, 199, 201, 206, 210, 211, 212, 213, 214, 216,
    217, 220, 221, 223, 226, 227, 228, 231, 232, 234, 235, 238,
    239, 240, 241, 242, 247, 249, 251, 252, 253, 254, 255, 256,
    259, 261, 262, 269, 271, 272, 276, 277, 278, 279, 281, 282,
    284, 285, 286, 288, 289, 291, 294, 296, 297, 298, 300, 301,
    303, 304, 306, 310, 311, 312, 313, 314, 315, 316, 319, 321,
    322, 323, 324, 326, 327, 328, 329, 330, 332, 335, 337, 339,
    340, 341, 342, 345, 346, 347, 349, 350, 351, 353, 354, 356,
    360, 361, 362, 365, 366, 368, 372, 375, 376, 377, 378, 379,
    381, 382, 383, 385, 388, 393, 394, 396, 397, 398, 402, 403,
    405, 406, 407, 409, 410, 411, 412, 413, 416, 417, 418, 419,
    421, 422, 423, 424, 427, 429, 430, 431, 432, 433, 435, 437,
    438, 439, 441, 442, 443, 444, 445, 447, 450, 451, 452, 453,
    454, 457, 458, 459, 461, 462, 463, 465, 466, 467, 469, 471,
    472, 478, 480, 481, 482, 484, 485, 489, 490, 491, 492, 494,
    498, 500, 504, 508, 510, 511, 515, 517, 518, 519, 520, 523,
    524, 525, 526, 527, 528, 529, 533, 534, 536, 538, 540, 542,
    546, 547, 548, 549, 550, 551, 553, 555, 556, 557, 564, 567,
    570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581,
    585, 586, 587, 590, 591, 597, 600, 601, 605, 606, 607, 608,
    609, 610, 611, 612, 615, 616, 619, 624, 626, 628, 630, 632,
    633, 634, 635, 636, 639, 642, 644, 645, 647, 648, 649, 650,
    651, 653, 654, 655, 656, 658, 659, 661, 662, 663, 666, 667,
    669, 670, 674, 675, 676, 677, 678, 679, 680, 682, 684, 686,
    689, 690, 692, 693, 694, 695, 696, 697, 699, 700, 701, 702,
    704, 706, 708, 709, 710, 712, 713, 716, 717, 718, 719, 720,
    721, 725, 726, 727, 729, 730, 731, 732, 734, 736, 737, 738,
    739, 742, 745, 747, 748, 750, 751, 752, 753, 754, 755, 758,
    760, 762, 764, 765, 768, 769, 772, 774, 775, 776, 778, 779,
    781, 783, 785, 786, 788, 789, 791, 794, 795, 796, 798, 800,
    802, 804, 805, 806, 808, 809, 810, 812, 815, 816, 819, 820,
    821, 822, 824, 825, 828, 829, 830, 832, 834, 835, 836, 837,
    839, 840, 841, 846, 847, 853, 854, 856, 857, 859, 861, 863,
    865, 866, 867, 870, 871, 873, 874, 876, 878, 880, 881, 882,
    883, 884, 888, 890, 893, 895, 896, 899, 900, 901, 904, 907,
    909, 911, 912, 914, 915, 916, 917, 918, 923, 924, 925, 927,
    929, 930, 931, 932, 934, 935, 936, 937, 938, 939, 941, 942,
    946, 948, 949, 950, 953, 955, 959, 961, 962, 963, 964, 965,
    966, 967, 968, 969, 971, 972, 973, 974, 975, 978, 979, 980,
    982, 984, 985, 987, 988, 992, 996, 997, 998,
})


# ------------------------------------------------------------------ readers

def column(result, name):
    """The record of `columns` for the column `name`."""
    found = [r for r in h.records(result.columns) if r["name"] == name]
    assert len(found) == 1, f"no single record for column {name!r}"
    return found[0]


def is_leak(record):
    return record["role"] == "excluded" and record["reason"] == "suspected_leak"


def is_feature(record):
    return record["role"] == "feature" and record["reason"] is None


def leaks(result):
    """The sorted names of the columns excluded as `suspected_leak`."""
    return sorted(r["name"] for r in h.records(result.columns) if is_leak(r))


def told(result, code, name=None):
    """The messages of the warnings with `code` (about the column `name`, if given)."""
    return [str(r["message"]) for r in h.records(result.warnings)
            if r["code"] == code and (name is None or r["column"] == name)]


def training_rows(result, frame):
    """The row numbers of `frame` that are not in the holdout (rows dropped, or in a
    gap, are the caller's to leave out)."""
    held = set(h.holdout_rows(result))
    return [r for r in range(len(frame)) if r not in held]


# ------------------------------------------------------- rules, by hand

def previous_by_hand(y, keys, gap=0):
    """For each row the mean of `y` over the rows at the timestamp `gap + 1` distinct
    timestamps before its own, or None. Dictionaries and plain sums."""
    y = [float(v) for v in y]
    keys = list(keys)
    stamps = sorted(set(keys))
    rank = {k: i for i, k in enumerate(stamps)}
    sums, counts = {}, {}
    for value, key in zip(y, keys):
        sums[rank[key]] = sums.get(rank[key], 0.0) + value
        counts[rank[key]] = counts.get(rank[key], 0) + 1
    out = []
    for key in keys:
        before = rank[key] - (gap + 1)
        out.append(None if before < 0 else sums[before] / counts[before])
    return out


def against_order(x, y):
    """The share against the order of step 8 (D76): the pairs `(x, y)` with neither
    missing, sorted by `x` then `y`; among neighbours whose `x` differ, the smaller of
    the share in which `y` falls and the share in which it rises. None with fewer than
    20 such neighbours."""
    pairs = []
    for a, b in zip(x, y):
        if a is None or b is None:
            continue
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b):
            continue
        pairs.append((a, b))
    pairs.sort()
    rises = falls = total = 0
    for (a0, b0), (a1, b1) in zip(pairs, pairs[1:]):
        if a1 == a0:
            continue
        total += 1
        if b1 > b0:
            rises += 1
        elif b1 < b0:
            falls += 1
    if total < ORDER_PAIRS:
        return None
    return min(rises, falls) / total


def far_outside(values):
    """The rule of `extreme_value` (spec step 11), value by value, over the distinct
    finite values of `values`: a list of `(value, rows that hold it, nearest value that
    is not far outside)`, the low end first."""
    finite = []
    for v in values:
        if v is None:
            continue
        v = float(v)
        if math.isfinite(v):
            finite.append(v + 0.0)
    count = {}
    for v in finite:
        count[v] = count.get(v, 0) + 1
    distinct = sorted(count)
    n = len(distinct)
    out = []
    if n < 3:
        return out
    for ordered in (distinct, distinct[::-1]):
        last = 0
        for j in range(1, EXTREME_OUTERMOST + 1):
            if j > n - 2:
                break
            gap = abs(ordered[j] - ordered[j - 1])
            m = min(EXTREME_SPREAD, n - j - 1)
            spread = abs(ordered[j + m] - ordered[j])
            if gap > EXTREME_GAP * spread:
                last = j
        for j in range(last):
            out.append((ordered[j], count[ordered[j]], ordered[last]))
    return out


def largest_row(rows, truth, guess):
    """`(share, row)` of the squared errors: the largest over their sum (`math.fsum`),
    the first row with it; `(None, None)` when the sum is zero."""
    errors = [(float(t) - float(p)) ** 2 for t, p in zip(truth, guess)]
    total = math.fsum(errors)
    if total == 0:
        return None, None
    top = max(errors)
    return top / total, int(list(rows)[errors.index(top)])


# ------------------------------------------------------------------- tables

def walk(n=600, seed=0, per=1, noise=0.0, shuffle=False):
    """A random walk at the timestamps `t` (0 to n - 1, int64, each `per` times), two
    noise columns `n1` and `n2` and the target `y`, the level plus `noise` times normal
    draws. Draws in the order: level, n1, n2, the noise of y, then the permutation."""
    rng = np.random.default_rng(seed)
    level = np.cumsum(rng.normal(size=n))
    n1 = rng.normal(size=n * per)
    n2 = rng.normal(size=n * per)
    y = np.repeat(level, per) + noise * rng.normal(size=n * per)
    frame = pd.DataFrame({"t": np.repeat(np.arange(n, dtype=np.int64), per),
                          "n1": n1, "n2": n2, "y": y})
    if shuffle:
        frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
    return frame


def with_column(frame, name, values, target="y"):
    """A copy of `frame` with the column `name` put just before the target."""
    out = frame.copy()
    out.insert(list(out.columns).index(target), name, values)
    return out


def rotated(values, share, seed=0, rows=None):
    """A float copy of `values` in which `max(2, round(share x len(rows)))` cells, drawn
    among `rows` (every row when None) without replacement, are moved round among
    themselves by one place."""
    out = np.array(values, dtype=float).copy()
    rows = np.arange(len(out)) if rows is None else np.asarray(list(rows))
    k = max(2, int(round(share * len(rows))))
    where = np.random.default_rng(seed).choice(rows, size=k, replace=False)
    out[where] = np.roll(out[where], 1)
    return out


SCHEMES = ("time", "time_repeated", "time_entities", "random", "grouped")


def scheme_table(scheme, seed):
    """`(frame, statements)`: a regression table with a noise column `n1` and no column
    that copies the target, with the statements of its call."""
    if scheme == "time":
        return walk(n=600, seed=seed), {"time_col": "t", "group_col": False}
    if scheme == "time_repeated":
        return (walk(n=200, seed=seed, per=3, noise=0.5),
                {"time_col": "t", "group_col": False})
    if scheme == "time_entities":
        frame = g.walks(n_series=6, n_times=100, seed=seed).drop(columns=["prev"])
        noise = np.random.default_rng(seed).normal(size=len(frame))
        frame.insert(2, "n1", noise)
        return frame, {"time_col": "t", "group_col": "who"}
    if scheme == "random":
        return h.reg_frame(n=600, seed=seed).rename(columns={"z1": "n1"}), dict(s.SAID)
    if scheme == "grouped":
        frame = s.sized_entities([5] * 120, seed=seed).rename(columns={"m1": "n1"})
        return frame, {"time_col": False, "group_col": "who"}
    raise ValueError(scheme)


def screened(frame, statements, **kw):
    """`s.run` with one recording family: what the screen and the warnings do is all
    such a test reads."""
    _, families = s.rec()
    return s.run(frame, families=families, **statements, **kw)


def dominant_entity(large=1490, small=255, per=2, seed=0):
    """`who` is `C0000` for `large` rows and `C0001`, ... for `small` entities of `per`
    rows; `n1` to `n4` and `y` are normal draws, in that order; rows permuted."""
    rng = np.random.default_rng(seed)
    who = ["C0000"] * large
    for k in range(small):
        who.extend([f"C{k + 1:04d}"] * per)
    n = len(who)
    frame = pd.DataFrame({"who": who})
    for j in range(4):
        frame[f"n{j + 1}"] = rng.normal(size=n)
    frame["y"] = rng.normal(size=n)
    return frame.iloc[rng.permutation(n)].reset_index(drop=True)
