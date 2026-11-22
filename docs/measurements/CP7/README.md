# CP7: the measurements behind the sixth pass (D62 to D66)

The scripts and the values behind spec section 10.8, the row that the sixth
pass added to section 10.1, and the plan of the pass (D35). Measured on
2026-11-21 and 2026-11-22 on the judge as changed for the sixth pass
(`verify\smoke_test.py`, `verify\reference`), and where it says so on the
judge and the application as they stood before the pass: a copy of `verify\`
and `tests\` taken before the first edit, when the last commit shown was
`de1eb4d`. Nothing here is
part of the gate, and no script writes inside the repo: each works in a
temporary folder. Every table below is printed by the script named above it;
the files it printed are kept in `output\`.

Run each from `E:\CODE\ml-professor`. Seeds: "data seed *k*" is the gate's
base seed 1115 + 100 x *k*, *k* = 0 to 29 (*k* = 0 is the gate's own data),
for a dataset written by the gate's generator; for a table made by a script
here it is the seed *k* = 0 to 29 passed to numpy. The tool's own seed is 0
unless a table says otherwise.

| Script | What it measures | Command |
|---|---|---|
| `count_calls.py`, `analyse_calls.py` | which existing tests leave a statement unsaid (D63, D65) | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP7\count_calls.py <lab folder> <log.jsonl>` then `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP7\analyse_calls.py <log.jsonl> <lab folder>\junit.xml` |
| `loosening.py` | the 39 replacements of the loosening, as a folder to lay over the suite | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP7\loosening.py <folder>`, or `--table` |
| `unloosened_failures.py` | whether every case the changed reference fails, before the loosening, is one the loosening covers | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP7\unloosened_failures.py <log.jsonl> <failed.txt>` |
| `caught_table.py` | how many cases fail on each broken copy, before the loosening and with it | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP7\caught_table.py <label>=<mutation check output> ...` |
| `named_tests.py` | whether each broken copy fails the test named for it | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP7\named_tests.py <mutation check output> [<folder of kept reports>]` |
| `group_band.py` | the band of `duplicates.group_no_skill`, and the tables tried | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP7\group_band.py gate - - --table` and `... explore -` |
| `grouped_holdout.py` | the grouped holdout under the rule as built and the rule of D64 | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP7\grouped_holdout.py --against-reference --table` |
| `candidates.py` | the two candidate warnings of section D | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP7\candidates.py dates` and `... series --table` |
| `short_panels.py` | how often the panel warning fires on a short panel | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP7\short_panels.py --table` |
| `gate_moved.py` | what moved at the gate seed between the two states of the judge | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP7\gate_moved.py <before.json> <after.json>`, the two files from `docs\measurements\CP2\long_rows.py values` |

## 1. Which existing tests leave a statement unsaid (D63, D65)

`count_calls.py` builds a throwaway project outside the repo: a copy of
`tests\` and a copy of the application whose `profile()` writes one line for
each call: the test (`PYTEST_CURRENT_TEST`, which a command-line subprocess
inherits), what was passed as `time_col` and `group_col`, whether the run got
as far as step 5, and how it ended. It was run on the application as built by
the fifth pass, before any file of this pass was written. "D63 changes the
outcome" means a call that leaves a statement unsaid, gets as far as step 5,
and does not already stop there with the code D63 gives.

```
suite: 1757 cases in 52 files; 0 failing in the lab
calls of profile(): 1731

   1178 calls: unsaid, D63 changes the outcome
     35 calls: unsaid, stops today with the same code
     67 calls: both said
    451 calls: unsaid, ends before step 5 (unchanged by D63)

| File | cases | call profile()/CLI | leave something unsaid | D63 changes the outcome | stop today with the same code |
|---|---|---|---|---|---|
| test_chosen_failure.py | 5 | 5 | 5 | 5 | 0 |
| test_console_summary.py | 3 | 3 | 3 | 2 | 2 |
| test_csv_edges.py | 16 | 8 | 8 | 1 | 0 |
| test_duplicates.py | 22 | 20 | 20 | 14 | 1 |
| test_duplicates_group.py | 13 | 13 | 13 | 8 | 0 |
| test_edges.py | 39 | 39 | 39 | 16 | 2 |
| test_fast_cap_error.py | 4 | 3 | 3 | 3 | 0 |
| test_fast_profile_error.py | 5 | 5 | 5 | 5 | 0 |
| test_feature_names.py | 9 | 7 | 7 | 7 | 0 |
| test_full_profile.py | 7 | 4 | 4 | 4 | 0 |
| test_groups_random.py | 25 | 24 | 23 | 19 | 0 |
| test_groups_time.py | 7 | 7 | 1 | 1 | 0 |
| test_guard_final_fit.py | 3 | 3 | 3 | 3 | 0 |
| test_guard_header_lines.py | 3 | 3 | 3 | 0 | 0 |
| test_guard_leak_screen.py | 2 | 2 | 2 | 2 | 0 |
| test_guard_review3.py | 6 | 6 | 6 | 6 | 0 |
| test_guard_review4.py | 6 | 6 | 6 | 5 | 0 |
| test_guard_uncertainty.py | 2 | 2 | 2 | 2 | 0 |
| test_leakage.py | 8 | 8 | 8 | 8 | 0 |
| test_model_families.py | 5 | 3 | 3 | 3 | 0 |
| test_non_finite.py | 18 | 15 | 15 | 14 | 0 |
| test_object_numbers.py | 24 | 24 | 24 | 24 | 0 |
| test_panel_stop.py | 5 | 5 | 4 | 2 | 2 |
| test_pipeline.py | 24 | 24 | 24 | 24 | 0 |
| test_read_once.py | 5 | 5 | 5 | 1 | 0 |
| test_readme_examples.py | 3 | 1 | 1 | 1 | 0 |
| test_result_schema.py | 15 | 7 | 7 | 6 | 0 |
| test_review4_small.py | 10 | 3 | 3 | 1 | 0 |
| test_run_hygiene.py | 9 | 7 | 7 | 4 | 0 |
| test_seed_range.py | 7 | 5 | 5 | 1 | 0 |
| test_skill_undefined.py | 10 | 9 | 9 | 9 | 0 |
| test_summary_digits.py | 1 | 1 | 1 | 1 | 0 |
| test_target_white_space.py | 6 | 4 | 4 | 4 | 0 |
| test_time_fraction.py | 28 | 3 | 3 | 1 | 0 |
| test_time_order_stated.py | 10 | 10 | 10 | 5 | 5 |
| test_time_pipeline.py | 15 | 15 | 15 | 13 | 3 |
| test_type_matrix.py | 805 | 805 | 805 | 505 | 20 |
| test_type_matrix_group.py | 353 | 353 | 299 | 246 | 0 |
| test_uncertainty.py | 16 | 16 | 16 | 16 | 0 |
| test_whole_numbers.py | 19 | 14 | 14 | 14 | 0 |
| total | 1757 | 1497 | 1435 | 1006 | 35 |
```

Where those calls are made, which is the list of helpers and tests that the
loosening changes:

```
where the calls that D63 changes are made (the function that calls profile()):
    505 cases,   505 calls: test_type_matrix.py::test_a_result_or_a_named_error   test_type_matrix.py 505
    246 cases,   246 calls: test_type_matrix_group.py::test_a_result_or_a_named_error_as_the_entity_column   test_type_matrix_group.py 246
    141 cases,   257 calls: cp3_helpers.py::run   test_chosen_failure.py 5, test_duplicates.py 12, test_duplicates_group.py 5, test_fast_cap_error.py 2, test_fast_profile_error.py 5, test_feature_names.py 6, test_groups_random.py 15, test_groups_time.py 1, test_guard_final_fit.py 3, test_guard_leak_screen.py 2, test_guard_review3.py 6, test_guard_review4.py 5, test_guard_uncertainty.py 2, test_non_finite.py 12, test_object_numbers.py 24, test_panel_stop.py 2, test_read_once.py 1, test_review4_small.py 1, test_run_hygiene.py 3, test_skill_undefined.py 8, test_target_white_space.py 4, test_time_fraction.py 1, test_time_order_stated.py 3, test_whole_numbers.py 13
     24 cases,    32 calls: test_pipeline.py::_run   test_pipeline.py 24
     18 cases,    27 calls: cp5_helpers.py::raised   test_duplicates.py 8, test_duplicates_group.py 2, test_fast_cap_error.py 3, test_groups_random.py 4, test_time_order_stated.py 1
     16 cases,    18 calls: test_edges.py::_run   test_edges.py 16
     16 cases,    23 calls: test_uncertainty.py::_run   test_uncertainty.py 16
     15 cases,    22 calls: command line (h.run_cli or a subprocess)   test_console_summary.py 2, test_duplicates.py 1, test_duplicates_group.py 1, test_feature_names.py 1, test_full_profile.py 1, test_groups_random.py 1, test_non_finite.py 1, test_readme_examples.py 1, test_result_schema.py 1, test_skill_undefined.py 1, test_summary_digits.py 1, test_time_fraction.py 1, test_time_order_stated.py 1, test_whole_numbers.py 1
     13 cases,    17 calls: test_time_pipeline.py::_run   test_time_pipeline.py 13
      8 cases,     8 calls: test_leakage.py::_run   test_leakage.py 8
      6 cases,     7 calls: test_result_schema.py::_profile   test_result_schema.py 6
      3 cases,     7 calls: test_model_families.py::_profile   test_model_families.py 3
      1 cases,     1 calls: test_csv_edges.py::test_profile_runs_on_a_file_with_leading_blank_lines   test_csv_edges.py 1
      1 cases,     2 calls: conftest.py::full_runs   test_full_profile.py 1
      1 cases,     1 calls: test_full_profile.py::test_same_result_at_one_and_two_jobs   test_full_profile.py 1
      1 cases,     1 calls: test_full_profile.py::test_n_jobs_minus_one_records_cpu_count   test_full_profile.py 1
      1 cases,     1 calls: test_non_finite.py::test_a_target_on_a_very_large_scale_gives_finite_numbers_or_the_named_error   test_non_finite.py 1
      1 cases,     1 calls: test_run_hygiene.py::_call_profile   test_run_hygiene.py 1
      1 cases,     2 calls: test_seed_range.py::test_the_largest_seed_runs_and_the_next_one_is_refused   test_seed_range.py 1
```

## 2. The loosening, and what the suite still catches (D65)

`loosening.py` holds the 39 replacements in 22 files, each of which must match
its file once. With them laid over a copy of the suite:

- the application as it stood: 0 of 1,757 cases fail;
- the reference as changed for this pass: 0 of 1,757 fail.

**Before the loosening**, the changed reference fails 275 of the 1,757 cases
in 37 files: 274 of the 1,747 that the mutation check counts, in 36 files,
and one of the ten README tests, which it reports apart.
`unloosened_failures.py` sets each against the log of section 1: 255 make a
call that D63 stops, and 20 take a fixture that makes such a call once for
many tests. None is left over. A case is counted under the first reason that
holds, so the last column is 0: every test that reads one of the two old
names also leaves a statement unsaid. The 275 cases, one a line, are in
`output\unloosened_failed_cases.txt`.

| File | fail | a statement unsaid | through a fixture | an old name |
|---|---|---|---|---|
| `test_chosen_failure.py` | 5 | 5 | 0 | 0 |
| `test_console_summary.py` | 2 | 2 | 0 | 0 |
| `test_csv_edges.py` | 1 | 1 | 0 | 0 |
| `test_duplicates.py` | 14 | 14 | 0 | 0 |
| `test_duplicates_group.py` | 8 | 8 | 0 | 0 |
| `test_edges.py` | 16 | 16 | 0 | 0 |
| `test_fast_cap_error.py` | 3 | 3 | 0 | 0 |
| `test_fast_profile_error.py` | 5 | 5 | 0 | 0 |
| `test_feature_names.py` | 7 | 7 | 0 | 0 |
| `test_full_profile.py` | 7 | 4 | 3 | 0 |
| `test_groups_random.py` | 19 | 19 | 0 | 0 |
| `test_groups_time.py` | 1 | 1 | 0 | 0 |
| `test_guard_final_fit.py` | 3 | 3 | 0 | 0 |
| `test_guard_leak_screen.py` | 2 | 2 | 0 | 0 |
| `test_guard_review3.py` | 6 | 6 | 0 | 0 |
| `test_guard_review4.py` | 5 | 5 | 0 | 0 |
| `test_guard_uncertainty.py` | 2 | 2 | 0 | 0 |
| `test_leakage.py` | 8 | 8 | 0 | 0 |
| `test_model_families.py` | 5 | 3 | 2 | 0 |
| `test_non_finite.py` | 14 | 14 | 0 | 0 |
| `test_object_numbers.py` | 24 | 24 | 0 | 0 |
| `test_panel_stop.py` | 2 | 2 | 0 | 0 |
| `test_pipeline.py` | 24 | 24 | 0 | 0 |
| `test_read_once.py` | 1 | 1 | 0 | 0 |
| `test_readme_examples.py` | 1 | 1 | 0 | 0 |
| `test_result_schema.py` | 14 | 6 | 8 | 0 |
| `test_review4_small.py` | 8 | 1 | 7 | 0 |
| `test_run_hygiene.py` | 4 | 4 | 0 | 0 |
| `test_seed_range.py` | 1 | 1 | 0 | 0 |
| `test_skill_undefined.py` | 9 | 9 | 0 | 0 |
| `test_summary_digits.py` | 1 | 1 | 0 | 0 |
| `test_target_white_space.py` | 4 | 4 | 0 | 0 |
| `test_time_fraction.py` | 1 | 1 | 0 | 0 |
| `test_time_order_stated.py` | 5 | 5 | 0 | 0 |
| `test_time_pipeline.py` | 13 | 13 | 0 | 0 |
| `test_uncertainty.py` | 16 | 16 | 0 | 0 |
| `test_whole_numbers.py` | 14 | 14 | 0 | 0 |
| total | 275 | 255 | 20 | 0 |

not explained: 0

For each broken copy that the suite caught before this pass, the number of
cases that pass on the good reference and fail on that copy. The first two
columns are the judge as it stood (`verify\mutation_check.py` and the
reference in the copy taken before the first edit), with the suite as it is
and with the loosening laid over it; the third is the judge as changed, with
the loosening and the prototypes of the new tests.

| Broken copy | judge as it stood, suite as it is | judge as it stood, suite loosened | judge as changed, suite loosened and the new tests |
|---|---|---|---|
| F1 | 26 | 26 | 28 |
| F2 | 11 | 11 | 11 |
| F3 | 21 | 21 | 21 |
| F4 | 12 | 12 | 12 |
| F5 | 50 | 50 | 52 |
| F6 | 140 | 145 | 146 |
| F7 | 139 | 139 | 147 |
| F8 | 16 | 16 | 16 |
| F9 | 13 | 13 | 16 |
| F6b | 49 | 49 | 50 |
| F6c | 18 | 18 | 18 |
| F6d | 6 | 6 | 6 |
| F7b | 33 | 33 | 35 |
| F7c | 12 | 12 | 12 |
| F10 | 14 | 14 | 28 |
| F7d | 5 | 5 | 6 |
| F8b | 4 | 4 | 5 |
| M1 | 1 | 1 | 1 |
| M2 | 2 | 2 | 2 |
| M3 | 2 | 2 | 2 |
| M4 | 3 | 3 | 3 |
| M5 | 1 | 1 | 1 |
| M6 | 1 | 1 | 1 |
| M7 | 4 | 4 | 4 |
| M8 | 1 | 1 | 1 |
| M9 | 6 | 6 | 7 |
| M10 | 1 | 1 | 1 |
| M11 | 1 | 1 | 1 |
| M12 | 1 | 1 | 1 |
| M13 | 1 | 1 | 1 |
| M14 | 4 | 4 | 4 |
| M15 | 3 | 3 | 4 |
| M16 | 5 | 5 | 6 |
| M17 | 1 | 1 | 1 |
| M18 | 6 | 6 | 6 |
| M19 | 2 | 2 | 7 |
| M20 | 3 | 3 | 3 |
| M21 | 1 | 1 | 1 |
| M22 | 4 | 4 | 11 |
| F11 |  |  | 25 |
| F12 |  |  | 5 |
| F13 |  |  | 11 |
| F14 |  |  | 16 |
| M23 |  |  | 1 |
| M24 |  |  | 2 |
| M25 |  |  | 3 |
| M26 |  |  | 4 |
| M27 |  |  | 3 |
| M28 |  |  | 1 |
| M29 |  |  | 3 |
| M30 |  |  | 2 |
judge as it stood, suite as it is: good   fails 0 of 1747
judge as it stood, suite loosened: good   fails 0 of 1747
judge as changed, suite loosened and the new tests: good   fails 0 of 1828

No copy loses a failing case, in either step. F6 gains five on the judge as
it stood: type-matrix cases that used to end in the stop for a date column,
and now go on to the step where F6 shows.

The run behind the third column (the judge as changed, with the loosening and
the prototypes of the new tests laid over the suite by `--overlay`) passes:
the good reference fails 0 of 1,828 cases and each of the 51 broken copies at
least one. Its README line shows 2 of 13 failing, the two tests of
`test_readme_statements.py` that wait for the README (T53). Whether each copy
fails the test named for it (`named_tests.py`; the mutation check prints five
names of a longer list, so the copies whose named test was not among the five
were run again with `--keep`):

| Broken copy | cases failing | the test named for it | among them |
|---|---|---|---|
| F1 | 28 | (caught by the gate; no one test named) | |
| F2 | 11 | (caught by the gate; no one test named) | |
| F3 | 21 | (caught by the gate; no one test named) | |
| F4 | 12 | (caught by the gate; no one test named) | |
| F5 | 52 | (caught by the gate; no one test named) | |
| F6 | 146 | (caught by the gate; no one test named) | |
| F7 | 147 | (caught by the gate; no one test named) | |
| F8 | 16 | (caught by the gate; no one test named) | |
| F9 | 16 | (caught by the gate; no one test named) | |
| F6b | 50 | (caught by the gate; no one test named) | |
| F6c | 18 | (caught by the gate; no one test named) | |
| F6d | 6 | (caught by the gate; no one test named) | |
| F7b | 35 | (caught by the gate; no one test named) | |
| F7c | 12 | (caught by the gate; no one test named) | |
| F10 | 28 | (caught by the gate; no one test named) | |
| F7d | 6 | `test_time_order_stated.py::test_a_date_column_and_no_word_about_time_stops_the_run` | yes, 2 case(s) |
| F8b | 5 | (caught by the gate; no one test named) | |
| F11 | 25 | `test_statements.py::test_nothing_said_about_time_order_stops_whatever_the_table_holds` | yes, 4 case(s) (from the kept report) |
| F12 | 5 | `test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_together_whatever_their_target` | yes, 1 case(s) |
| F13 | 11 | `test_grouped_holdout.py::test_a_large_entity_is_never_held_out_when_smaller_ones_can_fill_a_quarter` | yes, 3 case(s) |
| F14 | 16 | `test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date` | yes, 5 case(s) |
| M1 | 1 | `test_guard_final_fit.py::test_m1_final_fit_statistics_ignore_the_holdout` | yes, 1 case(s) |
| M2 | 2 | `test_guard_final_fit.py::test_m2_levels_come_from_the_rows_of_each_fit` | yes, 1 case(s) |
| M3 | 2 | `test_guard_leak_screen.py::test_m3_single_column_skill_is_fitted_on_fold_training_rows_only` | yes, 1 case(s) |
| M3 | 2 | `test_guard_leak_screen.py::test_m3_missing_by_class_counts_training_rows_only` | yes, 1 case(s) |
| M4 | 3 | `test_guard_uncertainty.py::test_m4_block_bootstrap_draws_whole_blocks` | yes, 1 case(s) |
| M5 | 1 | `test_guard_uncertainty.py::test_m5_ties_count_in_the_p_value` | yes, 1 case(s) |
| M6 | 1 | `test_non_finite.py::test_a_prediction_that_is_not_finite_counts_as_raising` | yes, 1 case(s) |
| M7 | 4 | `test_fast_profile_error.py::test_fast_error_stands_when_the_full_profile_would_run` | yes, 2 case(s) |
| M7 | 4 | `test_time_pipeline.py::test_wide_panel_under_fast_raises` | yes, 1 case(s) |
| M8 | 1 | `test_guard_review3.py::test_m8_the_interval_is_the_middle_95_percent` | yes, 1 case(s) |
| M9 | 7 | `test_guard_review3.py::test_m9_the_leak_screen_uses_the_previous_value` | yes, 1 case(s) (from the kept report) |
| M10 | 1 | `test_guard_review3.py::test_m10_the_fast_cap_is_a_sample_from_the_seed` | yes, 1 case(s) |
| M11 | 1 | `test_guard_review3.py::test_m11_cv_std_is_the_sample_deviation` | yes, 1 case(s) |
| M12 | 1 | `test_guard_review3.py::test_m12_tied_predictions_in_a_resample` | yes, 1 case(s) |
| M13 | 1 | `test_guard_review3.py::test_m13_n_imputed_counts_rows_that_reach_a_model` | yes, 1 case(s) |
| M14 | 4 | `test_read_once.py (its tests)` | yes, 4 case(s) |
| M15 | 4 | `test_groups_random.py::test_no_entity_on_both_sides_or_in_two_folds` | yes, 2 case(s) |
| M16 | 6 | `test_groups_time.py::test_the_leak_screen_measures_a_lag_against_the_entity_baseline` | yes, 1 case(s) |
| M17 | 1 | `test_groups_random.py::test_the_fast_cap_keeps_whole_entities` | yes, 1 case(s) |
| M18 | 6 | `test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite` | yes, 5 case(s) |
| M19 | 7 | `test_panel_stop.py::test_a_persistent_panel_with_no_entity_column_stops` | yes, 1 case(s) |
| M19 | 7 | `test_statements.py::test_nothing_said_about_entities_stops_whatever_the_table_holds` | yes, 3 case(s) |
| M20 | 3 | `test_guard_review4.py::test_m20_infinite_cells_of_the_target_and_the_time_column_are_counted` | yes, 1 case(s) |
| M21 | 1 | `test_groups_random.py::test_the_interval_resamples_whole_entities_and_there_is_no_p_value` | yes, 1 case(s) |
| M22 | 11 | `test_duplicates_group.py::test_each_set_of_copies_stays_on_one_side_and_in_one_fold` | yes, 1 case(s) (from the kept report) |
| M23 | 1 | `test_guard_review5.py::test_m23_the_panel_warning_starts_just_above_one_half` | yes, 1 case(s) |
| M24 | 2 | `test_statements.py::test_one_message_asks_for_everything_that_is_missing` | yes, 1 case(s) |
| M25 | 3 | `test_duplicates_features.py::test_the_count_is_of_rows_that_agree_in_every_feature` | yes, 1 case(s) |
| M26 | 4 | `test_grouped_holdout.py::test_what_the_holdout_can_spare_goes_back` | yes, 1 case(s) |
| M27 | 3 | `test_grouped_holdout.py::test_a_large_unit_kept_in_training_is_named` | yes, 1 case(s) |
| M28 | 1 | `test_date_numbers.py::test_eight_digit_text_is_accepted_as_the_time_column` | yes, 1 case(s) |
| M29 | 3 | `test_date_numbers.py::test_the_date_rule_comes_before_the_id_like_rule` | yes, 2 case(s) |
| M30 | 2 | `test_guard_review5.py::test_m30_a_copy_at_step_8a_still_needs_the_same_target` | yes, 1 case(s) |

every named test fails on its copy

## 3. The band of `duplicates.group_no_skill` (D62)

`group_band.py gate` writes the gate's `few_valued` dataset at each data seed
and profiles it as the gate does, with the reference and with F12, in which
the sets of `duplicates="group"` are formed with the target as the fifth pass
built them. The band is the reference's mean plus or minus 4.5 standard
deviations, rounded outwards: -0.20 to 0.20.

| Table | Copy | skill min | mean | sd | max | stops | holdout units | interval reported | interval holds 0 |
|---|---|---|---|---|---|---|---|---|---|
| the gate's `few_valued` | good | -0.121 | -0.000 | 0.044 | +0.084 | 0 | 5..6 | 0 of 30 | 0 of 0 |
|  | mean - 4.5 sd, mean + 4.5 sd | -0.198 |  |  | +0.198 |  |  |  |  |
| the gate's `few_valued` | F12 | -1.000 | -0.906 | 0.094 | -0.648 | 0 | 8..10 | 0 of 30 | 0 of 0 |

Per seed:

| table | copy | k | skill | model | holdout units | rows that agree with another | holdout copies | holdout share |
|---|---|---|---|---|---|---|---|---|
| few_valued | good | 0 | +0.0183 | tree-0 | 5 | 2000 | 0 | 0.295 |
| few_valued | good | 1 | +0.0000 | tree-0 | 5 | 2000 | 0 | 0.303 |
| few_valued | good | 2 | -0.0461 | tree-0 | 5 | 2000 | 0 | 0.307 |
| few_valued | good | 3 | -0.0199 | tree-0 | 5 | 2000 | 0 | 0.298 |
| few_valued | good | 4 | +0.0129 | linear-0 | 5 | 2000 | 0 | 0.3005 |
| few_valued | good | 5 | +0.0835 | linear-0 | 5 | 2000 | 0 | 0.2875 |
| few_valued | good | 6 | -0.0115 | linear-0 | 5 | 2000 | 0 | 0.295 |
| few_valued | good | 7 | -0.0879 | linear-0 | 5 | 2000 | 0 | 0.2985 |
| few_valued | good | 8 | +0.0471 | tree-0 | 5 | 2000 | 0 | 0.3015 |
| few_valued | good | 9 | +0.0376 | tree-0 | 5 | 2000 | 0 | 0.3065 |
| few_valued | good | 10 | -0.0123 | tree-0 | 5 | 2000 | 0 | 0.293 |
| few_valued | good | 11 | +0.0435 | tree-0 | 5 | 2000 | 0 | 0.3025 |
| few_valued | good | 12 | +0.0031 | linear-0 | 5 | 2000 | 0 | 0.294 |
| few_valued | good | 13 | -0.0046 | linear-0 | 5 | 2000 | 0 | 0.303 |
| few_valued | good | 14 | +0.0019 | linear-0 | 5 | 2000 | 0 | 0.302 |
| few_valued | good | 15 | +0.0078 | linear-0 | 5 | 2000 | 0 | 0.299 |
| few_valued | good | 16 | -0.0037 | linear-0 | 5 | 2000 | 0 | 0.3045 |
| few_valued | good | 17 | -0.0859 | tree-0 | 5 | 2000 | 0 | 0.2975 |
| few_valued | good | 18 | +0.0648 | linear-0 | 5 | 2000 | 0 | 0.284 |
| few_valued | good | 19 | -0.0113 | linear-0 | 5 | 2000 | 0 | 0.2935 |
| few_valued | good | 20 | -0.1214 | linear-0 | 5 | 2000 | 0 | 0.295 |
| few_valued | good | 21 | -0.0255 | linear-0 | 5 | 2000 | 0 | 0.3 |
| few_valued | good | 22 | -0.0083 | linear-0 | 6 | 2000 | 0 | 0.353 |
| few_valued | good | 23 | -0.0017 | linear-0 | 5 | 2000 | 0 | 0.2945 |
| few_valued | good | 24 | +0.0068 | tree-0 | 5 | 2000 | 0 | 0.306 |
| few_valued | good | 25 | +0.0656 | linear-0 | 5 | 2000 | 0 | 0.3055 |
| few_valued | good | 26 | +0.0000 | tree-0 | 6 | 2000 | 0 | 0.3515 |
| few_valued | good | 27 | +0.0000 | tree-0 | 5 | 2000 | 0 | 0.3005 |
| few_valued | good | 28 | +0.0319 | tree-0 | 6 | 2000 | 0 | 0.353 |
| few_valued | good | 29 | +0.0146 | linear-0 | 6 | 2000 | 0 | 0.3535 |
| few_valued | F12 | 0 | -1.0000 | tree-0 | 10 | 2000 | 0 | 0.2935 |
| few_valued | F12 | 1 | -1.0000 | tree-0 | 9 | 2000 | 0 | 0.272 |
| few_valued | F12 | 2 | -0.9343 | tree-0 | 9 | 2000 | 0 | 0.274 |
| few_valued | F12 | 3 | -0.6481 | linear-0 | 10 | 2000 | 0 | 0.298 |
| few_valued | F12 | 4 | -1.0000 | tree-0 | 9 | 2000 | 0 | 0.2745 |
| few_valued | F12 | 5 | -0.9626 | tree-0 | 10 | 2000 | 0 | 0.2945 |
| few_valued | F12 | 6 | -0.8319 | linear-0 | 10 | 2000 | 0 | 0.2805 |
| few_valued | F12 | 7 | -1.0000 | tree-0 | 10 | 2000 | 0 | 0.2865 |
| few_valued | F12 | 8 | -0.9549 | linear-0 | 9 | 2000 | 0 | 0.2795 |
| few_valued | F12 | 9 | -0.9496 | tree-0 | 9 | 2000 | 0 | 0.276 |
| few_valued | F12 | 10 | -0.9632 | tree-0 | 9 | 2000 | 0 | 0.274 |
| few_valued | F12 | 11 | -0.7493 | tree-0 | 10 | 2000 | 0 | 0.297 |
| few_valued | F12 | 12 | -0.8900 | tree-0 | 10 | 2000 | 0 | 0.2915 |
| few_valued | F12 | 13 | -0.7615 | tree-0 | 10 | 2000 | 0 | 0.293 |
| few_valued | F12 | 14 | -0.9661 | tree-0 | 9 | 2000 | 0 | 0.265 |
| few_valued | F12 | 15 | -0.8288 | linear-0 | 10 | 2000 | 0 | 0.2965 |
| few_valued | F12 | 16 | -0.9641 | tree-0 | 10 | 2000 | 0 | 0.2975 |
| few_valued | F12 | 17 | -0.7741 | tree-0 | 9 | 2000 | 0 | 0.2775 |
| few_valued | F12 | 18 | -0.8100 | tree-0 | 9 | 2000 | 0 | 0.2715 |
| few_valued | F12 | 19 | -0.9580 | linear-0 | 9 | 2000 | 0 | 0.2645 |
| few_valued | F12 | 20 | -0.8247 | tree-0 | 10 | 2000 | 0 | 0.292 |
| few_valued | F12 | 21 | -0.9619 | linear-0 | 10 | 2000 | 0 | 0.2965 |
| few_valued | F12 | 22 | -1.0000 | tree-0 | 9 | 2000 | 0 | 0.2775 |
| few_valued | F12 | 23 | -1.0000 | tree-0 | 8 | 2000 | 0 | 0.256 |
| few_valued | F12 | 24 | -0.9527 | tree-0 | 9 | 2000 | 0 | 0.2755 |
| few_valued | F12 | 25 | -0.9530 | tree-0 | 9 | 2000 | 0 | 0.27 |
| few_valued | F12 | 26 | -0.8340 | tree-0 | 10 | 2000 | 0 | 0.2895 |
| few_valued | F12 | 27 | -0.9413 | tree-0 | 9 | 2000 | 0 | 0.274 |
| few_valued | F12 | 28 | -0.8181 | tree-0 | 10 | 2000 | 0 | 0.293 |
| few_valued | F12 | 29 | -0.9538 | tree-0 | 10 | 2000 | 0 | 0.2925 |

**The tables tried** (`group_band.py explore`, each with a generator of its
own in the script, 30 data seeds). Where two columns cross, a linear model
softens the fault: the broken copy's best seed comes within reach of the good
band. The table chosen has a branch that alone says which combination a row
has (rows `p` and `q` are that shape).

| Table | Copy | skill min | mean | sd | max | stops | holdout units | interval reported | interval holds 0 |
|---|---|---|---|---|---|---|---|---|---|
| three columns of 4, 3 and 2 levels: 24 combinations, 1,500 rows | good | -0.130 | -0.004 | 0.062 | +0.157 | 0 | 7..8 | 0 of 30 | 0 of 0 |
| three columns of 4, 3 and 2 levels: 24 combinations, 1,500 rows | F12 | -1.000 | -0.607 | 0.192 | -0.217 | 0 | 12..14 | 0 of 30 | 0 of 0 |
| four columns of 4, 3, 3 and 2 levels: 72 combinations, 1,500 rows | good | -0.107 | -0.016 | 0.056 | +0.134 | 0 | 17..21 | 0 of 30 | 0 of 0 |
| four columns of 4, 3, 3 and 2 levels: 72 combinations, 1,500 rows | F12 | -0.686 | -0.371 | 0.126 | -0.129 | 0 | 34..41 | 30 of 30 | 14 of 30 |
| four columns of 5, 4, 3 and 3 levels: 180 combinations, 1,500 rows | good | -0.074 | +0.031 | 0.053 | +0.130 | 0 | 41..55 | 30 of 30 | 28 of 30 |
| four columns of 5, 4, 3 and 3 levels: 180 combinations, 1,500 rows | F12 | -0.405 | -0.240 | 0.082 | -0.043 | 0 | 83..97 | 30 of 30 | 18 of 30 |
| three columns of 6, 5 and 4 levels: 120 combinations, 1,800 rows | good | -0.094 | -0.004 | 0.057 | +0.103 | 0 | 30..34 | 30 of 30 | 29 of 30 |
| three columns of 6, 5 and 4 levels: 120 combinations, 1,800 rows | F12 | -0.612 | -0.327 | 0.114 | -0.179 | 0 | 58..68 | 30 of 30 | 13 of 30 |
| one column of 16 levels, 1,500 rows | good | +0.000 | +0.000 | 0.000 | +0.000 | 0 | 5..6 | 0 of 30 | 0 of 0 |
| one column of 16 levels, 1,500 rows | F12 | -1.000 | -0.870 | 0.140 | -0.563 | 0 | 8..10 | 0 of 30 | 0 of 0 |
| two columns of 8 and 2 levels: 16 combinations, 1,500 rows | good | -0.092 | +0.022 | 0.062 | +0.129 | 0 | 5..6 | 0 of 30 | 0 of 0 |
| two columns of 8 and 2 levels: 16 combinations, 1,500 rows | F12 | -1.000 | -0.803 | 0.177 | -0.367 | 0 | 8..10 | 0 of 30 | 0 of 0 |
| three columns of 3, 2 and 2 levels: 12 combinations, 1,500 rows | good | -0.143 | -0.003 | 0.062 | +0.109 | 0 | 4..5 | 0 of 30 | 0 of 0 |
| three columns of 3, 2 and 2 levels: 12 combinations, 1,500 rows | F12 | -1.000 | -0.607 | 0.255 | -0.115 | 0 | 6..8 | 0 of 30 | 0 of 0 |
| two columns of 4 and 3 levels: 12 combinations, 2,000 rows | good | -0.077 | +0.000 | 0.042 | +0.095 | 0 | 4..5 | 0 of 30 | 0 of 0 |
| two columns of 4 and 3 levels: 12 combinations, 2,000 rows | F12 | -1.000 | -0.588 | 0.239 | -0.127 | 0 | 6..8 | 0 of 30 | 0 of 0 |
| two columns of 5 and 4 levels: 20 combinations, 2,000 rows | good | -0.087 | +0.004 | 0.041 | +0.087 | 0 | 5..7 | 0 of 30 | 0 of 0 |
| two columns of 5 and 4 levels: 20 combinations, 2,000 rows | F12 | -1.000 | -0.726 | 0.191 | -0.324 | 0 | 10..12 | 0 of 30 | 0 of 0 |
| two columns of 8 and 2 levels: 16 combinations, 2,000 rows | good | -0.096 | -0.003 | 0.045 | +0.130 | 0 | 4..6 | 0 of 30 | 0 of 0 |
| two columns of 8 and 2 levels: 16 combinations, 2,000 rows | F12 | -1.000 | -0.766 | 0.168 | -0.370 | 0 | 8..10 | 0 of 30 | 0 of 0 |
| two columns of 10 and 2 levels: 20 combinations, 2,000 rows | good | -0.116 | -0.005 | 0.050 | +0.076 | 0 | 6..7 | 0 of 30 | 0 of 0 |
| two columns of 10 and 2 levels: 20 combinations, 2,000 rows | F12 | -1.000 | -0.753 | 0.164 | -0.406 | 0 | 11..12 | 0 of 30 | 0 of 0 |
| two columns of 12 and 2 levels: 24 combinations, 2,000 rows | good | -0.109 | -0.006 | 0.054 | +0.102 | 0 | 6..8 | 0 of 30 | 0 of 0 |
| two columns of 12 and 2 levels: 24 combinations, 2,000 rows | F12 | -1.000 | -0.730 | 0.173 | -0.256 | 0 | 12..14 | 0 of 30 | 0 of 0 |
| one column of 16 levels, 2,000 rows | good | +0.000 | +0.000 | 0.000 | +0.000 | 0 | 4..6 | 0 of 30 | 0 of 0 |
| one column of 16 levels, 2,000 rows | F12 | -1.000 | -0.902 | 0.094 | -0.638 | 0 | 8..10 | 0 of 30 | 0 of 0 |
| two columns of 6 and 2 levels: 12 combinations, 2,000 rows | good | -0.096 | -0.018 | 0.042 | +0.052 | 0 | 4..5 | 0 of 30 | 0 of 0 |
| two columns of 6 and 2 levels: 12 combinations, 2,000 rows | F12 | -1.000 | -0.725 | 0.207 | -0.118 | 0 | 7..8 | 0 of 30 | 0 of 0 |
| one column of 12 levels, 2,000 rows | good | +0.000 | +0.000 | 0.000 | +0.000 | 0 | 4..5 | 0 of 30 | 0 of 0 |
| one column of 12 levels, 2,000 rows | F12 | -1.000 | -0.849 | 0.122 | -0.628 | 0 | 7..8 | 0 of 30 | 0 of 0 |
| a region of 4 levels and a branch of 16, each branch in one region, 2,000 rows | good | -0.068 | +0.007 | 0.036 | +0.092 | 0 | 4..6 | 0 of 30 | 0 of 0 |
| a region of 4 levels and a branch of 16, each branch in one region, 2,000 rows | F12 | -1.000 | -0.884 | 0.110 | -0.620 | 0 | 8..10 | 0 of 30 | 0 of 0 |
| a region of 5 levels and a branch of 20, each branch in one region, 2,000 rows | good | -0.094 | -0.002 | 0.039 | +0.068 | 0 | 6..7 | 0 of 30 | 0 of 0 |
| a region of 5 levels and a branch of 20, each branch in one region, 2,000 rows | F12 | -1.000 | -0.884 | 0.113 | -0.577 | 0 | 10..12 | 0 of 30 | 0 of 0 |

## 4. The grouped holdout (D64, D66)

`grouped_holdout.py` holds both rules as plain functions: `as built`, the
units in an order drawn from the seed until they hold a quarter, and
`proposed`, the rule of D64. It checks them against the reference's own
function, as changed and with its switch F13:

```
`proposed` here and the reference's _split_grouped agree in 330 of 330 splits (holdout rows and folds, or the same stop)
`as built` here and the reference's _split_grouped as its broken copy F13 agree in 330 of 330 splits (holdout rows and folds, or the same stop)
```

The holdout's share of the rows over 30 seeds (seed *k*: the table from data
seed *k*, the split from seed *k*). "kept in training" and "is the holdout"
count the seeds on which the warning of that kind is raised (D66). *u* is the
unit of spec step 7, property 3.

| Table | Rule | min | mean | max | too_few_groups | warned: kept in training | warned: is the holdout | u* (rows) |
|---|---|---|---|---|---|---|---|---|
| equal: 400 entities of 5 rows | as built | 0.250 | 0.250 | 0.250 | 0 | n/a | n/a | 5 |
| equal: 400 entities of 5 rows | proposed | 0.250 | 0.250 | 0.250 | 0 | 0 | 0 | 5 |
| one entity of 30% of the rows | as built | 0.250 | 0.314 | 0.540 | 0 | n/a | n/a | 5 |
| one entity of 30% of the rows | proposed | 0.250 | 0.250 | 0.250 | 0 | 30 | 0 | 5 |
| one entity of 60% of the rows | as built | 0.250 | 0.453 | 0.792 | 0 | n/a | n/a | 5 |
| one entity of 60% of the rows | proposed | 0.250 | 0.250 | 0.250 | 0 | 30 | 0 | 5 |
| one entity of 74.5% of the rows | as built | 0.750 | 0.895 | 0.988 | 0 | n/a | n/a | 2 |
| one entity of 74.5% of the rows | proposed | 0.250 | 0.250 | 0.250 | 0 | 30 | 0 | 2 |
| one entity of 90% of the rows | as built | 0.902 | 0.948 | 0.988 | 4 | n/a | n/a | 1800 |
| one entity of 90% of the rows | proposed | 0.900 | 0.900 | 0.900 | 0 | 0 | 30 | 1800 |
| five entities of 12% each, the rest of 5 rows | as built | 0.250 | 0.288 | 0.367 | 0 | n/a | n/a | 5 |
| five entities of 12% each, the rest of 5 rows | proposed | 0.250 | 0.250 | 0.250 | 0 | 0 | 0 | 5 |
| eight large entities and no other | as built | 0.250 | 0.305 | 0.400 | 2 | n/a | n/a | 250 |
| eight large entities and no other | proposed | 0.250 | 0.259 | 0.275 | 0 | 0 | 0 | 250 |
| two of 40% each and five of 4% | as built | 0.400 | 0.414 | 0.440 | 13 | n/a | n/a | 800 |
| two of 40% each and five of 4% | proposed | 0.400 | 0.400 | 0.400 | 0 | 30 | 30 | 800 |
| one entity of exactly a quarter of the rows | as built | 0.250 | 0.278 | 0.472 | 0 | n/a | n/a | 5 |
| one entity of exactly a quarter of the rows | proposed | 0.250 | 0.250 | 0.250 | 0 | 30 | 0 | 5 |
| sizes drawn from a long-tailed law (1 to about 300 rows) | as built | 0.250 | 0.258 | 0.325 | 0 | n/a | n/a | 5..10 |
| sizes drawn from a long-tailed law (1 to about 300 rows) | proposed | 0.250 | 0.250 | 0.250 | 0 | 0 | 0 | 5..10 |
| classification: one entity of 40% of the rows, its label 1 | as built | 0.250 | 0.440 | 0.647 | 0 | n/a | n/a | (5, 5) |
| classification: one entity of 40% of the rows, its label 1 | proposed | 0.250 | 0.252 | 0.253 | 0 | 30 | 0 | (5, 5) |

```
per seed (seed 0, 1, ...): the holdout's share of the rows; `stop` is too_few_groups
equal: 400 entities of 5 rows | as built | .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500
equal: 400 entities of 5 rows | proposed | .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500
one entity of 30% of the rows | as built | .3700 .2500 .2500 .3800 .2500 .2500 .2500 .4125 .4625 .2500 .2500 .4425 .2500 .2500 .2500 .2500 .2500 .5400 .2500 .4225 .4625 .2500 .2500 .2500 .2500 .2500 .3275 .2500 .5325 .3150
one entity of 30% of the rows | proposed | .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500
one entity of 60% of the rows | as built | .2500 .2500 .7375 .6425 .2500 .7100 .2500 .2500 .7475 .2500 .2500 .2500 .6975 .2500 .2500 .2500 .6675 .2500 .7075 .6100 .2500 .2500 .6050 .2500 .7925 .6800 .7500 .2500 .6275 .6200
one entity of 60% of the rows | proposed | .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500
one entity of 74.5% of the rows | as built | .9250 .9460 .8850 .8370 .9810 .9880 .9360 .9550 .8330 .9570 .8170 .9200 .9480 .7720 .9840 .9620 .7600 .9300 .8620 .7500 .8780 .9480 .8220 .9400 .8930 .8800 .8650 .9280 .9550 .7790
one entity of 74.5% of the rows | proposed | .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500
one entity of 90% of the rows | as built | .9825 .9775 .9150 .9125 .9825 .9375 .9575 .9875 stop .9675 .9025 .9425 .9475 stop .9675 .9550 .9375 .9750 .9350 stop .9050 .9100 .9325 .9200 stop .9875 .9300 .9650 .9275 .9750
one entity of 90% of the rows | proposed | .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000 .9000
five entities of 12% each, the rest of 5 rows | as built | .2675 .2500 .2500 .2975 .2500 .3675 .3400 .2500 .3375 .2500 .3225 .3650 .3350 .3525 .2500 .2500 .2575 .2700 .2500 .2500 .3400 .2500 .2500 .2725 .2625 .3600 .3675 .2500 .2650 .2700
five entities of 12% each, the rest of 5 rows | proposed | .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500
eight large entities and no other | as built | .2750 .2500 .3750 .3750 .3250 .3750 .2500 .3000 .3000 .2500 .2500 .2750 .4000 .2500 .2750 .3250 .3750 .2750 stop .2750 .3250 .3250 .3000 stop .2750 .3250 .3250 .3000 .3250 .2750
eight large entities and no other | proposed | .2750 .2500 .2500 .2500 .2500 .2500 .2750 .2500 .2500 .2500 .2500 .2500 .2750 .2750 .2500 .2500 .2500 .2750 .2500 .2500 .2750 .2750 .2750 .2500 .2500 .2750 .2750 .2500 .2500 .2750
two of 40% each and five of 4% | as built | .4000 stop stop .4400 stop .4400 .4400 .4000 .4000 stop stop stop .4000 stop .4000 .4000 .4000 stop .4000 stop .4400 stop stop stop .4400 .4000 .4400 stop .4000 .4000
two of 40% each and five of 4% | proposed | .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000 .4000
one entity of exactly a quarter of the rows | as built | .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .4425 .2500 .2500 .4725 .2500 .4525 .2850 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .3025 .2500 .2500 .3800 .2500 .2500 .2500
one entity of exactly a quarter of the rows | proposed | .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500 .2500
sizes drawn from a long-tailed law (1 to about 300 rows) | as built | .2516 .2512 .2509 .2505 .2851 .2501 .2504 .2560 .2504 .2505 .2505 .2758 .2547 .2521 .2721 .2511 .2537 .2774 .2514 .2546 .2501 .2501 .2523 .2512 .2549 .2512 .3245 .2582 .2502 .2504
sizes drawn from a long-tailed law (1 to about 300 rows) | proposed | .2501 .2502 .2504 .2500 .2502 .2501 .2504 .2500 .2504 .2500 .2500 .2504 .2502 .2501 .2501 .2501 .2502 .2502 .2504 .2501 .2501 .2501 .2504 .2502 .2504 .2502 .2500 .2502 .2502 .2504
classification: one entity of 40% of the rows, its label 1 | as built | .4850 .6475 .2500 .2525 .5775 .2500 .2525 .4800 .2500 .6275 .6475 .4875 .4875 .2525 .6325 .2525 .4900 .6050 .2500 .6450 .5000 .2525 .6150 .6325 .5025 .2525 .2525 .2500 .4975 .6275
classification: one entity of 40% of the rows, its label 1 | proposed | .2525 .2525 .2500 .2525 .2525 .2500 .2525 .2525 .2500 .2525 .2500 .2525 .2525 .2525 .2500 .2525 .2525 .2525 .2500 .2525 .2500 .2525 .2525 .2525 .2525 .2525 .2525 .2500 .2525 .2525
```

```
gate dataset `entities`, 30 data seeds, tool seed 0: holdout and folds identical, row for row, under both rules in 30 of 30; holdout share 0.2500 to 0.2533
```

## 5. Dates written as whole numbers (D66; section D, candidate 1)

`candidates.py dates`. The rule: every non-missing value of a column other
than the target is a whole number that reads as a calendar date written
`YYYYMMDD`, years 1900 to 2100, and at least two differ. On every dataset of
the gate at 30 data seeds, on `examples\customers.csv`, and on sixteen tables
made for the purpose at seeds 0 to 29. The rule gives the same answer on every
seed of every table, so there is one row for each table.

| Table | data seeds | columns looked at | whole-number columns | flagged |
|---|---|---|---|---|
| gate `signal_clf` | 30 | 210 | 0 | none |
| gate `signal_reg` | 30 | 180 | 0 | none |
| gate `noise_clf` | 30 | 9000 | 0 | none |
| gate `noise_reg` | 30 | 9000 | 0 | none |
| gate `imbalanced` | 30 | 210 | 0 | none |
| gate `leak` | 30 | 300 | 0 | none |
| gate `leak_reg` | 30 | 210 | 0 | none |
| gate `binary_numeric` | 30 | 210 | 0 | none |
| gate `ambiguous_target` | 30 | 90 | 0 | none |
| gate `messy` | 30 | 360 | 60 | none |
| gate `time_blocks` | 30 | 270 | 240 | none |
| gate `time_dayfirst` | 30 | 270 | 240 | none |
| gate `random_walk` | 30 | 60 | 0 | none |
| gate `noise_small` | 30 | 300 | 0 | none |
| gate `signal_clf_small` | 30 | 210 | 0 | none |
| gate `signal_reg_small` | 30 | 180 | 0 | none |
| gate `near_leak` | 30 | 240 | 0 | none |
| gate `long_rows` | 30 | 210 | 0 | none |
| gate `flat_tail` | 30 | 90 | 0 | none |
| gate `dup_noise` | 30 | 300 | 0 | none |
| gate `int_stamps` | 30 | 60 | 30 | none |
| gate `entities` | 30 | 210 | 0 | none |
| gate `panel` | 30 | 90 | 0 | none |
| gate `few_valued` | 30 | 60 | 0 | none |
| gate `dominant_entity` | 30 | 150 | 0 | none |
| gate `day_numbers` (added by this pass for the rule: its column `day` is dates) | 30 | 150 | 30 | day |
| examples/customers.csv | - | 7 | 2 | none |

flagged on the gate's other datasets and the example table: 0; on `day_numbers`, the one column that is dates

| Table made for the purpose | should flag | flagged, over the seeds | verdict |
|---|---|---|---|
| finding 4: the day as a whole number, three rows a day | day | day | as it should |
| the same, as floats with empty cells | day | day | as it should |
| the same, as digit text | day | day | as it should |
| one row a day, in order (id-like today) | day | day | as it should |
| month ends over 40 years, shuffled | day | day | as it should |
| two date columns (opened, closed) | opened, closed | opened, closed | as it should |
| eight-digit identifiers drawn at random | none | none | as it should |
| identifiers counting up from 20230101 | none | none | as it should |
| whole numbers drawn at random between 19000101 and 21001231 | none | none | as it should |
| years only (1990 to 2024) | none | none | as it should |
| year and month as YYYYMM | none | none | as it should |
| seconds since 1970 | none | none | as it should |
| days since 1970 | none | none | as it should |
| prices in whole cents, counts, ages, post codes | none | none | as it should |
| dates as YYYYMMDD with one cell of 20231301 | none | none | as it should |
| dates as DDMMYYYY whole numbers | none | none | as it should |

tables made for the purpose on which the rule is wrong: 0 of 16
a whole number drawn at random between 19000101 and 21001231 reads as a date with probability 0.0369; a column of 100 such values all read as dates with probability about 4.6e-144

## 6. A column whose levels are series (section D, candidate 2; not built, D66)

`candidates.py series`. For every text or whole-number column whose levels
could be series (two levels or more, four rows a level on average, no level
with two rows at one timestamp): the squared error of the previous value
within a level over the squared error of the table's own baseline, on the
training rows, as 1 minus that ratio ("skill"); and the same against each
level's own training mean ("own mean"). The candidate as stated flags a
column at a cut-off on the first; the stricter form asks 0.5 on both.

| Table | column | levels | skill, min..max over the seeds | own mean, min..max | flagged at 0.5 | flagged at 0.9 | flagged at 0.95 | flagged at 0.98 | flagged at 0.5 on both |
|---|---|---|---|---|---|---|---|---|---|
| gate `random_walk` (--time-col date) | no column whose levels could be series | | | | | | | | |
| gate `flat_tail` (--time-col date) | no column whose levels could be series | | | | | | | | |
| gate `int_stamps` (--time-col t) | no column whose levels could be series | | | | | | | | |
| gate `panel` (--time-col date) | series | 12 | +0.993..+0.999 | +0.886..+0.950 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |

not in scope (no regression under a time column in the gate's own run): signal_clf, signal_reg, noise_clf, noise_reg, imbalanced, leak, leak_reg, binary_numeric, ambiguous_target, messy, time_blocks, time_dayfirst, noise_small, signal_clf_small, signal_reg_small, near_leak, long_rows, dup_noise, entities, few_valued, dominant_entity, day_numbers

| Table | column | levels | skill | own mean | flagged at 0.5 | flagged at 0.9 | flagged at 0.95 | flagged at 0.98 | flagged at 0.5 on both |
|---|---|---|---|---|---|---|---|---|---|
| examples/customers.csv, target monthly_spend, time column signup_date | age | 62 | -1.106..-1.106 | -1.503..-1.503 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 |
| examples/customers.csv, target monthly_spend, time column signup_date | plan | 3 | -0.050..-0.050 | -0.978..-0.978 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 |
| examples/customers.csv, target monthly_spend, time column signup_date | region | 4 | -1.062..-1.062 | -1.074..-1.074 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 |
| examples/customers.csv, target monthly_spend, time column signup_date | support_calls | 10 | -1.003..-1.003 | -1.067..-1.067 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 |
| examples/customers.csv, target age, time column signup_date | plan | 3 | -1.027..-1.027 | -1.038..-1.038 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 |
| examples/customers.csv, target age, time column signup_date | region | 4 | -1.046..-1.046 | -1.048..-1.048 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 |
| examples/customers.csv, target age, time column signup_date | support_calls | 10 | -1.133..-1.133 | -1.234..-1.234 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 |

| Table made for the purpose | column | levels | skill, min..max | own mean, min..max | flagged at 0.5 | flagged at 0.9 | flagged at 0.95 | flagged at 0.98 | flagged at 0.5 on both |
|---|---|---|---|---|---|---|---|---|---|
| finding 3: twelve walks, stamps offset by one second [should flag: series] | series | 12 | +0.992..+0.999 | +0.894..+0.967 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| twelve walks on the same stamps [should flag: series] | series | 12 | +0.996..+0.999 | +0.901..+0.956 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| finding 2: 150 walks of 12 stamps [should flag: series] | series | 150 | +0.997..+0.998 | +0.178..+0.344 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 0 of 30 |
| 40 walks of 8 stamps [should flag: series] | series | 40 | +0.995..+0.998 | -0.468..+0.014 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 0 of 30 |
| 40 walks of 36 stamps [should flag: series] | series | 40 | +0.996..+0.999 | +0.701..+0.824 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| twelve walks, the series named by a whole number [should flag: series] | series | 12 | +0.995..+0.999 | +0.883..+0.967 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| twelve series that return to their own level (0.9 a step) [should flag: series] | series | 12 | +0.992..+0.999 | +0.672..+0.810 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| twelve series that return quickly (0.5 a step) [should flag: series] | series | 12 | +0.991..+0.998 | -0.148..+0.070 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 0 of 30 |
| one random walk with a weekday, a month, a segment and a rating [should flag: none] | month | 12 | -30.418..-1.001 | +0.847..+0.946 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| one random walk with a weekday, a month, a segment and a rating [should flag: none] | rating | 5 | -5.235..-3.051 | +0.764..+0.984 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| one random walk with a weekday, a month, a segment and a rating [should flag: none] | segment | 4 | -3.698..-2.310 | +0.795..+0.988 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| one random walk with a weekday, a month, a segment and a rating [should flag: none] | weekday | 5 | -4.888..-3.222 | +0.753..+0.984 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| independent rows, a segment that sets the level (4 sd apart) [should flag: none] | segment | 4 | +0.889..+0.920 | -1.118..-0.792 | 30 of 30 | 25 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| one series with a strong weekly pattern and independent noise [should flag: none] | weekday | 5 | +0.846..+0.895 | -1.177..-0.819 | 30 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| one slow random walk under a strong weekly pattern [should flag: none] | weekday | 5 | +0.987..+0.991 | +0.753..+0.984 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| independent rows, a segment that sets the level (20 sd apart) [should flag: none] | segment | 4 | +0.996..+0.997 | -1.233..-0.725 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 0 of 30 |
| eight stores a day, the target independent of the store [should flag: none] | store | 8 | -1.182..-0.848 | -1.191..-0.872 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| eight stores a day, each store at a level of its own, no memory [should flag: none] | store | 8 | -0.161..+0.849 | -1.191..-0.872 | 23 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |

```
per seed (data seed 0, 1, ... for the tables made here; base seed 1115, 1215, ... for the gate's): the ratio against the table's own baseline, then against each level's own mean
gate `panel` (--time-col date) | series | table: +0.998 +0.998 +0.996 +0.998 +0.996 +0.998 +0.998 +0.993 +0.998 +0.998 +0.998 +0.998 +0.997 +0.998 +0.998 +0.999 +0.997 +0.999 +0.997 +0.998 +0.997 +0.999 +0.997 +0.998 +0.997 +0.997 +0.995 +0.995 +0.998 +0.998
gate `panel` (--time-col date) | series | own mean: +0.903 +0.939 +0.950 +0.940 +0.889 +0.950 +0.944 +0.912 +0.928 +0.924 +0.940 +0.886 +0.927 +0.904 +0.903 +0.927 +0.914 +0.916 +0.946 +0.945 +0.943 +0.941 +0.899 +0.923 +0.928 +0.930 +0.944 +0.923 +0.937 +0.937
examples/customers.csv, target monthly_spend, time column signup_date | age | table: -1.106
examples/customers.csv, target monthly_spend, time column signup_date | age | own mean: -1.503
examples/customers.csv, target monthly_spend, time column signup_date | plan | table: -0.050
examples/customers.csv, target monthly_spend, time column signup_date | plan | own mean: -0.978
examples/customers.csv, target monthly_spend, time column signup_date | region | table: -1.062
examples/customers.csv, target monthly_spend, time column signup_date | region | own mean: -1.074
examples/customers.csv, target monthly_spend, time column signup_date | support_calls | table: -1.003
examples/customers.csv, target monthly_spend, time column signup_date | support_calls | own mean: -1.067
examples/customers.csv, target age, time column signup_date | plan | table: -1.027
examples/customers.csv, target age, time column signup_date | plan | own mean: -1.038
examples/customers.csv, target age, time column signup_date | region | table: -1.046
examples/customers.csv, target age, time column signup_date | region | own mean: -1.048
examples/customers.csv, target age, time column signup_date | support_calls | table: -1.133
examples/customers.csv, target age, time column signup_date | support_calls | own mean: -1.234
finding 3: twelve walks, stamps offset by one second [should flag: series] | series | table: +0.996 +0.997 +0.996 +0.998 +0.997 +0.998 +0.992 +0.996 +0.998 +0.996 +0.998 +0.997 +0.998 +0.999 +0.999 +0.998 +0.999 +0.998 +0.997 +0.998 +0.998 +0.998 +0.993 +0.997 +0.998 +0.997 +0.997 +0.996 +0.997 +0.998
finding 3: twelve walks, stamps offset by one second [should flag: series] | series | own mean: +0.949 +0.940 +0.911 +0.928 +0.894 +0.949 +0.936 +0.948 +0.947 +0.924 +0.955 +0.925 +0.898 +0.920 +0.947 +0.920 +0.945 +0.956 +0.959 +0.944 +0.967 +0.915 +0.935 +0.939 +0.946 +0.918 +0.937 +0.916 +0.944 +0.942
twelve walks on the same stamps [should flag: series] | series | table: +0.998 +0.999 +0.998 +0.999 +0.997 +0.998 +0.997 +0.998 +0.999 +0.998 +0.997 +0.998 +0.998 +0.998 +0.998 +0.997 +0.997 +0.997 +0.996 +0.997 +0.998 +0.996 +0.998 +0.997 +0.998 +0.997 +0.998 +0.998 +0.996 +0.998
twelve walks on the same stamps [should flag: series] | series | own mean: +0.946 +0.927 +0.934 +0.911 +0.942 +0.943 +0.942 +0.929 +0.923 +0.939 +0.939 +0.943 +0.933 +0.906 +0.956 +0.950 +0.908 +0.935 +0.927 +0.901 +0.932 +0.926 +0.918 +0.925 +0.938 +0.938 +0.936 +0.930 +0.948 +0.931
finding 2: 150 walks of 12 stamps [should flag: series] | series | table: +0.997 +0.997 +0.998 +0.997 +0.998 +0.998 +0.998 +0.998 +0.998 +0.998 +0.997 +0.998 +0.998 +0.997 +0.998 +0.997 +0.997 +0.997 +0.998 +0.997 +0.997 +0.997 +0.998 +0.997 +0.998 +0.997 +0.998 +0.998 +0.998 +0.998
finding 2: 150 walks of 12 stamps [should flag: series] | series | own mean: +0.337 +0.237 +0.344 +0.300 +0.243 +0.217 +0.254 +0.208 +0.180 +0.248 +0.225 +0.326 +0.316 +0.178 +0.232 +0.260 +0.255 +0.314 +0.274 +0.243 +0.263 +0.268 +0.320 +0.245 +0.308 +0.321 +0.243 +0.236 +0.272 +0.188
40 walks of 8 stamps [should flag: series] | series | table: +0.997 +0.997 +0.998 +0.998 +0.996 +0.997 +0.998 +0.996 +0.997 +0.998 +0.997 +0.997 +0.997 +0.998 +0.997 +0.997 +0.997 +0.998 +0.998 +0.998 +0.998 +0.997 +0.997 +0.997 +0.995 +0.995 +0.997 +0.997 +0.997 +0.997
40 walks of 8 stamps [should flag: series] | series | own mean: -0.215 -0.232 -0.463 -0.468 -0.223 -0.076 -0.248 +0.014 -0.084 -0.256 -0.169 -0.116 -0.391 -0.312 -0.313 -0.084 -0.145 -0.055 -0.205 -0.233 -0.308 -0.264 -0.145 -0.299 -0.236 -0.302 -0.449 -0.224 -0.248 -0.018
40 walks of 36 stamps [should flag: series] | series | table: +0.998 +0.997 +0.998 +0.998 +0.997 +0.998 +0.998 +0.997 +0.998 +0.996 +0.998 +0.997 +0.997 +0.998 +0.999 +0.997 +0.997 +0.997 +0.998 +0.998 +0.997 +0.998 +0.997 +0.998 +0.998 +0.997 +0.996 +0.997 +0.998 +0.997
40 walks of 36 stamps [should flag: series] | series | own mean: +0.742 +0.794 +0.746 +0.792 +0.824 +0.756 +0.777 +0.747 +0.802 +0.792 +0.743 +0.760 +0.811 +0.764 +0.760 +0.767 +0.735 +0.742 +0.775 +0.701 +0.768 +0.796 +0.742 +0.761 +0.735 +0.777 +0.765 +0.787 +0.776 +0.738
twelve walks, the series named by a whole number [should flag: series] | series | table: +0.997 +0.996 +0.999 +0.995 +0.996 +0.997 +0.996 +0.998 +0.998 +0.998 +0.997 +0.997 +0.997 +0.996 +0.997 +0.998 +0.997 +0.998 +0.998 +0.997 +0.996 +0.998 +0.997 +0.997 +0.998 +0.997 +0.997 +0.998 +0.997 +0.999
twelve walks, the series named by a whole number [should flag: series] | series | own mean: +0.929 +0.949 +0.943 +0.934 +0.924 +0.912 +0.962 +0.937 +0.964 +0.912 +0.928 +0.883 +0.935 +0.952 +0.931 +0.937 +0.894 +0.916 +0.917 +0.941 +0.953 +0.916 +0.933 +0.936 +0.946 +0.903 +0.944 +0.967 +0.909 +0.933
twelve series that return to their own level (0.9 a step) [should flag: series] | series | table: +0.997 +0.992 +0.997 +0.996 +0.997 +0.994 +0.998 +0.998 +0.995 +0.996 +0.996 +0.994 +0.997 +0.997 +0.997 +0.998 +0.994 +0.997 +0.998 +0.997 +0.998 +0.996 +0.994 +0.997 +0.998 +0.996 +0.999 +0.998 +0.997 +0.997
twelve series that return to their own level (0.9 a step) [should flag: series] | series | own mean: +0.672 +0.722 +0.786 +0.753 +0.793 +0.725 +0.724 +0.736 +0.769 +0.795 +0.681 +0.711 +0.730 +0.716 +0.737 +0.768 +0.682 +0.771 +0.731 +0.810 +0.682 +0.761 +0.779 +0.734 +0.765 +0.760 +0.757 +0.737 +0.787 +0.780
twelve series that return quickly (0.5 a step) [should flag: series] | series | table: +0.995 +0.991 +0.994 +0.997 +0.996 +0.998 +0.997 +0.994 +0.993 +0.994 +0.998 +0.994 +0.996 +0.997 +0.995 +0.996 +0.993 +0.994 +0.995 +0.995 +0.996 +0.998 +0.998 +0.995 +0.995 +0.997 +0.998 +0.997 +0.995 +0.992
twelve series that return quickly (0.5 a step) [should flag: series] | series | own mean: -0.028 -0.037 -0.019 -0.015 +0.070 +0.005 -0.032 -0.106 -0.148 -0.050 +0.029 +0.022 +0.012 -0.088 -0.029 -0.011 -0.015 -0.084 -0.057 -0.014 -0.040 +0.009 -0.068 -0.077 -0.012 -0.023 -0.010 +0.036 -0.033 -0.006
one random walk with a weekday, a month, a segment and a rating [should flag: none] | month | table: -18.777 -3.108 -16.377 -30.418 -5.818 -6.473 -7.007 -4.385 -8.496 -1.801 -7.158 -4.507 -2.728 -2.599 -7.896 -1.001 -2.075 -2.865 -2.913 -4.623 -9.987 -3.945 -1.716 -1.102 -2.295 -2.633 -10.877 -9.932 -6.861 -8.179
one random walk with a weekday, a month, a segment and a rating [should flag: none] | month | own mean: +0.934 +0.876 +0.925 +0.881 +0.920 +0.913 +0.939 +0.927 +0.930 +0.915 +0.937 +0.932 +0.910 +0.921 +0.924 +0.847 +0.859 +0.916 +0.877 +0.927 +0.937 +0.909 +0.930 +0.858 +0.879 +0.863 +0.878 +0.931 +0.941 +0.946
one random walk with a weekday, a month, a segment and a rating [should flag: none] | rating | table: -4.045 -3.920 -5.235 -4.675 -3.863 -4.030 -3.774 -4.937 -3.346 -3.278 -3.695 -4.382 -3.852 -3.573 -3.735 -3.509 -3.051 -4.341 -3.863 -3.959 -3.729 -3.129 -4.290 -3.231 -3.265 -4.609 -4.358 -4.270 -3.723 -4.434
one random walk with a weekday, a month, a segment and a rating [should flag: none] | rating | own mean: +0.984 +0.859 +0.974 +0.979 +0.947 +0.949 +0.965 +0.926 +0.968 +0.889 +0.965 +0.940 +0.902 +0.908 +0.960 +0.830 +0.837 +0.902 +0.902 +0.947 +0.975 +0.928 +0.886 +0.764 +0.852 +0.827 +0.950 +0.971 +0.968 +0.969
one random walk with a weekday, a month, a segment and a rating [should flag: none] | segment | table: -2.932 -2.948 -3.698 -3.289 -2.823 -2.779 -2.831 -3.542 -2.813 -2.321 -2.880 -3.289 -3.084 -2.807 -2.773 -2.403 -2.506 -3.250 -2.670 -2.772 -2.587 -2.310 -3.003 -2.593 -2.697 -2.793 -3.283 -3.020 -2.815 -3.567
one random walk with a weekday, a month, a segment and a rating [should flag: none] | segment | own mean: +0.988 +0.887 +0.980 +0.984 +0.959 +0.962 +0.972 +0.944 +0.972 +0.914 +0.971 +0.953 +0.918 +0.924 +0.968 +0.873 +0.859 +0.922 +0.926 +0.960 +0.981 +0.943 +0.914 +0.795 +0.875 +0.883 +0.960 +0.978 +0.974 +0.974
one random walk with a weekday, a month, a segment and a rating [should flag: none] | weekday | table: -4.039 -4.131 -4.720 -4.455 -4.123 -4.110 -3.863 -4.888 -3.722 -3.453 -3.910 -4.249 -4.268 -3.770 -3.941 -3.658 -3.282 -4.248 -3.710 -3.872 -3.828 -3.222 -3.709 -3.435 -3.362 -3.956 -4.010 -4.272 -4.190 -4.208
one random walk with a weekday, a month, a segment and a rating [should flag: none] | weekday | own mean: +0.984 +0.855 +0.976 +0.980 +0.945 +0.949 +0.965 +0.927 +0.966 +0.884 +0.964 +0.943 +0.894 +0.903 +0.958 +0.826 +0.830 +0.904 +0.905 +0.948 +0.975 +0.927 +0.899 +0.753 +0.849 +0.848 +0.954 +0.971 +0.965 +0.971
independent rows, a segment that sets the level (4 sd apart) [should flag: none] | segment | table: +0.904 +0.908 +0.899 +0.903 +0.911 +0.913 +0.911 +0.900 +0.907 +0.916 +0.901 +0.901 +0.906 +0.899 +0.905 +0.902 +0.889 +0.904 +0.904 +0.910 +0.907 +0.912 +0.920 +0.904 +0.894 +0.906 +0.902 +0.908 +0.909 +0.908
independent rows, a segment that sets the level (4 sd apart) [should flag: none] | segment | own mean: -1.107 -1.055 -0.921 -1.092 -0.945 -0.958 -0.981 -1.019 -1.029 -0.998 -1.047 -1.052 -1.007 -0.982 -1.002 -0.929 -1.109 -0.954 -1.071 -1.105 -0.967 -0.877 -0.792 -1.061 -1.118 -1.038 -0.941 -1.008 -1.031 -0.954
one series with a strong weekly pattern and independent noise [should flag: none] | weekday | table: +0.855 +0.858 +0.865 +0.846 +0.860 +0.866 +0.872 +0.874 +0.857 +0.895 +0.859 +0.862 +0.866 +0.877 +0.861 +0.870 +0.870 +0.863 +0.881 +0.871 +0.853 +0.883 +0.873 +0.857 +0.873 +0.860 +0.874 +0.861 +0.861 +0.872
one series with a strong weekly pattern and independent noise [should flag: none] | weekday | own mean: -1.177 -1.002 -0.920 -1.158 -1.037 -0.993 -0.993 -0.992 -1.045 -0.886 -1.016 -1.025 -0.895 -0.948 -0.819 -1.002 -0.989 -0.962 -1.000 -0.990 -1.053 -0.963 -1.087 -1.163 -0.921 -1.041 -1.027 -1.082 -1.065 -0.992
one slow random walk under a strong weekly pattern [should flag: none] | weekday | table: +0.990 +0.987 +0.988 +0.989 +0.990 +0.988 +0.989 +0.987 +0.991 +0.990 +0.990 +0.989 +0.987 +0.987 +0.989 +0.988 +0.989 +0.987 +0.989 +0.989 +0.991 +0.990 +0.988 +0.988 +0.989 +0.988 +0.989 +0.990 +0.991 +0.990
one slow random walk under a strong weekly pattern [should flag: none] | weekday | own mean: +0.984 +0.855 +0.976 +0.980 +0.945 +0.949 +0.965 +0.927 +0.966 +0.884 +0.964 +0.943 +0.894 +0.903 +0.958 +0.826 +0.830 +0.904 +0.905 +0.948 +0.975 +0.927 +0.899 +0.753 +0.849 +0.848 +0.954 +0.971 +0.965 +0.971
independent rows, a segment that sets the level (20 sd apart) [should flag: none] | segment | table: +0.996 +0.996 +0.996 +0.997 +0.996 +0.996 +0.996 +0.996 +0.996 +0.996 +0.996 +0.996 +0.996 +0.996 +0.996 +0.996 +0.996 +0.997 +0.996 +0.996 +0.996 +0.996 +0.997 +0.996 +0.996 +0.996 +0.996 +0.996 +0.996 +0.996
independent rows, a segment that sets the level (20 sd apart) [should flag: none] | segment | own mean: -1.123 -1.233 -1.119 -0.725 -1.052 -1.037 -0.988 -0.961 -0.806 -1.133 -1.004 -0.920 -1.045 -0.926 -1.001 -0.958 -0.954 -0.815 -1.011 -1.052 -0.961 -1.047 -0.956 -0.878 -0.959 -1.072 -0.946 -0.974 -1.066 -1.039
eight stores a day, the target independent of the store [should flag: none] | store | table: -1.149 -1.092 -1.037 -1.182 -0.973 -1.038 -1.015 -0.955 -1.081 -0.930 -1.129 -1.021 -0.977 -1.037 -0.868 -0.947 -0.987 -0.848 -1.032 -1.035 -1.002 -0.986 -1.000 -0.883 -0.998 -1.117 -0.956 -1.065 -1.008 -0.848
eight stores a day, the target independent of the store [should flag: none] | store | own mean: -1.158 -1.101 -1.050 -1.191 -0.994 -1.054 -1.037 -0.978 -1.091 -0.946 -1.146 -1.035 -0.988 -1.060 -0.889 -0.968 -0.999 -0.891 -1.055 -1.051 -1.009 -0.998 -1.025 -0.898 -1.026 -1.146 -0.989 -1.085 -1.042 -0.872
eight stores a day, each store at a level of its own, no memory [should flag: none] | store | table: +0.509 +0.702 +0.577 -0.161 +0.732 +0.579 +0.461 +0.840 +0.778 +0.692 +0.659 +0.849 +0.396 +0.736 +0.671 +0.418 +0.642 +0.736 +0.622 +0.584 +0.471 +0.766 +0.756 +0.844 +0.608 +0.716 +0.735 +0.235 +0.488 +0.681
eight stores a day, each store at a level of its own, no memory [should flag: none] | store | own mean: -1.158 -1.101 -1.050 -1.191 -0.994 -1.054 -1.037 -0.978 -1.091 -0.946 -1.146 -1.035 -0.988 -1.060 -0.889 -0.968 -0.999 -0.891 -1.055 -1.051 -1.009 -0.998 -1.025 -0.898 -1.026 -1.146 -0.989 -1.085 -1.042 -0.872
```

The candidate as stated names every panel and also a segment that only sets
the level; the stricter form names no such segment and misses the panels of 8
and 12 timestamps and the series that return quickly. Neither was built.

## 7. The panel warning on short panels (the fifth review, finding 2)

`short_panels.py`. Random walks on the same timestamps, each at a level of its
own, under a time column with `group_col=False`. The warning is raised when
the lag-1 autocorrelation of the per-timestamp mean over the training
timestamps is above 0.5; every table here is a persistent panel.

| series | timestamps | training timestamps | warned | autocorrelation, min | median | max |
|---|---|---|---|---|---|---|
| 12 | 8 | 6 | 15 of 30 | -0.71 | +0.50 | +0.98 |
| 12 | 12 | 9 | 22 of 30 | +0.10 | +0.66 | +0.98 |
| 12 | 16 | 12 | 19 of 30 | -0.43 | +0.66 | +0.95 |
| 12 | 24 | 18 | 28 of 30 | +0.41 | +0.82 | +0.97 |
| 12 | 36 | 27 | 30 of 30 | +0.58 | +0.86 | +0.99 |
| 12 | 60 | 45 | 30 of 30 | +0.65 | +0.92 | +0.98 |
| 12 | 125 | 93 | 30 of 30 | +0.80 | +0.96 | +0.99 |
| 40 | 8 | 6 | 15 of 30 | -0.66 | +0.47 | +1.00 |
| 40 | 12 | 9 | 21 of 30 | -0.27 | +0.64 | +0.91 |
| 40 | 16 | 12 | 22 of 30 | +0.16 | +0.65 | +0.96 |
| 40 | 24 | 18 | 24 of 30 | +0.02 | +0.76 | +0.95 |
| 40 | 36 | 27 | 29 of 30 | +0.49 | +0.86 | +0.97 |
| 40 | 60 | 45 | 30 of 30 | +0.51 | +0.94 | +0.99 |
| 40 | 125 | 93 | 30 of 30 | +0.81 | +0.96 | +0.99 |
| 150 | 8 | 6 | 17 of 30 | -0.77 | +0.59 | +0.97 |
| 150 | 12 | 9 | 21 of 30 | -0.40 | +0.62 | +0.98 |
| 150 | 16 | 12 | 24 of 30 | -0.06 | +0.76 | +0.94 |
| 150 | 24 | 18 | 27 of 30 | +0.27 | +0.78 | +0.96 |
| 150 | 36 | 27 | 29 of 30 | +0.41 | +0.86 | +0.97 |
| 150 | 60 | 45 | 30 of 30 | +0.70 | +0.92 | +0.98 |
| 150 | 125 | 93 | 30 of 30 | +0.81 | +0.95 | +0.99 |

```
per seed: series | timestamps | the autocorrelation at seeds 0, 1, ...
12 | 8 | +0.88 -0.23 +0.98 +0.40 -0.71 +0.74 -0.09 +0.47 +0.09 -0.58 +0.79 +0.21 -0.06 -0.47 +0.93 +0.80 +0.79 +0.16 +0.53 +0.87 +0.92 -0.17 +0.83 +0.80 +0.96 +0.28 +0.84 -0.23 +0.68 +0.35
12 | 12 | +0.42 +0.71 +0.53 +0.42 +0.60 +0.98 +0.67 +0.85 +0.57 +0.57 +0.92 +0.68 +0.23 +0.33 +0.10 +0.86 +0.81 +0.14 +0.59 +0.77 +0.84 +0.76 +0.55 +0.67 +0.92 +0.40 +0.66 +0.66 +0.88 +0.17
12 | 16 | +0.49 +0.74 +0.80 +0.70 +0.86 +0.91 +0.65 +0.91 -0.09 +0.25 +0.95 +0.40 +0.88 +0.47 +0.04 +0.81 +0.23 +0.44 +0.86 +0.68 +0.94 +0.47 +0.73 +0.64 +0.77 -0.43 +0.67 +0.59 +0.34 +0.54
12 | 24 | +0.72 +0.92 +0.85 +0.53 +0.80 +0.50 +0.89 +0.95 +0.63 +0.64 +0.97 +0.79 +0.85 +0.74 +0.88 +0.84 +0.41 +0.75 +0.75 +0.64 +0.92 +0.88 +0.86 +0.91 +0.96 +0.42 +0.76 +0.91 +0.85 +0.54
12 | 36 | +0.88 +0.86 +0.67 +0.96 +0.67 +0.81 +0.86 +0.97 +0.69 +0.77 +0.99 +0.83 +0.90 +0.58 +0.89 +0.89 +0.89 +0.91 +0.84 +0.95 +0.80 +0.92 +0.91 +0.84 +0.95 +0.78 +0.80 +0.80 +0.68 +0.97
12 | 60 | +0.73 +0.97 +0.93 +0.95 +0.92 +0.95 +0.88 +0.97 +0.81 +0.97 +0.96 +0.91 +0.87 +0.89 +0.83 +0.93 +0.84 +0.97 +0.73 +0.94 +0.98 +0.83 +0.96 +0.78 +0.65 +0.93 +0.89 +0.95 +0.88 +0.79
12 | 125 | +0.92 +0.99 +0.98 +0.97 +0.80 +0.97 +0.89 +0.98 +0.98 +0.93 +0.97 +0.95 +0.93 +0.97 +0.95 +0.98 +0.96 +0.95 +0.94 +0.97 +0.99 +0.91 +0.93 +0.96 +0.82 +0.98 +0.80 +0.98 +0.96 +0.97
40 | 8 | +0.68 +0.93 -0.33 -0.01 +0.96 +0.39 +0.79 -0.24 +0.91 +0.41 +1.00 +0.25 +0.27 -0.66 -0.66 +0.28 +0.19 +0.76 +0.51 -0.23 +0.66 +0.23 +0.73 -0.32 +0.99 +0.67 +0.44 +0.89 +0.64 +0.73
40 | 12 | +0.33 +0.11 +0.86 +0.89 +0.36 -0.27 +0.76 +0.91 +0.62 +0.71 +0.91 +0.51 +0.91 +0.19 +0.77 +0.53 +0.26 +0.65 +0.17 +0.60 +0.70 +0.51 +0.59 +0.68 +0.86 +0.80 +0.90 +0.23 +0.76 +0.43
40 | 16 | +0.61 +0.83 +0.93 +0.96 +0.65 +0.72 +0.65 +0.95 +0.74 +0.43 +0.94 +0.79 +0.16 +0.48 +0.70 +0.62 +0.83 +0.75 +0.87 +0.21 +0.72 +0.44 +0.62 +0.29 +0.38 +0.77 +0.55 +0.64 +0.41 +0.60
40 | 24 | +0.87 +0.76 +0.78 +0.88 +0.46 +0.39 +0.02 +0.95 +0.73 +0.70 +0.92 +0.87 +0.81 +0.87 +0.58 +0.72 +0.71 +0.60 +0.64 +0.92 +0.92 +0.76 +0.65 +0.93 +0.50 +0.78 +0.44 +0.90 +0.45 +0.89
40 | 36 | +0.83 +0.91 +0.97 +0.96 +0.68 +0.71 +0.65 +0.49 +0.89 +0.91 +0.96 +0.80 +0.93 +0.79 +0.75 +0.96 +0.85 +0.70 +0.82 +0.86 +0.93 +0.94 +0.93 +0.88 +0.86 +0.97 +0.71 +0.88 +0.79 +0.71
40 | 60 | +0.94 +0.87 +0.94 +0.97 +0.95 +0.94 +0.89 +0.94 +0.95 +0.90 +0.99 +0.92 +0.80 +0.77 +0.88 +0.51 +0.92 +0.73 +0.95 +0.77 +0.99 +0.97 +0.90 +0.91 +0.96 +0.94 +0.95 +0.83 +0.94 +0.95
40 | 125 | +0.89 +0.97 +0.96 +0.97 +0.97 +0.98 +0.99 +0.98 +0.94 +0.95 +0.99 +0.93 +0.93 +0.95 +0.95 +0.93 +0.98 +0.98 +0.96 +0.93 +0.99 +0.96 +0.81 +0.98 +0.98 +0.97 +0.92 +0.97 +0.99 +0.93
150 | 8 | +0.01 +0.75 +0.86 +0.95 -0.05 +0.62 +0.30 -0.34 +0.97 +0.10 +0.62 +0.65 -0.20 -0.22 +0.61 -0.13 +0.58 +0.58 +0.44 +0.32 +0.90 -0.07 +0.66 +0.85 +0.87 +0.79 +0.74 +0.97 -0.43 -0.77
150 | 12 | +0.56 +0.94 +0.96 +0.72 +0.79 +0.51 +0.18 +0.72 +0.33 +0.46 +0.77 +0.77 +0.76 -0.40 +0.51 +0.81 +0.58 +0.65 +0.61 +0.71 +0.98 +0.64 +0.50 +0.27 +0.81 +0.30 +0.49 +0.90 +0.28 +0.41
150 | 16 | +0.59 +0.88 +0.76 +0.85 +0.60 +0.21 +0.59 +0.72 +0.82 +0.86 +0.94 +0.79 +0.21 +0.17 +0.76 +0.52 +0.75 +0.56 +0.08 +0.84 +0.90 +0.88 -0.06 +0.77 +0.44 +0.89 +0.61 +0.92 +0.52 +0.78
150 | 24 | +0.92 +0.75 +0.77 +0.44 +0.85 +0.93 +0.44 +0.60 +0.56 +0.27 +0.92 +0.65 +0.88 +0.58 +0.96 +0.86 +0.94 +0.78 +0.95 +0.67 +0.93 +0.78 +0.58 +0.76 +0.70 +0.88 +0.72 +0.95 +0.90 +0.66
150 | 36 | +0.84 +0.75 +0.57 +0.90 +0.83 +0.89 +0.85 +0.64 +0.93 +0.75 +0.92 +0.84 +0.86 +0.88 +0.96 +0.88 +0.87 +0.91 +0.70 +0.91 +0.97 +0.94 +0.65 +0.78 +0.91 +0.86 +0.83 +0.41 +0.91 +0.78
150 | 60 | +0.93 +0.92 +0.96 +0.88 +0.97 +0.98 +0.70 +0.95 +0.98 +0.90 +0.98 +0.94 +0.88 +0.88 +0.91 +0.82 +0.98 +0.88 +0.96 +0.84 +0.95 +0.87 +0.79 +0.92 +0.85 +0.90 +0.97 +0.93 +0.95 +0.95
150 | 125 | +0.99 +0.93 +0.97 +0.93 +0.93 +0.98 +0.97 +0.97 +0.84 +0.91 +0.91 +0.92 +0.97 +0.96 +0.96 +0.98 +0.95 +0.98 +0.96 +0.97 +0.87 +0.86 +0.98 +0.97 +0.92 +0.81 +0.92 +0.90 +0.98 +0.95
```

## 8. The gate: each new check seen failing, and what else moved

The changed gate against the reference as it stood (the copy), through
`docs\measurements\CP2\long_rows.py values`: 43 checks, of which the six new
ones fail and the 37 others pass.

```
43 checks, 6 failed, 34 runs, 27 s
  FAIL statements.time_order_asked: the tool exited 0 on `signal_clf`; it must stop with the named error `time_order_undeclared` (this run says nothing about time order and nothing about entities. `signal_clf` holds no date, and that ma
  FAIL statements.entities_asked: the tool exited 0 on `random_walk`; it must stop with the named error `entities_undeclared` (this run passes `--time-col date` and says nothing about entities. `random_walk` is one series, a row for e
  FAIL duplicates.group_on_features: 8 of the 16 combinations of `region` and `branch` have rows on both sides of the split (for example east / b11). With `--duplicates group` the rows that agree in every feature column are kept together
  FAIL duplicates.group_no_skill: skill on few_valued is -0.808219, expected between -0.2 and 0.2: the label is drawn at random, so nothing predicts it. Skill far below zero here means that the sets of `--duplicates group` were formed
  FAIL groups.holdout_bounded: the holdout of `dominant_entity` holds 1500 of the 2000 rows (75.0%). One entity holds 1490 rows and the 255 others 510, which is more than a quarter: the holdout is made of small entities, at least a
  FAIL time.date_numbers_seen: the column `day` of `day_numbers` has role 'feature' and reason None. Every value of it is a whole number that reads as a calendar date written YYYYMMDD (20150101, 20150102, ...): such a column is a d
```

The changed gate against the application as built by the fifth pass gives the
same six failures and no other:

```
43 checks, 6 failed, 34 runs, 27 s
  FAIL statements.time_order_asked: the tool exited 0 on `signal_clf`; it must stop with the named error `time_order_undeclared` (this run says nothing about time order and nothing about entities. `signal_clf` holds no date, and that ma
  FAIL statements.entities_asked: the tool exited 0 on `random_walk`; it must stop with the named error `entities_undeclared` (this run passes `--time-col date` and says nothing about entities. `random_walk` is one series, a row for e
  FAIL duplicates.group_on_features: 5 of the 16 combinations of `region` and `branch` have rows on both sides of the split (for example east / b03). With `--duplicates group` the rows that agree in every feature column are kept together
  FAIL duplicates.group_no_skill: skill on few_valued is -0.806726, expected between -0.2 and 0.2: the label is drawn at random, so nothing predicts it. Skill far below zero here means that the sets of `--duplicates group` were formed
  FAIL groups.holdout_bounded: the holdout of `dominant_entity` holds 1510 of the 2000 rows (75.5%). One entity holds 1490 rows and the 255 others 510, which is more than a quarter: the holdout is made of small entities, at least a
  FAIL time.date_numbers_seen: the column `day` of `day_numbers` has role 'feature' and reason None. Every value of it is a whole number that reads as a calendar date written YYYYMMDD (20150101, 20150102, ...): such a column is a d
```

The gate's self-test on the changed judge: 34 of 34
pairs of a check and the failure mode it names seen failing, over 21 broken
copies. Its proof table:

```
PROOF: every check that names a failure mode, run on that mode's own fixture
| Check | Names | Failed on that fixture | Also failed on the fixture of |
|---|---|---|---|
| data.rows_reconcile | F6 | F6: yes, 2 of 2 runs | - |
| data.columns_accounted | F6b | F6b: yes, 2 of 2 runs | F4 |
| data.coercion_reported | F6 | F6: yes, 2 of 2 runs | - |
| data.long_rows_stop | F6d | F6d: yes, 2 of 2 runs | - |
| data.holdout_copies_stop | F9 | F9: yes, 2 of 2 runs | - |
| models.all_accounted | F6c | F6c: yes, 2 of 2 runs | - |
| task.binary_numeric | F5 | F5: yes, 2 of 2 runs | F4 |
| task.ambiguous_stops | F5 | F5: yes, 2 of 2 runs | - |
| leak.excluded | F2 | F2: yes, 2 of 2 runs | F6 |
| leak.regression | F2 | F2: yes, 2 of 2 runs | - |
| leak.near_leak_warned | F2 | F2: yes, 2 of 2 runs | - |
| model.noise_clf | F1 | F1: yes, 2 of 2 runs | - |
| model.noise_p_value | F1 | F1: yes, 2 of 2 runs | - |
| model.noise_reg | F1 | F1: yes, 2 of 2 runs | - |
| model.headline_recomputed | F3, F4 | F3: yes, 2 of 2 runs; F4: yes, 2 of 2 runs | - |
| model.metric_clf | F4 | F4: yes, 2 of 2 runs | F3 |
| model.selection_full | F3 | F3: yes, 2 of 2 runs | F1, F4, F6c |
| time.order_respected | F7 | F7: yes, 2 of 2 runs | - |
| time.no_skill_on_blocks | F7 | F7: yes, 2 of 2 runs | F1 |
| time.gap_applied | F7 | F7: yes, 2 of 2 runs | - |
| time.dayfirst_stops | F7b | F7b: yes, 2 of 2 runs | - |
| time.persistence_baseline | F8 | F8: yes, 2 of 2 runs | F3, F7 |
| time.integer_keys_distinct | F7c | F7c: yes, 2 of 2 runs | F3, F7, F8 |
| time.undeclared_stops | F7d | F7d: yes, 2 of 2 runs | F11 |
| time.panel_previous_value | F8b | F8b: yes, 2 of 2 runs | F3, F7, F8 |
| groups.kept_together | F10 | F10: yes, 2 of 2 runs | - |
| groups.no_skill_on_entities | F10 | F10: yes, 2 of 2 runs | - |
| statements.time_order_asked | F11 | F11: yes, 2 of 2 runs | - |
| statements.entities_asked | F11 | F11: yes, 2 of 2 runs | - |
| duplicates.group_on_features | F12 | F12: yes, 2 of 2 runs | F10 |
| duplicates.group_no_skill | F12 | F12: yes, 2 of 2 runs | F10 |
| groups.holdout_bounded | F13 | F13: yes, 2 of 2 runs | F10 |
| time.date_numbers_seen | F14 | F14: yes, 2 of 2 runs | - |
34 of 34 pairs of a check and a failure mode it names were seen failing, over 21 broken fixtures.

FOR INFORMATION: checks that give a reason in place of a failure mode, and the broken fixtures they failed on (no part of the verdict)
| Check | Failed on the fixture of |
|---|---|
| leak.control_kept | F6 |
| model.signal_clf | none: never seen to fail here |
| model.signal_reg | none: never seen to fail here |
| time.undeclared_warns | none: never seen to fail here |
| time.flat_holdout_reported | F3, F7, F8 |
| uncertainty.reported | F1, F4 |
| result.matches_json | none: never seen to fail here |
| output.complete | none: never seen to fail here |
| output.deterministic | none: never seen to fail here |
| output.same_at_any_n_jobs | none: never seen to fail here |
SELFTEST: PASS (good fixture passes twice with identical results; 21 failure mode(s) each caught by a check that names it; temp and project dirs agree)
```

What moved at the gate seed between the judge as it stood and as changed
(`gate_moved.py` on the two files of `long_rows.py values`). No dataset, no
check and no recorded value that existed before moved; of the runs' fields,
the command lines, the two recorded statements, two messages, and the key
that D66 renamed.

```
datasets: 69 values before under 23 names; 0 moved, 0 gone, 0 new under an existing name; only after: day_numbers, dominant_entity, few_valued
checks: 102 values before under 10 names; 0 moved, 0 gone, 6 new under an existing name; only after: duplicates, statements
    new   date_numbers_seen.catches[0]: 1 run(s)
    new   date_numbers_seen.message: 1 run(s)
    new   date_numbers_seen.ok: 1 run(s)
    new   holdout_bounded.catches[0]: 1 run(s)
    new   holdout_bounded.message: 1 run(s)
    new   holdout_bounded.ok: 1 run(s)
records: 113 values before under 112 names; 0 moved, 0 gone, 0 new under an existing name; only after: day_numbers role of `day`, dominant_entity holdout rows, feature combinations on both sides, skill on few_valued, unsaid_both exit code, unsaid_entities exit code
runs: 15142 values before under 29 names; 69 moved, 25 gone, 25 new under an existing name; only after: day_numbers, dominant_entity, few_valued, unsaid_both, unsaid_entities
    moved  command: 27 run(s): ambiguous_target, binary_numeric, dup_noise, entities, flat_tail, full_noise_small, full_signal_clf_j1, full_signal_clf_j4, full_signal_reg_j1, full_signal_reg_j4, imbalanced, int_stamps, leak, leak_reg, long_rows, messy, near_leak, noise_clf, noise_reg, random_walk, signal_clf, signal_reg, time_blocks, time_blocks_gap, time_blocks_undeclared, time_blocks_unstated, time_dayfirst
    moved  profile.input.group_col: 22 run(s): api_signal_clf, binary_numeric, flat_tail, full_noise_small, full_signal_clf_j1, full_signal_clf_j4, full_signal_reg_j1, full_signal_reg_j4, imbalanced, int_stamps, leak, leak_reg, messy, near_leak, noise_clf, noise_reg, random_walk, signal_clf, signal_reg, time_blocks, time_blocks_gap, time_blocks_undeclared
           None -> False (22)
    moved  profile.input.time_col: 16 run(s): api_signal_clf, binary_numeric, entities, full_noise_small, full_signal_clf_j1, full_signal_clf_j4, full_signal_reg_j1, full_signal_reg_j4, imbalanced, leak, leak_reg, near_leak, noise_clf, noise_reg, signal_clf, signal_reg
           None -> False (16)
    moved  stderr: 2 run(s): dup_noise, time_blocks_unstated
           "error: holdout_copies_in_training: 277  -> "error: holdout_copies_in_training: 277  (1); "error: time_order_undeclared: column(s) -> "error: time_order_undeclared: Nothing w (1)
    moved  stdout.result.to_dict.input.group_col: 1 run(s): api_signal_clf
           None -> False (1)
    moved  stdout.result.to_dict.input.time_col: 1 run(s): api_signal_clf
           None -> False (1)
    gone   profile.validation.n_rows_with_copies: 24 run(s)
    gone   stdout.result.to_dict.validation.n_rows_with_copies: 1 run(s)
    new   profile.validation.n_rows_same_features: 24 run(s)
    new   stdout.result.to_dict.validation.n_rows_same_features: 1 run(s)
```

## 9. The harness after the change

`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B loop\run_loop.py doctor`:

```
Loop doctor
  ok   interpreter E:\CODE\ml-professor\mlprof_env\Scripts\python.exe (3.12.13)
  ok   hooks registered with an absolute interpreter path
  ok   Claude Code 2.1.330 (Claude Code)
       harness   -> fable   effort=xhigh  turns<=-, $-/run
       tests     -> opus    effort=high   turns<=150, $15/run
       review    -> fable   effort=high   turns<=80, $10/run
       diagnose  -> fable   effort=high   turns<=40, $5/run
       support   -> haiku   effort=medium turns<=60, $2/run
       routine   -> sonnet  effort=medium turns<=100, $6/run
       complex   -> opus    effort=high   turns<=200, $20/run
       total budget $250, permission mode 'auto'
  ok   pytest importable
  ok   gate lists 43 check(s)
  ok   gate self-test: SELFTEST: PASS (good fixture passes twice with identical results; 21 failure mode(s) each caught by a check that names it; temp and project dirs agree)
  ok   gate timing: 17.2s per edit, 26.3s in full
  ok   plan valid: 53 task(s) (37 complex, 9 routine, 7 support)
  WARN 12 uncommitted change(s) in the harness; commit before an unattended run
Doctor: OK, 1 warning(s)
```

`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B verify\smoke_test.py --matrix`:

```
| Failure mode | Description | Caught by |
|---|---|---|
| F1 | the chosen model is refitted on all rows before the holdout is scored | model.noise_clf, model.noise_p_value, model.noise_reg |
| F2 | the leak screen is removed, so a copy of the target is a feature | leak.excluded, leak.regression, leak.near_leak_warned |
| F3 | the headline is the best cross-validation score, not the holdout score | model.headline_recomputed, model.selection_full |
| F4 | the headline metric is accuracy | model.headline_recomputed, model.metric_clf |
| F5 | the task is decided from the column's type alone | task.binary_numeric, task.ambiguous_stops |
| F6 | rows with any missing value are dropped before counting | data.rows_reconcile, data.coercion_reported |
| F7 | the split is shuffled even when a time column is declared | time.order_respected, time.no_skill_on_blocks, time.gap_applied |
| F8 | the regression baseline is always the training mean | time.persistence_baseline |
| F6b | a column is lost from the accounting (constant columns are dropped with no record) | data.columns_accounted |
| F6c | a model family is lost from the models table (only the chosen configuration's family is listed) | models.all_accounted |
| F6d | data rows longer than the header are read silently, with no error | data.long_rows_stop |
| F7b | a day-first time column is parsed leniently, so the run goes ahead on the wrong order | time.dayfirst_stops |
| F7c | whole-number time keys are read as floats, so distinct instants above 2**53 become one timestamp | time.integer_keys_distinct |
| F9 | holdout rows with copies among the training rows are scored with no error | data.holdout_copies_stop |
| F10 | an entity has rows on both sides of the split although the entity column was declared | groups.kept_together, groups.no_skill_on_entities |
| F8b | a panel is scored against the training mean although the entity column was declared | time.panel_previous_value |
| F7d | a table with an undeclared date column is profiled under a random split without the caller's say | time.undeclared_stops |
| F11 | a statement the caller did not make is assumed: nothing said about time order, or about entities, and the run goes ahead | statements.time_order_asked, statements.entities_asked |
| F12 | duplicates='group' forms its sets with the target, so rows that agree in every feature are on both sides of the split | duplicates.group_on_features, duplicates.group_no_skill |
| F13 | the grouped holdout takes far more than a quarter because a large entity was drawn early | groups.holdout_bounded |
| F14 | dates written as whole numbers (YYYYMMDD) are a numeric feature | time.date_numbers_seen |
```

`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B loop\run_loop.py build --dry-run`:

```
[02:01:21] Dry run. Nothing will be executed.
[02:01:21]   T1     support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  Package skeleton, command-line stub, requirements.txt
[02:01:21]   T2     complex  opus x3                            gate=0 tests=1  Load one table; count usable rows
[02:01:21]   T3     complex  opus x3                            gate=0 tests=1  Column roles that need no target
[02:01:21]   T4     complex  opus x3                            gate=0 tests=1  Task decision
[02:01:21]   T5     complex  opus x3                            gate=0 tests=1  Random holdout and folds
[02:01:21]   T6     complex  opus x3                            gate=0 tests=1  Time column, time-ordered holdout, expanding folds, gap
[02:01:21]   T7     complex  opus x3                            gate=0 tests=1  Metrics, skill and the previous-value baseline
[02:01:21]   T8     complex  opus x3                            gate=0 tests=1  Preprocessing fitted on the rows it is given
[02:01:21]   T9     complex  opus x3                            gate=0 tests=1  Fit, select, score once: the run in memory, random scheme, fast profile
[02:01:21]   T10    complex  opus x3                            gate=12 tests=1  profile.json, predictions.csv and the command line: the first run the gate can judge
[02:01:21]   T11    complex  opus x3                            gate=5 tests=1  Time-ordered scheme wired into the run, with its baselines
[02:01:21]   T12    complex  opus x3                            gate=4 tests=1  Leak screen
[02:01:21]   T13    complex  opus x3                            gate=2 tests=1  Interval and p-value, row and block forms
[02:01:21]   T14    routine  sonnet x2 -> opus x3               gate=0 tests=1  Random forest and histogram gradient boosting behind the existing family interface
[02:01:21]   T15    complex  opus x3                            gate=2 tests=1  The full profile's grids, and the same result at any n_jobs
[02:01:21]   T16    routine  sonnet x2 -> opus x3               gate=1 tests=1  Edges, limitations statement, console summary
[02:01:21]   T17    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  README with run instructions
[02:01:21]   T18    complex  opus x3                            gate=0 tests=4  Guards for the first review's test holes, and two CSV edges: no code change expected
[02:01:21]   T19    complex  opus x3                            gate=1 tests=1  Skill when a baseline's error is zero
[02:01:21]   T20    routine  sonnet x2 -> opus x3               gate=0 tests=1  Colliding feature names stop the run
[02:01:21]   T21    complex  opus x3                            gate=0 tests=1  A configuration that fails after it is chosen
[02:01:21]   T22    routine  sonnet x2 -> opus x3               gate=0 tests=2  Small defects: the output folder, process settings, the no-features message, the console summary
[02:01:21]   T23    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README's commands use a table kept in the repo
[02:01:21]   T24    complex  opus x3                            gate=0 tests=1  Guard of the third pass: a line pandas reads as the header. No code change expected
[02:01:21]   T25    complex  opus x3                            gate=0 tests=1  No number that is not finite is reported, or turned into null
[02:01:21]   T26    complex  opus x3                            gate=0 tests=1  White space round a target value; a target cell of only white space
[02:01:21]   T27    routine  sonnet x2 -> opus x3               gate=0 tests=1  The seed's range
[02:01:21]   T28    routine  sonnet x2 -> opus x3               gate=0 tests=1  The fast profile's error only when the full profile would run
[02:01:21]   T29    routine  sonnet x2 -> opus x3               gate=0 tests=1  The console summary prints four significant digits
[02:01:21]   T30    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: which rows the fast profile keeps, and what this pass changed
[02:01:21]   T31    complex  opus x3                            gate=0 tests=1  Guards of the fourth pass: six pinned behaviours the third review could break unseen. No code change expected
[02:01:21]   T32    complex  opus x3                            gate=1 tests=1  Copies of a holdout row among the training rows stop the run; the `duplicates` argument
[02:01:21]   T33    complex  opus x3                            gate=1 tests=1  Whole numbers are compared exactly: integer time keys, distinct values, and whole numbers too large for 64 bits
[02:01:21]   T34    complex  opus x3                            gate=0 tests=1  Number columns as a database driver returns them; a warning for every unsupported column
[02:01:21]   T35    complex  opus x3                            gate=0 tests=1  The type matrix: every kind of column, in every role, ends in a result or a named error
[02:01:21]   T36    routine  sonnet x2 -> opus x3               gate=0 tests=1  One to nine digits in a fraction of a second; a refusal that says what is wrong
[02:01:21]   T37    routine  sonnet x2 -> opus x3               gate=0 tests=1  The fast profile's cap leaving too few rows: its own error when the full profile would run
[02:01:21]   T38    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: copies of holdout rows, and what this pass changed
[02:01:21]   T39    complex  opus x3                            gate=0 tests=1  Guards of the fifth pass: what the fourth review pinned and the application already does. No code change expected
[02:01:21]   T40    complex  opus x3                            gate=0 tests=1  Each object column is read once, over all loaded rows (the fourth review, finding 1)
[02:01:21]   T41    complex  opus x3                            gate=0 tests=1  No part of the output can hold a number that is not finite; the boundary of a failed configuration; the three-class message (the fourth review, findings 4 and 5)
[02:01:21]   T42    complex  opus x3                            gate=5 tests=1  A date column with nothing said about time order stops the run; `time_col=False` and `--no-time-order`; the README's commands
[02:01:21]   T43    complex  opus x3                            gate=2 tests=1  A declared entity column under the random scheme: `group_col`, the grouped split, the cap, the interval
[02:01:21]   T44    complex  opus x3                            gate=1 tests=2  Entities under time order: each entity's own previous value, the leak screen's baseline, and the stop for a panel with no entity declared
[02:01:21]   T45    complex  opus x3                            gate=0 tests=1  `duplicates="group"`: each set of copies on one side of the split and of every fold
[02:01:21]   T46    complex  opus x3                            gate=0 tests=1  The type matrix gains a role: every kind of column as the entity column
[02:01:21]   T47    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: entities, the two statements, `duplicates="group"`, and what is not detected
[02:01:21]   T48    complex  opus x3                            gate=0 tests=1  Guards of the sixth pass: the cut-off of the panel warning, and what a copy is. No code change expected
[02:01:21]   T49    complex  opus x3                            gate=2 tests=1  Both statements on every call: `time_col` and `group_col` left unsaid stop the run, whatever the table holds
[02:01:21]   T50    complex  opus x3                            gate=2 tests=1  `duplicates="group"` keeps together the rows that agree in every feature column; the count and the warning renamed; the copies stop says when each answer fits
[02:01:21]   T51    complex  opus x3                            gate=1 tests=1  The grouped holdout holds a quarter and no more than it must; a unit of a quarter or more is named
[02:01:21]   T52    complex  opus x3                            gate=1 tests=1  Dates written as whole numbers (YYYYMMDD) are a date form: excluded as a date, named, and accepted as the time column
[02:01:21]   T53    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: both statements in every command and example, what `group` means, the two warnings for a large entity, dates as whole numbers, and the limits
[02:01:21] Then: full gate (43 checks) + full suite, on failure an integration task at tier 'complex'.
```
