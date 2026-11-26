
---

# Ninth pass (CP10): what the headline rests on, the reading by levels, and the screen's repairs

D77 is the decision this pass builds, with the rulings D78, D80 and D81. Read
them in `docs/DECISIONS.md`, and spec step 12a, step 8, step 7, 2.2, 2.5, 4.2,
4.3, 5.2, 5.3 (rules 17 to 21), 5.4, 7, 8, 8.1, 8.2, 10.11 and 11 before you
write anything. Every test below was proved by the harness on a prototype
outside the repo: on the application as it stands, on the reference and on
each broken copy of it (spec 10.11; `docs/measurements/CP10/`).

## Ground rules of the ninth pass

1. **The application already exists, and this is the ninth pass.**
   `mlprofessor/` is built and passes all 2,140 cases of the first eight
   passes. Most of your new tests must fail today; the guards and the
   controls must pass today; every existing test must pass today and after
   the build. Rule 5 says which, by name.

2. **The first eight passes are final. Do not edit them, with the five
   exceptions of rule 3, which D59 allows and which are given there exactly.**
   That means every existing test file, `tests/conftest.py`,
   `tests/mlp_helpers.py`, `tests/cp3_helpers.py` to `tests/cp9_helpers.py`
   and everything under `tests/fixtures/`. New shared helpers go in one new
   file, `tests/cp10_helpers.py`. The owner's lead reads the diff of every
   existing file against the committed one, and any change other than those
   of rule 3 sends the whole stage back.

3. **THE LOOSENINGS (D59). Five existing files, each changed only as written
   here.** Two behaviours of this pass reverse what five cases held. The
   statement of step 12a has the chosen configuration predict the holdout
   again, five times for each set of copies, after the headline's own call:
   two tests pinned that the holdout reached the chosen configuration once.
   And D77 point 5 has rule 2 leave its rows unscored where too many rows are
   marked to set any aside: three cases pinned that every row was then
   scored and the copy kept. D78 point 8 does the same for the two older
   judge-tools files: the judge of this pass asserts rules the application
   does not have yet, and both files are in the loop's ratchet. Each change
   makes its test accept the application as it stands and as T69 to T74 build
   it. D65 is not needed: no helper of an earlier pass changes. Keep each
   file's line endings as they are; each text below matches its file exactly
   once.

   **(a) `tests/test_chosen_failure.py`,
   `test_holdout_rows_reach_one_configuration_at_a_time`.** Add, after the
   helpers above the test, a helper:

   ```python
   def _in_turn(log, hold):
       """The families whose predict calls hold holdout rows, in call order, with the calls of
       one family in a row counted once.

       D59, D78 point 7: until the ninth pass the configuration that scored the holdout
       predicted it once. Since then it predicts it again, five times for each set of
       columns, to state what the headline rests on; a family's calls in a row are one turn,
       so what is asserted is still the order in which the holdout reaches configurations.
       """
       predicted = [ev["family"] for ev in log.events
                    if ev["action"] == "predict" and set(ev["rows"]) & hold]
       return [f for k, f in enumerate(predicted) if k == 0 or f != predicted[k - 1]]
   ```

   and in the test, in both places, replace

   ```python
       reached = [ev["family"] for ev in log.events
                  if ev["action"] == "predict" and set(ev["rows"]) & hold]
   ```

   with

   ```python
       reached = _in_turn(log, hold)
   ```

   (the two `assert reached == ...` lines that follow stay as they are).

   **(b) `tests/test_guard_final_fit.py`,
   `test_m2_levels_come_from_the_rows_of_each_fit`.** Replace

   ```python
       assert len(holdout_predicts) == 1
   ```

   with

   ```python
       # D59, D78 point 7: since the ninth pass the chosen configuration predicts the holdout
       # again, shuffled, to state what the headline rests on. The first call is the
       # headline's, and it is the one read here.
       assert len(holdout_predicts) >= 1
   ```

   `predict = holdout_predicts[0]` after it is unchanged.

   **(c) `tests/test_screen_set_aside.py`,** the cases
   `test_at_most_one_percent_of_the_training_rows_is_set_aside[120-2-False]`
   and `[600-5-False]`, and
   `test_under_time_order_the_row_after_a_wild_cell_counts_too[3-False]`: the
   three cases that reach the branch where more rows are marked than may be
   set aside. Add, after `_assert_copy_is_kept_and_named`:

   ```python
   def _assert_copy_is_kept_and_named_or_excluded(result):
       """Where more rows are marked than may be set aside.

       D59, D77 point 5, D78 point 7: as D74 built it every row was then scored, and the copy
       was kept and named by the order reading. Since the ninth pass rule 2 still leaves its
       rows unscored and is also read against each fold's training median, and the copy is
       excluded, the only leak. Either is accepted here; tests/test_screen_marked.py pins the
       second.
       """
       if q.leaks(result):
           _assert_copy_is_the_only_leak(result)
       else:
           _assert_copy_is_kept_and_named(result)
   ```

   and in the `else` branch of each of the two tests replace
   `_assert_copy_is_kept_and_named(result)` with
   `_assert_copy_is_kept_and_named_or_excluded(result)`. In the module's
   docstring, after `set aside and`, the words `every row is scored.` become:
   `as D74 had it, every row is scored. Since the ninth pass (D77 point 5, D78
   point 7) rule 2 then still leaves its rows unscored and is also read against
   each fold's training median: tests/test_screen_marked.py pins that, and this
   file accepts either (D59).` The first test's docstring becomes: `F25 where
   the rows are set aside; where they are too many, M44 as D74 built it (all
   scored, the copy kept and named by the order reading) or the copy excluded
   as the ninth pass has it (D59).` M44 stays held by
   `test_order_reading.py` (six cases).

   **(d) `tests/test_judge_tools.py`.** `import cp9_helpers as q` becomes
   `import cp10_helpers as q`; in the two assertion messages `the eighth pass
   does not own` becomes `the ninth pass does not own`; in the docstring,
   `D74).` on the first line becomes `D74, D78 point 8).`, `the application of
   the seventh pass` becomes `the application of the pass before`, and the
   sentence on the owned lists and the strict form says: every pair (every
   table) that fails is one the pass being built owns, since D78 point 8 the
   ninth pass, T69 to T76 (`cp10_helpers.BATTERY_OWNED`,
   `cp10_helpers.TABLES_OWNED`), and the other forms,
   `tests/test_judge_tools_pass8.py` and the strict
   `tests/test_judge_tools_pass9.py` (T76), ask `run_tool` for the same run.
   Nothing else changes.

   **(e) `tests/test_judge_tools_pass8.py`.** Until this pass the strict form.
   It takes the accepting form: remove `_assert_exit_0`, and make the bodies of
   its two tests, which keep their names (the loop's ratchet holds them by
   name), `j.test_the_perturbation_battery_holds(tmp_path_factory)` and
   `j.test_the_generated_tables_agree_with_the_reference(tmp_path_factory)`.
   The docstring of the module says: T68: the judge's two other tools on the
   application, in the form the eighth pass left (spec 8.2; D73, D74 point 7,
   D76, D78 point 8); until the ninth pass this file was strict; since D78
   point 8 it takes the accepting form of `tests/test_judge_tools.py`, on the
   same session run; the strict form is `tests/test_judge_tools_pass9.py`
   (T76). Each test's docstring adds `or is one the ninth pass owns`.

   | What the existing case pinned | The cases | What pins the new behaviour |
   |---|---|---|
   | the holdout reaches the chosen configuration once | `test_chosen_failure.py::test_holdout_rows_reach_one_configuration_at_a_time`, `test_guard_final_fit.py::test_m2_levels_come_from_the_rows_of_each_fit` | `test_rests_on.py::test_the_shuffles_move_one_set_of_columns_among_the_holdout_rows` |
   | with too many rows marked, rule 2 scores every row and the copy is kept | `test_screen_set_aside.py`, the three cases of (c) | `test_screen_marked.py::test_a_copy_with_five_cells_moved_far_is_excluded` |
   | each of the judge's two tools exits 0 on the application | `test_judge_tools_pass8.py`, both tests | `test_judge_tools_pass9.py`, both tests (T76) |

   **No existing test is ever edited so that it fails against the library as
   built:** run the whole suite against the application before you finish and
   see all 2,140 cases pass.

4. **You write nine new test files and the helper file, and nothing else.**
   The files are listed below, each with the tests it must hold, **under the
   names given**: the lists of rule 5 and the mutation check go by name. Split
   a named test into parameter cases as the text says, with the case names
   given. If you add a test of your own, say so in your final report, with
   the list of rule 5 it belongs to.

5. **Which tests fail today and which pass today is fixed, by name.** The
   lists are at the end of this section. Run the new files before you finish
   and compare. **Do not weaken a guard to make it fail, and do not weaken a
   red test to make it pass.** If a test on the green list fails, or one on
   the red list passes, and you believe your test is what the spec says, stop
   and report it with the output.

6. **Every new variant must fail a test, and `verify/mutation_check.py` is how
   that will be judged.** Twenty-five are new: F29 to F35 and M52 to M69
   (spec 8 and 8.1; `verify/reference/mlprofessor/_variant.py`). Each has a
   test of this pass named for it:

   | Variant | The test that must fail on it | Cases that fail on it in the new files |
   |---|---|---|
   | F29 | `test_rests_on.py::test_every_feature_has_a_statement_and_no_other_column_has_one` | 19 |
   | F30 | `test_near_perfect.py::test_two_columns_that_give_the_target_together_are_named` | 5 |
   | F31 | `test_rests_on.py::test_copies_are_shuffled_together_and_share_one_fall` | 5 |
   | F32 | `test_screen_levels.py::test_a_coded_copy_of_the_target_is_excluded` | 11 |
   | F33 | `test_order_ties.py::test_noise_whose_values_repeat_is_not_named` | 6 |
   | F34 | `test_screen_marked.py::test_a_copy_with_five_cells_moved_far_is_excluded` | 4 |
   | F35 | `test_large_holdout.py::test_a_holdout_of_half_the_rows_or_more_is_named` | 3 |
   | M52 | `test_rests_on.py::test_the_statement_is_the_same_at_any_n_jobs` | 2 |
   | M53 | `test_rests_on.py::test_every_feature_has_a_statement_and_no_other_column_has_one` | 2 |
   | M54 | `test_rests_on.py::test_the_list_holds_the_columns_whose_fall_reaches_the_floor` | 9 |
   | M55 | `test_rests_on.py::test_equal_falls_go_in_file_order` | 2 |
   | M56 | `test_rests_on.py::test_copies_are_shuffled_together_and_share_one_fall` | 4 |
   | M57 | `test_near_perfect.py::test_two_columns_that_give_the_target_together_are_named` | 3 |
   | M58 | `test_screen_levels.py::test_rows_whose_level_the_fold_has_not_seen_are_not_scored` | 5 |
   | M59 | `test_screen_levels.py::test_a_reading_by_levels_on_too_few_rows_of_a_class_has_no_value` | 2 |
   | M60 | `test_screen_levels.py::test_few_values_are_counted_over_the_training_rows` | 2 |
   | M61 | `test_screen_levels.py::test_a_coded_copy_of_the_target_is_excluded` | 8 |
   | M62 | `test_order_ties.py::test_a_column_whose_means_rise_with_the_target_is_not_named` | 4 |
   | M63 | `test_screen_marked.py::test_wild_target_cells_beyond_the_share_make_no_column_a_leak` | 7 |
   | M64 | `test_large_holdout.py::test_a_holdout_of_half_the_rows_or_more_is_named` | 2 |
   | M65 | `test_rests_on.py::test_the_summary_prints_one_line_for_the_list` | 2 |
   | M66 | `test_rests_on.py::test_a_model_that_cannot_predict_a_shuffled_matrix_leaves_that_column_unstated` | 4 |
   | M67 | `test_rests_on.py::test_the_floor_is_the_half_width_of_the_skill_interval` | 5 |
   | M68 | `test_rests_on.py::test_where_the_headline_has_no_interval_the_floor_stands_in` | 2 |
   | M69 | `test_rests_on.py::test_copies_are_shuffled_together_and_share_one_fall` | 3 |

   **Run the mutation check on the new variants only:**

   ```
   E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/mutation_check.py --only good,F29,F30,F31,F32,F33,F34,F35,M52,M53,M54,M55,M56,M57,M58,M59,M60,M61,M62,M63,M64,M65,M66,M67,M68,M69 --jobs 8
   ```

   (about 25 minutes). The good reference must fail none of the tests (the
   README tests apart: `test_readme_pass9.py` fails on it, since the README is
   the application's), and every case listed must fail at least one. **The
   owner runs the full check, of all 113 variants, after the stage.** The
   cases of the five loosened files that failed on a broken copy before still
   fail on it: on F1, F23, F24, F25, M2, M42, M43 and M47 the number of failing
   cases is what it was or more; M44 is caught by six cases of
   `test_order_reading.py` and no longer by the three loosened cases of (c).

7. **The owned lists of the judge's tools are fixed, by name.**
   `cp10_helpers.BATTERY_OWNED` is a `frozenset` of 1,196 pairs, written
   `"run:change"`, and `cp10_helpers.TABLES_OWNED` a `frozenset` of 704 table
   numbers: exactly the pairs and the tables that fail when
   `verify/battery.py mlprofessor` and `verify/generated_tables.py mlprofessor`
   are run on the application as it stands. They are in
   `docs/measurements/CP10/owned/owned_by_pass9.json` (keys `battery` and
   `tables`), with `owned.py`, which writes them again from the tools'
   `--json` reports: copy them as literals, sorted, eight to a line. Why they
   fail, and which task owns each (2,000 entries, grouped): every finished
   pair and every finished table lacks the three keys of the statement (T69);
   145 carrier pairs also lack `headline.rests_on_floor` (T69), coded copies
   are kept (T71), two columns together are not named (T69, T70); 26 noise
   pairs with repeated values are warned about (T72); in the tables, the
   near-perfect warning (T70), coded copies and an undeclared entity kept
   (T71), and `strong_single_column` on columns with repeated values (T72).
   No entry belongs to T73 or T74.

8. **The seam sees the statement.** Step 12a pins that each shuffle is one
   call of the fitted configuration's prediction on every holdout row, handed
   as the holdout was for the headline: a family supplied through the seam is
   asked five times for each set of copies, after the headline's own call.
   Which set comes first is open: no test may depend on it. The
   recording fakes of earlier passes see those calls; rule 3 (a) and (b) are
   the only tests that counted them.

9. **What is open, and what is pinned to a tolerance.** Which permutations the
   shuffles draw and which resamples the floor's stand-in draws are open
   (spec 11): a test compares a shuffled score with its own computation only
   through a seam model whose predictions it knows (rule 8), or within a
   stated tolerance; `test_where_the_headline_has_no_interval_the_floor_stands_in`
   compares the floor with the test's own bootstrap of `predictions.csv` to
   within a third. A share is compared with `math.isclose` or
   `pytest.approx`, never with `==`, but for the cases of
   `test_large_holdout.py` built to be exactly one half.

10. **The rules of the earlier passes still hold**: the documented surface
    only (spec 2.1 to 2.7); recompute, do not trust; assert on values as the
    application produced them; **no skipped or expected-to-fail test**;
    nothing written outside `tmp_path`; the application imported inside test
    functions; the command line run one process at a time; small tables
    (nothing here needs more than 2,000 rows), and the fast profile.

## Files of the ninth pass

Each file belongs to exactly one task of `plan/tasks.json`.

| File | Task | About | Today |
|---|---|---|---|
| `tests/test_rests_on.py` | T69 | the statement of step 12a: the three keys, the shuffles, copies, the list and its floor, the summary line, `shuffled_score_not_computed` (D77 point 1, D78, D80) | red |
| `tests/test_near_perfect.py` | T70 | the `near_perfect_headline` warning (D77 point 2) | red, one control green |
| `tests/test_screen_levels.py` | T71 | the reading by levels, unseen levels, its floor, "few", the `group_col` sentence, and the controls (D77 point 3, D78 points 4 to 6) | red; six guards and controls green |
| `tests/test_order_ties.py` | T72 | the order reading between rows of neighbouring values (D77 point 4) | red |
| `tests/test_screen_marked.py` | T73 | rule 2 where too many rows are marked (D77 point 5) | red; the controls green |
| `tests/test_large_holdout.py` | T74 | `holdout_far_above_quarter` (D77 point 6) | red; two controls green |
| `tests/test_readme_pass9.py` | T75 | the README on this pass | red |
| `tests/test_judge_tools_pass9.py` | T76 | both tools exit 0 on the application, strictly | red |
| `tests/test_guard_review8.py` | T76 | a second route for M52 to M69 that one test held | red; four green |
| `tests/cp10_helpers.py` | none | shared helpers of the ninth pass, and the two owned lists; holds no test | |
| `tests/test_chosen_failure.py`, `test_guard_final_fit.py`, `test_screen_set_aside.py`, `test_judge_tools.py`, `test_judge_tools_pass8.py` | (T21, T18, T63, T60, T68) | as rule 3 says; nothing else | green before and after |

### `tests/cp10_helpers.py`

Import the earlier helper files and build on them; do not copy from them.
Nothing imports `mlprofessor` at module level.

- **Pinned values**: `REST_SHUFFLES` 5, `REST_FALL` 0.05, `REST_MOST` 5,
  `NEAR_PERFECT` 0.98, `COPY_TOLERANCE` 1e-9, `LEVEL_ROWS` 5,
  `LEVEL_SCORED` 20, `LEVEL_SCORED_CLASS` 10, `FAR_ABOVE` 0.5, the fixed
  passages of spec 5.4.
- **Readers of the output**: the three keys of a column by name; the list as
  `(column, fall)`; the floor; the warnings of one code; `leaks(result)`, the
  columns excluded as suspected leaks (as `cp9_helpers.leaks`).
- **Recomputations in plain arithmetic**: the list from the column records and
  the floor (5.3 rule 19); the floor from the skill interval (rule 18); a
  bootstrap of the holdout rows of `predictions.csv` taken as if independent,
  for the stand-in, with the test's own seeded generator; the share against
  the order between rows of neighbouring values (step 8), and the one-point-
  per-value form of M62 for contrast; the mean target per level of a fold's
  training rows.
- **Table makers**, each from `numpy.random.default_rng` with the seed the
  test gives: two columns that give the target only together (`pair_a` = the
  target plus a noise of its spread, `pair_b` that noise; for a class, the
  0/1 target plus noise of 0.5); a coded copy of the target (bands of equal
  count, labels in random order, text or whole numbers; for a class, a code
  of 40 levels each of one class); noise with repeated values; a log-normal
  target and a copy of it with cells moved; a time-ordered table with a share
  of its rows on the last date.
- **Seam families**: one that records every matrix it is handed and predicts a
  known function of it (for the shuffles, M52, M56, M69); one that raises, or
  predicts infinity, on any matrix whose given column differs from the
  holdout's (M66).
- **`BATTERY_OWNED`, `TABLES_OWNED`**: rule 7.

## A: what the headline rests on (task T69)

### `tests/test_rests_on.py`

Every table is small (a few hundred rows), the fast profile, both statements
made.

- `test_every_feature_has_a_statement_and_no_other_column_has_one` (F29, M53):
  a regression with two features that carry the target, a constant, an
  identifier and a copy of the target: each feature's three keys are numbers
  and its fall is of a shuffle (the strong features fall by more than the
  floor); the target, the constant, the identifier and the suspected leak have
  three nulls; 5.3 rule 17 holds.
- `test_a_classification_states_its_falls_in_skill_of_roc_auc`:
  `shuffled_skill` is 2 × `shuffled_score` − 1, and `skill_fall` the headline
  skill less it.
- `test_the_shuffles_move_one_set_of_columns_among_the_holdout_rows` (F29,
  F31, M56, M69), through the seam with a recording family: after the
  headline's call the family is handed the holdout five times for each set of
  copies; each time one set's columns are a permutation of their holdout
  values and every other column is as it was; `shuffled_score` is the mean of
  the five scores, recomputed from the recorded predictions.
- `test_copies_are_shuffled_together_and_share_one_fall` (F31, M55, M56, M69):
  an exact copy, a copy 2 × x + 3, a copy −x of a numeric column, and a copy
  of a categorical column with its levels renamed: each shares the three
  numbers of what it copies.
- `test_a_column_that_differs_in_one_holdout_row_is_no_copy`: a column equal to
  `x1` but in one holdout cell has a statement of its own, and each of the two
  falls by less than the two together.
- `test_the_statement_is_the_same_at_any_n_jobs` (M52): at `n_jobs` 1 and 3
  the three keys of every column, the list and the floor are equal.
- `test_the_list_holds_the_columns_whose_fall_reaches_the_floor` (M54): two
  columns carry the headline and two are noise: the list names the two, and
  equals the recomputation of rule 19.
- `test_at_most_five_are_listed_largest_first`: seven columns of one weight
  each: five listed, by fall from the largest.
- `test_equal_falls_go_in_file_order` (M55): two exact copies, the first in
  file order named `z_first` and the second `a_second`: one fall, `z_first`
  listed first.
- `test_the_floor_is_the_half_width_of_the_skill_interval` (M67): a table
  whose skill interval has a half-width above 0.05 (about 0.085): the floor is
  that half-width.
- `test_where_the_headline_has_no_interval_the_floor_stands_in` (M68): the
  grouped scheme with 40 entities, fewer than 30 in the holdout, so no skill
  interval: the floor is above 0.05 and within a third of the test's own
  bootstrap of the holdout rows.
- `test_with_no_headline_skill_a_statement_holds_the_score_alone` (D36): a
  baseline with no error: each feature has a `shuffled_score` and null skill
  and fall, the list is empty, the floor null.
- `test_the_summary_prints_one_line_for_the_list` (M65), by the command line:
  one line that begins `rests on`, with the floor and each listed column in
  quotes with its fall; on a table of noise the line holds `none`.
- `test_a_model_that_cannot_predict_a_shuffled_matrix_leaves_that_column_unstated`,
  cases `[False]` (raises) and `[True]` (predicts infinity) (M66): a seam
  family that fails on any shuffle of `x2`: the run finishes with the headline
  it would have had; `x2` has three nulls and one `shuffled_score_not_computed`
  warning names it; the other features have their statements.

## B: the near-perfect warning (task T70)

### `tests/test_near_perfect.py`

- `test_two_columns_that_give_the_target_together_are_named`, cases
  `[regression]` and `[classification]` (F30, M57): two columns that give the
  target only together: skill 0.98 or more, one `near_perfect_headline`
  warning with a null column, whose message names both columns in quotes.
- `test_under_time_order_the_warning_is_raised_too` (M57): a regression under
  time order, against the better of its baselines.
- `test_under_the_grouped_scheme_the_warning_is_raised_too`: a classification
  of whole entities.
- `test_no_warning_below_the_level`: a headline of about 0.7: none (green
  today).
- `test_a_near_perfect_headline_that_rests_on_no_one_column_is_named_too`:
  sixty columns of one weight each give the target together: no fall reaches
  the floor, the list is empty, and the warning is raised all the same.

## C: the reading by levels (task T71)

### `tests/test_screen_levels.py`

- `test_a_coded_copy_of_the_target_is_excluded`, four cases
  `[regression-True]`, `[regression-False]`, `[classification-True]`,
  `[classification-False]` (text or whole numbers) (F32, M61): the target in
  30 bands (a regression) or in 40 levels each of one class (a
  classification), labels in random order: excluded as a suspected leak with
  the rule `predicts_target`, the message holding the fixed sentence of 5.4;
  the other columns are features.
- `test_under_the_grouped_scheme_too` (F32): a text code of a regression's
  target, the rows of 120 entities of five.
- `test_a_column_excluded_by_another_reading_has_no_sentence_on_group_col`: an
  exact numeric copy of the target, every value distinct: excluded, and its
  message does not hold the sentence (green today).
- `test_rows_whose_level_the_fold_has_not_seen_are_not_scored`, cases
  `[regression]`, `[classification]` (M58): 40% of the rows hold a level of
  their own; on the others the column is the target: excluded.
- `test_a_reading_by_levels_on_too_few_rows_of_a_class_has_no_value` (M59): a
  classification in which fewer than ten of the positive validation rows have
  a level the fold has seen: the column, which reads as perfect on the rows
  it scores, is not excluded.
- `test_few_values_are_counted_over_the_training_rows` (M60): a numeric code
  of the target with six rows for each value, so its 450 training rows hold
  4.5 for each: not read by its levels, a feature named by no rule (green
  today).
- `test_numeric_bands_with_enough_rows_are_read_by_their_levels` (F32): ten
  rows for each value: excluded.
- `test_eight_bands_are_warned_about_and_kept` (D78 point 4): a coded copy of
  eight bands: a feature with `strong_single_column` (green today).
- `test_an_honest_column_of_ten_levels_is_a_quiet_feature`, `test_in_a_classification_too`,
  `test_a_numeric_column_of_ten_values_is_a_quiet_feature`: a column of ten
  levels that explains about half of the target, and noise of 60 levels:
  features named by no rule (green today).

## D: the order reading (task T72)

### `tests/test_order_ties.py`

- `test_noise_whose_values_repeat_is_not_named`, cases `[9601]`, `[9602]`
  (seeds) (F33): beside a target with signal, a whole number from 1 to 40 and
  noise rounded to one decimal place: features, no `strong_single_column`.
- `test_a_column_whose_means_rise_with_the_target_is_not_named`, cases
  `[9603]`, `[9605]` (M62): 22 distinct values (at least 21: spec 4.2), 2,000
  rows, the means of the values in the target's order and the rows not: not
  named.
- `test_a_column_in_the_targets_order_with_repeated_values_is_named` (F33,
  M62): a copy of a walk's target with 4% of its training cells moved round,
  two rows a timestamp, so each value is held twice: warned about, its message
  holding the share counted pair by pair (about 4%; spec 11).

## E: rule 2 with too many rows marked (task T73)

### `tests/test_screen_marked.py`

- `test_a_copy_with_five_cells_moved_far_is_excluded`, cases `[9501]`,
  `[9502]`, `[9503]` (F34): a log-normal target whose tail marks more than 1%
  of the training rows, and a copy 1.07 × y + 3 with five cells moved up by 20
  spreads: excluded, the only leak.
- `test_whatever_the_size_of_the_change`, cases `[1.0]` (green today) and
  `[1000.0]` (F34).
- `test_wild_target_cells_beyond_the_share_make_no_column_a_leak`, five cases
  `[random-999999.0-0.013]`, `[random--999.0-0.013]`, `[random-999999.0-0.015]`,
  `[grouped-999999.0-0.013]`, `[grouped--999.0-0.015]` (M63): noise, with 1.3% or
  1.5% of the training targets set to a sentinel: no column excluded (green
  today).

## F: a large time-ordered holdout (task T74)

### `tests/test_large_holdout.py`

- `test_a_holdout_of_half_the_rows_or_more_is_named`, cases `[0.6]` and
  `[0.5]` (F35, M64): 1,000 rows, that share on the last date: one
  `holdout_far_above_quarter` warning, null column, its message holding `600
  of the 1000 rows` (`500 of the 1000 rows`).
- `test_the_rows_split_count_the_gap_rows`: with `gap=1` the holdout still
  holds 500 of the 1000 rows split: named.
- `test_below_half_there_is_no_warning`: 45% on the last date (green today).
- `test_an_ordinary_time_split_has_no_warning`: one row a timestamp (green
  today).

## The README (task T75)

### `tests/test_readme_pass9.py`

- `test_the_readme_names_what_the_ninth_pass_added`: the names
  `shuffled_score`, `shuffled_skill`, `skill_fall`, `rests_on`,
  `rests_on_floor`, `near_perfect_headline`, `holdout_far_above_quarter` and
  `shuffled_score_not_computed`.
- `test_the_readme_says_what_the_statement_and_its_floor_are`: the words
  `shuffled`, `fall`, `0.05`, `half-width`, `interval` and `five`.
- `test_the_readme_says_what_a_near_perfect_headline_asks_of_the_caller`:
  `0.98`, `easy`, `known before the outcome`, and `half` (of the holdout).

Until T75 the README is unchanged and every existing README test stays green;
only this file is red.

## The judge's other tools and the guards (task T76)

### `tests/test_judge_tools_pass9.py`

The strict form, as `tests/test_judge_tools_pass8.py` was until this pass:
`import test_judge_tools as j`, and two tests,
`test_the_perturbation_battery_exits_0` and
`test_the_generated_tables_exit_0`, each asking `j.run_tool` for the session's
run and asserting exit code 0.

### `tests/test_guard_review8.py`

A second route for each small fault of this pass that the files above hold
with one test, or on one table:
`test_m52_the_shuffled_matrices_are_the_same_at_any_n_jobs` (by the seam, a
classification, `n_jobs` 1 and 4),
`test_m53_under_time_order_no_column_outside_the_models_has_a_statement` (the
time column, a constant, free text and a suspected leak),
`test_m54_a_classification_of_noise_lists_nothing`,
`test_m56_a_relabelled_categorical_copy_shares_one_fall` (and 100 × x − 7),
`test_m69_a_negated_copy_is_moved_with_what_it_copies` (by the seam, −3 × x1 +
1), `test_m66_a_model_that_can_predict_no_shuffle_leaves_every_feature_unstated`
(cases `[regression]`, `[classification]`),
`test_m67_a_classification_with_a_short_holdout` (200 rows),
`test_m68_a_grouped_classification_with_no_interval`,
`test_m65_with_no_skill_the_summary_still_has_its_line` (D36, by the command
line), `test_m58_under_the_grouped_scheme_unseen_levels_are_not_scored`,
`test_m59_too_few_scored_rows_of_the_negative_class`,
`test_m60_a_classification_code_of_six_rows_a_value` (green today),
`test_m62_a_falling_trend_under_the_grouped_scheme` (24 values),
`test_m63_a_target_of_noise_with_seven_sentinel_cells` (cases `[9713]`,
`[9714]`; green today) and `test_m63_under_time_order_with_entities` (green
today).

## What must fail today, and what must pass today

**Red** (fails against the application as it stands, every case): all of
`test_rests_on.py` (15 cases); `test_near_perfect.py` but
`test_no_warning_below_the_level` (5); `test_screen_levels.py`:
`test_a_coded_copy_of_the_target_is_excluded` (4),
`test_under_the_grouped_scheme_too`,
`test_rows_whose_level_the_fold_has_not_seen_are_not_scored` (2),
`test_a_reading_by_levels_on_too_few_rows_of_a_class_has_no_value`,
`test_numeric_bands_with_enough_rows_are_read_by_their_levels` (9 in all);
all of `test_order_ties.py` (5); `test_screen_marked.py`:
`test_a_copy_with_five_cells_moved_far_is_excluded` (3) and
`test_whatever_the_size_of_the_change[1000.0]` (4 in all);
`test_large_holdout.py`: `test_a_holdout_of_half_the_rows_or_more_is_named`
(2) and `test_the_rows_split_count_the_gap_rows` (3 in all); all of
`test_readme_pass9.py` (3) and `test_judge_tools_pass9.py` (2);
`test_guard_review8.py` but the four below (13). **78 new cases, 59 red.**

**Green** (passes today, every case): `test_near_perfect.py::test_no_warning_below_the_level`;
`test_screen_levels.py::test_a_column_excluded_by_another_reading_has_no_sentence_on_group_col`,
`::test_few_values_are_counted_over_the_training_rows`,
`::test_eight_bands_are_warned_about_and_kept`,
`::test_an_honest_column_of_ten_levels_is_a_quiet_feature`,
`::test_in_a_classification_too`, `::test_a_numeric_column_of_ten_values_is_a_quiet_feature`;
`test_screen_marked.py::test_whatever_the_size_of_the_change[1.0]` and
`::test_wild_target_cells_beyond_the_share_make_no_column_a_leak` (5);
`test_large_holdout.py::test_below_half_there_is_no_warning` and
`::test_an_ordinary_time_split_has_no_warning`;
`test_guard_review8.py::test_m60_a_classification_code_of_six_rows_a_value`,
`::test_m63_a_target_of_noise_with_seven_sentinel_cells` (2) and
`::test_m63_under_time_order_with_entities`. **19 cases.** And every case of
the five files of rule 3, and every other existing case: 2,140.

Measured by the harness with prototypes of these files laid over the suite:
on the application, 2,155 passed and 57 failed, the 57 being the red cases
above but those of the two judge-tools files; on the reference, every case
passes but the three README cases.

## Not tested in the ninth pass

- Which permutations the shuffles draw, and which resamples the floor's
  stand-in draws (open, spec 11).
- The wording of the new warnings' messages beyond what spec 11 pins.
- The cost of the statement at 100,000 rows (measured in spec 10.11, not
  asserted).
- Coarse bands of the target (four or fewer), and coded copies under time
  order on a wandering target: stated limits (spec 11; D80 point 3); no test
  asserts that they are excluded or warned about.
