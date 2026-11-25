---
name: mlprofessor-build
description: Build or repair a tabular profiling library of the mlprofessor kind (one table, one target, a holdout headline with leak, time-order and row accounting guarantees). Use when implementing any step of spec.md, when a gate check or a test fails, or when a score looks too good.
---

# Building a tabular profiling library that cannot fool its user

The product is one number, skill on a holdout, and the accounting that makes it
believable. Every bug that matters here produces a plausible number and no
error. Build in the order below. Each step has a way to prove it before the
next one starts.

The contract is `spec.md`. The judge is `verify/smoke_test.py` and `tests/`.
This file is the method.

## Order of work

Steps 2 to 7 each deliver one function of `mlprofessor.stages` (spec 2.7) and
are judged by that step's own test file, before anything runs end to end.
`profile()` must then call those same functions, never a second copy.

1. **Skeleton.** Package folder, `__main__.py`, the command line with every
   option, `requirements.txt` with the four pinned libraries. Nothing is
   installed into the venv; the package runs from the repo root.
2. **Load and count.** One loader for a CSV path and a DataFrame. Count rows
   before anything is dropped. Detect duplicate header names on the raw header
   line, because pandas renames them silently. `row` is the position in the
   loaded table and never changes afterwards.
3. **Column roles that need no target.** Decide them on the usable rows, before
   the fast profile's row cap. Keep the counts (missing, coerced, infinite) on
   every loaded row. Every source column gets exactly one record.
4. **Task.** Two distinct values is classification whatever the type. Decide it
   before the row cap. Never guess: raise the named error.
5. **Split.** Random: 25% holdout first, then five folds inside the rest. Time
   order: sort by the time column, cut at a whole timestamp, six blocks, gap
   before the holdout and before every validation block. Write the split as a
   pure function from keys to row positions and test it on tiny tables.
6. **Metrics and baselines.** ROC AUC by ranks with ties averaged; squared
   error; skill; the previous-value baseline. Test against values worked out by
   hand.
7. **Preprocessing as an object with `fit` and `transform`.** It is created
   fresh for every fit and only ever sees that fit's training rows.
8. **Fit, select, score once.** One loop over configurations and folds. The
   holdout is touched once, by the chosen configuration, after selection.
9. **End to end.** Result object, `profile.json`, `predictions.csv`. From here
   the gate can judge. Run it after every edit.
10. **Time-ordered scheme wired in**, then the **leak screen**, then the
    **interval and p-value**, then **forest and boosting**, then the **full
    profile's grids**, then **edges and warnings**, then the **README**.

### Second pass, after the first review (tasks T18 to T23)

The package exists. Change as little as each task needs.

11. **Guards first (T18).** Tests that pass today and must go on passing: the
    built-in preprocessing, the leak screen, the block bootstrap and the
    p-value each proved against a faulty copy of the reference. Expect to
    change nothing. A guard that fails means the application is wrong.
12. **Skill where it has no value (T19)**, then **colliding feature names
    (T20)**, then **a configuration that fails after it is chosen (T21)**, then
    the **small defects (T22)**, then the **README's examples (T23)**.

### Third pass, after the second review (tasks T24 to T30)

Change as little as each task needs. Every earlier test stays green.

13. **The guard first (T24).** A line pandas reads as the header, before a
    header of two columns. Expect to change nothing.
14. **No number that is not finite (T25).** The largest task of the pass. Work
    through spec section 3, "Numbers that are not finite", in its order: the
    baselines before any fit, the leak screen, a configuration's predictions
    and fold scores, the chosen configuration, skill, the check when the
    result is assembled. Then the feature values too large for the tree
    models, and the complex target (D48).
15. **White space round a target value (T26)**, then **the seed's range
    (T27)**, then **the fast profile's error (T28)**, then **the summary's
    significant digits (T29)**, then the **README (T30)**.

### Fourth pass, after the third review (tasks T31 to T38)

Change as little as each task needs. Every earlier test stays green.

16. **The guards first (T31).** Six tests, one for each fault the third review
    planted unseen: the interval's percentiles, the leak screen's
    previous-value baseline, the fast profile's sample, `cv_std`, ties in a
    resampled ROC AUC, `n_imputed`. Expect to change nothing.
17. **Copies of holdout rows (T32).** The `duplicates` argument, the count, the
    rule (more than 1% of the holdout rows and at least 10), the error, the
    warning. Count after the leak screen and before any fit, on the values a
    seam step would be handed.
18. **Whole numbers (T33).** Integer time keys above 2**53 stay integers;
    distinct values of an integer column are counted on the integers; whole
    numbers too large for 64 bits are excluded or a named error, never a raw
    one.
19. **Object columns of numbers (T34).** Type them once, before any step, and
    let every role read the typed column. Then one warning per column still
    excluded as `unsupported_type`.
20. **The type matrix (T35).** About 800 small tables. Run it, group what fails
    by where the raw exception starts, and fix each place. No broad `except`.
21. **Fractions of a second (T36)**, then **the fast profile's cap leaving too
    few rows (T37)**, then the **README (T38)**.

### Fifth pass, after the fourth review and the real-data run (tasks T39 to T47)

Change as little as each task needs. Every earlier test stays green; nine of
them were loosened by the tests stage so that they pass before and after
(D59), and none of them is yours to edit.

22. **The guards first (T39).** Six tests of what the application already
    does: the infinite cells of the target and the time column are counted, an
    infinite float in an object target is a missing target, the constructor
    `ProfileResult(document, predictions)`, a column of predictions. Expect to
    change nothing.
23. **Read each object column once (T40).** Over all loaded rows, before any
    row is dropped. An odd cell in a row that step 2 drops still decides what
    its column is.
24. **Findings 4 and 5, the three-class message (T41).** The finite check
    covers every part of the document and sits in the constructor. Only what
    the model does may fail a configuration; the tool's own scoring is outside
    that `try`. This second part has no test: read spec step 9.
25. **The undeclared date column (T42).** `time_col=None` with a date column
    stops at step 5; `time_col=False` is the caller's statement and gives the
    old behaviour. Change the README's commands in the same task: the example
    table has a date column. (T49 makes the stop unconditional: step 31.)
26. **Entities under the random scheme (T43).** `group_col`, the role `group`,
    the split of whole entities, the cap of whole entities, the interval from
    whole entities, no p-value. The five properties of spec step 7 are the
    contract; no row number is.
27. **Entities under time order (T44).** Each entity's own previous value, in
    the headline baseline and in the leak screen; a row with none gets the
    training mean, counted and warned. Then the stop for a panel with nothing
    said about entities.
28. **`duplicates="group"` (T45).** The exact copies of step 8a, found before
    the split, each set on one side. Nothing wider: the broader screen is not
    in this version (D60). (T50 changes which rows are a set: step 32.)
29. **The entity role of the type matrix (T46)**, then the **README (T47)**.
30. **Guards of the sixth pass (T48).** The 0.5 cut-off of the panel warning,
    held from both sides, and the count of step 8a, which still reads the
    target. Expect to change nothing.
31. **Both statements on every call (T49).** `None` for `time_col` or
    `group_col` is "not said", and stops the run at step 5 whatever the table
    holds: `time_order_undeclared` when time order is unsaid (its message asks
    for both when both are), `entities_undeclared` when only the entities are.
    Exit code 3 on the command line, never argparse's 2. The stop for entities
    moves from step 11 to step 5; the two conditions that used to decide a
    stop (a date form, an autocorrelation) now decide only a warning.
32. **`duplicates="group"` on the features alone (T50).** The sets are the
    rows that agree in every feature column; the target plays no part. The
    count and the warning are renamed (`n_rows_same_features`,
    `same_features_grouped`). Step 8a is not touched.
33. **The grouped holdout (T51).** A unit joins while the holdout stays under
    a quarter; the smallest unit passed over completes it; what it can spare
    goes back. Then the two warnings for a unit of a quarter or more.
34. **Dates written as whole numbers (T52).** Rule 2a of step 5, before the
    id-like rules: every value a calendar date `YYYYMMDD`, two distinct.
    Excluded as `datetime`; eight-digit text accepted as the time column.
35. **The README (T53).** Every command and every example says both things.

### Seventh pass, after the sixth review (tasks T54 to T60)

Change as little as each task needs. Every earlier test stays green; one was
loosened by the tests stage so that it passes before and after (D59:
`test_date_numbers.py`, eleven cases), and it is not yours to edit.

36. **The guards first (T54).** A second test, by another route, for eleven
    rules that one test each held; a date beyond the nanosecond range in one
    cell of a time column; one `fold_not_scored` record for each fold; repeated
    rows counted beside a column of lists. Expect to change nothing.
37. **A column of dates with a sentinel (T55).** Rule 2a of step 5 is judged
    by the column's distinct values: at least two read as dates, and at least
    nine do for each one that does not, or at least five beside at most two.
    Never by a share of the cells. As the time column, a value that is no date
    stops the run and is named.
38. **`duplicates="group"` after a removed leak (T56).** Once the leak screen
    has excluded a column, one holdout row that agrees with a training row in
    every feature that is left stops the run. No threshold. Then the passage
    that says what `"group"` measures.
39. **The fast profile's cap with entities (T57).** An entity of a quarter or
    more of the usable rows is kept first, on every seed; the others in an
    order drawn from the seed. When an entity is larger than the cap, or the
    large ones do not fit together: `fast_profile_group_too_large`, or the
    full profile's own error when it would stop too.
40. **Two small things (T58).** The date warning ends with the caveat of the
    scheme the run used, so finish its text after the split. One branch that
    turned a number that is not finite into null becomes a raise.
41. **The README (T59).**
42. **The judge's two other tools (T60).** Run the battery and the
    generated-table test on the build. Fix what fails on neither of their
    lists; leave what they list as open or as a stated limit.

### Eighth pass, after the judge's two other tools (tasks T61 to T68)

The two tools found, on their first run, faults that no review had (D73). This
pass builds each as D74 and D76 rule, and nothing is left open. Seven earlier
cases were loosened by the tests stage (D59): they pinned the previous value
as it was at repeated timestamps. `tests/test_judge_tools.py` accepts exactly
the failures the plan lists until the pass is built; the strict test of the
two tools belongs to the last task. `plan/tasks.json` has the order and which
task owns which check. The rules, in the order of the spec:

43. **A whole-number identifier in any row order.** Rule 8b of step 5 keeps
    D19's clause and gains a second: more than 95% of the values distinct, and
    the distinct values filling at least half of their range. Count the range
    in whole numbers of any size.
44. **The leak screen's second readers.** Beside the tree: a line through the
    median of paired slopes for a numeric column of a regression, and the
    column read directly, with no minimum leaf, in a classification. The
    better reader counts; against every baseline, the lower.
45. **Rows set aside, rows left unscored.** A training row whose target is
    more than 10 median deviations from the median, or 10 median steps from
    its previous value, leaves the whole screen when such rows are at most 1%
    of the training rows. Then two readings of one measurement: 0.98 with the
    worst 1% unscored, 0.90 with every row.
46. **The order reading.** For a numeric column of a regression whose skill
    has a value below 0.90: two conditions, at least 20 pairs, a warning only.
47. **No choice where the folds cannot choose.** Under the grouped scheme,
    count the folds' training and validation rows; rank in configuration
    order; one `selection_not_made` warning. The fallback of step 10 follows
    the same order.
48. **The order of the rows of a fit.** Time order under time order, the rows
    of one timestamp in file order; otherwise file order.
49. **The previous value at repeated timestamps.** With no entity column, the
    mean target of the timestamp `gap + 1` distinct timestamps before, for
    the headline and the screen alike. Write the panel warning last: it names
    the baseline the run used.
50. **The largest row and the extreme value.** The share of each squared
    error that its largest row carries, in `headline`, `cv` and every record
    of `models`; one warning for each statement above 0.5; one `extreme_value`
    warning for each value far outside its column. No number changes.
51. **The README, then the judge's two other tools.** Both must end with
    nothing failing and nothing open.

## Rules that keep the number honest

- The holdout rows are chosen before anything is fitted, and nothing fitted
  ever receives one: not the model, not the imputer, scaler or encoder, not the
  leak screen. If a helper takes "the data", ask which rows.
- Selection uses cross-validation scores. The headline uses the holdout score.
  They are different variables with different names.
- A count in the output is a count somebody can redo from the source file.
- One seed drives everything. Linear algebra runs on one thread always;
  forests predict on one thread. Without both, the same seed gives different
  bits on different runs or at different `n_jobs`.
- When a rule cannot decide, stop with the named error. Never fall back.
- A number that cannot be computed is null with a reason beside it. Skill
  against a baseline whose error is exactly zero is the case in point: test
  for the zero before dividing, and never put an error, an infinity or a zero
  in its place. The score keeps its interval; count what was left out, per
  interval. Two warning codes, never together: `skill_undefined` when the
  headline skill has no value, `no_skill_interval` when it has a value and
  its interval has none.
- A categorical level is the cell's text without the white space round it,
  everywhere a level is used: the counts, the leak screen, what a step
  receives, the matrix. A header cell is blank only when it is empty; one that
  holds white space is a name like any other.
- The holdout never chooses. If the chosen configuration raises on the final
  fit or on the holdout, the next in the ranking takes its place (the next
  best by `cv_mean`; the next in configuration order where step 9 made no
  choice, D74); a holdout
  score is never a reason to pass one over, and a failed one keeps the fold
  scores it earned.
- Every feature listed in the output reaches the models under a name of its
  own. Two that would share a name stop the run.
- Leave the caller's process as you found it: put back any process-wide
  setting in a `finally`.
- A null is "does not apply" or "cannot be computed, and the output says
  why". It is never what became of an infinity or a not-a-number (D44). A
  squared error overflows above about 1e154, and `numpy.mean` and `numpy.std`
  overflow sooner than their answers do. So: check a number where it is
  computed; never convert a number that is not finite into `None` on the way
  out; and do arithmetic that can overflow under `numpy.errstate(...)`, so
  that no warning reaches standard error and the caller's settings stay as
  they were.
- When the target is the cause, the caller gets one error that says so. The
  baselines need no model: score them on the holdout before anything is
  fitted, and stop there with `non_finite_score` if one is not finite. Never
  let an extreme target fail every configuration in turn and end in
  `all_models_failed`.
- Which thing a non-finite number belongs to decides what happens. A
  prediction that is not finite is the model's doing: the configuration is
  failed, like one that raised, and after the choice the next best takes its
  place. A holdout score that is not finite, from finite predictions, is a
  score: the run stops, and the next configuration is not tried, because the
  holdout never chooses.
- A value is read one way everywhere it is used. A text target value is
  without the white space round it (`yes ` is `yes`), and one that holds only
  white space is a missing target. A feature value that is infinite, or too
  large for the 32-bit floats the tree models read (above
  3.4028234663852886e+38), is a missing cell: counted, warned, imputed.
- An error that tells the caller what to do must be true. The fast profile
  says "use the full profile" only when the full profile would get past that
  point; otherwise it raises what the full profile would raise.
- Check an argument's whole range before anything is loaded. A seed of 2³² or
  more reaches scikit-learn as a raw error from deep inside the run.
- "No fitted step has seen the holdout rows" must be true of what the rows
  hold, not only of their positions. A holdout row whose exact copy is among
  the training rows can be scored by recall. Count such rows under the random
  scheme, on the feature values a step is handed and the target as read, and
  stop when more than 1% of the holdout rows and at least 10 of them have a
  copy, unless the caller said `duplicates="keep"` (D49, D53). Under time
  order the rule does not apply.
- A whole number stays a whole number. A 64-bit float holds every whole number
  only up to 2**53; above it, distinct integers become one float. Never send
  an integer through a float to compare it, sort it or count it: integer time
  keys above 2**53 are compared as integers, and the distinct values of an
  integer column are counted on the integers (D50). Compare neighbours rather
  than subtract them: `numpy.diff` overflows on the ends of the int64 range.
- Type a column once, where it is loaded, and let every role read the typed
  column. An object column that holds only numbers (`Decimal`, booleans or
  whole numbers with `None`) is that typed column in every role (D51). Teaching
  each rule about objects separately is how one role gets forgotten.
- A run ends in a result or in a named error, whatever type a column has and
  whatever it holds (D52). A raw `TypeError` or `OverflowError` is fixed where
  it starts. A broad `except` round the run that renames it would hide the
  next fault of the same class.
- Nothing is excluded in silence: a column the tool cannot read is named in a
  warning, so the caller who meant it as a feature finds out.
- The tool assumes nothing about time order or about entities (D63). Every
  call says both: a column's name, or `False`. `None` means "nothing was
  said", and stops the run at step 5 with a named error, whatever the table
  holds; `False` is the caller's statement, and it is recorded in `input`.
  Never treat the two alike, and never let what the table looks like decide
  whether to ask: a detector for one kind of table leaves the next kind open
  (a panel of twelve timestamps, a day held as a number). What the table
  shows is a warning beside the statement, nothing more. Copies of holdout
  rows in training are a named error too (D49).
- One message tells the caller everything that is missing. With both
  statements unsaid the error is `time_order_undeclared` and it asks for both.
  On the command line a missing pair is that same error, exit code 3: do not
  make the pair `required` for argparse, whose exit 2 could not name the date
  columns the table holds.
- Rows that belong together stay together. With `group_col` and no time
  column, split entities, not rows: for the holdout, for every fold, under the
  fast profile's cap, and in every resample of the interval. A row-by-row
  shuffle or bootstrap of rows that are not independent gives a p-value and an
  interval that are too sure: under the grouped scheme there is no p-value.
  Reporting `grouped` and splitting by row is the failure the gate is built to
  catch (F10).
- Under time order an entity has its own past. The previous value of a row is
  the latest earlier row of the same entity, `gap + 1` timestamps back or
  more, and it is used wherever the previous value is: the headline baseline
  and the leak screen. Forget it in the screen and the lag of a persistent
  panel is thrown out as a leak.
- With no entity column the previous value is the mean target of the
  timestamp before (D74). One row per timestamp is the case of one row. Never
  drop the baseline because timestamps repeat: one row written twice then
  turns a random walk into skill 0.99, or throws its lag out as a leak.
- A tree is not enough to read a column (D74). It cannot predict beyond the
  targets it was fitted on, it sees no split in a column whose values lie
  within 1e-7 of each other, and a leaf of five rows cannot hold a class of
  fewer than five. So a second reader stands beside it, and the better one
  counts.
- One row can be most of a squared error, and a mean is no defence against
  it. In the screen, find such rows with medians, and take a row that is set
  aside out of every fold's training rows as well as out of the sums: left in
  a fold's mean it makes every column look like a leak. In the result, say
  which row carries the error and which value lies far outside its column,
  and leave both in every number (D74, D76).
- The warning tier reads every row; only the cut-off that excludes leaves the
  worst 1% unscored. The order reading warns and never excludes, and it is
  not taken of a column whose skill has no value (D36, D76).
- Where the folds cannot choose, do not choose (D74). A fold of ten training
  rows that the other four validate on decides nothing: score every fold,
  report it, and use the first configuration.
- Hand a fit its rows in one fixed order (D74). A forest draws its rows by
  position, so the same rows in another order are another model.
- A copy is an exact copy (D49, D60): the same value in every feature and
  the same target. That is what step 8a counts, and it did not change. A copy
  that differs in any column is not detected in this version, and the README
  says so plainly.
- `duplicates="group"` keeps together the rows that agree in every feature
  column, and never looks at the target to do it (D62). Form the sets with the
  target and a held-out set has only its opposite class among the training
  rows: noise then scores far below zero, which looks like a result.
- The grouped holdout holds a quarter and no more than it must (D64). Take a
  unit while the holdout stays under a quarter, complete with the smallest
  unit passed over, return what it can spare. Units taken in seed order until
  a quarter is held make a holdout of 70% when a large one comes early. A
  unit of a quarter or more is always named: kept in training and so never
  tested, or the whole holdout because nothing smaller can fill it (D66).
- A date is a date however it is written. Whole numbers that all read as
  `YYYYMMDD` are excluded like a date column, before the id-like rule can
  call them an identifier (D66). Only that form: a day count or `YYYYMM`
  cannot be told from a measure, and the spec says so.
- A sentinel in a column of dates is one value, not a reason to call the
  column a number (D69, D72). Count distinct values: dates, and values that
  are no dates. A share of the cells is the wrong count, because `99991231`
  can fill a third of a column of end dates. And never order such a value in
  time: as the time column it stops the run.
- What `"group"` promises is checked after the leak screen too (D69). The sets
  are formed before the split; the screen then removes a column, and rows that
  differed only there are on both sides. One such holdout row stops the run.
  Counting only copies, target and all, misses a column that copies a
  two-valued target: the two classes of a combination are different rows.
- A cap that keeps whole entities must not leave out a large one (D70, D72).
  Keep every entity of a quarter or more of the usable rows first; stop when
  that cannot be done. Left to the draw, an entity of half the table is
  dropped on most seeds and only a row count says so.
- The reference is not the answer (D56). When the generated-table test shows
  the application and the reference disagreeing, the spec says which is
  wrong. If it says neither, stop and report the table: do not copy the
  reference's behaviour.
- Nothing is open since the eighth pass (D73, D74). What the judge's tools
  list as a stated limit is accepted by a decision: do not add a rule for it
  on the way past. A failure on neither list is yours.
- Read a column once. What an object column is, is decided over all its loaded
  rows; a second reading over the usable rows only lets a cell that should
  stop the run slip out with its row.
- A model fails for what the model does: its making, its `fit`, its
  prediction, the shape and finiteness of what it hands back. The tool's own
  scoring is outside that `try`. A fault there must end the run, not fill the
  models table with `failed`.

## Common failures and their real causes

| What you see | The real cause | Where to look |
|---|---|---|
| `model.noise_clf` or `model.noise_reg` fails with skill well above the band | The chosen configuration was refitted on training plus holdout rows before scoring (F1). | The refit after selection: which row positions does it receive? |
| A recording-fake test reports that a step's `fit` saw holdout rows | The preprocessing chain was fitted before the split, or once for all folds (F1, small form). No band can see this; the fake sees it for a model and for a step it supplied. | Where preprocessing objects are created. One per fit, inside the fold loop. |
| `test_guard_final_fit.py` fails: a matrix handed to `fit` moved when only holdout values changed, or has a column for a level its rows never held | The built-in imputer, scaler or encoder learned from rows outside the fit (M1, M2). The fake cannot see inside the built-in preprocessing, so this guard compares the matrices. The application is wrong, not the test. | Which frame the built-in `Preprocessor` is fitted on, in the fold loop and for the final fit. |
| `test_guard_leak_screen.py` fails: a column is excluded as a leak that only holdout rows make look like one | The screen's trees were fitted on rows that include the holdout, or its missing-by-class shares were counted over every row (M3). | The rows handed to the leak screen: training rows, and each fold's own training part. |
| `test_guard_uncertainty.py` fails with an interval of some width where every block sums the same | The "block" bootstrap draws timestamps one at a time (M4). Each draw is a start position; the block is that timestamp and the next L − 1. | The resampling under time order. |
| `test_guard_uncertainty.py` fails with p = 1/(R+1) for a constant prediction | The p-value counts shuffles strictly above the observed skill (M5). A tie counts: "at least". | The comparison in the p-value. |
| `time.flat_holdout_reported` fails with a `ZeroDivisionError`, or with a skill of 0 | Skill was divided by a baseline error of exactly zero, or a zero was written in its place (D36). It is null, with a `skill_undefined` warning; the score keeps its interval, and only the skill interval loses the resample. | Where skill is computed: the headline, each resample, the leak screen, `stages.skill`. |
| A categorical column gives more one-hot columns than it has categories, or the leak screen misses a copy of the target written with stray spaces | Levels were taken as written, so `red` and `red ` counted as two (D42). | Strip once, where a column is typed, so the counts, the screen, the steps and the matrix all see one level. |
| A feature is listed in `columns` and no model ever saw it | Two matrix columns had one name and the second overwrote the first: a numeric column `c=red` beside a categorical `c` with the level `red` (D37). | Stop with `feature_name_collision` at step 5, judged on every level of the usable rows. |
| A configuration with status `failed` has null fold scores although its folds were scored | It failed after it was chosen and its record was wiped (D38). Only one that fails during selection has null fold scores. | The fallback after selection. |
| `csv.field_size_limit()` is different after a run | The loader raised the limit for the whole process and did not put it back. | A `try` / `finally` round the read. |
| A bare `UnicodeDecodeError`, or the wrong line taken as the header | The header line is read outside the handler that turns a decode error into `input_error`; or leading lines were skipped with `str.strip()`, which also removes a form feed and a no-break space, where pandas skips only spaces and tabs. | The loader's own reading of the header line. |
| Fold scores look better than the holdout score, by a lot | Preprocessing was fitted once on all training rows and reused inside the folds, so each validation block leaked into its own training statistics. | The fold loop: `fit` must be called on the fold's training rows. |
| `leak.excluded` fails with skill near 1 | The leak screen is missing, runs after the model, or screens on rows that include the holdout (F2). | Step 8 of the spec: the screen runs first, on training rows, with the selection folds. |
| Every strong feature is flagged as a leak under time order | The screen uses random folds, or measures skill against the training mean only. Under time order it uses the expanding folds, and for regression the previous-value baseline, with unique timestamps and with repeated ones (D74). | The screen's folds and baselines. |
| An id column is among the features and skill is odd | The id-like rule was applied after the row cap, or by dtype: one blank cell turns an integer id into `float64`. | Roles are decided before the cap, by value. |
| A real integer measure (price, volume) is excluded as id-like | The 95%-distinct rule was applied to a number column on its own. For whole numbers it is half of a clause: the distinct values must also fill at least half of their range (D74). | Rule 8b: distinct, and consecutive or strictly ordered; or more than 95% distinct and filling half of the range. A distinct measure in a table sorted by it is still caught: a stated limit (D31). |
| `task.binary_numeric` fails: 0/1 target treated as regression | The task was read from the dtype. With a blank cell a 0/1 column loads as `float64` (F5). | Count distinct values; ignore the type. |
| `data.rows_reconcile` fails by a few rows | `dropna()` somewhere, or blank lines counted as rows, or rows dropped before `loaded` was counted (F6). | The first line that changes the number of rows. |
| `spend`-like column is all missing or became categories | Coercion was done with `astype(float)`, or not at all. Use `to_numeric(errors="coerce")`, count what failed, warn (F6). | Column typing. |
| `time.no_skill_on_blocks` fails with skill near 1 | The split is shuffled although a time column was declared, or the file order was used instead of the time column (F7). | The split under time order; the sort. |
| `time.gap_applied` fails on the folds | The gap was removed before the holdout but not before each validation block (F7). | `validation.folds[].gap_rows`. |
| A day-first date column runs and gives a result | The time column went through pandas' lenient parser. Text must match ISO 8601 with the year first, or the run stops (F7). | Time column validation. |
| `time.persistence_baseline` fails with skill near 1 | The regression baseline is the training mean only (F8). | Baselines under time order (step 11). |
| `model.headline_recomputed` fails | The headline is a cross-validation score (F3), or accuracy (F4), or was computed from other predictions than the ones written. | The variable assigned to `headline.score`. |
| `model.signal_reg` fails with skill near -1 | The worst configuration was chosen: squared error is lower-is-better. | The comparison in selection. |
| `output.deterministic` fails | A clock value outside `meta.timing`, an unseeded generator, a set written in iteration order, or a library thread pool left at its default. | Diff the two files. |
| `output.same_at_any_n_jobs` fails in the last digits | Forest predictions were summed across threads, or BLAS was not held to one thread. `n_jobs=-1` or unset thread variables make a score move between runs. | `threadpoolctl.threadpool_limits`; `n_jobs=1` at predict time for forests. |
| A score moves after `pip install` | A library changed version. | `requirements.txt` against `meta.versions`. |
| Several commands started together each take 35 or 70 seconds longer than they should; or a gate run reports "the tool did not finish" and passes when repeated | Not the code. Until D32 the interpreter's base was on a mapped drive, and Python processes started at the same instant blocked each other for exactly 35 or 70 seconds. The base is a local path now and the wait was not seen in 60 rounds of eight simultaneous starts. The gate still spaces its own runs and retries once. | Run the check again before changing anything. If it comes back, check `home` in `mlprof_env\pyvenv.cfg` and tell the human. |
| NaN in `profile.json` | A number that is not finite was written as it was. If it has no value for a stated reason (skill over a baseline error of exactly zero), write `null` and the reason. If it overflowed, do not write `null`: fail the configuration or stop with `non_finite_score` (D44). | `json.dumps(..., allow_nan=False)` will find it. |
| `score`, `baseline_score` or `cv_mean` is null on a run that finished; or a p-value beside a null skill | A number that was infinite or not-a-number was turned into `None` on the way out, where the result is built. A target value near 1e155 or more does it: its square is beyond the largest double (D44). | Where each number is computed, not where the result is built. Baselines first, before any fit. |
| `all_models_failed` on a table with one huge target value | Every configuration's squared error overflowed, one after another, and nobody said the target was the cause. | Score the baselines on the holdout before the leak screen and the models; stop there with `non_finite_score`. |
| A raw `ValueError: Input X contains infinity or a value too large for dtype('float32')` | A finite feature value above 3.4028234663852886e+38 reached a tree: scikit-learn's trees and forests read 32-bit floats. The leak screen's tree reads the column as it is. | Read such a cell as an infinite one is, where features are typed: missing, counted in `n_infinite`, warned (D48). |
| A raw `InvalidParameterError` about `random_state`, or a traceback from the command line for a large `--seed` | The seed was checked for "at least 0" only. scikit-learn takes 0 to 4,294,967,295 (D45). | The argument checks, before anything is loaded; `stages.split_random` too. |
| `task_ambiguous` with "4 distinct values" on a yes/no target; or a positive class reported as `yes ` | Target text was taken as written. White space round a target value is not part of it, and a cell of only white space is a missing target (D46, D48). | Strip once, where the target is first read, so the task, the classes, the split and `y_true` all see one value. |
| The fast profile says "use profile='full'" and the full profile then stops too | The fast error was raised whenever the cap had dropped a row. It is raised only when the split of every usable row can be formed (D46). | Try the split on all usable rows with the caller's gap; raise what it raises, or the fast error if it succeeds. |
| The summary shows `mse 0.0000` or a p-value of `0.0050` | Fixed decimals. A number in the summary agrees with `profile.json` to four significant digits at any scale (D46). | The number format of the summary: significant digits, not decimals. |
| Extra lines on standard error before `error: non_finite_score: ...` | numpy printed its overflow warnings. Exit 3 carries exactly one line. | `numpy.errstate(over="ignore", invalid="ignore")` round the arithmetic that can overflow. |
| `test_guard_header_lines.py` fails: a file whose first line is a form feed or a no-break space loads | The loader took a line for blank that pandas reads as the header (F6d's ground). `str.strip()` removes more than spaces and tabs. | The loader's reading of the header line, and its check for rows longer than the header. |
| `test_a_prediction_that_is_not_finite_counts_as_raising` fails | A NaN prediction was scored and written. ROC AUC is a count of ranks, so the score looks fine and the NaN reaches `y_pred` (M6). | Check every prediction is finite before it is scored, in the folds and on the holdout. |
| Skill near 0.9 on a table that should have none, with a `duplicate_rows` warning; or `data.holdout_copies_stop` fails with exit 0 | Rows are present more than once, so holdout rows have copies among the training rows and a tree recalls them (F9). | Spec step 8a: count the copies after the leak screen, before any fit, and stop with `holdout_copies_in_training` when the rule is met. |
| `time.undeclared_warns` fails with exit 2 and "unrecognized arguments: --duplicates keep" | The command line does not take `--duplicates` yet. That run passes `keep` because every one of its holdout rows has a copy once the date column is excluded (D53). | The command line and `profile()`: the tenth parameter, `duplicates`. |
| `holdout_copies_in_training` on a table with a date column and no repeated rows | The date was not declared, so it is excluded, and rows that differ only in their date are copies. The message names the column. | Not a fault: the caller declares `time_col`, or passes `duplicates="keep"`. |
| `time.integer_keys_distinct` fails with 439 and 148 timestamps where the gate counts 1,125 and 375 | Whole-number time keys went through float64, which holds only every 256th whole number at 1.7e18 (F7c). | Where the time column becomes keys: integers above 2**53 stay `int64` or `uint64`, through the cap, the split, the previous value and the bootstrap blocks. |
| A column of large distinct whole numbers is reported `constant`, or an id column above 2**53 is a feature | Distinct values were counted on floats (D50). | Rules 8, 8a and 8b of step 5, on the integers of an integer-typed column. |
| A raw `TypeError` naming `isinf`, or any raw error on a CSV with a 26-digit number | pandas loads a whole number beyond 64 bits as a Python integer in an `object` column, and numpy's functions refuse it (D50). | Where object cells are first looked at: such a column is `unsupported_type` with a warning, `task_unsupported` as the target, `time_column_invalid` as the time column. |
| A table read from a database scores no skill, and its number columns are `excluded` / `unsupported_type` | The driver returned `Decimal`, or booleans and whole numbers with `None`: object columns (D51). | Spec step 1a: type such a column once, before any step. |
| A raw `decimal.InvalidOperation` | pandas' `isna` raises on `Decimal("sNaN")`. | Find the empty cells of an object column yourself before pandas is asked; a decimal not-a-number of either kind is an empty cell. |
| A raw `TypeError: unhashable type` | The target holds lists, dictionaries or sets, and counting its distinct values hashes them (D53). | Step 4: `task_unsupported`. |
| A run finishes and then a raw `TypeError` says an object "is not JSON serializable" | A class value of a type JSON cannot hold (bytes, a period, a complex number) was put into `task.classes` as it was (D53). | Where `task.classes` is built: a boolean, a number or text; anything else as `str(value)`. |
| `time_column_invalid` on a CSV pandas wrote from microsecond timestamps | Exactly three digits of a fraction were accepted. One to nine are (D52). | The ISO 8601 pattern of the time column, and the refusal's message, which names the fraction when that is all that is wrong. |
| `too_few_rows: 60 rows are left` under `fast` on a table the full profile runs | The cap kept whole timestamps holding at most 2,000 rows, and the latest one is small; or a rare class fell out of the sample (D52). | After the cap: if rows were left out and the full profile would get past this point and form its split, `fast_profile_too_few_rows`; otherwise the full profile's own error. |
| `test_guard_review3.py` fails | One of six pinned details moved: the percentiles (2.5 and 97.5), the leak screen's previous-value baseline, the seeded sample of the fast cap, the sample deviation in `cv_std`, shared ranks for ties in every resample, `n_imputed` over training and holdout rows only (M8 to M13). The application is wrong, not the test. | The test's name says which. |
| `test_type_matrix.py` fails with a raw exception in a case named kind-role-extreme | A kind of column, in that role, with that value, reached code not written for it. | The first frame of the traceback inside the package. Decide what the spec says of that kind (steps 1a to 5), and make that place do it. |
| `data.rows_reconcile`, `data.columns_accounted`, `data.coercion_reported` or `time.undeclared_warns` fails with exit 2 and "unrecognized arguments: --no-time-order" | The command line does not take `--no-time-order` yet (D58). Those runs are on tables with a date column and no time column, and say so. | T42: `time_col=False` in `profile()`, `--no-time-order` on the command line. |
| `time.undeclared_stops` fails with exit 0 | A table with a date column and nothing said about time order was split at random (F7d). | Step 5: stop with `time_order_undeclared` when `time_col` is `None`, whatever the columns hold (D63). |
| `statements.time_order_asked` or `statements.entities_asked` fails with exit 0 | A statement the caller did not make was assumed (F11): `None` was read as `False`. | Step 5: `time_col is None` stops with `time_order_undeclared`; otherwise `group_col is None` stops with `entities_undeclared`. No condition on the table. |
| The same two fail with exit 2 | The pair was made `required` for argparse. A missing pair is the function's own error, exit 3, with its message. | The command line: pass `None` through and let `profile()` stop. |
| `statements.time_order_asked` fails: the message does not name `--group-col` | With both unsaid the message asked for time order only (M24). | The message of `time_order_undeclared`: when `group_col` is `None` too, it asks for both. |
| An older test fails with `time_order_undeclared` or `entities_undeclared` after T49 | It reaches `profile()` by a road that does not answer for it. The suite's helpers (`mlp_helpers.stated_profile`, `run_cli`, `cp5_helpers.raised`) repeat a call with the missing statement only when the run stops for one. | Not the test: check that the stop is raised at step 5 with exactly these two codes, and that an error of steps 1 to 4 still comes first. |
| `duplicates.group_on_features` fails: combinations on both sides; or `duplicates.group_no_skill` fails with skill near -0.9 | The sets of `duplicates="group"` were formed with the target (F12). | Step 7: the key of a set is the feature columns of step 5, nothing else. |
| `validation` has no key `n_rows_same_features`, or the summary shows `copies_grouped` | The two names D66 changed. | Where `validation` and the warning are built: `n_rows_same_features`, `same_features_grouped`; the old names are not written. |
| `groups.holdout_bounded` fails with a holdout of three quarters of the rows or more | Units were taken in seed order until a quarter was held, and the large one came before the small ones had filled it (F13). | Step 7, property 3: take while the holdout stays under a quarter, complete with the smallest unit passed over, return what it can spare. |
| The same check fails on the warnings | A unit of a quarter or more was not named, or by the wrong code. | One `large_group_kept_in_training` for each such unit among the training rows; one `large_group_is_holdout` when it is the holdout. |
| `time.date_numbers_seen` fails: the column `day` is a `feature`, or `id_like` | Dates written as whole numbers were read as a measure, or the id-like rule was asked first (F14, M29). | Step 5, rule 2a, after rules 1 and 2 and before every other. |
| `groups.kept_together` fails: entities on both sides, with `validation.scheme` saying `grouped` | The entity column was read and reported, and the split was made row by row all the same (F10). | The split of step 7: it takes the entities, not the rows. Then the folds, the cap and the bootstrap. |
| `groups.no_skill_on_entities` fails with skill near 0.8 | The same: a row of an entity on each side lets a model know the entity again. | As above. |
| `time.panel_previous_value` fails: `baselines` lists only `training_mean`, skill near 1 | With entities declared the previous value was not computed per entity, or was dropped because the timestamps repeat (F8b). | Step 11: each entity's latest row at least `gap + 1` timestamps earlier. |
| The lag column of a panel is excluded as `suspected_leak` although the entity column is declared | The leak screen measured it against the training mean only (M16). | Step 8: the screen's baseline uses each entity's previous value too. |
| `entities_undeclared` or `time_order_undeclared` on a table that ran before | Nothing was said about entities, or about time order. Since D63 every call says both. | Not a defect: pass the column's name, or `False`. |
| A table with rows that have a copy runs under `duplicates="group"` and then stops with `holdout_copies_in_training` | The leak screen excluded a column, and rows that differed only there became copies on opposite sides. | Not a defect: the message names the column; the caller removes it. |
| `test_type_matrix_group.py` fails with a raw exception | A kind of column, as the entity column, reached code not written for it. | Where the entity column's cells are compared: fix it there. No broad `except`. |
| A result holds `null` where a number belongs in `task`, `columns` or `input`; or `test_review4_small.py` fails | The finite check looked at the five scoring parts only (finding 4). | The constructor of `ProfileResult`: every part of the document. |
| `time.date_sentinel_seen` fails: the column `day` is a `feature` | Every value was asked to read as a date, or 95% of the cells: one sentinel, or a sentinel in a third of the rows, and the column is a number again (F14b, M31). | Step 5, rule 2a: count the distinct values that read as dates and those that do not; nine for each, or five beside at most two. |
| A run with a date-number time column finishes with `train_end` of 99991231, or with its last rows first | A value that is no date was ordered as a number (M32). | Step 3: such a time column stops with `time_column_invalid`, naming the value and its rows. |
| `duplicates.group_leak_stops` fails with exit 0, and skill near -1 | After the leak screen removed a column, combinations of the features that are left were on both sides, and only exact copies were counted (F12b); or the count kept the threshold of 1% and ten rows (M33). | Step 8a, under `"group"` and after a removed leak: one holdout row that agrees with a training row in every feature left stops the run. |
| `fast.large_entity_stops` fails with exit 0 | An entity larger than the cap was left out and the rest profiled (F15). | Step 6: stop with `fast_profile_group_too_large` before any row is left out. |
| `fast.large_entity_kept` fails: 1,500 rows kept and 1,200 left out | The large entity was taken in seed order like any other and did not fit when its turn came (F15, M35). | Step 6: keep every entity of a quarter or more of the usable rows first. |
| `fast_profile_group_too_large` on a table the full profile cannot split either | The advice is not true (M34). | Try the full profile's own checks on every usable row first: a class under 10 rows, the split of whole entities. Raise what they raise. |
| The `date_column_undeclared` warning says the rows were split at random under `--group-col` | Its text was written at step 5, before the scheme was known (M36). | Finish the message after the split, with `validation.caveat`. |
| The interpreter ends with exit code 3221225477 and nothing on standard error, on a time column with one date such as `9999-12-31` | pandas was asked to turn a column that holds a date beyond the nanosecond range into nanoseconds, and did not raise. No `except` catches that. | Ask the range before converting: compare with `pandas.Timestamp.min` and `.max`, and raise `time_column_invalid`. |
| `verify/battery.py` fails a pair of the kind `same` | A change that cannot move a number moved one: a constant, an identifier or a renamed column reached a model, or changed the split. | The message names the run, the change and the first number that moved; open the two output folders it names. |
| `verify/generated_tables.py` reports a disagreement | The application and the reference differ on one generated table. Agreement proves neither right. | Make the table again with `--only i --describe`, read the spec for that case, and fix the application only where the spec says it is wrong. Otherwise stop and report. |
| Either tool lists a pair as a `stated limit` | It is known, and accepted by a decision (spec 8.2: four kinds for the battery, none for the generated tables). Nothing is open since the eighth pass. | Leave it. It does not fail the tool. A failure on neither list does. |
| `time.previous_value_at_repeats` fails: `baselines` lists only `training_mean`; or a walk with one repeated row stops with `no_features` | The previous value was computed only while the timestamps were unique (F23). | Step 11: the mean target of the timestamp `gap + 1` distinct timestamps before, for the headline and for the leak screen. |
| `leak.copy_excluded_under_time` fails: `level_scaled` is a feature and skill is 0.9 or more | The screen reads with its tree alone, and a tree cannot follow a trending target beyond the range it was fitted on (F24). | Step 8: the line beside the tree; the better reader counts. |
| `leak.excluded_despite_extreme_target` fails: the copy is a feature beside one target cell of -999 | No row was set aside, or every row was scored for the cut-off that excludes (F25). | Step 8: rows set aside by medians; the worst 1% unscored for 0.98 only. |
| Every column is excluded as a leak on a table with one huge target cell | The row was left out of the screen's sums and is still in each fold's training mean (M42). | Step 8: a row set aside is in no fold's training rows and in no fold's validation rows. |
| A copy of a two-valued target stays a feature where a class is rare | The screen's tree needs five rows in a leaf (M47). | Step 8: the direct reading, with no minimum leaf. |
| A `strong_single_column` warning on the lag of a series that only rises | The order reading was taken without its second condition (M43). | Step 8: the same share for the previous value must be above 0.10. |
| A column is excluded as `suspected_leak` for its order alone | The order reading excluded it (M44). It warns and nothing more. | Step 8. |
| `result.largest_row_reported` fails on a share | The share was taken of absolute errors (F26), or of other rows than the gate counts. | Step 11: the largest squared error over the sum; the first row among equals. |
| An `extreme_value` warning on a long-tailed column, or none on a sentinel | The gap was compared with the whole range of the column, or with the wrong ten values (M46). | Step 11: the three outermost distinct values at each end; the gap against the spread of the ten distinct values after the next one inward; more than 30 times. |
| `data.identifier_survives_shuffle` fails: `record_id` is a `feature` | D19's clause alone was applied: shuffled, the values are neither consecutive nor ordered (F27). | Rule 8b's second clause. |
| `groups.choice_not_made` fails: no `selection_not_made` warning, or `tree-0` chosen | The choice was made on folds that validate on a few rows (F28). | Step 9: count each fold's training and validation rows; rank in configuration order. |
| A forest's fold scores differ from the reference's under time order while the linear model's agree | The rows of a fit were handed over in file order (M49). | Step 9: time order, the rows of one timestamp in file order. |
| `verify/battery.py` fails a pair of the kind `band-or-named` | One extreme cell moved skill out of its band, and the result names neither the row nor the value; or it is still out of band without the named rows. | The two statements of step 11: the largest row's share, and `extreme_value`. |

## How to debug a gate failure

1. Read the whole message. It names the dataset, the value and the likely cause.
2. Open the files the gate just read: `verify/artifacts/runs/<run>/profile.json`
   and `predictions.csv`. The dataset is `verify/artifacts/data/<name>.csv`.
   That folder is always the last gate run on the build: the gate's self-test
   writes to `verify/artifacts/selftest_runs/` instead.
3. Re-run the one check: `python verify/smoke_test.py mlprofessor --only <id>`.
4. Fix the code. Never the band, the dataset or the check: they are locked, and
   a band you could widen proves nothing.

## The judge's two other tools

Neither is part of the gate. Run them from the repo root when a task says so,
and before you call the last task of a pass finished (D71, D72):

```
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/battery.py mlprofessor
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/generated_tables.py mlprofessor
```

- **The battery** changes one small thing in each of the gate's tables and
  asks whether the conclusion still holds. `--list` prints every pair of a run
  and a change, and what must hold for it; `--only run:change` runs one again.
  Its tables and outputs are under `verify/artifacts/battery/`.
- **The generated-table test** runs 1,000 generated tables through your build
  and through the reference and compares them. `--only i` runs table i again
  and `--describe` prints what it holds; its files are under
  `verify/artifacts/generated/`.
- Both end with a line of three counts. Only a failure outside their two fixed
  lists fails the tool; that failure is yours. Since the eighth pass nothing
  is listed as open; the battery lists four kinds of stated limit, and the
  generated tables none (spec 8.2).
- `tests/test_judge_tools.py` runs both, so the full suite takes about seven
  minutes longer once that file exists (the battery about five minutes at
  its default of 32 runs side by side, the generated tables about two,
  measured on 2026-11-25 with nothing else on the machine and the pairs of
  the eighth pass). For the eighth pass that file accepts exactly the failures the plan
  lists, and the strict test of both tools belongs to the pass's last task.
  While working on one task, run that task's own test files.
