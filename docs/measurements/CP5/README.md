# CP5 measurements

The scripts behind the CP5 session (2026-11-20): the third review's findings
(D49 to D52) and the rulings that followed the same day (D53, D54). Kept here
by D35. The conclusions are in `spec.md` (10.6, section 7, section 9) and
`plan/test-plan.md` ("Fourth pass"); this folder is how to get the numbers
again.

| File | What it is |
|---|---|
| `copies.py` | How many holdout rows have a copy among the training rows: every run of the gate, the example table and duplicated data, over a number of data seeds. The count the tool reports and a count of the script's own, through the seam. Behind the rule of D53. |
| `copies_in_tests.py` | The same count in every run the locked suite makes, to find the tests that profile a table with repeated rows. Behind D53's count of 10. |
| `type_matrix.py` | The type matrix of D52, run against one implementation: every kind of column, role and extreme, and how each case ends. |
| `facts.py` | The pandas and numpy facts behind D50, D51 and D52: how `read_csv` loads whole numbers, where floats merge them, what pandas' type tests say of an object column of numbers, fractions of a second, dates beyond the nanosecond range. |

Three older scripts were used again and are not copied:
`docs/measurements/CP2/long_rows.py` (`values`, `compare`: every value of a
full gate run, in two states), `docs/measurements/CP4/prototypes.py` (prototype
tests against the application, the reference and its variants) and
`docs/measurements/CP4/fixtures.py`.

Raw outputs are not kept. Every script prints its result; none writes inside
the repo. The prototype tests themselves are not kept: the tests stage writes
the real ones from the plan.

## Running a measurement again

From the repo root, with the project interpreter, one command at a time:

```
$env:MLPROF_LAB_WORK = "C:\somewhere\outside\the\repo"
$py = "E:\CODE\ml-professor\mlprof_env\Scripts\python.exe"
$W  = "$env:MLPROF_LAB_WORK\cp5"
```

| What it backs | Command | Takes |
|---|---|---|
| The rule on copies, 30 data seeds | `& $py -B docs/measurements/CP5/copies.py measure 30 "$W\copies.jsonl"`, then `... copies.py table "$W\copies.jsonl"` | 5 min |
| Which locked tests hold copies | `& $py -B docs/measurements/CP5/copies_in_tests.py "$W"` | 2 min |
| The type matrix on the application | `& $py -B docs/measurements/CP5/type_matrix.py . application "$W\matrix_app.tsv"` | 35 s |
| The type matrix on the reference | `& $py -B docs/measurements/CP5/type_matrix.py verify/reference reference "$W\matrix_ref.tsv"` | 35 s |
| The pandas and numpy facts | `& $py -B docs/measurements/CP5/facts.py` | 2 s |
| Every `catches` of the gate proved | `& $py -B verify/smoke_test.py --selftest` | 10 min |
| Every value of a full gate run on the reference | `& $py -B docs/measurements/CP2/long_rows.py values - - "$W\gate_after.json"` | 20 s |
| "No existing gate value moved" | `& $py -B docs/measurements/CP2/long_rows.py compare "$W\gate_before.json" "$W\gate_after.json"` | |
| The suite against the reference and its 27 variants | `& $py -B verify/mutation_check.py --jobs 8` | 6 min |
| The same with prototype tests laid over the suite | `& $py -B verify/mutation_check.py --jobs 8 --overlay <folder of prototypes>` | 15 min |
| The new tests, red or green today | `& $py -B docs/measurements/CP4/prototypes.py <folder of prototypes> app good F9 F7c F6b M8 M9 M10 M11 M12 M13` | 4 min |

`gate_before.json` comes from the CP2 `values` command given the state before
CP5: `values <old>\verify\smoke_test.py <old>\verify\reference
"$W\gate_before.json"`, where `<old>` is a copy of the repo at commit
`13695d8`, outside this folder. The human makes it, since agents do not run
git here; in the session it was a copy of `verify/` taken before the first
edit. The reference "as it stood" in the tables below is that copy.

## Seeds

- `copies.py`: the gate's datasets at the base seeds 1115, 1215, …, 4015 (the
  gate's own is 1115; a dataset's seed is the base plus its position); the
  tool at seed 0 for the gate's runs, and at seeds 0 to 29 for the example
  table. Duplicated tables: the rows to repeat, and the order, from
  `random.Random(f"dup.{base}.{share}")`.
- `type_matrix.py`: every table from `numpy.random.default_rng(0)`, 200 rows;
  the tool at seed 0, fast profile, one core.
- The prototypes: the fixed seeds written in the test plan's recipes; the tool
  at seed 0 unless the test says otherwise.
- Two gate datasets were added, last in the list, so their seeds are 1134
  (`dup_noise`) and 1135 (`int_stamps`) and every other dataset's seed and
  bytes are what they were.

## What was found

pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1; 32 cores, 64 threads.

### A: copies of holdout rows among the training rows (D49, D53)

`copies.py measure 30`, then `table`. 1,440 runs; the tool's count and the
script's own differ in none.

| Kind | Run | Runs | Outcome | Holdout rows | Rows with a copy | Share | Meets the rule |
|---|---|---|---|---|---|---|---|
| gate run | signal_clf | 30 | random | 375 | 0 | 0% | 0 of 30 |
| gate run | signal_reg | 30 | random | 375 | 0 | 0% | 0 of 30 |
| gate run | noise_clf | 30 | random | 375 | 0 | 0% | 0 of 30 |
| gate run | noise_reg | 30 | random | 375 | 0 | 0% | 0 of 30 |
| gate run | imbalanced | 30 | random | 500 | 0 | 0% | 0 of 30 |
| gate run | leak | 30 | random | 375 | 0 | 0% | 0 of 30 |
| gate run | leak_reg | 30 | random | 375 | 0 | 0% | 0 of 30 |
| gate run | near_leak | 30 | random | 375 | 0 | 0% | 0 of 30 |
| gate run | binary_numeric | 30 | random | 375 | 0 | 0% | 0 of 30 |
| gate run | messy | 30 | random | 375 | 0 | 0% | 0 of 30 |
| gate run | full_noise_small | 30 | random | 100 | 0 | 0% | 0 of 30 |
| gate run | full_signal_clf_j1 | 30 | random | 125 | 0 | 0% | 0 of 30 |
| gate run | full_signal_reg_j1 | 30 | random | 125 | 0 | 0% | 0 of 30 |
| gate run | ambiguous_target, long_rows, time_dayfirst | 30 each | a named error, as the gate expects | - | - | - | - |
| gate run | time_blocks, time_blocks_gap, random_walk, int_stamps, flat_tail | 30 each | time_ordered | - | no count is made | - | - |
| gate run | **time_blocks_undeclared** | 30 | random | 500 | **494 to 500** | 98.8% to 100% | **30 of 30** |
| gate run | **dup_noise** (new) | 30 | random | 375 | **263 to 305** | 70.1% to 81.3% | **30 of 30** |
| gate dataset, time column not declared | time_dayfirst | 30 | random | 500 | 485 to 500 | 97.0% to 100% | 30 of 30 |
| gate dataset, time column not declared | random_walk | 30 | `no_features` in 28, random in 2 | 375 | 0 | 0% | 0 of 2 |
| gate dataset, time column not declared | int_stamps | 30 | random | 375 | 0 to 1 | 0% to 0.27% | 0 of 30 |
| gate dataset, time column not declared | flat_tail | 30 | random | 375 | 0 | 0% | 0 of 30 |
| example table | examples/customers.csv, fast and full | 30 each | random | 100 | 0 | 0% | 0 of 60 |
| example table | tests/fixtures/example.csv, fast and full | 30 each | random | 60 | 0 | 0% | 0 of 60 |
| duplicated | signal_clf + 0.5% of its rows | 30 | random | 377 | 1 to 5 | 0.27% to 1.33% | 0 of 30 |
| duplicated | signal_clf + 1% | 30 | random | 379 | 2 to 9 | 0.53% to 2.37% | 0 of 30 |
| duplicated | signal_clf + 2% | 30 | random | 383 | 4 to 15 | 1.04% to 3.92% | 24 of 30 |
| duplicated | signal_clf + 5% | 30 | random | 394 | 21 to 36 | 5.33% to 9.14% | 30 of 30 |
| duplicated | signal_clf + 10% | 30 | random | 413 | 46 to 71 | 11.14% to 17.19% | 30 of 30 |
| duplicated | signal_clf + 25% | 30 | random | 469 | 123 to 153 | 26.23% to 32.62% | 30 of 30 |
| duplicated | signal_clf + 50% | 30 | random | 500 | 199 to 252 | 39.80% to 50.40% | 30 of 30 |
| duplicated | signal_clf + 100% | 30 | random | 500 | 231 to 276 | 46.20% to 55.20% | 30 of 30 |
| duplicated | signal_reg + 0.5% | 30 | random | 377 | 0 to 6 | 0% to 1.59% | 0 of 30 |
| duplicated | signal_reg + 1% | 30 | random | 379 | 1 to 10 | 0.26% to 2.64% | 1 of 30 |
| duplicated | signal_reg + 2% | 30 | random | 383 | 6 to 15 | 1.57% to 3.92% | 23 of 30 |
| duplicated | signal_reg + 5% | 30 | random | 394 | 20 to 37 | 5.08% to 9.39% | 30 of 30 |
| duplicated | signal_reg + 10% | 30 | random | 413 | 47 to 65 | 11.38% to 15.74% | 30 of 30 |
| duplicated | signal_reg + 25% | 30 | random | 469 | 121 to 158 | 25.80% to 33.69% | 30 of 30 |
| duplicated | signal_reg + 50% | 30 | random | 500 | 198 to 242 | 39.60% to 48.40% | 30 of 30 |
| duplicated | signal_reg + 100% | 30 | random | 500 | 222 to 285 | 44.40% to 57.00% | 30 of 30 |

(The duplicated tables of 50% and 100% have more than 2,000 rows, so the fast
profile's cap keeps 2,000 and the holdout is 500.)

The two ranges CP5 asked for:

- **Data with no repeated rows:** 0 rows on every one of the 13 gate runs
  under the random scheme and on the example table, all 30 seeds; at most 1
  row, 0.27%, on any table measured (`int_stamps` with its time column left
  undeclared, which is not a gate run).
- **Duplicated data:** from 5% of the rows repeated upwards, at least 20 rows
  and 5.08% on every seed. `dup_noise`, the gate's new dataset: 263 to 305
  rows, 70% to 81%.

Both parts of the rule (more than 1%, and at least 10 rows) sit between the
two with room on either side. Between them, at 1% to 2% of the rows repeated
in a table of 1,500, the count of 10 is what decides, and the outcome varies
with the seed: 1 of 60 runs meets the rule at 1%, 47 of 60 at 2%. That is a
property of the count D53 chose, not a fault: nine repeated rows in a holdout
of 380 go ahead with the count reported.

One existing gate run meets the rule: `time_blocks` with no time column. The
label of that dataset is constant over blocks of five dates and the features
identify the block, so with the date excluded the rows of a block are alike:
494 to 500 of its 500 holdout rows have a copy. By D53 that run passes
`--duplicates keep`. `time_dayfirst` run the same way is the same table with
other date text.

`copies_in_tests.py`, the locked suite against the reference with the rule
switched off and each count logged:

```
pytest: 1 failed, 410 passed
132 runs under the random scheme reached the count
2 test(s) in which a holdout row had a copy among the training rows:
     2 of   31 (  6.45%)  below the rule  test_result_schema.py::test_warnings_from_steps_one_to_eleven
     2 of   38 (  5.26%)  below the rule  test_edges.py::test_duplicate_rows_kept_and_reported
```

(The one failing test is `test_skeleton.py::test_profile_signature_and_defaults`,
until the tests stage has loosened it: D54.)

So two locked tests profile a table with repeated rows under the random
scheme, and neither meets the rule, by its count; a share alone at 1% would
have stopped both (5.3% and 6.5%). Against the application's own split,
measured in the session by the same instrumenting applied to the application
as it stood, only the first of the two has copies, 2 of 31. (That script
patched lines of the application that T32 will change, so it is not kept.)

### B: whole numbers (D50)

`facts.py`:

```
-- how read_csv loads whole numbers
int64 max                         dtype int64;   values [9223372036854775807, 1]
int64 max + 1 (fits uint64)       dtype uint64;  values [9223372036854775808, 1]
uint64 max                        dtype uint64;  values [18446744073709551615, 1]
uint64 max + 1                    dtype object;  python types ['int']; values [18446744073709551616, 1]
26 digits                         dtype object;  python types ['int']
26 digits and a blank             dtype object;  python types ['float', 'int']; values [12345678901234567890123456, nan, 1]
26 digits and a negative          dtype object;  python types ['int']
above int64 and a negative        dtype str;     values ['9223372036854775808', '-5']
below int64 and small ones        dtype object;  python types ['int']; values [-9223372036854775809, 5]
above and below int64 together    dtype str
2**53 + 1 and a blank             dtype float64; values [9007199254740992.0, nan, 9007199254740992.0]
2**53 + 1, no blank               dtype int64;   values [9007199254740993, 9007199254740992]
26 digits and text                dtype str
1e400                             dtype float64; values [inf, 1.5]
-- floats merge whole numbers above 2**53
200 consecutive int64 from 2**53 + 1, as float64: distinct        101
1500 int64 nanosecond stamps 100 ns apart, as float64: distinct   587
float64 spacing at 1.7e18                                         256.0
50 distinct uint64 near 2**63, as float64: distinct               1
-- object columns as a database driver returns them
pd.isna(Decimal('NaN'))                                           True
float(Decimal('1E+400')), float(Decimal('-Infinity'))             (inf, -inf)
float(10**400)                                                    OverflowError: int too large to convert to float
object ints and None: is_numeric / is_bool / is_integer / is_string   (False, False, False, True)
object Decimals:      is_numeric / is_bool / is_integer / is_string   (False, False, False, True)
np.isinf(10**25)                                                  TypeError: ufunc 'isinf' not supported for the input types ...
int64 min/max diff overflows silently                             [-1]
-- equality of rows: missing cells and negative zero
np.unique([0.0, -0.0, nan, nan, 1.5], return_inverse)             (array([0. , 1.5, nan]), array([0, 0, 2, 2, 1]))
-- ISO 8601 fractions
2024-01-05 09:15:00.123456                                        2024-01-05T09:15:00.123456
2024-01-05 09:15:00.1234567891                                    2024-01-05T09:15:00.123456789   (ten digits, cut to nine)
how pandas writes a microsecond timestamp to CSV                  ['2024-01-05', '09:15:00.123456']
-- datetime columns beyond the nanosecond range
datetime64[s] year 9999 .to_numpy('datetime64[ns]')               OverflowError: Overflow when converting between datetime64 units
-- what the reference relies on to read objects as typed columns
pd.isna(Decimal('sNaN'))                                          RAISES InvalidOperation
Series([Decimal('sNaN'), Decimal(1)]).isna()                      RAISES InvalidOperation
(boolean == True).to_numpy() with a missing cell                  array([True, <NA>, False, True], dtype=object)
```

(Abridged: the script prints about 60 lines.)

What follows from it:

- Integer time keys above 2**53 merge as floats: 1,500 stamps 100 nanoseconds
  apart at 1.7e18 are 587 floats (`int_stamps`), and the 600 of the review are
  235.
- `read_csv` hands the tool exact integers in three shapes (`int64`,
  `uint64`, Python integers in an `object` column) and already-rounded floats
  in one: **a whole-number column with an empty cell**. A value above the
  signed range beside a negative one arrives as text. D50 cannot reach the
  last two: the digits are gone, or the column is text, before the tool is
  called. `spec.md` section 6 says so.
- A `Decimal` with no fractional part is a whole number held exactly. Read as
  a float (the first form of step 1a in the session), a `Decimal` time column
  of nanosecond stamps merged exactly as the integers did (76 and 25
  timestamps of 200), and a `Decimal` column of 26-digit numbers was reported
  `constant`. Step 1a therefore reads such a column as whole numbers.

### H: the type matrix (D52)

`type_matrix.py`, 805 cases:

| Implementation | Result | Named error | Raw exception |
|---|---|---|---|
| the application as it stands | 439 | 329 | **37** |
| the reference as it stood | 458 | 329 | **18** |
| the reference after CP5 | 484 | 321 | 0 |

The application's 37, by where the exception starts:

```
[application] 805 cases in 34 s: RAW 37, named 329, result 439
   15  RAW: TypeError @ load.py:167 _is_infinite_number
       object_int and csv, each of feature, target and time, each of beyond_64 and beyond_64_few;
       mixed|feature|e300, mixed|target|e300, mixed|time|e300
   12  RAW: TypeError @ task.py:77 decide_task
       list, dict and set as the target, each of ordinary, two, one_value and nan
    7  RAW: TypeError @ result.py:154 to_json
       bytes|target|two, timedelta|target|two, period|target|two, interval|target|two,
       complex128|target|two, complex64|target|two, decimal|target|two
    1  RAW: InvalidOperation @ columns.py:256 column_roles      decimal|feature|nan
    1  RAW: InvalidOperation @ task.py:33 read_target_values    decimal|target|nan
    1  RAW: InvalidOperation @ load.py:146 missing_mask         decimal|time|nan
```

The reference's 18, as it stood:

```
[reference_before] 805 cases in 31 s: RAW 18, named 329, result 458
   12  RAW: TypeError @ core.py:422 _decide_task               list, dict and set as the target
    1  RAW: TypeError @ core.py:1060 _run                       boolean|target|nan
    1  RAW: OverflowError @ core.py:383 _time_keys              iso_text|time|max_min
    1  RAW: OverflowError @ core.py:348 _time_keys              datetime_s|time|max_min
    1  RAW: InvalidOperation @ core.py:493 _roles               decimal|feature|nan
    1  RAW: InvalidOperation @ core.py:453 _no_target           decimal|target|nan
    1  RAW: InvalidOperation @ core.py:354 _time_keys           decimal|time|nan
[reference_after] 805 cases in 34 s: named 321, result 484
```

When the matrix was first run, for the session's plan, it had 645 cases (no
`one_value` extreme, no ISO-text kind, no signalling not-a-number): 24 raw
exceptions and 7 results that could not be written, in the application. The
7 show as raw exceptions above because the script now passes an output
folder, as the test does.

Four causes in the application: whole numbers beyond 64 bits (15, the review's
finding 1); a target of lists, dictionaries or sets (12); a two-valued target
of a type JSON cannot hold (7: the run finishes and `profile.json` cannot be
written); a `Decimal` signalling not-a-number (3: pandas' own `isna` raises).
The reference shared the second and the fourth and had three of its own: a
nullable boolean target with a missing cell, and a date of the year 9999 in a
time column, as ISO text and as a datetime in seconds (pandas 3 keeps such a
date in a coarser unit and raises `OverflowError` on conversion to
nanoseconds). The reference also ordered a complex-number time column by its
real part, with a result and no error; the application refused it.

### G: the two new gate checks, and every older gate value

The new gate on the reference as it stood (`long_rows.py values`):

```
33 checks, 3 failed, 26 runs
  FAIL data.holdout_copies_stop: the tool exited 0 on `dup_noise`; it must stop with the named error `holdout_copies_in_training` ...
  FAIL time.integer_keys_distinct: validation on `int_stamps` reports 439 training and 148 holdout timestamps; the gate counts 1125 and 375, one per row (587 of the 1500 are left when the keys are read as floats). ...
  FAIL time.undeclared_warns: the tool exited 2 on dataset `time_blocks` where it should have succeeded. ...
```

The third is the older check whose run now passes `--duplicates keep`, which
the reference as it stood did not take. On the reference after CP5: 33
checks, 0 failed.

The same gate on the application as it stands (`values <gate> . <out>`, which
runs in the lab folder and leaves `verify/artifacts/runs/` alone): 33 checks,
the same 3 failed and for the same reasons, the other 30 pass. What the two
runs reported, in both implementations:

| Run | Application | Reference as it stood |
|---|---|---|
| `dup_noise` (pure noise, every row twice; fast profile) | exit 0; skill 0.8085; p = 1/201; `tree-0` | exit 0; skill 0.7224; p = 1/201; `tree-0` |
| `int_stamps` (a random walk, 1,500 integer nanosecond stamps) | exit 0; 439 and 148 timestamps, not unique; baseline `training_mean`; skill 0.9953 | the same: 439 and 148; `training_mean`; skill 0.9953 |

Which task makes each pass: `data.holdout_copies_stop` and
`time.undeclared_warns`, T32; `time.integer_keys_distinct`, T33.

Every value of a full gate run on the reference, before CP5 and after
(`long_rows.py compare`):

```
datasets (seed, target, bytes): 19 before, 21 after
  57 values before; 0 moved, 0 gone, 0 new under an existing name
  only after: dup_noise, int_stamps
checks (verdict, message, failure modes named): 31 before, 33 after
  94 values before; 0 moved, 0 gone, 0 new under an existing name
  only after: data.holdout_copies_stop, time.integer_keys_distinct
values the gate recorded: 100 before, 108 after
  100 values before; 0 moved, 0 gone, 0 new under an existing name
runs (exit code, both streams, files, every field of profile.json but meta.timing, bytes of predictions.csv): 24 before, 26 after
  14302 values before; 2 moved, 0 gone, 69 new under an existing name
    moved  time_blocks_undeclared: command: '... --n-jobs 1' -> '... --n-jobs 1 --duplicates keep'
    moved  time_blocks_undeclared: stdout: (the summary gained the line of the holdout_copies_kept warning)
  only after: dup_noise, int_stamps
```

The 69 new values under existing runs are what D49 adds to the output: three
keys (`input.duplicates`, `validation.n_holdout_copies`,
`validation.holdout_copy_share`) in each of the 21 runs that write a
`profile.json` (63), the same three in the result the run through the Python
function prints (3), and the three fields of the one new warning in
`time_blocks_undeclared` (3). No number of any existing `profile.json`, no
byte of any `predictions.csv` and no line of any other console summary
moved. No band was changed and none was added.

### The suite against the reference and its variants

`verify/mutation_check.py`, against the 411 cases of the first three passes
(405 counted, 6 README), after CP5:

```
mutation check: 28 case(s), 8 at a time
good   fails 1 of 405: test_skeleton.py::test_profile_signature_and_defaults   [88 s]
F1     fails 13      F2     fails 10      F3     fails 17      F4     fails 10
F5     fails 36      F6     fails 25      F7     fails 40      F8     fails 6
F9     fails 0   NOT CAUGHT: no test fails on this variant
F6b    fails 8       F6c    fails 17      F6d    fails 6       F7b    fails 6
F7c    fails 0   NOT CAUGHT: no test fails on this variant
M1     fails 1       M2     fails 2       M3     fails 2       M4     fails 3
M5     fails 1       M6     fails 1       M7     fails 4
M8     fails 0   NOT CAUGHT        M9     fails 0   NOT CAUGHT
M10    fails 0   NOT CAUGHT        M11    fails 0   NOT CAUGHT
M12    fails 0   NOT CAUGHT        M13    fails 0   NOT CAUGHT
README tests (not counted: they judge README.md, the same file in every case): 0 of 6 failing
MUTATION CHECK: FAIL
```

(The lines are shortened here: the script names up to five failing tests per
case. The session's final report has them in full.)

**The changed reference fails one of the 405:
`test_skeleton.py::test_profile_signature_and_defaults`,** which pins nine
parameters where D49 adds a tenth. D54 has the tests stage loosen that one
test. Every one of the 19 older variants is still caught. The 8 new ones are
caught by nothing in the existing suite, which is what the third review
found and why the test plan demands their tests.

The same with the prototypes of every test of the fourth pass laid over the
suite, the loosened `test_skeleton.py` among them
(`mutation_check.py --overlay`):

```
mutation check: 28 case(s), 8 at a time
  note: overlay: tests/cp5_helpers.py added ...            (8 new test files and the helper)
  note: overlay: tests/test_skeleton.py replaced by ...    (the one loosened test)
good   fails 0 of 1312   [161 s]
F1     fails 23      F2     fails 10      F3     fails 19      F4     fails 10
F5     fails 41      F6     fails 103     F7     fails 133     F8     fails 13
F9     fails 9: test_duplicates.py::test_command_line_takes_duplicates, ...::test_every_row_twice_stops_the_run[classification], ...[regression], ...::test_the_message_names_an_undeclared_date_column, ...::test_the_rule_in_regression (and 4 more)
F6b    fails 49      F6c    fails 18      F6d    fails 6       F7b    fails 33
F7c    fails 12: test_object_numbers.py::test_object_whole_numbers_are_exact[python_integers], ...[whole_decimals], test_whole_numbers.py::test_missing_and_repeated_integer_stamps, ...::test_nanosecond_stamps_are_distinct_timestamps[Int64], ...[UInt64] (and 7 more)
M1     fails 1       M2     fails 2       M3     fails 2       M4     fails 3
M5     fails 1       M6     fails 1       M7     fails 4
M8     fails 1: test_guard_review3.py::test_m8_the_interval_is_the_middle_95_percent
M9     fails 1: test_guard_review3.py::test_m9_the_leak_screen_uses_the_previous_value
M10    fails 1: test_guard_review3.py::test_m10_the_fast_cap_is_a_sample_from_the_seed
M11    fails 1: test_guard_review3.py::test_m11_cv_std_is_the_sample_deviation
M12    fails 1: test_guard_review3.py::test_m12_tied_predictions_in_a_resample
M13    fails 1: test_guard_review3.py::test_m13_n_imputed_counts_rows_that_reach_a_model
README tests (not counted: they judge README.md, the same file in every case): 2 of 8 failing: test_readme_duplicates.py::test_readme_names_the_argument_and_the_error, test_readme_duplicates.py::test_the_claim_about_unseen_holdout_rows_is_true_as_worded
MUTATION CHECK: PASS (the good reference fails nothing; each of 27 variant(s) fails at least one test)
```

With the loosened test and the new tests in place the reference fails
nothing, and every variant is caught: each of M8 to M13 by its own guard and
by no other test. (This run had 1,312 counted cases; one CSV case was added
to `test_whole_numbers.py` afterwards, which makes 1,313.) The matrix adds
cases to several older variants: F6 and F7 fail many of its cases, F6b the
`one_value` cases of the feature role.

### The tests the plan demands, checked ahead

Every test named in the fourth-pass section of `plan/test-plan.md` was
written once as a prototype, outside the repo, and run with
`CP4/prototypes.py` against the application, the changed reference, and the
reference carrying F9, F7c, F6b and each of M8 to M13.

927 cases: 910 of the eight new files and the 17 of `test_skeleton.py` with
its one test loosened.

| File | Cases | Failing on the application | Failing on the reference |
|---|---|---|---|
| `test_guard_review3.py` | 6 | 0 | 0 |
| `test_duplicates.py` | 22 | 22 | 0 |
| `test_whole_numbers.py` | 19 | 19 | 0 |
| `test_object_numbers.py` | 24 | 24 | 0 |
| `test_type_matrix.py` | 805 | 37 | 0 |
| `test_time_fraction.py` | 28 | 28 | 0 |
| `test_fast_cap_error.py` | 4 | 4 | 0 |
| `test_readme_duplicates.py` | 2 | 2 | 2 |
| `test_skeleton.py`, loosened | 17 | 0 | 0 |
| all | 927 | 136 | 2 |

- against the application 136 cases fail: the 37 red tests of the plan in
  all their 99 cases, and the 37 red cases of the matrix; the six guards and
  the other 768 cases of the matrix pass. No test is red in some of its cases
  and green in others, the matrix apart;
- against the reference all pass but the two README tests, which judge
  `README.md`;
- **the loosened `test_profile_signature_and_defaults` passes against the
  application (nine parameters) and against the reference (ten)**;
- each guard fails on its variant and on no other of the six: the M8 guard on
  M8 alone, and so on;
- on F9, nine cases of `test_duplicates.py` fail; on F7c, ten cases of
  `test_whole_numbers.py` and both of `test_object_whole_numbers_are_exact`;
  on F6b, 41 cases, the `one_value` cases of the matrix's feature role among
  them;
- the 927 cases take 59 seconds against the application and 82 against the
  reference, with eleven such suites running side by side.

Four prototypes had to be changed, and the plan says why in each recipe:

- **M9.** A plain random walk with its lag does not trip the fault: against
  the training mean alone the lag scores 0.92 (a warning, not an exclusion).
  A slow wave does: 0.993 against the mean alone, −0.85 against the previous
  value.
- **M8.** A tolerance of 0.15 left the reference's upper end (−0.04) under
  four standard errors from it; 0.18 leaves more than four on both sides.
  Measured: the reference +0.03 and −0.04; M8 +0.26 and −0.35.
- **Tests with cases that pass today** were made one test each, so that the
  test is red as a whole: the four kinds of object target (booleans and whole
  numbers already work), the tables on which the fast profile must go on
  raising the full profile's error, the day-first message beside the
  ten-digit one, and the boundary at 2**53.
- **`test_distinct_values_are_counted_exactly`** needs one ordinary column.
  Its other columns reach the models as floats, where many of their values
  are one, so without it 12 of 50 holdout rows had a copy among the training
  rows and the run stopped with `holdout_copies_in_training`. That is step 8a
  doing what it says (it compares what a model is given), and it is worth
  knowing: two rules of this pass meet there.

### The self-test and the loop's doctor

`verify/smoke_test.py --selftest` on the final state:

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
24 of 24 pairs of a check and a failure mode it names were seen failing, over 14 broken fixtures.

FOR INFORMATION: checks that give a reason in place of a failure mode, and the broken fixtures they failed on (no part of the verdict)
| leak.control_kept | F6 |
| time.flat_holdout_reported | F3, F7, F8 |
| uncertainty.reported | F1, F4 |
| model.signal_clf, model.signal_reg, time.undeclared_warns, result.matches_json, output.complete, output.deterministic, output.same_at_any_n_jobs | none: never seen to fail here |
SELFTEST: PASS (good fixture passes twice with identical results; 14 failure mode(s) each caught by a check that names it; temp and project dirs agree)
```

What each new fixture fails: F9 `data.holdout_copies_stop` alone; F7c
`time.integer_keys_distinct` alone. The new check for integer keys also fails
on F3, F7 and F8, as `time.persistence_baseline` does on F3 and F7: it
recomputes the score and both baselines. `time.undeclared_warns` has still
never been seen to fail on a fixture of the gate; it fails on the reference
as it stood and on the application, for want of the `--duplicates` option.

`loop/run_loop.py doctor` on the final state:

```
  ok   gate lists 33 check(s)
  ok   gate self-test: SELFTEST: PASS (good fixture passes twice with identical results; 14 failure mode(s) each caught by a check that names it; temp and project dirs agree)
  ok   gate timing: 13.0s per edit, 18.3s in full
  ok   plan valid: 38 task(s) (24 complex, 9 routine, 5 support)
  WARN 12 uncommitted change(s) in the harness; commit before an unattended run
Doctor: OK, 1 warning(s)
```

13.0 seconds per edit against a budget of 20 (12.5 before CP5: two more fast
runs, started half a second apart). The self-test takes about ten minutes
with fourteen broken fixtures. `verify/artifacts/runs/` was left alone
throughout: every gate run of the session on a copy of the reference or on
the application was made in a lab folder outside the repo, and the self-test
writes to `selftest_runs/`.

## Two things to know about these files

- The loop driver classifies anything under `docs/` as application code. Its
  build report lists application files that mention the gate; these scripts
  name the judge's folders in passing and may appear there. They are
  measurement tools for the harness stage and no part of the package.
- For the same reason a build-stage agent is not prevented from editing them.
  If a number here ever disagrees with `spec.md`, run the command again.
