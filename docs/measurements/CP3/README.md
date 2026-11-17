# CP3 measurements

The scripts behind the CP3 session (2026-11-17): the first review's findings
(D36 to D39) and the rulings that followed it the same day (D40 to D43). Kept
here by D35. The conclusions are in `spec.md` and `plan/test-plan.md`; this
folder is how to get the numbers again.

| File | What it is |
|---|---|
| `built_library.py` | Part G: the built library, run as a user would, timed at 100,000 rows by 100 columns and compared at `n_jobs` 1 and 64. |
| `mutants_differ.py` | For each of the five small faults M1 to M5, one crafted table on which the faulty reference and the good one give different answers, with both values. |
| `gate_moved.py` | Sorts the differences between two recorded states of the gate into the two CP3 meant and the rest. |
| `csv_edges.py` | Part E: files that are not valid UTF-8 and lines before the header, in pandas, the application and the reference. |
| `make_example.py` | Writes `examples/customers.csv`, the table the README's commands use. |

Raw outputs are not kept. Every command prints its result and writes its files
outside the repo, apart from `make_example.py`, whose one output is the
example table.

## Running a measurement again

From the repo root, with the project interpreter, one command at a time:

```
$env:MLPROF_LAB_WORK = "C:\somewhere\outside\the\repo"
$py = "E:\CODE\ml-professor\mlprof_env\Scripts\python.exe"
$W  = "$env:MLPROF_LAB_WORK\cp3"
```

`MLPROF_LAB_WORK` is where generated files and results go; without it they go
to `mlprof-lab` in the system's temporary folder.

| What it backs | Command | Takes |
|---|---|---|
| The built library inside the 10-minute budget | `& $py -B docs/measurements/CP3/built_library.py time` | 7 min; run it alone, it uses every core; writes two files of 79 MB |
| The same files at `n_jobs` 1 and 64 | `& $py -B docs/measurements/CP3/built_library.py threads` | 3 min |
| Each of M1 to M5 changes behaviour | `& $py -B docs/measurements/CP3/mutants_differ.py` | 15 s |
| Every value of a full gate run on the reference | `& $py -B docs/measurements/CP2/long_rows.py values - - "$W\gate_after.json"` | 20 s |
| "No other gate value moved" | `& $py -B docs/measurements/CP3/gate_moved.py "$W\gate_before.json" "$W\gate_after.json"` | |
| The two CSV edges, application and reference | `& $py -B docs/measurements/CP3/csv_edges.py app`, then `... ref` | |
| The suite against the reference and its 13 variants | `& $py -B verify/mutation_check.py --jobs 7` | 2 min |
| The example table, byte for byte | `& $py -B docs/measurements/CP3/make_example.py` | |

`gate_before.json` comes from the CP2 `values` command given the state before
CP3: `values <old>\verify\smoke_test.py <old>\verify\reference
"$W\gate_before.json"`, where `<old>` is a copy of the repo at commit
`37d71c0`, outside this folder. The human makes it, since agents do not run
git here; in the session it was a copy taken before the first edit.

## Seeds

- `built_library.py`: both tables come from
  `numpy.random.default_rng([3917, n_rows, n_columns])`. One column in ten is
  text with six levels and the rest are numbers; the first, fifth and sixth
  numeric columns have 2% of their cells blank. The target depends on four
  numeric columns and on the first text column. The tool runs with its
  default seed, 0.
- `mutants_differ.py`: the tables of the guard tests, built from the fixed
  seeds 41, 42, 51, 52, 61 and 62; the tool at seed 0, fast profile.
- The gate's dataset `flat_tail` is the base seed plus 18, its position in
  the gate's `DATASETS` (after `long_rows`, +17). At the gate's base seed,
  1115, that is 1133.
- `make_example.py`: `random.Random(2026)`.

## What was found

pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1; 32 cores, 64 threads.

### G: the built library

Full profile, 18 configurations, run through the command line in a new
process, on all cores (`n_jobs` −1, recorded as 64):

| Table | Task | Seconds in all | Inside `profile()` | Chosen | Skill | Budget (D16) |
|---|---|---|---|---|---|---|
| 100,000 × 100 | classification | 174.4 | 173.6 | `boosting-1` | 0.6395 | 600 |
| 100,000 × 100 | regression | 202.1 | 201.2 | `boosting-5` | 0.8754 | 600 |

The reference implementation on its own table was measured at 178.6 and 205.6
seconds (spec section 9). The two are not the same table, so the figures agree
only roughly; both are about a third of the budget.

Full profile on 5,000 rows × 30 columns, at `--n-jobs 1` and `--n-jobs 64`:

| Task | `profile.json` apart from `meta.timing` and `meta.n_jobs` | `predictions.csv` | Chosen | Skill |
|---|---|---|---|---|
| classification | identical | identical (sha256 `85ef2b5a3a540565…`) | `linear-4` | 0.599299280840863 |
| regression | identical | identical (sha256 `c5e252b02976eca6…`) | `boosting-1` | 0.8573300039006714 |

The seconds of those four runs (53, 31, 48, 28) were taken while other work
ran on the machine and are not a measurement.

### The gate, before CP3 and after

The new check against three packages:

- the reference as it was before CP3: **fails**, "headline.skill on
  `flat_tail` is 0.0; it must be null";
- the reference after CP3: passes;
- the application as built: **fails**, exit 1 with
  `ZeroDivisionError: float division by zero` at `metrics.py:58`. That is
  finding 1 of the review, and it stays so until task T19 is built.

Every value of a full gate run on the reference, in the two states:

```
datasets: 18 before, 19 after; 54 values identical; new: flat_tail
checks: 30 before, 31 after; 91 values identical; new: time.flat_holdout_reported
records: 92 before, 100 after; 92 values identical; new: flat_tail baselines, ...
runs: 23 before, 24 after; 14056 values identical; 19 console summaries reworded; 22 times the new key headline.n_resamples_skill_used; new: flat_tail
NOTHING ELSE MOVED
```

No dataset's seed or bytes moved, no check's verdict or message, no recorded
value, and no number in any `profile.json` or `predictions.csv`.

On `flat_tail` the reference gives: score 349.385 (`linear-0`), training mean
297.946, previous value 0, skill null, interval on the score 333.403 to
367.694, 200 resamples used for the score and 0 for skill.

### The suite against the reference and its variants, before the new tests

`verify/mutation_check.py` on the 308 tests of the first pass:

```
good   fails 0 of 306
F1     fails 7      F5     fails 30
F2     fails 7      F6     fails 19
F3     fails 10     F7     fails 16
F4     fails 6      F8     fails 1
M1     fails 0   NOT CAUGHT: no test fails on this variant
M2     fails 0   NOT CAUGHT: no test fails on this variant
M3     fails 0   NOT CAUGHT: no test fails on this variant
M4     fails 0   NOT CAUGHT: no test fails on this variant
M5     fails 0   NOT CAUGHT: no test fails on this variant
README tests (not counted: they judge README.md, the same file in every case): 0 of 2 failing
MUTATION CHECK: FAIL
```

That is the review's finding, measured: the reference agrees with all 308
tests, and none of the five small faults is caught by any of them. F8 is
caught by a single test.

A project built as CP3 first described it (tests, reference, README,
requirements) fails one of the 308 for a reason that has nothing to do with
the implementation: `test_readme.py::test_every_command_names_things_that_exist`
needs `verify/artifacts/data/signal_clf.csv`, which the README names and only
a gate run writes. The check therefore puts the files the README names into
each project, written by the gate's generator when they are gate datasets.
Task T23 removes the cause.

### Each small fault changes behaviour

| Fault | Crafted table | Good reference | With the fault |
|---|---|---|---|
| M1 | holdout feature values replaced by 1e6, −1e6, 5e5 | the final fit's matrix does not change (largest change 0); mean of its `x1` column 0.0000 | largest change 4.283; mean of `x1` −0.5774 |
| M2 | a level `holdout_only` in eight holdout rows | the final fit's matrix has `colour=blue`, `green`, `red` | also `colour=holdout_only` |
| M3, rule 1 | `late` missing for 70% of class 0 in the training rows, 97% over all rows | `late` is a feature (single-column skill 0.593) | excluded, `missing_by_class` |
| M3, trees | `drift`: two values a block, readable only from holdout rows | a feature; single-column skill 0.000 | excluded, `predicts_target`; skill 1.000 |
| M4 | squared errors alternating 0 and 1 over 64 holdout timestamps | interval 0.5 to 0.5 | 0.359375 to 0.625 |
| M5 | a model that predicts 0.5 for every row | p-value 1.0 | 0.004975 (1/201) |

### The tests the plan demands, checked ahead

Every test named in the second-pass section of `plan/test-plan.md` was written
once as a prototype, outside the repo, and run against the application, the
good reference and M1 to M5. The prototypes are not kept: the tests stage
writes the real ones from the plan. What they showed:

- all 46 pass against the good reference, apart from the three README tests,
  which judge `README.md` and pass against a README rewritten as T23 asks;
- against the application, 26 fail and 20 pass, as the plan's two lists say;
- each of the six guards passes against the application and the good
  reference and fails against its own fault and no other.

### After D40 to D43

D42 changed the reference in three places and the application in none.

| Ruling | Test in the plan | Application | Reference before D42 | Reference after |
|---|---|---|---|---|
| (1) `no_skill_interval` is its own code | `test_skill_undefined_and_no_skill_interval_are_two_codes` | fails (red until T19) | fails: the code was `skill_undefined` | passes |
| (2) white space is not part of a level | `test_surrounding_white_space_is_not_part_of_a_level` | passes | fails: a copy of the target written with stray spaces stays a feature | passes |
| (3) a header cell of white space is a name | `test_header_cells_of_only_white_space_are_names_like_any_other` | passes | fails: a header `a, , ,y` loads | passes |

A categorical cell that holds only white space, in both implementations: the
level of empty text (matrix column `c=`), not a missing cell.

The gate on the reference, after CP3 against after D42: 19 datasets, 31
checks, 100 recorded values and 14,302 values of 24 runs, none moved. Only the
reference's source changed. The suite of the first pass against the changed
reference: the good one fails 0 of 306, as before, with F1 to F8 caught and
M1 to M5 not.

### E: the two CSV edges

| File | pandas alone | Application | Reference before CP3 | Reference after |
|---|---|---|---|---|
| a byte `0xFF` in the header line, the first data row, the third data row or the last cell | `UnicodeDecodeError` | `input_error` | a bare `UnicodeDecodeError` | `input_error` |
| a byte `0xFF` past 200 kB | `UnicodeDecodeError` | `input_error` | `input_error` | `input_error` |
| leading lines of spaces and tabs, with or without a byte-order mark | skipped: header `a, b, y` | the same | `input_error` ("the first data row has 3 fields and the header has 1") | the same as pandas |

**The application has neither fault.** The reference had both.

Which first line pandas skips, and which it reads as the header (file: one
line, then `a,b,y` and two rows):

| First line | pandas | Application |
|---|---|---|
| empty; spaces; a tab; spaces and tabs | skipped: header `a, b, y`, 2 rows | the same |
| a form feed; a vertical tab; a no-break space; an em space; two quotes; a quoted space | a header of one column, 3 rows | `input_error` |
| commas | a header of three blank names, 3 rows | the same, with three `blank_header` warnings |
| spaces between commas | header `' '`, `' .1'`, `Unnamed: 2` | `duplicate_columns` |

### The application on A to D, before the build

Measured on crafted tables: a flat holdout under time order, and a training
mean that is exact under the random scheme, both end in `ZeroDivisionError`;
`stages.skill("mse", x, 0)` raises it too. A numeric column `c=red` beside
`c` with the level `red` reaches the models as one column, not two. A
configuration that fails after it was chosen is marked failed with all its
fold scores null. An `out_dir` that names a file raises `FileExistsError`
after the run. `csv.field_size_limit()` is 131,072 before a load and
2,147,483,647 after. The summary prints the skill and "warnings: 2".

## Two things to know about these files

- The loop driver classifies anything under `docs/` as application code. Its
  build report lists application files that mention the gate; these scripts
  name the judge's folders in passing and may appear there. They are
  measurement tools for the harness stage and no part of the package.
- For the same reason a build-stage agent is not prevented from editing them.
  If a number here ever disagrees with `spec.md`, run the command again.
