# CP8: the measurements behind the seventh pass (D69 to D73)

The scripts and the values behind spec section 10.9, section 8.2 and the plan
of the pass (D35). Measured on 2026-11-22 in the harness session of CP8, on
this machine (Threadripper PRO 5975WX, Python 3.12.13, pandas 3.1.1, numpy
2.5.4, scikit-learn 1.9.1). Nothing here is part of the tool, the gate or the
suite. No script writes inside the repo but the judge's two tools, which write
under `verify\artifacts\` as the gate does: lab copies and work files go to
`MLPROF_LAB_WORK`, or to `mlprof-lab` in the system's temporary folder. Every
table below is the text a script printed; the files are kept in `output\`,
`battery\` and `generated_tables\`.

Three packages are measured:

- **the application**: `mlprofessor` at the repo root, as built by the sixth
  pass (the last commit shown was `f8edec4`; the build is `e917ba8`);
- **the reference**: `verify\reference\mlprofessor`, as changed by this pass;
- **the reference as it stood**: the same folder as committed at `f8edec4`,
  from a copy of `verify\` taken before the first edit of the pass and kept
  outside the repo (`<old verify>` below; `<old reference>` is
  `<old verify>\reference\mlprofessor`).

Run each command from `E:\CODE\ml-professor`.

## Seeds

- **Data seed *k*** of a gate dataset: the gate's base seed 1115 + 100 x *k*,
  *k* = 0 to 29 (*k* = 0 is the gate's own data), each dataset at its own
  position, as `verify\smoke_test.py` makes them.
- **Tool seed**: 0, but for `cap_order.py`, which runs tool seeds 0 to 29 on
  one table.
- **Tables made by a script**: from `numpy.random.default_rng(...)` with the
  seed the script's docstring gives: `8000 + k` for planted sentinels, `k` for
  the controls, `9000 + k` for code columns, `100 + k` for `group_bias.py`.
- **The battery** changes each gate table from `random.Random` seeded with the
  names of the run and the change, so a pair is the same table on every run.
- **The generated tables**: table *i* of seed 20261122 (the default), and of
  seeds 1 and 2 for the prototype's other runs.

## The scripts

| Script | What it measures | Command |
|---|---|---|
| `lab.py` | shared: a lab folder outside the repo, a copy of the reference carrying one fault | (imported) |
| `date_rule.py` | seven forms of the rule for dates written `YYYYMMDD`, on columns that are dates and columns that are not (A1; D69, D72) | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\date_rule.py --seeds 30`, and with `--per-seed` |
| `sentinel_skill.py` | what a sentinel costs in skill on `day_numbers`, and what the rule gives back (A1) | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\sentinel_skill.py mlprofessor --per-seed`; the same with `verify\reference\mlprofessor` |
| `group_bias.py` | `duplicates="group"` on noise and on a parity label, 8 to 120 combinations (A3; D69 point 3) | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\group_bias.py verify\reference\mlprofessor --per-seed` |
| `group_leak_and_cap.py` | `duplicates="group"` beside a removed leak, and entities near the fast cap, run by run (A2, B) | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\group_leak_and_cap.py mlprofessor`; the same with `verify\reference\mlprofessor` |
| `cap_order.py` | how often the fast cap leaves out a large entity (B; D70, D72) | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\cap_order.py mlprofessor --per-seed`; the same with `verify\reference\mlprofessor` |
| `crash_probe.py` | a date beyond the nanosecond range among ordinary ones | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B -X faulthandler docs\measurements\CP8\crash_probe.py pandas`; then with `<old reference>`, with `<old reference> 400`, with `verify\reference\mlprofessor` and with `mlprofessor` in place of `pandas` |
| `broken_copies.py` | the gate on each new broken copy, and chosen checks on any package (proofs 1 and 2) | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\broken_copies.py copies`; `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\broken_copies.py package <old reference> time.date_sentinel_seen,duplicates.group_leak_stops,fast.large_entity_stops,fast.large_entity_kept`; `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\broken_copies.py package mlprofessor` |
| `gate_timing.py` | seconds for one gate run at four launch settings (D72). Run alone | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\gate_timing.py verify\smoke_test.py <a copy of the reference>`; the same with `<old verify>\smoke_test.py <a copy of the old reference>` |
| `..\CP2\long_rows.py`, `..\CP7\gate_moved.py` | every value of every gate run before and after (proof 3) | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP2\long_rows.py values <old verify>\smoke_test.py <old verify>\reference before.json`; `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP2\long_rows.py values - - after.json`; `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP2\long_rows.py compare before.json after.json`; `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP7\gate_moved.py before.json after.json` |
| `loosening.py` | the one replacement of the loosening (D59), as a folder to lay over the suite | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\loosening.py <folder>`, or `--table` |
| `red_green.py` | whether each prototype is red or green today, as the test plan lists it (proof 5) | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\red_green.py <junit.xml>` |
| `named_tests.py` | whether each broken copy fails the test named for it (proof 5) | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\named_tests.py <folder of kept reports> --new-files` |
| `..\CP7\caught_table.py` | how many cases fail on each broken copy, before the prototypes and with them (proof 5) | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP7\caught_table.py "suite as it is=docs\measurements\CP8\output\mutation_judge_changed_suite_as_is.txt" "with the prototypes=docs\measurements\CP8\output\mutation_judge_changed_suite_loosened_and_new.txt"` |
| `battery\pairs_csv.py`, `battery\by_owner.py` | one line for each pair of a battery run; its failing pairs by the task that owns them | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\battery\pairs_csv.py <run.json> <pairs.csv>`; `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\battery\by_owner.py <pairs.csv>` |
| `generated_tables\summarise.py` | the failing tables of a generated-table run by the task that owns them, and what the tables reach | `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP8\generated_tables\summarise.py <run.json>` |
| `battery\prototype_battery.py`, `generated_tables\prototype_generated_tables.py`, `generated_tables\row_order.py`, `generated_tables\variants.py` | the prototypes of the two tools, as they were run for the plan, with two scripts that ran beside the second. Kept as evidence; they name folders of the session and are not meant to be run again: `verify\battery.py` and `verify\generated_tables.py` replace them | |

The judge's own commands used below:

```
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B verify\smoke_test.py --selftest
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B verify\smoke_test.py --matrix
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B verify\mutation_check.py --jobs 8
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B verify\mutation_check.py --jobs 12 --keep --overlay <folder of the prototypes>
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B verify\battery.py verify\reference\mlprofessor --json <run.json>
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B verify\battery.py mlprofessor --json <run.json>
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B verify\generated_tables.py verify\reference\mlprofessor --json <run.json>
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B verify\generated_tables.py mlprofessor --json <run.json>
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B loop/run_loop.py doctor
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B loop/run_loop.py build --dry-run
```

## 1. The date rule (A1; D69, D72)

`date_rule.py` builds each column 30 times and asks seven forms of the rule
whether it is a column of dates. *d* is the number of distinct values that
read as dates written `YYYYMMDD`, years 1900 to 2100; *x* the number of
distinct values that do not. A cell of a table is the number of seeds, of 30,
on which the form calls the column a date column. The form adopted (D72) is
the last: *d* >= 9 x *x*, or *x* <= 2 and *d* >= 5; always with *d* >= 2.

```
d: distinct values that read as dates; x: distinct values that do not
  A every value (D66)
  B 95% of cells
  C x<=2
  D x<=2, d>=5
  E d>=9x
  F d>=9x, x<=2
  U adopted: E or D

MUST BE a date column (each cell: the seeds, of 30, on which the rule calls the column a date column; it must be: all of them)

| Column | d, x over the seeds | share of cells that are dates | A every value (D66) | B 95% of cells | C x<=2 | D x<=2, d>=5 | E d>=9x | F d>=9x, x<=2 | U adopted: E or D |
|---|---|---|---|---|---|---|---|---|---|
| clean (the gate's day_numbers) | d 600..600, x 0..0 | 1.000..1.000 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| one cell of 99991231 | d 600..600, x 1..1 | 0.999..0.999 | 0 | 30 | 30 | 30 | 30 | 30 | 30 |
| one cell of 0 | d 600..600, x 1..1 | 0.999..0.999 | 0 | 30 | 30 | 30 | 30 | 30 | 30 |
| one cell of 20240230 | d 600..600, x 1..1 | 0.999..0.999 | 0 | 30 | 30 | 30 | 30 | 30 | 30 |
| 5% of the rows 99991231 | d 599..600, x 1..1 | 0.950..0.950 | 0 | 30 | 30 | 30 | 30 | 30 | 30 |
| 30% of the rows 99991231 | d 579..590, x 1..1 | 0.700..0.700 | 0 | 0 | 30 | 30 | 30 | 30 | 30 |
| 60% of the rows 99991231 | d 456..479, x 1..1 | 0.400..0.400 | 0 | 0 | 30 | 30 | 30 | 30 | 30 |
| two sentinels, one cell each (0 and 99991231) | d 600..600, x 2..2 | 0.999..0.999 | 0 | 30 | 30 | 30 | 30 | 30 | 30 |
| two sentinels in 30% of the rows (0 and 99991231) | d 574..592, x 2..2 | 0.700..0.700 | 0 | 0 | 30 | 30 | 30 | 30 | 30 |
| three sentinels, one cell each (0, 99991231, 99999999) | d 600..600, x 3..3 | 0.998..0.998 | 0 | 30 | 0 | 0 | 30 | 0 | 30 |
| floats with gaps, one cell of 99991231 | d 599..600, x 1..1 | 0.999..0.999 | 0 | 30 | 30 | 30 | 30 | 30 | 30 |
| floats with gaps, clean | d 599..600, x 0..0 | 1.000..1.000 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| eight-digit text, one cell of 99991231 | d 600..600, x 1..1 | 0.999..0.999 | 0 | 30 | 30 | 30 | 30 | 30 | 30 |
| eight-digit text, one cell of 0 | d 600..600, x 1..1 | 0.999..0.999 | 0 | 30 | 30 | 30 | 30 | 30 | 30 |
| eight-digit text, 30% of the rows 00000000 | d 577..591, x 1..1 | 0.700..0.700 | 0 | 0 | 30 | 30 | 30 | 30 | 30 |
| eight-digit text, clean | d 600..600, x 0..0 | 1.000..1.000 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| CP7 control, reversed by D69: one cell of 20231301 | d 600..600, x 1..1 | 0.999..0.999 | 0 | 30 | 30 | 30 | 30 | 30 | 30 |
| 96 month ends, 30% of the rows 99991231 | d 95..96, x 1..1 | 0.700..0.700 | 0 | 0 | 30 | 30 | 30 | 30 | 30 |
| 12 month ends, 30% of the rows 99991231 | d 12..12, x 1..1 | 0.700..0.700 | 0 | 0 | 30 | 30 | 30 | 30 | 30 |
| 8 quarter ends, 30% of the rows 99991231 | d 8..8, x 1..1 | 0.700..0.700 | 0 | 0 | 30 | 30 | 0 | 0 | 30 |
| 5 month ends, 0 and 99991231 in 30% of the rows | d 5..5, x 2..2 | 0.700..0.700 | 0 | 0 | 30 | 30 | 0 | 0 | 30 |
| columns on which the rule is wrong | | | 18 | 8 | 1 | 1 | 2 | 3 | 0 |

MUST NOT be: the controls of the CP7 measurement (each cell: the seeds, of 30, on which the rule calls the column a date column; it must be: none)

| Column | d, x over the seeds | share of cells that are dates | A every value (D66) | B 95% of cells | C x<=2 | D x<=2, d>=5 | E d>=9x | F d>=9x, x<=2 | U adopted: E or D |
|---|---|---|---|---|---|---|---|---|---|
| eight-digit identifiers drawn at random | d 0..4, x 596..600 | 0.000..0.007 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| identifiers counting up from 20230101 | d 181..181, x 419..419 | 0.302..0.302 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| whole numbers drawn at random between 19000101 and 21001231 | d 12..32, x 568..588 | 0.020..0.053 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| years only (1990 to 2024) | d 0..0, x 35..35 | 0.000..0.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| year and month as YYYYMM | d 0..0, x 240..240 | 0.000..0.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| seconds since 1970 | d 0..0, x 600..600 | 0.000..0.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| days since 1970 | d 0..0, x 200..200 | 0.000..0.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| prices in whole cents | d 0..0, x 599..600 | 0.000..0.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| counts | d 0..0, x 9..12 | 0.000..0.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ages | d 0..0, x 72..72 | 0.000..0.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| post codes | d 0..0, x 594..600 | 0.000..0.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dates as DDMMYYYY whole numbers | d 0..0, x 200..200 | 0.000..0.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| columns on which the rule is wrong | | | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

MUST NOT be: code columns of a few distinct eight-digit values (each cell: the seeds, of 30, on which the rule calls the column a date column; it must be: none)

| Column | d, x over the seeds | share of cells that are dates | A every value (D66) | B 95% of cells | C x<=2 | D x<=2, d>=5 | E d>=9x | F d>=9x, x<=2 | U adopted: E or D |
|---|---|---|---|---|---|---|---|---|---|
| 2 distinct codes, 1 of them read as dates | d 1..1, x 1..1 | 0.452..0.543 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 distinct codes, 1 read as dates, as text | d 1..1, x 1..1 | 0.452..0.543 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 distinct codes, 1 of them read as dates | d 1..1, x 2..2 | 0.285..0.363 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 distinct codes, 1 read as dates, as text | d 1..1, x 2..2 | 0.285..0.363 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 distinct codes, 2 of them read as dates | d 2..2, x 1..1 | 0.640..0.693 | 0 | 0 | 30 | 0 | 0 | 0 | 0 |
| 3 distinct codes, 2 read as dates, as text | d 2..2, x 1..1 | 0.640..0.693 | 0 | 0 | 30 | 0 | 0 | 0 | 0 |
| 5 distinct codes, 1 of them read as dates | d 1..1, x 4..4 | 0.145..0.223 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 distinct codes, 1 read as dates, as text | d 1..1, x 4..4 | 0.145..0.223 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 distinct codes, 2 of them read as dates | d 2..2, x 3..3 | 0.372..0.438 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 distinct codes, 2 read as dates, as text | d 2..2, x 3..3 | 0.372..0.438 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 distinct codes, 3 of them read as dates | d 3..3, x 2..2 | 0.562..0.645 | 0 | 0 | 30 | 0 | 0 | 0 | 0 |
| 5 distinct codes, 3 read as dates, as text | d 3..3, x 2..2 | 0.562..0.645 | 0 | 0 | 30 | 0 | 0 | 0 | 0 |
| 5 distinct codes, 4 of them read as dates | d 4..4, x 1..1 | 0.772..0.877 | 0 | 0 | 30 | 0 | 0 | 0 | 0 |
| 5 distinct codes, 4 read as dates, as text | d 4..4, x 1..1 | 0.772..0.877 | 0 | 0 | 30 | 0 | 0 | 0 | 0 |
| columns on which the rule is wrong | | | 0 | 0 | 6 | 0 | 0 | 0 | 0 |

The limits of the adopted rule: columns of dates it does not name (each cell: the seeds, of 30, on which the rule calls the column a date column; it must be: shown for the record)

| Column | d, x over the seeds | share of cells that are dates | A every value (D66) | B 95% of cells | C x<=2 | D x<=2, d>=5 | E d>=9x | F d>=9x, x<=2 | U adopted: E or D |
|---|---|---|---|---|---|---|---|---|---|
| 4 quarter ends, 30% of the rows 99991231 | d 4..4, x 1..1 | 0.700..0.700 | 0 | 0 | 30 | 0 | 0 | 0 | 0 |
| 26 days and three sentinels | d 26..26, x 3..3 | 0.900..0.900 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

MUST NOT be: every column of every gate dataset but the two made for the rule (day_numbers, day_sentinel), and of examples/customers.csv (the target included: a rule that named it would show)

| Table | data seeds | columns | of them, columns rule 2a looks at | A every value (D66) | B 95% of cells | C x<=2 | D x<=2, d>=5 | E d>=9x | F d>=9x, x<=2 | U adopted: E or D |
|---|---|---|---|---|---|---|---|---|---|---|
| gate `signal_clf` | 30 | 240 | 240 | none | none | none | none | none | none | none |
| gate `signal_reg` | 30 | 210 | 210 | none | none | none | none | none | none | none |
| gate `noise_clf` | 30 | 9030 | 9030 | none | none | none | none | none | none | none |
| gate `noise_reg` | 30 | 9030 | 9030 | none | none | none | none | none | none | none |
| gate `imbalanced` | 30 | 240 | 240 | none | none | none | none | none | none | none |
| gate `leak` | 30 | 330 | 330 | none | none | none | none | none | none | none |
| gate `leak_reg` | 30 | 240 | 240 | none | none | none | none | none | none | none |
| gate `binary_numeric` | 30 | 240 | 240 | none | none | none | none | none | none | none |
| gate `ambiguous_target` | 30 | 120 | 120 | none | none | none | none | none | none | none |
| gate `messy` | 30 | 390 | 390 | none | none | none | none | none | none | none |
| gate `time_blocks` | 30 | 300 | 300 | none | none | none | none | none | none | none |
| gate `time_dayfirst` | 30 | 300 | 300 | none | none | none | none | none | none | none |
| gate `random_walk` | 30 | 90 | 90 | none | none | none | none | none | none | none |
| gate `noise_small` | 30 | 330 | 330 | none | none | none | none | none | none | none |
| gate `signal_clf_small` | 30 | 240 | 240 | none | none | none | none | none | none | none |
| gate `signal_reg_small` | 30 | 210 | 210 | none | none | none | none | none | none | none |
| gate `near_leak` | 30 | 270 | 270 | none | none | none | none | none | none | none |
| gate `long_rows` | 30 | 240 | 240 | none | none | none | none | none | none | none |
| gate `flat_tail` | 30 | 120 | 120 | none | none | none | none | none | none | none |
| gate `dup_noise` | 30 | 330 | 330 | none | none | none | none | none | none | none |
| gate `int_stamps` | 30 | 90 | 90 | none | none | none | none | none | none | none |
| gate `entities` | 30 | 240 | 240 | none | none | none | none | none | none | none |
| gate `panel` | 30 | 120 | 120 | none | none | none | none | none | none | none |
| gate `few_valued` | 30 | 90 | 90 | none | none | none | none | none | none | none |
| gate `dominant_entity` | 30 | 180 | 180 | none | none | none | none | none | none | none |
| gate `few_valued_leak` | 30 | 120 | 120 | none | none | none | none | none | none | none |
| gate `large_entity` | 30 | 180 | 180 | none | none | none | none | none | none | none |
| gate `quarter_entity` | 30 | 180 | 180 | none | none | none | none | none | none | none |
| examples/customers.csv | - | 8 | 8 | none | none | none | none | none | none | none |
| columns named in all | | | | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The gate's two tables made for the rule, the column `day`, over the same seeds:
  `day_numbers`: A every value (D66): 30, B 95% of cells: 30, C x<=2: 30, D x<=2, d>=5: 30, E d>=9x: 30, F d>=9x, x<=2: 30, U adopted: E or D: 30
  `day_sentinel`: A every value (D66): 0, B 95% of cells: 0, C x<=2: 30, D x<=2, d>=5: 30, E d>=9x: 30, F d>=9x, x<=2: 30, U adopted: E or D: 30

The smallest change that defeats each rule on the gate's `day_numbers` (600 distinct dates, 1,800 cells): the number of distinct values that are no dates, one cell each (99991231, 99991230, ...), at which the column is no longer a date column

| Rule | cells changed, min..max over the seeds |
|---|---|
| A every value (D66) | 1..1 |
| B 95% of cells | 91..91 |
| C x<=2 | 3..3 |
| D x<=2, d>=5 | 3..3 |
| E d>=9x | 67..67 |
| F d>=9x, x<=2 | 3..3 |
| U adopted: E or D | 67..67 |
```

**Seed by seed:** `output\date_rule_per_seed.txt` holds, for every column
above and each of the 30 seeds, *d*, *x*, the share of cells that are dates
and the verdict of each form (the same command with `--per-seed`). Its first
lines:

```
d: distinct values that read as dates; x: distinct values that do not
  A every value (D66)
  B 95% of cells
  C x<=2
  D x<=2, d>=5
  E d>=9x
  F d>=9x, x<=2
  U adopted: E or D

MUST BE a date column (each cell: the seeds, of 30, on which the rule calls the column a date column; it must be: all of them)

| Column | d, x over the seeds | share of cells that are dates | A every value (D66) | B 95% of cells | C x<=2 | D x<=2, d>=5 | E d>=9x | F d>=9x, x<=2 | U adopted: E or D |
|---|---|---|---|---|---|---|---|---|---|
| clean (the gate's day_numbers) | d 600..600, x 0..0 | 1.000..1.000 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
    seed 0: d 600, x 0, r 1.0000: 1 1 1 1 1 1 1
    seed 1: d 600, x 0, r 1.0000: 1 1 1 1 1 1 1
    seed 2: d 600, x 0, r 1.0000: 1 1 1 1 1 1 1
    seed 3: d 600, x 0, r 1.0000: 1 1 1 1 1 1 1
    seed 4: d 600, x 0, r 1.0000: 1 1 1 1 1 1 1
    seed 5: d 600, x 0, r 1.0000: 1 1 1 1 1 1 1
    seed 6: d 600, x 0, r 1.0000: 1 1 1 1 1 1 1
    seed 7: d 600, x 0, r 1.0000: 1 1 1 1 1 1 1
    seed 8: d 600, x 0, r 1.0000: 1 1 1 1 1 1 1
    seed 9: d 600, x 0, r 1.0000: 1 1 1 1 1 1 1
... (1577 more lines in output\date_rule_per_seed.txt)
```

## 2. What a sentinel costs in skill, and what the rule gives back (A1)

The gate's `day_numbers` (a label that is random per block of five days, the
day written as a whole number: the truth is skill about 0), with `--no-time-order
--no-entities`, fast profile, tool seed 0, data seeds 0 to 29. The summary,
then skill at each seed.

The application as it stood:

```
package: E:\CODE\ml-professor\mlprofessor
| `day_numbers` with | `day` excluded as a date, of the seeds | named by the warning | skill: range (mean) |
|---|---|---|---|
| clean | 30 of 30 | 30 | -0.182 to +0.089 (-0.009) |
| one cell of 99991231 | 0 of 30 | 0 | +0.820 to +1.000 (+0.948) |
| one cell of 0 | 0 of 30 | 0 | +0.844 to +0.995 (+0.949) |
| one cell of 20240230 | 0 of 30 | 0 | +0.820 to +1.000 (+0.947) |
| 5% of the rows 99991231 | 0 of 30 | 0 | +0.004 to +0.458 (+0.339) |
| 30% of the rows 99991231 | 0 of 30 | 0 | -0.049 to +0.231 (+0.101) |
| 60% of the rows 99991231 | 0 of 30 | 0 | -0.183 to +0.119 (+0.010) |
| two sentinels, 0 and 99991231, one cell each | 0 of 30 | 0 | +0.832 to +1.000 (+0.946) |
| three sentinels, one cell each | 0 of 30 | 0 | +0.829 to +0.995 (+0.948) |
| 70 distinct values that are no dates, one cell each (a stated limit) | 0 of 30 | 0 | +0.187 to +0.582 (+0.409) |

skill at each data seed (1115, 1215, ...), in that order; `d` marks a seed on which `day` was excluded as a date
clean: +0.015d -0.011d +0.018d +0.039d +0.026d -0.003d -0.182d +0.053d -0.004d +0.005d -0.050d -0.030d +0.009d -0.071d +0.089d -0.056d +0.031d -0.068d -0.000d +0.043d -0.003d +0.001d -0.007d -0.042d +0.005d -0.035d -0.043d -0.016d +0.005d +0.010d
one cell of 99991231: +0.924 +0.960 +0.862 +0.995 +0.937 +0.982 +0.972 +1.000 +0.942 +0.986 +0.908 +0.902 +0.898 +0.991 +0.990 +0.820 +0.929 +0.911 +0.932 +0.972 +0.969 +0.958 +0.982 +0.964 +0.953 +0.978 +0.951 +0.974 +0.978 +0.906
one cell of 0: +0.937 +0.960 +0.876 +0.995 +0.937 +0.982 +0.981 +0.987 +0.938 +0.986 +0.931 +0.902 +0.898 +0.977 +0.978 +0.844 +0.942 +0.898 +0.932 +0.977 +0.969 +0.972 +0.982 +0.973 +0.953 +0.978 +0.937 +0.960 +0.983 +0.906
one cell of 20240230: +0.924 +0.960 +0.876 +0.995 +0.937 +0.982 +0.972 +0.987 +0.942 +1.000 +0.908 +0.916 +0.898 +0.991 +0.964 +0.820 +0.929 +0.907 +0.932 +0.972 +0.969 +0.958 +0.982 +0.964 +0.953 +0.969 +0.937 +0.974 +0.978 +0.906
5% of the rows 99991231: +0.317 +0.434 +0.004 +0.444 +0.279 +0.167 +0.411 +0.367 +0.458 +0.299 +0.421 +0.413 +0.276 +0.431 +0.400 +0.401 +0.379 +0.302 +0.373 +0.324 +0.293 +0.303 +0.333 +0.455 +0.265 +0.339 +0.232 +0.307 +0.414 +0.324
30% of the rows 99991231: -0.013 -0.041 +0.147 +0.231 +0.095 -0.017 +0.148 +0.093 +0.200 +0.102 +0.120 +0.111 +0.129 +0.174 +0.136 +0.196 +0.022 +0.147 +0.155 +0.114 +0.151 +0.036 +0.170 +0.050 +0.181 -0.049 +0.078 +0.040 +0.140 -0.026
60% of the rows 99991231: +0.014 -0.024 -0.027 +0.070 -0.069 -0.011 -0.183 +0.060 -0.013 -0.016 -0.048 +0.022 -0.058 +0.040 +0.100 +0.072 +0.082 -0.082 +0.119 -0.044 +0.096 +0.010 -0.016 -0.030 +0.022 +0.020 +0.071 +0.081 +0.017 +0.026
two sentinels, 0 and 99991231, one cell each: +0.933 +0.960 +0.898 +0.995 +0.932 +0.982 +0.981 +0.978 +0.929 +1.000 +0.931 +0.902 +0.889 +0.977 +0.978 +0.832 +0.938 +0.893 +0.928 +0.949 +0.969 +0.949 +0.978 +0.982 +0.949 +0.974 +0.937 +0.960 +0.983 +0.906
three sentinels, one cell each: +0.933 +0.973 +0.898 +0.995 +0.923 +0.986 +0.981 +0.978 +0.924 +0.986 +0.931 +0.902 +0.907 +0.977 +0.990 +0.829 +0.942 +0.907 +0.923 +0.960 +0.969 +0.949 +0.978 +0.982 +0.949 +0.965 +0.933 +0.960 +0.983 +0.916
70 distinct values that are no dates, one cell each (a stated limit): +0.365 +0.582 +0.187 +0.461 +0.455 +0.405 +0.447 +0.286 +0.516 +0.468 +0.454 +0.418 +0.262 +0.507 +0.520 +0.525 +0.448 +0.458 +0.420 +0.368 +0.381 +0.292 +0.403 +0.437 +0.365 +0.350 +0.218 +0.436 +0.510 +0.339
```

The reference as changed:

```
package: E:\CODE\ml-professor\verify\reference\mlprofessor
| `day_numbers` with | `day` excluded as a date, of the seeds | named by the warning | skill: range (mean) |
|---|---|---|---|
| clean | 30 of 30 | 30 | -0.052 to +0.081 (+0.012) |
| one cell of 99991231 | 30 of 30 | 30 | -0.052 to +0.081 (+0.012) |
| one cell of 0 | 30 of 30 | 30 | -0.052 to +0.081 (+0.012) |
| one cell of 20240230 | 30 of 30 | 30 | -0.052 to +0.081 (+0.012) |
| 5% of the rows 99991231 | 30 of 30 | 30 | -0.052 to +0.081 (+0.012) |
| 30% of the rows 99991231 | 30 of 30 | 30 | -0.052 to +0.081 (+0.012) |
| 60% of the rows 99991231 | 30 of 30 | 30 | -0.052 to +0.081 (+0.012) |
| two sentinels, 0 and 99991231, one cell each | 30 of 30 | 30 | -0.052 to +0.081 (+0.012) |
| three sentinels, one cell each | 30 of 30 | 30 | -0.052 to +0.081 (+0.012) |
| 70 distinct values that are no dates, one cell each (a stated limit) | 0 of 30 | 0 | +0.226 to +0.551 (+0.420) |

skill at each data seed (1115, 1215, ...), in that order; `d` marks a seed on which `day` was excluded as a date
clean: -0.033d +0.065d +0.062d +0.024d +0.029d +0.081d -0.026d +0.004d +0.036d +0.009d -0.046d -0.052d -0.035d -0.038d +0.049d -0.044d -0.006d -0.004d -0.038d +0.073d +0.014d +0.046d +0.020d +0.020d +0.059d -0.009d +0.030d -0.031d +0.021d +0.066d
one cell of 99991231: -0.033d +0.065d +0.062d +0.024d +0.029d +0.081d -0.026d +0.004d +0.036d +0.009d -0.046d -0.052d -0.035d -0.038d +0.049d -0.044d -0.006d -0.004d -0.038d +0.073d +0.014d +0.046d +0.020d +0.020d +0.059d -0.009d +0.030d -0.031d +0.021d +0.066d
one cell of 0: -0.033d +0.065d +0.062d +0.024d +0.029d +0.081d -0.026d +0.004d +0.036d +0.009d -0.046d -0.052d -0.035d -0.038d +0.049d -0.044d -0.006d -0.004d -0.038d +0.073d +0.014d +0.046d +0.020d +0.020d +0.059d -0.009d +0.030d -0.031d +0.021d +0.066d
one cell of 20240230: -0.033d +0.065d +0.062d +0.024d +0.029d +0.081d -0.026d +0.004d +0.036d +0.009d -0.046d -0.052d -0.035d -0.038d +0.049d -0.044d -0.006d -0.004d -0.038d +0.073d +0.014d +0.046d +0.020d +0.020d +0.059d -0.009d +0.030d -0.031d +0.021d +0.066d
5% of the rows 99991231: -0.033d +0.065d +0.062d +0.024d +0.029d +0.081d -0.026d +0.004d +0.036d +0.009d -0.046d -0.052d -0.035d -0.038d +0.049d -0.044d -0.006d -0.004d -0.038d +0.073d +0.014d +0.046d +0.020d +0.020d +0.059d -0.009d +0.030d -0.031d +0.021d +0.066d
30% of the rows 99991231: -0.033d +0.065d +0.062d +0.024d +0.029d +0.081d -0.026d +0.004d +0.036d +0.009d -0.046d -0.052d -0.035d -0.038d +0.049d -0.044d -0.006d -0.004d -0.038d +0.073d +0.014d +0.046d +0.020d +0.020d +0.059d -0.009d +0.030d -0.031d +0.021d +0.066d
60% of the rows 99991231: -0.033d +0.065d +0.062d +0.024d +0.029d +0.081d -0.026d +0.004d +0.036d +0.009d -0.046d -0.052d -0.035d -0.038d +0.049d -0.044d -0.006d -0.004d -0.038d +0.073d +0.014d +0.046d +0.020d +0.020d +0.059d -0.009d +0.030d -0.031d +0.021d +0.066d
two sentinels, 0 and 99991231, one cell each: -0.033d +0.065d +0.062d +0.024d +0.029d +0.081d -0.026d +0.004d +0.036d +0.009d -0.046d -0.052d -0.035d -0.038d +0.049d -0.044d -0.006d -0.004d -0.038d +0.073d +0.014d +0.046d +0.020d +0.020d +0.059d -0.009d +0.030d -0.031d +0.021d +0.066d
three sentinels, one cell each: -0.033d +0.065d +0.062d +0.024d +0.029d +0.081d -0.026d +0.004d +0.036d +0.009d -0.046d -0.052d -0.035d -0.038d +0.049d -0.044d -0.006d -0.004d -0.038d +0.073d +0.014d +0.046d +0.020d +0.020d +0.059d -0.009d +0.030d -0.031d +0.021d +0.066d
70 distinct values that are no dates, one cell each (a stated limit): +0.358 +0.510 +0.338 +0.462 +0.405 +0.237 +0.454 +0.431 +0.498 +0.393 +0.427 +0.551 +0.382 +0.508 +0.398 +0.520 +0.520 +0.302 +0.538 +0.235 +0.317 +0.486 +0.474 +0.534 +0.399 +0.226 +0.313 +0.428 +0.515 +0.445
```

## 3. `duplicates="group"`: what it measures (A3; D69 point 3)

2,000 rows, the reference, fast profile, 30 tables for each row (seed `100 +
k`). Noise: a label drawn at random. Parity: a label fixed by the combination
of the feature values. Under each row of a table, both skills at every seed.

```
package: E:\CODE\ml-professor\verify\reference\mlprofessor

Noise: a label drawn at random, 2,000 rows (the truth is 0)

| Combinations | `group`: skill, range | mean | sd | runs below -0.2 | `keep`: skill, range | mean | holdout units | runs with an interval | stopped |
|---|---|---|---|---|---|---|---|---|---|
| 8, crossed 4 x 2 | -0.116 to +0.091 | -0.002 | 0.049 | 0 | -0.082 to +0.070 | -0.006 | 2 to 3 | 0 of 26 | {'too_few_groups': 4} |
    group, seeds 0 to 29: +0.014 +0.011 -0.002 -0.009 -0.069 +0.021 -0.009 +0.015 -0.116 -0.089 +0.041 +0.091 +0.042 stop -0.072 stop -0.010 stop -0.054 -0.041 +0.034 stop +0.034 +0.024 +0.015 +0.030 +0.038 -0.054 +0.043 +0.015
    keep,  seeds 0 to 29: -0.043 +0.070 -0.056 +0.008 +0.022 -0.033 -0.008 +0.003 -0.012 +0.014 +0.027 +0.016 -0.025 stop +0.052 stop -0.048 stop -0.046 +0.040 +0.013 stop +0.014 -0.021 -0.011 -0.082 +0.050 -0.041 -0.007 -0.057
| 16, crossed 4 x 4 | -0.082 to +0.070 | +0.007 | 0.036 | 0 | -0.123 to +0.065 | -0.015 | 4 to 6 | 0 of 30 | none |
    group, seeds 0 to 29: -0.007 -0.014 +0.025 -0.003 +0.044 +0.007 +0.024 -0.016 +0.063 +0.022 -0.002 +0.030 +0.004 +0.034 +0.042 -0.067 -0.024 +0.013 +0.014 -0.075 -0.008 +0.070 -0.007 +0.042 -0.082 -0.003 +0.042 +0.025 +0.012 -0.005
    keep,  seeds 0 to 29: -0.049 +0.045 -0.058 -0.044 -0.034 -0.092 +0.010 +0.022 -0.027 +0.021 +0.031 +0.043 -0.007 -0.047 +0.042 +0.004 -0.071 -0.077 -0.025 +0.005 +0.027 -0.002 +0.024 -0.042 -0.051 -0.123 +0.028 -0.060 +0.065 +0.006
| 24, crossed 6 x 4 | -0.113 to +0.107 | +0.002 | 0.054 | 0 | -0.114 to +0.046 | -0.015 | 7 to 8 | 0 of 30 | none |
    group, seeds 0 to 29: -0.103 -0.094 -0.024 -0.041 -0.008 -0.006 +0.002 +0.018 -0.059 +0.090 -0.012 +0.072 -0.025 +0.059 +0.107 +0.004 -0.035 +0.019 -0.006 +0.047 +0.006 -0.020 +0.034 +0.060 +0.004 +0.077 -0.004 -0.023 -0.113 +0.021
    keep,  seeds 0 to 29: -0.040 +0.044 -0.085 -0.005 -0.064 -0.079 +0.032 +0.046 +0.018 -0.022 -0.015 +0.022 -0.060 +0.009 +0.046 -0.114 -0.023 -0.078 +0.009 +0.037 +0.030 +0.008 +0.031 -0.014 -0.048 -0.075 -0.057 -0.080 +0.045 +0.034
| 48, crossed 8 x 6 | -0.138 to +0.127 | -0.008 | 0.059 | 0 | -0.167 to +0.068 | -0.029 | 12 to 14 | 0 of 30 | none |
    group, seeds 0 to 29: +0.033 -0.033 -0.039 +0.009 -0.012 -0.103 +0.017 +0.022 -0.060 -0.138 +0.011 +0.004 +0.043 +0.127 +0.055 +0.008 -0.017 -0.036 +0.032 -0.040 -0.100 +0.045 -0.049 -0.018 +0.070 -0.068 -0.044 -0.049 -0.007 +0.097
    keep,  seeds 0 to 29: -0.066 +0.068 -0.068 -0.014 -0.018 -0.124 -0.025 +0.068 -0.006 -0.073 -0.059 +0.026 -0.032 -0.057 -0.003 -0.014 -0.039 -0.167 +0.000 +0.031 -0.064 -0.107 +0.039 -0.061 -0.020 -0.050 +0.036 -0.051 -0.052 +0.035
| 120, crossed 12 x 10 | -0.093 to +0.095 | +0.005 | 0.054 | 0 | -0.130 to +0.072 | -0.019 | 30 to 34 | 30 of 30 | none |
    group, seeds 0 to 29: -0.014 +0.021 +0.023 +0.005 -0.015 -0.075 -0.025 +0.033 +0.036 -0.034 +0.092 +0.086 -0.093 +0.070 -0.015 -0.066 -0.016 +0.035 +0.017 -0.004 +0.055 -0.023 +0.095 -0.035 -0.074 -0.063 +0.069 -0.049 +0.080 +0.029
    keep,  seeds 0 to 29: -0.060 +0.019 -0.096 +0.036 -0.077 -0.099 -0.012 +0.016 -0.021 +0.054 -0.085 +0.068 +0.054 -0.035 -0.038 +0.063 -0.014 -0.130 +0.065 +0.016 -0.011 -0.084 +0.034 -0.084 -0.085 -0.095 -0.008 -0.042 +0.016 +0.072
| 8, a branch within a region | -0.099 to +0.068 | -0.013 | 0.041 | 0 | -0.126 to +0.105 | +0.016 | 3 to 3 | 0 of 24 | {'too_few_groups': 6} |
    group, seeds 0 to 29: stop +0.010 -0.048 -0.010 -0.064 stop -0.034 stop +0.018 -0.004 stop stop -0.083 -0.036 +0.023 -0.019 +0.055 -0.001 -0.099 -0.035 -0.006 -0.020 +0.008 -0.058 +0.042 +0.018 stop -0.042 +0.068 -0.002
    keep,  seeds 0 to 29: stop -0.021 +0.028 -0.126 +0.013 stop +0.031 stop -0.052 +0.080 stop stop +0.059 +0.060 -0.054 -0.070 +0.021 -0.016 +0.085 +0.105 +0.080 +0.096 -0.057 +0.014 +0.022 +0.051 stop -0.033 +0.060 +0.001
| 16, a branch within a region | -0.105 to +0.090 | -0.005 | 0.044 | 0 | -0.133 to +0.103 | +0.023 | 4 to 6 | 0 of 30 | none |
    group, seeds 0 to 29: -0.016 -0.004 +0.000 -0.047 -0.025 +0.013 +0.004 +0.024 -0.059 +0.000 +0.019 -0.023 +0.000 +0.039 +0.048 +0.021 +0.000 -0.005 -0.050 -0.087 +0.031 +0.018 +0.000 +0.090 +0.025 -0.086 +0.000 -0.035 +0.070 -0.105
    keep,  seeds 0 to 29: +0.079 +0.002 +0.026 -0.092 +0.097 -0.043 +0.060 +0.035 -0.052 +0.079 +0.034 -0.001 +0.075 +0.024 -0.024 -0.020 +0.005 -0.032 +0.092 +0.082 +0.068 +0.103 -0.133 +0.000 +0.006 +0.072 +0.013 +0.010 +0.053 +0.063
| 24, a branch within a region | -0.096 to +0.080 | -0.004 | 0.049 | 0 | -0.126 to +0.194 | +0.015 | 7 to 8 | 0 of 30 | none |
    group, seeds 0 to 29: +0.041 -0.014 +0.032 -0.006 +0.018 +0.027 +0.075 -0.004 +0.021 -0.070 +0.035 +0.053 -0.020 -0.056 -0.011 -0.053 +0.026 -0.026 -0.056 +0.030 -0.044 -0.094 +0.044 +0.010 +0.080 -0.041 -0.041 -0.056 -0.096 +0.074
    keep,  seeds 0 to 29: +0.044 -0.000 +0.030 -0.126 -0.013 -0.031 +0.030 +0.063 -0.005 +0.051 -0.008 -0.061 +0.057 -0.001 -0.016 +0.007 +0.018 -0.040 +0.194 +0.096 +0.063 +0.093 -0.086 +0.060 -0.016 +0.023 +0.033 -0.055 +0.041 -0.007

Signal: the label is the parity of the two columns, one label in ten flipped (the sixth review, finding 3)

| Combinations | `group`: skill, range | mean | sd | runs below -0.2 | `keep`: skill, range | mean | holdout units | runs with an interval | stopped |
|---|---|---|---|---|---|---|---|---|---|
| 8, crossed 4 x 2 | -0.828 to +0.041 | -0.661 | 0.293 | 12 | +0.723 to +0.840 | +0.799 | 2 to 3 | 0 of 14 | {'too_few_groups': 16} |
    group, seeds 0 to 29: stop +0.041 -0.782 stop stop stop -0.759 -0.822 -0.791 -0.797 stop stop stop -0.763 stop stop stop -0.724 stop stop stop stop -0.775 stop +0.012 -0.752 -0.755 -0.828 -0.758 stop
    keep,  seeds 0 to 29: stop +0.831 +0.812 stop stop stop +0.808 +0.807 +0.833 +0.761 stop stop stop +0.807 stop stop stop +0.818 stop stop stop stop +0.809 stop +0.794 +0.768 +0.723 +0.840 +0.772 stop
| 16, crossed 4 x 4 | -0.839 to +0.565 | -0.492 | 0.320 | 26 | +0.746 to +0.844 | +0.793 | 4 to 6 | 0 of 30 | none |
    group, seeds 0 to 29: -0.794 -0.711 -0.163 -0.822 -0.375 -0.566 +0.565 -0.602 -0.762 -0.454 +0.091 -0.577 -0.431 -0.214 -0.283 -0.798 -0.529 -0.741 -0.791 +0.040 -0.670 -0.804 -0.795 -0.497 -0.334 -0.466 -0.573 -0.839 -0.401 -0.451
    keep,  seeds 0 to 29: +0.801 +0.780 +0.781 +0.820 +0.779 +0.746 +0.798 +0.844 +0.786 +0.753 +0.796 +0.755 +0.817 +0.761 +0.775 +0.831 +0.809 +0.796 +0.812 +0.791 +0.792 +0.788 +0.790 +0.792 +0.797 +0.828 +0.791 +0.791 +0.748 +0.839
| 24, crossed 6 x 4 | -0.818 to +0.630 | -0.246 | 0.362 | 17 | +0.733 to +0.854 | +0.801 | 6 to 8 | 0 of 30 | none |
    group, seeds 0 to 29: -0.357 +0.014 -0.185 -0.133 -0.344 -0.545 -0.553 -0.380 -0.265 -0.504 -0.111 -0.718 -0.303 +0.313 -0.311 -0.228 +0.630 -0.481 -0.510 -0.006 -0.796 +0.106 +0.043 -0.607 -0.084 +0.411 -0.044 -0.818 -0.782 +0.168
    keep,  seeds 0 to 29: +0.839 +0.801 +0.812 +0.798 +0.826 +0.854 +0.853 +0.845 +0.787 +0.733 +0.779 +0.828 +0.854 +0.754 +0.803 +0.815 +0.823 +0.785 +0.746 +0.847 +0.759 +0.751 +0.757 +0.788 +0.796 +0.803 +0.768 +0.847 +0.771 +0.815
| 48, crossed 8 x 6 | -0.806 to +0.380 | -0.262 | 0.298 | 17 | +0.737 to +0.848 | +0.794 | 12 to 14 | 0 of 30 | none |
    group, seeds 0 to 29: +0.040 -0.457 -0.303 -0.806 +0.030 +0.088 -0.580 -0.623 -0.364 -0.539 -0.490 -0.618 -0.224 -0.136 -0.702 +0.035 -0.269 -0.109 +0.380 -0.351 -0.077 -0.393 -0.189 -0.386 -0.554 -0.529 -0.097 +0.062 +0.045 +0.241
    keep,  seeds 0 to 29: +0.806 +0.813 +0.828 +0.812 +0.808 +0.775 +0.770 +0.835 +0.737 +0.764 +0.758 +0.794 +0.848 +0.742 +0.787 +0.800 +0.834 +0.793 +0.837 +0.777 +0.745 +0.794 +0.755 +0.809 +0.824 +0.818 +0.759 +0.827 +0.762 +0.805
| 120, crossed 12 x 10 | -0.574 to +0.524 | -0.074 | 0.283 | 9 | +0.760 to +0.844 | +0.793 | 29 to 33 | 29 of 30 | none |
    group, seeds 0 to 29: -0.525 -0.333 +0.416 +0.168 -0.177 +0.344 +0.524 -0.177 +0.165 -0.356 -0.389 +0.133 -0.039 -0.177 -0.130 +0.068 -0.293 -0.029 -0.002 -0.574 -0.020 -0.091 -0.313 -0.343 -0.192 -0.465 +0.194 +0.003 +0.380 +0.002
    keep,  seeds 0 to 29: +0.810 +0.766 +0.785 +0.831 +0.760 +0.798 +0.794 +0.826 +0.764 +0.777 +0.768 +0.787 +0.763 +0.785 +0.793 +0.795 +0.803 +0.768 +0.844 +0.803 +0.781 +0.808 +0.787 +0.822 +0.792 +0.820 +0.784 +0.825 +0.771 +0.792
```

## 4. The stop after a removed leak, and entities near the cap, run by run (A2, B)

`group_leak_and_cap.py` runs a handful of tables once each and prints what
each run did: `few_valued` with a column that copies the label under each
value of `duplicates`; entities at, just over and far over the fast cap; two
large entities; a large entity beside too few others; a few-valued table of
40,000 rows under `duplicates="group"` with no entity column.

The application as it stood:

```
package: E:\CODE\ml-professor\mlprofessor

-- D69: duplicates='group' and a column the leak screen removes (noise: the truth is 0)
copy of the label: None; duplicates=group: ran: scheme grouped, skill +0.046, training 1291, holdout 709, left out by the cap 0, copies 0, warnings ['duplicate_rows', 'no_interval', 'same_features_grouped']
copy of the label: None; duplicates=keep: ran: scheme random, skill +0.010, training 1500, holdout 500, left out by the cap 0, copies 500, warnings ['duplicate_rows', 'holdout_copies_kept']
copy of the label: None; duplicates=error: holdout_copies_in_training: 500 of the 500 holdout rows (100.0%) have a copy among the training rows: the same value in every feature column and the same target. A model that recalls  ...
copy of the label: two-valued; duplicates=group: ran: scheme grouped, skill -0.943, training 1459, holdout 541, left out by the cap 0, copies 0, warnings ['duplicate_rows', 'no_interval', 'same_features_grouped', 'suspected_leak']
copy of the label: two-valued; duplicates=keep: ran: scheme random, skill +0.010, training 1500, holdout 500, left out by the cap 0, copies 500, warnings ['duplicate_rows', 'holdout_copies_kept', 'suspected_leak']
copy of the label: two-valued; duplicates=error: holdout_copies_in_training: 500 of the 500 holdout rows (100.0%) have a copy among the training rows: the same value in every feature column and the same target. A model that re ...
copy of the label: distinct in every row; duplicates=group: holdout_copies_in_training: 500 of the 500 holdout rows (100.0%) have a copy among the training rows: the same value in every feature column and the same target. A mo ...
copy of the label: distinct in every row; duplicates=keep: ran: scheme random, skill +0.010, training 1500, holdout 500, left out by the cap 0, copies 500, warnings ['holdout_copies_kept', 'suspected_leak']
copy of the label: distinct in every row; duplicates=error: holdout_copies_in_training: 500 of the 500 holdout rows (100.0%) have a copy among the training rows: the same value in every feature column and the same target. A mo ...

-- D70, D72: one entity and 300 of five rows, the fast profile
an entity of 1200 rows among 2700: ran: scheme grouped, skill +0.029, training 695, holdout 1305, left out by the cap 700, copies 0, warnings ['large_group_is_holdout', 'no_interval']
an entity of 1500 rows among 3000: ran: scheme grouped, skill +0.057, training 440, holdout 1560, left out by the cap 1000, copies 0, warnings ['large_group_is_holdout', 'no_interval']
an entity of 2000 rows among 3500: ran: scheme grouped, skill -0.071, training 1125, holdout 375, left out by the cap 2000, copies 0, warnings []
an entity of 2001 rows among 3501: ran: scheme grouped, skill -0.033, training 1120, holdout 380, left out by the cap 2001, copies 0, warnings []
an entity of 2500 rows among 4000: ran: scheme grouped, skill -0.132, training 1125, holdout 375, left out by the cap 2500, copies 0, warnings []
an entity of 6000 rows among 7500: ran: scheme grouped, skill +0.106, training 1125, holdout 375, left out by the cap 6000, copies 0, warnings []
the 2,500, full profile: ran: scheme grouped, skill -0.004, training 1310, holdout 2690, left out by the cap 0, copies 0, warnings ['large_group_is_holdout']
2,500 and three small entities (the full profile cannot split either): too_few_groups: The rows split hold 4 entities; a split of whole entities needs at least 6.
two entities of 1,100 rows among 4000 (each a quarter or more; 2,200 together): ran: scheme grouped, skill -0.023, training 1500, holdout 500, left out by the cap 2000, copies 0, warnings ['large_group_kept_in_training']
a few-valued table of 40,000 rows (sets of 2,500), duplicates='group', no entity column: ran: scheme grouped, skill +0.016, training 1386, holdout 614, left out by the cap 38000, copies 0, warnings ['duplicate_rows', 'no_inter ...
an entity of 2,500 rows under a time column (the rule does not apply): ran: scheme time_ordered, skill +0.042, training 1500, holdout 500, left out by the cap 2000, copies None, warnings ['interval_approximate']
```

The reference as changed:

```
package: E:\CODE\ml-professor\verify\reference\mlprofessor

-- D69: duplicates='group' and a column the leak screen removes (noise: the truth is 0)
copy of the label: None; duplicates=group: ran: scheme grouped, skill -0.004, training 1293, holdout 707, left out by the cap 0, copies 0, warnings ['duplicate_rows', 'no_interval', 'same_features_grouped']
copy of the label: None; duplicates=keep: ran: scheme random, skill -0.038, training 1500, holdout 500, left out by the cap 0, copies 500, warnings ['duplicate_rows', 'holdout_copies_kept']
copy of the label: None; duplicates=error: holdout_copies_in_training: 500 of the 500 holdout rows (100.0%) have a copy among the training rows: the same value in every feature column and the same target. A model can score on  ...
copy of the label: two-valued; duplicates=group: holdout_copies_in_training: 585 of the 585 holdout rows (100.0%) agree with a training row in every feature column that is left: 10 combination(s) of feature values are on both  ...
copy of the label: two-valued; duplicates=keep: ran: scheme random, skill -0.038, training 1500, holdout 500, left out by the cap 0, copies 500, warnings ['duplicate_rows', 'holdout_copies_kept', 'suspected_leak']
copy of the label: two-valued; duplicates=error: holdout_copies_in_training: 500 of the 500 holdout rows (100.0%) have a copy among the training rows: the same value in every feature column and the same target. A model can sco ...
copy of the label: distinct in every row; duplicates=group: holdout_copies_in_training: 500 of the 500 holdout rows (100.0%) agree with a training row in every feature column that is left: 16 combination(s) of feature values a ...
copy of the label: distinct in every row; duplicates=keep: ran: scheme random, skill -0.038, training 1500, holdout 500, left out by the cap 0, copies 500, warnings ['holdout_copies_kept', 'suspected_leak']
copy of the label: distinct in every row; duplicates=error: holdout_copies_in_training: 500 of the 500 holdout rows (100.0%) have a copy among the training rows: the same value in every feature column and the same target. A mo ...

-- D70, D72: one entity and 300 of five rows, the fast profile
an entity of 1200 rows among 2700: ran: scheme grouped, skill +0.019, training 695, holdout 1305, left out by the cap 700, copies 0, warnings ['large_group_is_holdout', 'no_interval']
an entity of 1500 rows among 3000: ran: scheme grouped, skill +0.010, training 440, holdout 1560, left out by the cap 1000, copies 0, warnings ['large_group_is_holdout', 'no_interval']
an entity of 2000 rows among 3500: fast_profile_too_few_rows: the fast profile's cap of 2000 rows kept too few entities: the rows hold 1 entity; at least 6 are needed, one for the holdout and one for each of the 5 folds. Every ...
an entity of 2001 rows among 3501: fast_profile_group_too_large: one entity of the column 'who' holds 2001 of the 3501 usable rows (57.2%), more than the fast profile's cap of 2000 rows. The cap keeps whole entities and never  ...
an entity of 2500 rows among 4000: fast_profile_group_too_large: one entity of the column 'who' holds 2500 of the 4000 usable rows (62.5%), more than the fast profile's cap of 2000 rows. The cap keeps whole entities and never  ...
an entity of 6000 rows among 7500: fast_profile_group_too_large: one entity of the column 'who' holds 6000 of the 7500 usable rows (80.0%), more than the fast profile's cap of 2000 rows. The cap keeps whole entities and never  ...
the 2,500, full profile: ran: scheme grouped, skill -0.034, training 1310, holdout 2690, left out by the cap 0, copies 0, warnings ['large_group_is_holdout']
2,500 and three small entities (the full profile cannot split either): too_few_groups: the rows hold 4 entities; at least 6 are needed, one for the holdout and one for each of the 5 folds
two entities of 1,100 rows among 4000 (each a quarter or more; 2,200 together): fast_profile_group_too_large: 2 entities of the column 'who' each hold a quarter or more of the 4000 usable rows and together 2200, more than the  ...
a few-valued table of 40,000 rows (sets of 2,500), duplicates='group', no entity column: ran: scheme grouped, skill -0.066, training 1385, holdout 615, left out by the cap 38000, copies 0, warnings ['duplicate_rows', 'no_inter ...
an entity of 2,500 rows under a time column (the rule does not apply): ran: scheme time_ordered, skill +0.042, training 1500, holdout 500, left out by the cap 2000, copies None, warnings ['interval_approximate']
```

## 5. How often the fast cap leaves out a large entity (B; D70, D72)

One entity of the size shown beside 300 entities of five rows, fast profile,
tool seeds 0 to 29 on the one table. The summary, then each seed.

The application as it stood:

```
package: E:\CODE\ml-professor\mlprofessor
| rows of the large entity | of the usable rows | left out by the cap, of 30 seeds | stopped | rows profiled when left out |
|---|---|---|---|---|
| 500 | 25% | 0 | none | - |
| 1000 | 40% | 12 | none | [1500] |
| 1500 | 50% | 21 | none | [1500] |
| 1900 | 56% | 28 | none | [1500] |
| 2000 | 57% | 30 | none | [1500] |
| 2001 | 57% | 30 | none | [1500] |
| 2500 | 62% | 30 | none | [1500] |

at each tool seed, 0 to 29: kept, out (left out by the cap) or stop
500: kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept
1000: kept kept kept kept out kept kept kept kept kept kept kept out kept out out kept out kept out kept out out kept out out out out kept kept
1500: kept out out out out out kept kept out out out kept out out out out kept out out kept kept out out kept out out out kept out out
1900: out kept out out out out out out out out out out out out out out out out out out out out out out out kept out out out out
2000: out out out out out out out out out out out out out out out out out out out out out out out out out out out out out out
2001: out out out out out out out out out out out out out out out out out out out out out out out out out out out out out out
2500: out out out out out out out out out out out out out out out out out out out out out out out out out out out out out out
```

The reference as changed:

```
package: E:\CODE\ml-professor\verify\reference\mlprofessor
| rows of the large entity | of the usable rows | left out by the cap, of 30 seeds | stopped | rows profiled when left out |
|---|---|---|---|---|
| 500 | 25% | 0 | none | - |
| 1000 | 40% | 0 | none | - |
| 1500 | 50% | 0 | none | - |
| 1900 | 56% | 0 | none | - |
| 2000 | 57% | 0 | {'fast_profile_too_few_rows': 30} | - |
| 2001 | 57% | 0 | {'fast_profile_group_too_large': 30} | - |
| 2500 | 62% | 0 | {'fast_profile_group_too_large': 30} | - |

at each tool seed, 0 to 29: kept, out (left out by the cap) or stop
500: kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept
1000: kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept
1500: kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept
1900: kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept kept
2000: stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop
2001: stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop
2500: stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop
```

## 6. A date beyond the nanosecond range among ordinary dates

Found by the battery's first run: with one cell of `9999-12-31` in an ISO 8601
time column of 1,500 rows, the reference as it stood ended with exit code
3221225477 and no message. `crash_probe.py` shows where: pandas 3.1.1 does not
raise there, the process dies. The reference now asks the range first; the
application always did.

```
with one cell of 9999-12-31: parsed as datetime64[us], latest 9999-12-31 00:00:00; beyond the nanosecond range, asked without converting: True
ordinary dates: parsed as datetime64[us], latest 2022-02-08 00:00:00; beyond the nanosecond range, asked without converting: False
--- the reference as it stood before this pass
package: C:\Users\uahma\AppData\Local\Temp\claude\0233f86d4baa4dd4\769d56fc-4162-4286-9c77-ea847bfed92a\scratchpad\before\verify\reference\mlprofessor; 1500 rows
Windows fatal exception: access violation

Current thread 0x0000ff78 (most recent call first):
  File "E:\CODE\ml-professor\mlprof_env\Lib\site-packages\pandas\core\base.py", line 739 in to_numpy
  File "C:\Users\uahma\AppData\Local\Temp\claude\0233f86d4baa4dd4\769d56fc-4162-4286-9c77-ea847bfed92a\scratchpad\before\verify\reference\mlprofessor\core.py", line 537 in _to_nanoseconds
  File "C:\Users\uahma\AppData\Local\Temp\claude\0233f86d4baa4dd4\769d56fc-4162-4286-9c77-ea847bfed92a\scratchpad\before\verify\reference\mlprofessor\core.py", line 618 in _time_keys
  File "C:\Users\uahma\AppData\Local\Temp\claude\0233f86d4baa4dd4\769d56fc-4162-4286-9c77-ea847bfed92a\scratchpad\before\verify\reference\mlprofessor\core.py", line 1788 in _run
  File "C:\Users\uahma\AppData\Local\Temp\claude\0233f86d4baa4dd4\769d56fc-4162-4286-9c77-ea847bfed92a\scratchpad\before\verify\reference\mlprofessor\core.py", line 255 in profile
  File "E:\CODE\ml-professor\docs\measurements\CP8\crash_probe.py", line 43 in <module>
C:/Users/uahma/AppData/Local/Temp/claude/0233f86d4baa4dd4/769d56fc-4162-4286-9c77-ea847bfed92a/scratchpad/work/record_measurements.sh: line 31:  2365 Segmentation fault      "$PY" -B -X faulthandler $M/crash_probe.py "$OLD"
exit code 139
--- the same, 400 rows
package: C:\Users\uahma\AppData\Local\Temp\claude\0233f86d4baa4dd4\769d56fc-4162-4286-9c77-ea847bfed92a\scratchpad\before\verify\reference\mlprofessor; 400 rows
time_column_invalid: time column 'date' holds a date outside the range a nanosecond timestamp can hold (1677-09-21 to 2262-04-11), so its order cannot be read
exit code 0
--- the reference
package: E:\CODE\ml-professor\verify\reference\mlprofessor; 1500 rows
time_column_invalid: time column 'date' holds a date outside the range a nanosecond timestamp can hold (1677-09-21 to 2262-04-11), so its order cannot be read
exit code 0
--- the application
package: E:\CODE\ml-professor\mlprofessor; 1500 rows
time_column_invalid: time column: a timestamp is out of range (Out of bounds nanosecond timestamp: 9999-12-31 00:00:00)
exit code 0
```

## 7. The gate

### 7.1 Each new check seen failing (proofs 1 and 2)

The four new checks on the reference as it stood, before any of it was changed:

```
C:\Users\uahma\AppData\Local\Temp\claude\0233f86d4baa4dd4\769d56fc-4162-4286-9c77-ea847bfed92a\scratchpad\before\verify\reference\mlprofessor: exit 1, 4 checks, 4 failing [8 s]: time.date_sentinel_seen, duplicates.group_leak_s ...
    time.date_sentinel_seen: the column `day` of `day_sentinel` has role 'feature' and reason None. It holds 582 distinct days written YYYYMMDD and one value that is no date, 99991231, in 540 of its 1800 cells: an open-ended d ...
    duplicates.group_leak_stops: the tool exited 0 on `few_valued_leak`; it must stop with the named error `holdout_copies_in_training` (`few_valued_leak` is `few_valued` with a column `status` that copies the label. With `--d ...
    fast.large_entity_stops: the tool exited 0 on `large_entity`; it must stop with the named error `fast_profile_group_too_large` (one entity of `large_entity` holds 2500 of its 4,000 rows, more than the fast profile's cap of ...
    fast.large_entity_kept: rows on `quarter_entity`: dropped_row_cap 1200, training + holdout 1500. One entity holds 1200 of the 2700 rows (44%): a quarter or more, and fewer than the fast profile's cap of 2000. Such an entit ...
```

The whole gate on the reference as changed and on each broken copy of this
pass (`good` is the reference with no fault):

```
good: exit 0, 47 checks, 0 failing [25 s]: -
F12b: exit 1, 47 checks, 1 failing [20 s]: duplicates.group_leak_stops
    duplicates.group_leak_stops: the tool exited 0 on `few_valued_leak`; it must stop with the named error `holdout_copies_in_training` (`few_valued_leak` is `few_valued` with a column `status` that copies the label. With `--d ...
F14b: exit 1, 47 checks, 1 failing [20 s]: time.date_sentinel_seen
    time.date_sentinel_seen: the column `day` of `day_sentinel` has role 'feature' and reason None. It holds 582 distinct days written YYYYMMDD and one value that is no date, 99991231, in 540 of its 1800 cells: an open-ended d ...
F15: exit 1, 47 checks, 2 failing [21 s]: fast.large_entity_stops, fast.large_entity_kept
    fast.large_entity_stops: the tool exited 0 on `large_entity`; it must stop with the named error `fast_profile_group_too_large` (one entity of `large_entity` holds 2500 of its 4,000 rows, more than the fast profile's cap of ...
    fast.large_entity_kept: rows on `quarter_entity`: dropped_row_cap 1200, training + holdout 1500. One entity holds 1200 of the 2700 rows (44%): a quarter or more, and fewer than the fast profile's cap of 2000. Such an entit ...
F16: exit 1, 47 checks, 4 failing [20 s]: data.columns_accounted, leak.excluded, model.signal_clf, uncertainty.reported
    data.columns_accounted: skill on messy is -0.364004, expected between 0.55 and 0.91: the signal is still there under the mess; a wrong column role or lost rows remove it
    leak.excluded: skill on leak is -0.468376, expected between 0.54 and 0.91: with both leaks excluded this is the skill of `signal_clf`; near 1 means a leak is still a feature
    model.signal_clf: skill on signal_clf is -0.404286, expected between 0.54 and 0.91: the outcome depends on f1, f2 and seg. Too low: the model is not fitted on them, or the positive class is inverted. Too high: something leaks
    uncertainty.reported: headline.p_value on `signal_clf` is 1; with a clear signal no shuffle of the holdout targets matches the score, so it is (1 + 0) / (1 + 200)
F17: exit 1, 47 checks, 5 failing [23 s]: leak.regression, model.signal_reg, model.noise_reg, time.persistence_baseline, time.panel_previous_value
    leak.regression: skill on leak_reg is 0.602126, expected between 0.76 and 0.93: with the copy excluded this is the skill of `signal_reg`
    model.signal_reg: skill on signal_reg is 0.647989, expected between 0.76 and 0.93: amount = 2.0*f1 - 1.2*f2 + noise. Too low: the model is not fitted on them, the worst configuration was chosen (squared error: lower is bet ...
    model.noise_reg: skill on noise_reg is -1.07033, expected between -0.65 and -0.05: the target is independent of all 300 columns, so honest skill is below zero (a linear model pays for 300 useless columns). Above the band:  ...
    time.persistence_baseline: skill on random_walk is -11.9137, expected between -0.3 and 0.05: tomorrow's level is today's plus noise, so nothing beats the previous value
    time.panel_previous_value: skill on panel is -5.06899, expected between -0.04 and 0.03: each series is a random walk: tomorrow's level is today's plus noise, so nothing beats the series' own previous value
F18: exit 1, 47 checks, 3 failing [26 s]: time.undeclared_warns, time.date_numbers_seen, time.date_sentinel_seen
    time.undeclared_warns: no warning with code `date_column_undeclared` names the column `date`: a date column is present and the rows were split at random
    time.date_numbers_seen: no warning with the code `date_column_undeclared` names the column `day`: the run says `--no-time-order` on a table that holds dates, and the tool's doubt about that statement is this warning (spec. ...
    time.date_sentinel_seen: no warning with the code `date_column_undeclared` names the column `day` of `day_sentinel` (spec.md 2.5; D63, D69)
F19: exit 1, 47 checks, 1 failing [28 s]: result.matches_json
    result.matches_json: ProfileResult.headline differs from profile.json: headline.interval_high: 0.906 != 0.9056175377998613
F20: exit 1, 47 checks, 1 failing [30 s]: output.complete
    output.complete: the limitations statement in profile.json is not the fixed text of spec.md, 'Limitations' (9 sentences, in order); found 0 entries, and sentence(s) beyond the shorter list differ
F21: exit 1, 47 checks, 2 failing [29 s]: output.deterministic, output.same_at_any_n_jobs
    output.deterministic: two runs on `signal_clf` with seed 0 (one through the command line, one through mlprofessor.profile) wrote different profile.json files: profile.json.meta.finished: 1795373847.6546977 != 1795373861.96 ...
    output.same_at_any_n_jobs: the full profile on `signal_clf_small` gives different results with --n-jobs 1 and --n-jobs 4: profile.json.meta.finished: 1795373855.2997687 != 1795373858.5027971. The same seed must give the sa ...
F22: exit 1, 47 checks, 1 failing [29 s]: output.same_at_any_n_jobs
    output.same_at_any_n_jobs: the full profile on `signal_clf_small` gives different results with --n-jobs 1 and --n-jobs 4: profile.json.headline.interval_high: 0.8897528176229508 != 0.8889967532467532. The same seed must gi ...
```

The whole gate on the application as it stands:

```
E:\CODE\ml-professor\mlprofessor: exit 1, 47 checks, 4 failing [362 s]: time.date_sentinel_seen, duplicates.group_leak_stops, fast.large_entity_stops, fast.large_entity_kept
    time.date_sentinel_seen: the column `day` of `day_sentinel` has role 'feature' and reason None. It holds 582 distinct days written YYYYMMDD and one value that is no date, 99991231, in 540 of its 1800 cells: an open-ended d ...
    duplicates.group_leak_stops: the tool exited 0 on `few_valued_leak`; it must stop with the named error `holdout_copies_in_training` (`few_valued_leak` is `few_valued` with a column `status` that copies the label. With `--d ...
    fast.large_entity_stops: the tool exited 0 on `large_entity`; it must stop with the named error `fast_profile_group_too_large` (one entity of `large_entity` holds 2500 of its 4,000 rows, more than the fast profile's cap of ...
    fast.large_entity_kept: rows on `quarter_entity`: dropped_row_cap 1200, training + holdout 1500. One entity holds 1200 of the 2700 rows (44%): a quarter or more, and fewer than the fast profile's cap of 2000. Such an entit ...
```

### 7.2 What else moved at the gate seed (proof 3)

Every value of every dataset, check, recorded value and run of the gate, on
the judge as committed (with the reference as it stood) and on the judge as
changed. `gate_moved.py` gives the counts; `output\gate_compare.txt` names
every value that moved, was added or went.

```
datasets: 78 values before under 26 names; 0 moved, 0 gone, 0 new under an existing name; only after: day_sentinel, few_valued_leak, large_entity, quarter_entity
checks: 120 values before under 12 names; 0 moved, 0 gone, 13 new under an existing name; only after: fast
    new   complete.catches[0]: 1 run(s)
    new   date_sentinel_seen.catches[0]: 1 run(s)
    new   date_sentinel_seen.message: 1 run(s)
    new   date_sentinel_seen.ok: 1 run(s)
    new   deterministic.catches[0]: 1 run(s)
    new   group_leak_stops.catches[0]: 1 run(s)
    new   group_leak_stops.message: 1 run(s)
    new   group_leak_stops.ok: 1 run(s)
    new   matches_json.catches[0]: 1 run(s)
    new   same_at_any_n_jobs.catches[0]: 1 run(s)
    new   signal_clf.catches[0]: 1 run(s)
    new   signal_reg.catches[0]: 1 run(s)
    new   undeclared_warns.catches[0]: 1 run(s)
records: 119 values before under 118 names; 0 moved, 0 gone, 0 new under an existing name; only after: day_sentinel role of `day`, few_valued_leak exit code, large_entity exit code, quarter_entity rows left out by the cap
runs: 15818 values before under 34 names; 8 moved, 0 gone, 0 new under an existing name; only after: day_sentinel, few_valued_leak, large_entity, quarter_entity
    moved  profile.warnings[0].message: 1 run(s): day_numbers
           "column 'day' holds dates but no time_co -> "column 'day' holds dates and time_col=F (1)
    moved  profile.warnings[1].message: 3 run(s): few_valued, messy, time_blocks_undeclared
           "column 'signup_date' holds dates but no -> "column 'signup_date' holds dates and ti (1); "column 'date' holds dates but no time_c -> "column 'date' holds dates and time_col= (1); "2000 of the 2000 rows agree with a ...
    moved  stdout: 4 run(s): day_numbers, few_valued, messy, time_blocks_undeclared
           "task: classification (rule); scheme: ra -> "task: classification (rule); scheme: ra (3); "task: classification (rule); scheme: gr -> "task: classification (rule); scheme: gr (1)
```

### 7.3 The launch settings (D72)

```
Seconds for one gate run on a known-good package, three runs for each setting, with nothing
else loading the machine (2026-11-22). Printed by docs/measurements/CP8/gate_timing.py, once
for each gate file; the first column is added here.

| Gate | start spacing, side by side | full: seconds | per edit (--fast): seconds | verdicts |
|---|---|---|---|---|
| as committed at f8edec4 (38 runs, 43 checks), on the reference as it stood | 0.5 s, 8 | 22.6, 22.6, 22.5 | 17.3, 17.3, 17.2 | GATE: PASS (41 checks); GATE: PASS (43 checks) |
| as committed at f8edec4 (38 runs, 43 checks), on the reference as it stood | 0.25 s, 8 | 19.7, 19.7, 19.5 | 13.6, 13.6, 13.7 | GATE: PASS (41 checks); GATE: PASS (43 checks) |
| as committed at f8edec4 (38 runs, 43 checks), on the reference as it stood | 0.25 s, 16 | 13.1, 13.1, 13.0 | 10.5, 10.6, 10.5 | GATE: PASS (41 checks); GATE: PASS (43 checks) |
| as committed at f8edec4 (38 runs, 43 checks), on the reference as it stood | 0.1 s, 16 | 12.4, 12.3, 12.4 | 8.4, 8.4, 8.5 | GATE: PASS (41 checks); GATE: PASS (43 checks) |
| during the pass, before `quarter_entity` was added (41 runs, 46 checks) | 0.5 s, 8 | 24.0, 24.0, 24.0 | 18.7, 18.8, 18.7 | GATE: PASS (44 checks); GATE: PASS (46 checks) |
| during the pass, before `quarter_entity` was added (41 runs, 46 checks) | 0.25 s, 8 | 21.0, 21.2, 20.9 | 15.1, 15.2, 15.3 | GATE: PASS (44 checks); GATE: PASS (46 checks) |
| during the pass, before `quarter_entity` was added (41 runs, 46 checks) | 0.25 s, 16 | 14.0, 13.8, 13.9 | 11.3, 11.3, 11.2 | GATE: PASS (44 checks); GATE: PASS (46 checks) |
| during the pass, before `quarter_entity` was added (41 runs, 46 checks) | 0.1 s, 16 | 12.7, 12.7, 12.6 | 10.0, 10.0, 9.9 | GATE: PASS (44 checks); GATE: PASS (46 checks) |
| as it is after CP8 (42 runs, 47 checks), on the reference as changed | 0.5 s, 8 | 29.2, 28.9, 29.0 | 20.9, 20.8, 20.9 | GATE: PASS (45 checks); GATE: PASS (47 checks) |
| as it is after CP8 (42 runs, 47 checks), on the reference as changed | 0.25 s, 8 | 27.1, 27.1, 27.1 | 18.8, 19.0, 18.9 | GATE: PASS (45 checks); GATE: PASS (47 checks) |
| as it is after CP8 (42 runs, 47 checks), on the reference as changed | 0.25 s, 16 (D72: the gate's own) | 17.7, 17.6, 17.6 | 12.9, 12.9, 12.8 | GATE: PASS (45 checks); GATE: PASS (47 checks) |
| as it is after CP8 (42 runs, 47 checks), on the reference as changed | 0.1 s, 16 | 16.8, 16.8, 16.6 | 12.4, 12.5, 12.4 | GATE: PASS (45 checks); GATE: PASS (47 checks) |

The budget for a per-edit run is 20 seconds (loop/loop.json, hook_gate_budget_sec). At the
settings the gate had until D72 (0.5 s, 8) the gate as it is after CP8 takes 20.9 seconds
on each edit: over the budget. At the settings of D72 it takes 12.9.
```

### 7.4 The self-test (proof 1)

`output\selftest.txt` is the whole output. Its last part, the table that
proves every pair of a check and the mode it names:

```
(10 lines before these are in output\selftest.txt)
ok  [temp dir] F6d caught by data.long_rows_stop
ok  [temp dir] F7b caught by time.dayfirst_stops
ok  [temp dir] F7c caught by time.integer_keys_distinct
ok  [temp dir] F9 caught by data.holdout_copies_stop
ok  [temp dir] F10 caught by groups.kept_together, groups.no_skill_on_entities
ok  [temp dir] F8b caught by time.panel_previous_value
ok  [temp dir] F7d caught by time.undeclared_stops
ok  [temp dir] F11 caught by statements.time_order_asked, statements.entities_asked
ok  [temp dir] F12 caught by duplicates.group_on_features, duplicates.group_no_skill
ok  [temp dir] F13 caught by groups.holdout_bounded
ok  [temp dir] F14 caught by time.date_numbers_seen
ok  [temp dir] F12b caught by duplicates.group_leak_stops
ok  [temp dir] F14b caught by time.date_sentinel_seen
ok  [temp dir] F15 caught by fast.large_entity_stops, fast.large_entity_kept
ok  [temp dir] F16 caught by model.signal_clf
ok  [temp dir] F17 caught by model.signal_reg
ok  [temp dir] F18 caught by time.undeclared_warns
ok  [temp dir] F19 caught by result.matches_json
ok  [temp dir] F20 caught by output.complete
ok  [temp dir] F21 caught by output.deterministic
ok  [temp dir] F22 caught by output.same_at_any_n_jobs
ok  [project dir] F1 caught by model.noise_clf, model.noise_p_value, model.noise_reg
ok  [project dir] F2 caught by leak.excluded, leak.regression, leak.near_leak_warned
ok  [project dir] F3 caught by model.headline_recomputed, model.selection_full
ok  [project dir] F4 caught by model.headline_recomputed, model.metric_clf
ok  [project dir] F5 caught by task.binary_numeric, task.ambiguous_stops
ok  [project dir] F6 caught by data.rows_reconcile, data.coercion_reported
ok  [project dir] F7 caught by time.order_respected, time.no_skill_on_blocks, time.gap_applied
ok  [project dir] F8 caught by time.persistence_baseline
ok  [project dir] F6b caught by data.columns_accounted
ok  [project dir] F6c caught by models.all_accounted
ok  [project dir] F6d caught by data.long_rows_stop
ok  [project dir] F7b caught by time.dayfirst_stops
ok  [project dir] F7c caught by time.integer_keys_distinct
ok  [project dir] F9 caught by data.holdout_copies_stop
ok  [project dir] F10 caught by groups.kept_together, groups.no_skill_on_entities
ok  [project dir] F8b caught by time.panel_previous_value
ok  [project dir] F7d caught by time.undeclared_stops
ok  [project dir] F11 caught by statements.time_order_asked, statements.entities_asked
ok  [project dir] F12 caught by duplicates.group_on_features, duplicates.group_no_skill
ok  [project dir] F13 caught by groups.holdout_bounded
ok  [project dir] F14 caught by time.date_numbers_seen
ok  [project dir] F12b caught by duplicates.group_leak_stops
ok  [project dir] F14b caught by time.date_sentinel_seen
ok  [project dir] F15 caught by fast.large_entity_stops, fast.large_entity_kept
ok  [project dir] F16 caught by model.signal_clf
ok  [project dir] F17 caught by model.signal_reg
ok  [project dir] F18 caught by time.undeclared_warns
ok  [project dir] F19 caught by result.matches_json
ok  [project dir] F20 caught by output.complete
ok  [project dir] F21 caught by output.deterministic
ok  [project dir] F22 caught by output.same_at_any_n_jobs
TIMING: full gate 19.8s, per-edit (--fast) 12.6s on the good fixture (2 slow check(s) skipped per edit)

PROOF: every check that names a failure mode, run on that mode's own fixture
| Check | Names | Failed on that fixture | Also failed on the fixture of |
|---|---|---|---|
| data.rows_reconcile | F6 | F6: yes, 2 of 2 runs | - |
| data.columns_accounted | F6b | F6b: yes, 2 of 2 runs | F4, F16 |
| data.coercion_reported | F6 | F6: yes, 2 of 2 runs | - |
| data.long_rows_stop | F6d | F6d: yes, 2 of 2 runs | - |
| data.holdout_copies_stop | F9 | F9: yes, 2 of 2 runs | - |
| models.all_accounted | F6c | F6c: yes, 2 of 2 runs | - |
| task.binary_numeric | F5 | F5: yes, 2 of 2 runs | F4 |
| task.ambiguous_stops | F5 | F5: yes, 2 of 2 runs | - |
| leak.excluded | F2 | F2: yes, 2 of 2 runs | F6, F16 |
| leak.regression | F2 | F2: yes, 2 of 2 runs | F17 |
| leak.near_leak_warned | F2 | F2: yes, 2 of 2 runs | - |
| model.signal_clf | F16 | F16: yes, 2 of 2 runs | - |
| model.signal_reg | F17 | F17: yes, 2 of 2 runs | - |
| model.noise_clf | F1 | F1: yes, 2 of 2 runs | - |
| model.noise_p_value | F1 | F1: yes, 2 of 2 runs | - |
| model.noise_reg | F1 | F1: yes, 2 of 2 runs | F17 |
| model.headline_recomputed | F3, F4 | F3: yes, 2 of 2 runs; F4: yes, 2 of 2 runs | - |
| model.metric_clf | F4 | F4: yes, 2 of 2 runs | F3 |
| model.selection_full | F3 | F3: yes, 2 of 2 runs | F1, F4, F6c |
| time.order_respected | F7 | F7: yes, 2 of 2 runs | - |
| time.no_skill_on_blocks | F7 | F7: yes, 2 of 2 runs | F1 |
| time.gap_applied | F7 | F7: yes, 2 of 2 runs | - |
| time.dayfirst_stops | F7b | F7b: yes, 2 of 2 runs | - |
| time.persistence_baseline | F8 | F8: yes, 2 of 2 runs | F3, F7, F17 |
| time.integer_keys_distinct | F7c | F7c: yes, 2 of 2 runs | F3, F7, F8 |
| time.undeclared_warns | F18 | F18: yes, 2 of 2 runs | - |
| time.undeclared_stops | F7d | F7d: yes, 2 of 2 runs | F11 |
| time.panel_previous_value | F8b | F8b: yes, 2 of 2 runs | F3, F7, F8, F17 |
| groups.kept_together | F10 | F10: yes, 2 of 2 runs | - |
| groups.no_skill_on_entities | F10 | F10: yes, 2 of 2 runs | - |
| statements.time_order_asked | F11 | F11: yes, 2 of 2 runs | - |
| statements.entities_asked | F11 | F11: yes, 2 of 2 runs | - |
| duplicates.group_on_features | F12 | F12: yes, 2 of 2 runs | F10 |
| duplicates.group_no_skill | F12 | F12: yes, 2 of 2 runs | F10 |
| groups.holdout_bounded | F13 | F13: yes, 2 of 2 runs | F10 |
| time.date_numbers_seen | F14 | F14: yes, 2 of 2 runs | F18 |
| time.date_sentinel_seen | F14b | F14b: yes, 2 of 2 runs | F14, F18 |
| duplicates.group_leak_stops | F12b | F12b: yes, 2 of 2 runs | F2 |
| fast.large_entity_stops | F15 | F15: yes, 2 of 2 runs | - |
| fast.large_entity_kept | F15 | F15: yes, 2 of 2 runs | F10, F13 |
| result.matches_json | F19 | F19: yes, 2 of 2 runs | - |
| output.complete | F20 | F20: yes, 2 of 2 runs | - |
| output.deterministic | F21 | F21: yes, 2 of 2 runs | - |
| output.same_at_any_n_jobs | F22 | F22: yes, 2 of 2 runs | F21 |
45 of 45 pairs of a check and a failure mode it names were seen failing, over 31 broken fixtures.

FOR INFORMATION: checks that give a reason in place of a failure mode, and the broken fixtures they failed on (no part of the verdict)
| Check | Failed on the fixture of |
|---|---|
| leak.control_kept | F6 |
| time.flat_holdout_reported | F3, F7, F8 |
| uncertainty.reported | F1, F4, F16 |
SELFTEST: PASS (good fixture passes twice with identical results; 31 failure mode(s) each caught by a check that names it; temp and project dirs agree)
```

## 8. The suite against the changed judge (proofs 4 and 5)

### 8.1 The mutation check against the suite as it is (proof 4)

The good reference may fail only the eleven cases the plan names under D59,
all of one test; the twelve new variants have no test yet. Lines are cut
here; `output\mutation_judge_changed_suite_as_is.txt` has them whole.

```
mutation check: 71 case(s), 8 at a time, projects under C:\Users\uahma\AppData\Local\Temp\mlprof-mutation-pnqihe0t
  note: docs/DECISIONS.md: copied from the repo
  note: examples/customers.csv: copied from the repo
good   fails 11 of 1828: test_date_numbers.py::test_values_that_are_no_dates_leave_the_column_what_it_was[29_feb_1900], test_date_numbers.py::test_values_that_are_no_ ...
F1     fails 28: test_chosen_failure.py::test_every_configuration_failing_after_selection_raises_all_models_failed, test_chosen_failure.py::test_failure_after_the_cho ...
F2     fails 11: test_duplicates_group.py::test_copies_that_appear_when_the_leak_screen_excludes_a_column_stop_the_run, test_guard_final_fit.py::test_surrounding_whit ...
F3     fails 21: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_cho ...
F4     fails 12: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_cho ...
F5     fails 52: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_cho ...
F6     fails 147: test_date_numbers.py::test_a_date_number_column_is_a_time_column_like_any_number[nullable_int], test_duplicates.py::test_values_are_compared_as_a_st ...
F7     fails 147: test_console_summary.py::test_summary_under_time_order_says_why_there_is_no_p_value, test_date_numbers.py::test_a_date_number_column_is_a_time_colum ...
F8     fails 16: test_groups_time.py::test_each_entity_has_its_own_previous_value[0], test_groups_time.py::test_each_entity_has_its_own_previous_value[2], test_groups ...
F9     fails 15: test_date_numbers.py::test_the_copies_stop_names_a_date_number_column, test_duplicates.py::test_command_line_takes_duplicates, test_duplicates.py::te ...
F6b    fails 50: test_columns.py::test_f6_every_column_once_in_file_order, test_columns.py::test_roles_judged_on_usable_rows, test_columns.py::test_rule_table_one_col ...
F6c    fails 18: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_cho ...
F6d    fails 6: test_csv_edges.py::test_a_line_pandas_reads_as_the_header_is_not_skipped[form_feed], test_csv_edges.py::test_a_line_pandas_reads_as_the_header_is_not_ ...
F7b    fails 34: test_edges.py::test_edge_raises_its_named_error[time_column_invalid], test_split_time.py::test_f7_read_time_refuses_other_forms[day_first], test_spli ...
F7c    fails 12: test_object_numbers.py::test_object_whole_numbers_are_exact[decimal], test_object_numbers.py::test_object_whole_numbers_are_exact[python_int], test_w ...
F10    fails 28: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_t ...
F7d    fails 6: test_date_numbers.py::test_it_is_named_when_nothing_is_said_about_time_order, test_time_order_stated.py::test_a_date_column_and_no_word_about_time_sto ...
F8b    fails 5: test_groups_time.py::test_an_entity_with_no_earlier_row_gets_the_training_mean, test_groups_time.py::test_each_entity_has_its_own_previous_value[0], t ...
F11    fails 25: test_date_numbers.py::test_it_is_named_when_nothing_is_said_about_time_order, test_panel_stop.py::test_a_persistent_panel_with_no_entity_column_stops ...
F12    fails 4: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_to ...
F13    fails 11: test_grouped_holdout.py::test_a_large_entity_is_never_held_out_when_smaller_ones_can_fill_a_quarter[100], test_grouped_holdout.py::test_a_large_entit ...
F14    fails 16: test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[decimal], test_date_numbers.py::test_a_column_of_dates_wr ...
F12b   fails 0   NOT CAUGHT: no test fails on this variant   [602 s]
F14b   fails 0   NOT CAUGHT: no test fails on this variant   [591 s]
F15    fails 0   NOT CAUGHT: no test fails on this variant   [636 s]
F16    fails 13: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_cho ...
F17    fails 3: test_full_profile.py::test_f3_headline_is_chosen_configuration_on_the_holdout[regression], test_non_finite.py::test_a_holdout_score_that_is_not_finite ...
F18    fails 13: test_console_summary.py::test_summary_shows_both_intervals_the_p_value_and_each_warning, test_date_numbers.py::test_a_column_of_dates_written_as_whol ...
F19    fails 25: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_cho ...
F20    fails 4: test_edges.py::test_limitations_and_caveat_are_the_fixed_texts[random-classification], test_edges.py::test_limitations_and_caveat_are_the_fixed_texts[ ...
F21    fails 12: test_duplicates.py::test_the_rule_needs_a_share_and_a_count[0_of_1000], test_duplicates.py::test_the_rule_needs_a_share_and_a_count[10_of_1000], test ...
F22    fails 1: test_full_profile.py::test_same_result_at_one_and_two_jobs   [597 s]
M1     fails 1: test_guard_final_fit.py::test_m1_final_fit_statistics_ignore_the_holdout   [420 s]
M2     fails 2: test_feature_names.py::test_a_level_only_in_rows_with_no_target_does_not_count, test_guard_final_fit.py::test_m2_levels_come_from_the_rows_of_each_fit ...
M3     fails 2: test_guard_leak_screen.py::test_m3_missing_by_class_counts_training_rows_only, test_guard_leak_screen.py::test_m3_single_column_skill_is_fitted_on_fol ...
M4     fails 3: test_guard_uncertainty.py::test_m4_block_bootstrap_draws_whole_blocks, test_skill_undefined.py::test_no_skill_interval_when_fewer_than_half_the_resamp ...
M5     fails 1: test_guard_uncertainty.py::test_m5_ties_count_in_the_p_value   [474 s]
M6     fails 1: test_non_finite.py::test_a_prediction_that_is_not_finite_counts_as_raising   [477 s]
M7     fails 4: test_fast_profile_error.py::test_fast_error_stands_when_the_full_profile_would_run[eight_dates_of_450], test_fast_profile_error.py::test_fast_error_st ...
M8     fails 1: test_guard_review3.py::test_m8_the_interval_is_the_middle_95_percent   [480 s]
M9     fails 7: test_groups_time.py::test_an_entity_with_no_earlier_row_gets_the_training_mean, test_groups_time.py::test_each_entity_has_its_own_previous_value[0], t ...
M10    fails 1: test_guard_review3.py::test_m10_the_fast_cap_is_a_sample_from_the_seed   [488 s]
M11    fails 1: test_guard_review3.py::test_m11_cv_std_is_the_sample_deviation   [490 s]
M12    fails 1: test_guard_review3.py::test_m12_tied_predictions_in_a_resample   [479 s]
M13    fails 1: test_guard_review3.py::test_m13_n_imputed_counts_rows_that_reach_a_model   [484 s]
M14    fails 4: test_read_once.py::test_a_text_cell_of_an_object_target_in_a_row_with_no_time_value, test_read_once.py::test_a_text_cell_of_an_object_time_column_in_a ...
M15    fails 4: test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_together_whatever_their_target, test_duplicates_group.py::test_each_set_of_cop ...
M16    fails 6: test_groups_time.py::test_an_entity_with_no_earlier_row_gets_the_training_mean, test_groups_time.py::test_each_entity_has_its_own_previous_value[0], t ...
M17    fails 1: test_groups_random.py::test_the_fast_cap_keeps_whole_entities   [446 s]
M18    fails 6: test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite[columns], test_review4_small.py::test_no_part_of_the_output_can ...
M19    fails 7: test_panel_stop.py::test_a_persistent_panel_with_no_entity_column_stops, test_panel_stop.py::test_the_command_line_says_there_are_no_entities, test_st ...
M20    fails 3: test_guard_review4.py::test_an_infinite_float_in_an_object_target_is_a_missing_target, test_guard_review4.py::test_an_object_target_of_numbers_with_an ...
M21    fails 1: test_groups_random.py::test_the_interval_resamples_whole_entities_and_there_is_no_p_value   [428 s]
M22    fails 11: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_t ...
M23    fails 1: test_guard_review5.py::test_m23_the_panel_warning_starts_just_above_one_half[just_above]   [435 s]
M24    fails 2: test_statements.py::test_one_message_asks_for_everything_that_is_missing, test_statements.py::test_the_command_line_asks_for_each_missing_pair   [418 s]
M25    fails 3: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_the_count_is_of_rows_that_agree_in_every ...
M26    fails 4: test_grouped_holdout.py::test_a_unit_too_large_for_a_sound_split_is_the_holdout_and_is_named, test_grouped_holdout.py::test_one_warning_for_each_large ...
M27    fails 3: test_grouped_holdout.py::test_a_large_unit_kept_in_training_is_named, test_grouped_holdout.py::test_one_warning_for_each_large_unit_and_a_share_of_its ...
M28    fails 1: test_date_numbers.py::test_eight_digit_text_is_accepted_as_the_time_column   [424 s]
M29    fails 2: test_date_numbers.py::test_the_date_rule_comes_before_the_id_like_rule[int64], test_date_numbers.py::test_the_date_rule_comes_before_the_id_like_rule[ ...
M30    fails 2: test_duplicates.py::test_the_rule_in_regression, test_guard_review5.py::test_m30_a_copy_at_step_8a_still_needs_the_same_target   [425 s]
M31    fails 0   NOT CAUGHT: no test fails on this variant   [429 s]
M32    fails 0   NOT CAUGHT: no test fails on this variant   [428 s]
M33    fails 0   NOT CAUGHT: no test fails on this variant   [424 s]
M34    fails 0   NOT CAUGHT: no test fails on this variant   [421 s]
M35    fails 0   NOT CAUGHT: no test fails on this variant   [419 s]
M36    fails 0   NOT CAUGHT: no test fails on this variant   [417 s]
M37    fails 0   NOT CAUGHT: no test fails on this variant   [416 s]
M38    fails 0   NOT CAUGHT: no test fails on this variant   [412 s]
M39    fails 0   NOT CAUGHT: no test fails on this variant   [414 s]
README tests (not counted: they judge README.md, the same file in every case): 0 of 13 failing
MUTATION CHECK: FAIL
```

The eleven cases, by name (that one file run by itself against the
reference, and again with the replacement of 8.2 in place):

```
tests/test_date_numbers.py as it is (not loosened), run against verify/reference/mlprofessor as changed:
FAILED tests/test_date_numbers.py::test_values_that_are_no_dates_leave_the_column_what_it_was[before_1900]
FAILED tests/test_date_numbers.py::test_values_that_are_no_dates_leave_the_column_what_it_was[after_2100]
FAILED tests/test_date_numbers.py::test_values_that_are_no_dates_leave_the_column_what_it_was[29_feb_2023]
FAILED tests/test_date_numbers.py::test_values_that_are_no_dates_leave_the_column_what_it_was[29_feb_1900]
FAILED tests/test_date_numbers.py::test_values_that_are_no_dates_leave_the_column_what_it_was[month_13]
FAILED tests/test_date_numbers.py::test_values_that_are_no_dates_leave_the_column_what_it_was[day_0]
FAILED tests/test_date_numbers.py::test_values_that_are_no_dates_leave_the_column_what_it_was[31_april]
FAILED tests/test_date_numbers.py::test_values_that_are_no_dates_leave_the_column_what_it_was[a_fraction]
FAILED tests/test_date_numbers.py::test_values_that_are_no_dates_leave_the_column_what_it_was[an_infinite_cell]
FAILED tests/test_date_numbers.py::test_values_that_are_no_dates_leave_the_column_what_it_was[nine_digit_text]
FAILED tests/test_date_numbers.py::test_values_that_are_no_dates_leave_the_column_what_it_was[seven_digit_text]
11 failed, 22 passed in 4.42s

the same file with the one replacement of the loosening, against the reference as changed and against the application as it stands:
33 passed in 3.70s
33 passed in 3.26s
```

### 8.2 The loosening (D59)

One replacement in one existing file, `tests\test_date_numbers.py`:

```
| File | In | Lines before | Lines after |
|---|---|---|---|
| `test_date_numbers.py` | `test_values_that_are_no_dates_leave_the_column_what_it_was` | `record = _roles(_no_date_case(case))["day"]`<br>`if case == "one_date_throughout":`<br>`assert record["role"] == "excluded" and record["reason"] == "constant", record`<br>`else:`<br>`assert record["role"] == "feature" and record["kind"] == "numeric", record` | `record = _roles(_no_date_case(case))["day"]`<br>`if case == "one_date_throughout":`<br>`assert record["role"] == "excluded" and record["reason"] == "constant", record`<br>`elif case in ("day_first", "yyyymm"):`<br>`assert record["role"] == "feature" and record["kind"] == "numeric", record`<br>`else:`<br>`# D59, D69: 200 days and one value that is no date. A number as built, and a`<br>`# date column once D69 is built; `test_date_sentinels.py` holds the second.`<br>`as_built = record["role"] == "feature" and record["kind"] == "numeric"`<br>`as_decided = record["role"] == "excluded" and record["reason"] == "datetime"`<br>`assert as_built or as_decided, record` |
```

### 8.3 The prototypes, red and green today

A prototype was written of every new test file of the pass, of the helper
file and of the one loosened file: nine files. **They are outside the repo
and are not kept in it** (the change prompt asks for that: the tests stage
writes its own from the test plan). What they printed is here. The whole
suite with the prototypes laid over it, run against the application as it
stands in a throwaway project outside the repo (`tests\test_judge_tools.py`
apart: the two tools are run by themselves, sections 9 and 10):

```
=========================== short test summary info ===========================
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[one_cell_99991231]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[one_cell_0]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[one_cell_20240230]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[five_percent]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[thirty_percent]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[sixty_percent]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[two_sentinels]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[three_sentinels]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[float_with_gaps_and_an_infinite_cell]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[text_with_0]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[before_1900]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[after_2100]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[29_feb_2023]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[29_feb_1900]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[month_13]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[day_0]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[31_april]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[a_fraction]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[an_infinite_cell]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[nine_digit_text]
FAILED tests/test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[seven_digit_text]
FAILED tests/test_date_sentinels.py::test_a_sentinel_counts_once_however_many_cells_hold_it
FAILED tests/test_date_sentinels.py::test_the_rule_from_both_sides[five_dates_and_one_other]
FAILED tests/test_date_sentinels.py::test_the_rule_from_both_sides[five_dates_and_two_others]
FAILED tests/test_date_sentinels.py::test_the_rule_from_both_sides[twenty_seven_dates_and_three_others]
FAILED tests/test_date_sentinels.py::test_the_date_rule_is_judged_on_the_usable_rows
FAILED tests/test_date_sentinels.py::test_it_is_excluded_named_and_handed_to_no_step
FAILED tests/test_date_sentinels.py::test_it_is_named_when_nothing_is_said_and_by_the_copies_stop
FAILED tests/test_date_sentinels.py::test_under_another_time_column_it_is_excluded_without_a_word
FAILED tests/test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[int64]
FAILED tests/test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[float]
FAILED tests/test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[text]
FAILED tests/test_date_sentinels.py::test_the_message_names_the_value_that_most_rows_hold
FAILED tests/test_date_sentinels.py::test_the_time_column_is_judged_over_every_value_whatever_its_row
FAILED tests/test_date_sentinels.py::test_read_time_refuses_it_too[int64] - F...
FAILED tests/test_date_sentinels.py::test_read_time_refuses_it_too[float] - F...
FAILED tests/test_date_sentinels.py::test_read_time_refuses_it_too[text] - as...
FAILED tests/test_fast_cap_groups.py::test_the_error_is_named - AttributeErro...
FAILED tests/test_fast_cap_groups.py::test_an_entity_larger_than_the_cap_stops_the_fast_profile
FAILED tests/test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept
FAILED tests/test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run
FAILED tests/test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[exactly_the_cap]
FAILED tests/test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[one_row_more]
FAILED tests/test_fast_cap_groups.py::test_the_cap_keeps_whole_entities_every_large_one_and_no_more_than_it_can
FAILED tests/test_fast_cap_groups.py::test_the_command_line_exits_three - Ass...
FAILED tests/test_group_after_leak.py::test_a_copy_of_the_target_under_group_stops_the_run
FAILED tests/test_group_after_leak.py::test_the_stop_names_a_date_column_that_was_excluded_too
FAILED tests/test_group_after_leak.py::test_one_combination_on_both_sides_is_enough
FAILED tests/test_group_after_leak.py::test_the_grouped_warning_says_what_group_measures
FAILED tests/test_group_after_leak.py::test_the_command_line_stops_too - Asse...
FAILED tests/test_readme_review6.py::test_the_readme_names_what_the_seventh_pass_added
FAILED tests/test_review6_small.py::test_the_date_warning_holds_the_caveat_of_its_scheme[random]
FAILED tests/test_review6_small.py::test_the_date_warning_holds_the_caveat_of_its_scheme[grouped_by_an_entity_column]
FAILED tests/test_review6_small.py::test_the_date_warning_holds_the_caveat_of_its_scheme[grouped_by_rows_that_agree]
54 failed, 1881 passed, 1 warning in 366.80s (0:06:06)
```

```
1935 cases in C:/Users/uahma/AppData/Local/Temp/claude/0233f86d4baa4dd4/769d56fc-4162-4286-9c77-ea847bfed92a/scratchpad/work/app_suite.junit.xml
| File | passed | failed |
|---|---|---|
| the files that existed before the pass | 1841 | 0 |
| test_guard_review6.py | 17 | 0 |
| test_date_sentinels.py | 15 | 37 |
| test_group_after_leak.py | 5 | 5 |
| test_fast_cap_groups.py | 3 | 8 |
| test_review6_small.py | 0 | 3 |
| test_readme_review6.py | 0 | 1 |
| the six new files together | 40 | 54 |
RED AND GREEN: PASS (every case is as the test plan lists it)
```

### 8.4 The mutation check with the prototypes laid over the suite (proof 5)

The notes of this run name the folder `docs/measurements/CP8/prototypes`: for
the run a copy of the nine prototype files stood there, and it was taken out
of the repo again when the run had ended.

```
mutation check: 71 case(s), 12 at a time, projects under C:\Users\uahma\AppData\Local\Temp\mlprof-mutation-iw350vvc
  note: docs/DECISIONS.md: copied from the repo
  note: examples/customers.csv: copied from the repo
  note: overlay: tests/cp8_helpers.py added from docs/measurements/CP8/prototypes
  note: overlay: tests/test_date_numbers.py replaced by docs/measurements/CP8/prototypes
  note: overlay: tests/test_date_sentinels.py added from docs/measurements/CP8/prototypes
  note: overlay: tests/test_fast_cap_groups.py added from docs/measurements/CP8/prototypes
  note: overlay: tests/test_group_after_leak.py added from docs/measurements/CP8/prototypes
  note: overlay: tests/test_guard_review6.py added from docs/measurements/CP8/prototypes
  note: overlay: tests/test_judge_tools.py added from docs/measurements/CP8/prototypes
  note: overlay: tests/test_readme_review6.py added from docs/measurements/CP8/prototypes
  note: overlay: tests/test_review6_small.py added from docs/measurements/CP8/prototypes
  note: tests/test_judge_tools.py is not run here: it runs the judge's other tools on the application
good   fails 0 of 1921   [538 s]
F1     fails 29: test_chosen_failure.py::test_every_configuration_failing_after_selection_raises_all_models_failed, test_chosen_failure.py::test_failure_after_the_cho ...
F2     fails 18: test_duplicates_group.py::test_copies_that_appear_when_the_leak_screen_excludes_a_column_stop_the_run, test_group_after_leak.py::test_a_copy_of_the_t ...
F3     fails 24: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_cho ...
F4     fails 14: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_cho ...
F5     fails 59: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_cho ...
F6     fails 150: test_date_numbers.py::test_a_date_number_column_is_a_time_column_like_any_number[nullable_int], test_date_sentinels.py::test_the_time_column_is_judg ...
F7     fails 155: test_console_summary.py::test_summary_under_time_order_says_why_there_is_no_p_value, test_date_numbers.py::test_a_date_number_column_is_a_time_colum ...
F8     fails 16: test_groups_time.py::test_each_entity_has_its_own_previous_value[0], test_groups_time.py::test_each_entity_has_its_own_previous_value[2], test_groups ...
F9     fails 17: test_date_numbers.py::test_the_copies_stop_names_a_date_number_column, test_date_sentinels.py::test_it_is_named_when_nothing_is_said_and_by_the_copie ...
F6b    fails 50: test_columns.py::test_f6_every_column_once_in_file_order, test_columns.py::test_roles_judged_on_usable_rows, test_columns.py::test_rule_table_one_col ...
F6c    fails 19: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_cho ...
F6d    fails 6: test_csv_edges.py::test_a_line_pandas_reads_as_the_header_is_not_skipped[form_feed], test_csv_edges.py::test_a_line_pandas_reads_as_the_header_is_not_ ...
F7b    fails 34: test_edges.py::test_edge_raises_its_named_error[time_column_invalid], test_split_time.py::test_f7_read_time_refuses_other_forms[day_first], test_spli ...
F7c    fails 12: test_object_numbers.py::test_object_whole_numbers_are_exact[decimal], test_object_numbers.py::test_object_whole_numbers_are_exact[python_int], test_w ...
F10    fails 39: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_t ...
F7d    fails 7: test_date_numbers.py::test_it_is_named_when_nothing_is_said_about_time_order, test_date_sentinels.py::test_it_is_named_when_nothing_is_said_and_by_the ...
F8b    fails 6: test_groups_time.py::test_an_entity_with_no_earlier_row_gets_the_training_mean, test_groups_time.py::test_each_entity_has_its_own_previous_value[0], t ...
F11    fails 26: test_date_numbers.py::test_it_is_named_when_nothing_is_said_about_time_order, test_date_sentinels.py::test_it_is_named_when_nothing_is_said_and_by_th ...
F12    fails 6: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_to ...
F13    fails 11: test_grouped_holdout.py::test_a_large_entity_is_never_held_out_when_smaller_ones_can_fill_a_quarter[100], test_grouped_holdout.py::test_a_large_entit ...
F14    fails 47: test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[decimal], test_date_numbers.py::test_a_column_of_dates_wr ...
F12b   fails 4: test_group_after_leak.py::test_a_copy_of_the_target_under_group_stops_the_run, test_group_after_leak.py::test_one_combination_on_both_sides_is_enough, ...
F14b   fails 37: test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[29_feb_1900], test_date_sentinels.py::test_a_colum ...
F15    fails 7: test_fast_cap_groups.py::test_an_entity_larger_than_the_cap_stops_the_fast_profile, test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_ ...
F16    fails 14: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_cho ...
F17    fails 4: test_full_profile.py::test_f3_headline_is_chosen_configuration_on_the_holdout[regression], test_guard_review6.py::test_m6_an_infinite_holdout_predicti ...
F18    fails 17: test_console_summary.py::test_summary_shows_both_intervals_the_p_value_and_each_warning, test_date_numbers.py::test_a_column_of_dates_written_as_whol ...
F19    fails 27: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_cho ...
F20    fails 4: test_edges.py::test_limitations_and_caveat_are_the_fixed_texts[random-classification], test_edges.py::test_limitations_and_caveat_are_the_fixed_texts[ ...
F21    fails 12: test_duplicates.py::test_the_rule_needs_a_share_and_a_count[0_of_1000], test_duplicates.py::test_the_rule_needs_a_share_and_a_count[10_of_1000], test ...
F22    fails 1: test_full_profile.py::test_same_result_at_one_and_two_jobs   [491 s]
M1     fails 2: test_guard_final_fit.py::test_m1_final_fit_statistics_ignore_the_holdout, test_guard_review6.py::test_m1_the_final_fit_is_centred_on_its_own_rows   [4 ...
M2     fails 2: test_feature_names.py::test_a_level_only_in_rows_with_no_target_does_not_count, test_guard_final_fit.py::test_m2_levels_come_from_the_rows_of_each_fit ...
M3     fails 2: test_guard_leak_screen.py::test_m3_missing_by_class_counts_training_rows_only, test_guard_leak_screen.py::test_m3_single_column_skill_is_fitted_on_fol ...
M4     fails 3: test_guard_uncertainty.py::test_m4_block_bootstrap_draws_whole_blocks, test_skill_undefined.py::test_no_skill_interval_when_fewer_than_half_the_resamp ...
M5     fails 2: test_guard_review6.py::test_m5_a_shuffle_that_ties_counts_where_only_some_of_them_tie, test_guard_uncertainty.py::test_m5_ties_count_in_the_p_value    ...
M6     fails 2: test_guard_review6.py::test_m6_an_infinite_holdout_prediction_in_a_regression_falls_back, test_non_finite.py::test_a_prediction_that_is_not_finite_cou ...
M7     fails 4: test_fast_profile_error.py::test_fast_error_stands_when_the_full_profile_would_run[eight_dates_of_450], test_fast_profile_error.py::test_fast_error_st ...
M8     fails 2: test_guard_review3.py::test_m8_the_interval_is_the_middle_95_percent, test_guard_review6.py::test_m8_the_interval_from_whole_entities_is_the_middle_95 ...
M9     fails 8: test_groups_time.py::test_an_entity_with_no_earlier_row_gets_the_training_mean, test_groups_time.py::test_each_entity_has_its_own_previous_value[0], t ...
M10    fails 2: test_guard_review3.py::test_m10_the_fast_cap_is_a_sample_from_the_seed, test_guard_review6.py::test_m10_the_fast_cap_is_a_sample_whatever_the_order_of ...
M11    fails 2: test_guard_review3.py::test_m11_cv_std_is_the_sample_deviation, test_guard_review6.py::test_m11_cv_std_of_two_scored_folds_is_their_sample_deviation   ...
M12    fails 3: test_guard_review3.py::test_m12_tied_predictions_in_a_resample, test_guard_review6.py::test_m12_tied_predictions_in_a_resample_of_blocks_and_of_entiti ...
M13    fails 1: test_guard_review3.py::test_m13_n_imputed_counts_rows_that_reach_a_model   [491 s]
M14    fails 4: test_read_once.py::test_a_text_cell_of_an_object_target_in_a_row_with_no_time_value, test_read_once.py::test_a_text_cell_of_an_object_time_column_in_a ...
M15    fails 4: test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_together_whatever_their_target, test_duplicates_group.py::test_each_set_of_cop ...
M16    fails 7: test_groups_time.py::test_an_entity_with_no_earlier_row_gets_the_training_mean, test_groups_time.py::test_each_entity_has_its_own_previous_value[0], t ...
M17    fails 6: test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept, test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_tha ...
M18    fails 6: test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite[columns], test_review4_small.py::test_no_part_of_the_output_can ...
M19    fails 7: test_panel_stop.py::test_a_persistent_panel_with_no_entity_column_stops, test_panel_stop.py::test_the_command_line_says_there_are_no_entities, test_st ...
M20    fails 3: test_guard_review4.py::test_an_infinite_float_in_an_object_target_is_a_missing_target, test_guard_review4.py::test_an_object_target_of_numbers_with_an ...
M21    fails 2: test_groups_random.py::test_the_interval_resamples_whole_entities_and_there_is_no_p_value, test_guard_review6.py::test_m21_a_resample_of_whole_entitie ...
M22    fails 17: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_t ...
M23    fails 2: test_guard_review5.py::test_m23_the_panel_warning_starts_just_above_one_half[just_above], test_guard_review6.py::test_m23_a_panel_between_the_two_cut_ ...
M24    fails 2: test_statements.py::test_one_message_asks_for_everything_that_is_missing, test_statements.py::test_the_command_line_asks_for_each_missing_pair   [488 s]
M25    fails 3: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_the_count_is_of_rows_that_agree_in_every ...
M26    fails 5: test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run, test_grouped_holdout.py::test_a_unit_too_large_for_ ...
M27    fails 3: test_grouped_holdout.py::test_a_large_unit_kept_in_training_is_named, test_grouped_holdout.py::test_one_warning_for_each_large_unit_and_a_share_of_its ...
M28    fails 4: test_date_numbers.py::test_eight_digit_text_is_accepted_as_the_time_column, test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_st ...
M29    fails 5: test_date_numbers.py::test_the_date_rule_comes_before_the_id_like_rule[int64], test_date_numbers.py::test_the_date_rule_comes_before_the_id_like_rule[ ...
M30    fails 2: test_duplicates.py::test_the_rule_in_regression, test_guard_review5.py::test_m30_a_copy_at_step_8a_still_needs_the_same_target   [496 s]
M31    fails 19: test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[sixty_percent], test_date_sentinels.py::test_a_col ...
M32    fails 8: test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[float], test_date_sentinels.py::test_as_the_time_column_a_value_ ...
M33    fails 1: test_group_after_leak.py::test_one_combination_on_both_sides_is_enough   [488 s]
M34    fails 1: test_fast_cap_groups.py::test_when_the_full_profile_would_stop_too_its_error_is_raised   [487 s]
M35    fails 4: test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept, test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_tha ...
M36    fails 2: test_review6_small.py::test_the_date_warning_holds_the_caveat_of_its_scheme[grouped_by_an_entity_column], test_review6_small.py::test_the_date_warning ...
M37    fails 1: test_group_after_leak.py::test_the_grouped_warning_says_what_group_measures   [490 s]
M38    fails 2: test_guard_review6.py::test_m38_one_fold_not_scored_record_for_each_fold_not_scored[three_folds], test_guard_review6.py::test_m38_one_fold_not_scored_ ...
M39    fails 1: test_guard_review6.py::test_m39_rows_with_a_cell_that_cannot_be_hashed_are_counted_as_repeats   [482 s]
README tests (not counted: they judge README.md, the same file in every case): 1 of 14 failing: test_readme_review6.py::test_the_readme_names_what_the_seventh_pass_added
MUTATION CHECK: PASS (the good reference fails nothing; each of 70 variant(s) fails at least one test)
```

Each broken copy and the test named for it, from the kept reports:

```
70 broken copies; the good reference fails 1 case(s): test_readme_review6.py::test_the_readme_names_what_the_seventh_pass_added
| Copy | Cases failing on it | Test named for it | Cases of that test failing |
|---|---|---|---|
| F1 | 29 | (the gate's; none named) | |
| F2 | 18 | (the gate's; none named) | |
| F3 | 24 | (the gate's; none named) | |
| F4 | 14 | (the gate's; none named) | |
| F5 | 59 | (the gate's; none named) | |
| F6 | 150 | (the gate's; none named) | |
| F6b | 50 | (the gate's; none named) | |
| F6c | 19 | (the gate's; none named) | |
| F6d | 6 | (the gate's; none named) | |
| F7 | 155 | (the gate's; none named) | |
| F7b | 34 | (the gate's; none named) | |
| F7c | 12 | (the gate's; none named) | |
| F7d | 7 | (the gate's; none named) | |
| F8 | 16 | (the gate's; none named) | |
| F8b | 6 | (the gate's; none named) | |
| F9 | 17 | (the gate's; none named) | |
| F10 | 39 | (the gate's; none named) | |
| F11 | 26 | (the gate's; none named) | |
| F12 | 6 | (the gate's; none named) | |
| F12b | 4 | `test_group_after_leak.py::test_a_copy_of_the_target_under_group_stops_the_run` | 1 |
| F13 | 11 | (the gate's; none named) | |
| F14 | 47 | (the gate's; none named) | |
| F14b | 37 | `test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column` | 21 |
| F15 | 7 | `test_fast_cap_groups.py::test_an_entity_larger_than_the_cap_stops_the_fast_profile` | 1 |
| F16 | 14 | (the gate's; none named) | |
| F17 | 4 | (the gate's; none named) | |
| F18 | 17 | (the gate's; none named) | |
| F19 | 27 | (the gate's; none named) | |
| F20 | 4 | (the gate's; none named) | |
| F21 | 12 | (the gate's; none named) | |
| F22 | 1 | (the gate's; none named) | |
| M1 | 2 | `test_guard_final_fit.py::test_m1_final_fit_statistics_ignore_the_holdout` | 1 |
|  |  | `test_guard_review6.py::test_m1_the_final_fit_is_centred_on_its_own_rows` | 1 |
| M2 | 2 | `test_guard_final_fit.py::test_m2_levels_come_from_the_rows_of_each_fit` | 1 |
| M3 | 2 | `test_guard_leak_screen.py::test_m3_single_column_skill_is_fitted_on_fold_training_rows_only` | 1 |
|  |  | `test_guard_leak_screen.py::test_m3_missing_by_class_counts_training_rows_only` | 1 |
| M4 | 3 | `test_guard_uncertainty.py::test_m4_block_bootstrap_draws_whole_blocks` | 1 |
| M5 | 2 | `test_guard_uncertainty.py::test_m5_ties_count_in_the_p_value` | 1 |
|  |  | `test_guard_review6.py::test_m5_a_shuffle_that_ties_counts_where_only_some_of_them_tie` | 1 |
| M6 | 2 | `test_non_finite.py::test_a_prediction_that_is_not_finite_counts_as_raising` | 1 |
|  |  | `test_guard_review6.py::test_m6_an_infinite_holdout_prediction_in_a_regression_falls_back` | 1 |
| M7 | 4 | `test_fast_profile_error.py::test_fast_error_stands_when_the_full_profile_would_run` | 2 |
|  |  | `test_time_pipeline.py::test_wide_panel_under_fast_raises` | 1 |
| M8 | 2 | `test_guard_review3.py::test_m8_the_interval_is_the_middle_95_percent` | 1 |
|  |  | `test_guard_review6.py::test_m8_the_interval_from_whole_entities_is_the_middle_95_percent` | 1 |
| M9 | 8 | `test_guard_review3.py::test_m9_the_leak_screen_uses_the_previous_value` | 1 |
| M10 | 2 | `test_guard_review3.py::test_m10_the_fast_cap_is_a_sample_from_the_seed` | 1 |
|  |  | `test_guard_review6.py::test_m10_the_fast_cap_is_a_sample_whatever_the_order_of_the_file` | 1 |
| M11 | 2 | `test_guard_review3.py::test_m11_cv_std_is_the_sample_deviation` | 1 |
|  |  | `test_guard_review6.py::test_m11_cv_std_of_two_scored_folds_is_their_sample_deviation` | 1 |
| M12 | 3 | `test_guard_review3.py::test_m12_tied_predictions_in_a_resample` | 1 |
|  |  | `test_guard_review6.py::test_m12_tied_predictions_in_a_resample_of_blocks_and_of_entities` | 2 |
| M13 | 1 | `test_guard_review3.py::test_m13_n_imputed_counts_rows_that_reach_a_model` | 1 |
| M14 | 4 | `test_read_once.py::` | 4 |
| M15 | 4 | `test_groups_random.py::test_no_entity_on_both_sides_or_in_two_folds` | 2 |
| M16 | 7 | `test_groups_time.py::test_the_leak_screen_measures_a_lag_against_the_entity_baseline` | 1 |
| M17 | 6 | `test_groups_random.py::test_the_fast_cap_keeps_whole_entities` | 1 |
|  |  | `test_guard_review6.py::test_m17_the_fast_cap_keeps_whole_entities_of_seven_rows` | 1 |
| M18 | 6 | `test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite` | 6 |
| M19 | 7 | `test_panel_stop.py::test_a_persistent_panel_with_no_entity_column_stops` | 1 |
|  |  | `test_statements.py::test_nothing_said_about_entities_stops_whatever_the_table_holds` | 3 |
| M20 | 3 | `test_guard_review4.py::test_m20_infinite_cells_of_the_target_and_the_time_column_are_counted` | 1 |
| M21 | 2 | `test_groups_random.py::test_the_interval_resamples_whole_entities_and_there_is_no_p_value` | 1 |
|  |  | `test_guard_review6.py::test_m21_a_resample_of_whole_entities_has_one_score` | 1 |
| M22 | 17 | `test_duplicates_group.py::test_each_set_of_copies_stays_on_one_side_and_in_one_fold` | 1 |
| M23 | 2 | `test_guard_review5.py::test_m23_the_panel_warning_starts_just_above_one_half` | 1 |
|  |  | `test_guard_review6.py::test_m23_a_panel_between_the_two_cut_offs_is_warned_about_on_the_command_line` | 1 |
| M24 | 2 | `test_statements.py::test_one_message_asks_for_everything_that_is_missing` | 1 |
| M25 | 3 | `test_duplicates_features.py::test_the_count_is_of_rows_that_agree_in_every_feature` | 1 |
| M26 | 5 | `test_grouped_holdout.py::test_what_the_holdout_can_spare_goes_back` | 1 |
| M27 | 3 | `test_grouped_holdout.py::test_a_large_unit_kept_in_training_is_named` | 1 |
| M28 | 4 | `test_date_numbers.py::test_eight_digit_text_is_accepted_as_the_time_column` | 1 |
|  |  | `test_guard_review6.py::test_m28_eight_digit_text_orders_a_panel_and_read_time_reads_it` | 1 |
| M29 | 5 | `test_date_numbers.py::test_the_date_rule_comes_before_the_id_like_rule` | 2 |
| M30 | 2 | `test_guard_review5.py::test_m30_a_copy_at_step_8a_still_needs_the_same_target` | 1 |
| M31 | 19 | `test_date_sentinels.py::test_a_sentinel_counts_once_however_many_cells_hold_it` | 1 |
| M32 | 8 | `test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run` | 3 |
| M33 | 1 | `test_group_after_leak.py::test_one_combination_on_both_sides_is_enough` | 1 |
| M34 | 1 | `test_fast_cap_groups.py::test_when_the_full_profile_would_stop_too_its_error_is_raised` | 1 |
| M35 | 4 | `test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept` | 1 |
| M36 | 2 | `test_review6_small.py::test_the_date_warning_holds_the_caveat_of_its_scheme` | 2 |
| M37 | 1 | `test_group_after_leak.py::test_the_grouped_warning_says_what_group_measures` | 1 |
| M38 | 2 | `test_guard_review6.py::test_m38_one_fold_not_scored_record_for_each_fold_not_scored` | 2 |
| M39 | 1 | `test_guard_review6.py::test_m39_rows_with_a_cell_that_cannot_be_hashed_are_counted_as_repeats` | 1 |

Cases of the new test files that fail on each copy:
  F1: test_guard_review6.py::test_m1_the_final_fit_is_centred_on_its_own_rows
  F2: test_group_after_leak.py::test_a_copy_of_the_target_under_group_stops_the_run, test_group_after_leak.py::test_no_combination_of_the_features_is_on_both_sides_in_a_finished_run[a_leak_beside_measures], test_group_after_le ...
  F3: test_guard_review6.py::test_m5_a_shuffle_that_ties_counts_where_only_some_of_them_tie, test_guard_review6.py::test_m6_an_infinite_holdout_prediction_in_a_regression_falls_back, test_guard_review6.py::test_m8_the_interval ...
  F4: test_guard_review6.py::test_m12_tied_predictions_in_a_resample_of_blocks_and_of_entities[blocks], test_guard_review6.py::test_m12_tied_predictions_in_a_resample_of_blocks_and_of_entities[entities]
  F5: test_fast_cap_groups.py::test_when_the_full_profile_would_stop_too_its_error_is_raised, test_guard_review6.py::test_m11_cv_std_of_two_scored_folds_is_their_sample_deviation, test_guard_review6.py::test_m12_tied_predictio ...
  F6: test_date_sentinels.py::test_the_time_column_is_judged_over_every_value_whatever_its_row, test_guard_review6.py::test_m1_the_final_fit_is_centred_on_its_own_rows, test_guard_review6.py::test_m28_eight_digit_text_orders_a ...
  F6c: test_guard_review6.py::test_m6_an_infinite_holdout_prediction_in_a_regression_falls_back
  F7: test_date_sentinels.py::test_a_time_column_that_is_no_date_column_is_a_number_as_before, test_date_sentinels.py::test_under_another_time_column_it_is_excluded_without_a_word, test_guard_review6.py::test_m11_cv_std_of_two ...
  F7d: test_date_sentinels.py::test_it_is_named_when_nothing_is_said_and_by_the_copies_stop
  F8b: test_guard_review6.py::test_m28_eight_digit_text_orders_a_panel_and_read_time_reads_it
  F9: test_date_sentinels.py::test_it_is_named_when_nothing_is_said_and_by_the_copies_stop, test_group_after_leak.py::test_under_keep_and_error_nothing_changes
  F10: test_fast_cap_groups.py::test_it_is_for_an_entity_column_with_no_time_column_only, test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[exactly_the_cap], test_fast_cap_groups.py::test_when_the_full_profile_would ...
  F11: test_date_sentinels.py::test_it_is_named_when_nothing_is_said_and_by_the_copies_stop
  F12: test_fast_cap_groups.py::test_it_is_for_an_entity_column_with_no_time_column_only, test_group_after_leak.py::test_no_combination_of_the_features_is_on_both_sides_in_a_finished_run[few_valued]
  F12b: test_group_after_leak.py::test_a_copy_of_the_target_under_group_stops_the_run, test_group_after_leak.py::test_one_combination_on_both_sides_is_enough, test_group_after_leak.py::test_the_command_line_stops_too, test_gro ...
  F14: test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[29_feb_1900], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[29_feb_ ...
  F14b: test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[29_feb_1900], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[29_feb ...
  F15: test_fast_cap_groups.py::test_an_entity_larger_than_the_cap_stops_the_fast_profile, test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept, test_fast_cap_groups.py::test_entities_of_a_quar ...
  F16: test_guard_review6.py::test_m5_a_shuffle_that_ties_counts_where_only_some_of_them_tie
  F17: test_guard_review6.py::test_m6_an_infinite_holdout_prediction_in_a_regression_falls_back
  F18: test_date_sentinels.py::test_it_is_excluded_named_and_handed_to_no_step, test_review6_small.py::test_the_date_warning_holds_the_caveat_of_its_scheme[grouped_by_an_entity_column], test_review6_small.py::test_the_date_war ...
  F19: test_guard_review6.py::test_m6_an_infinite_holdout_prediction_in_a_regression_falls_back, test_guard_review6.py::test_m8_the_interval_from_whole_entities_is_the_middle_95_percent
  M1: test_guard_review6.py::test_m1_the_final_fit_is_centred_on_its_own_rows
  M5: test_guard_review6.py::test_m5_a_shuffle_that_ties_counts_where_only_some_of_them_tie
  M6: test_guard_review6.py::test_m6_an_infinite_holdout_prediction_in_a_regression_falls_back
  M8: test_guard_review6.py::test_m8_the_interval_from_whole_entities_is_the_middle_95_percent
  M9: test_guard_review6.py::test_m28_eight_digit_text_orders_a_panel_and_read_time_reads_it
  M10: test_guard_review6.py::test_m10_the_fast_cap_is_a_sample_whatever_the_order_of_the_file
  M11: test_guard_review6.py::test_m11_cv_std_of_two_scored_folds_is_their_sample_deviation
  M12: test_guard_review6.py::test_m12_tied_predictions_in_a_resample_of_blocks_and_of_entities[blocks], test_guard_review6.py::test_m12_tied_predictions_in_a_resample_of_blocks_and_of_entities[entities]
  M16: test_guard_review6.py::test_m28_eight_digit_text_orders_a_panel_and_read_time_reads_it
  M17: test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept, test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run, test_fast_cap_groups.py::test_the ...
  M21: test_guard_review6.py::test_m21_a_resample_of_whole_entities_has_one_score
  M22: test_fast_cap_groups.py::test_it_is_for_an_entity_column_with_no_time_column_only, test_group_after_leak.py::test_no_combination_of_the_features_is_on_both_sides_in_a_finished_run[few_valued], test_group_after_leak.py:: ...
  M23: test_guard_review6.py::test_m23_a_panel_between_the_two_cut_offs_is_warned_about_on_the_command_line
  M26: test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run
  M28: test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[text], test_date_sentinels.py::test_read_time_refuses_it_too[text], test_guard_review6.py::test_m28_eight_digit_text_orders_a_panel_a ...
  M29: test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[an_infinite_cell], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[fl ...
  M31: test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[sixty_percent], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[thirt ...
  M32: test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[float], test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[int64], test_date_sentinels.py::test_as ...
  M33: test_group_after_leak.py::test_one_combination_on_both_sides_is_enough
  M34: test_fast_cap_groups.py::test_when_the_full_profile_would_stop_too_its_error_is_raised
  M35: test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept, test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run, test_fast_cap_groups.py::test_the ...
  M36: test_review6_small.py::test_the_date_warning_holds_the_caveat_of_its_scheme[grouped_by_an_entity_column], test_review6_small.py::test_the_date_warning_holds_the_caveat_of_its_scheme[grouped_by_rows_that_agree]
  M37: test_group_after_leak.py::test_the_grouped_warning_says_what_group_measures
  M38: test_guard_review6.py::test_m38_one_fold_not_scored_record_for_each_fold_not_scored[three_folds], test_guard_review6.py::test_m38_one_fold_not_scored_record_for_each_fold_not_scored[two_folds]
  M39: test_guard_review6.py::test_m39_rows_with_a_cell_that_cannot_be_hashed_are_counted_as_repeats
NAMED TESTS: PASS (every broken copy fails each test named for it)
```

The number of cases that fail on each broken copy, with the suite as it is
and with the prototypes laid over it:

```
| Broken copy | suite as it is | with the prototypes |
|---|---|---|
| F1 | 28 | 29 |
| F2 | 11 | 18 |
| F3 | 21 | 24 |
| F4 | 12 | 14 |
| F5 | 52 | 59 |
| F6 | 147 | 150 |
| F7 | 147 | 155 |
| F8 | 16 | 16 |
| F9 | 15 | 17 |
| F6b | 50 | 50 |
| F6c | 18 | 19 |
| F6d | 6 | 6 |
| F7b | 34 | 34 |
| F7c | 12 | 12 |
| F10 | 28 | 39 |
| F7d | 6 | 7 |
| F8b | 5 | 6 |
| F11 | 25 | 26 |
| F12 | 4 | 6 |
| F13 | 11 | 11 |
| F14 | 16 | 47 |
| F12b | 0 | 4 |
| F14b | 0 | 37 |
| F15 | 0 | 7 |
| F16 | 13 | 14 |
| F17 | 3 | 4 |
| F18 | 13 | 17 |
| F19 | 25 | 27 |
| F20 | 4 | 4 |
| F21 | 12 | 12 |
| F22 | 1 | 1 |
| M1 | 1 | 2 |
| M2 | 2 | 2 |
| M3 | 2 | 2 |
| M4 | 3 | 3 |
| M5 | 1 | 2 |
| M6 | 1 | 2 |
| M7 | 4 | 4 |
| M8 | 1 | 2 |
| M9 | 7 | 8 |
| M10 | 1 | 2 |
| M11 | 1 | 2 |
| M12 | 1 | 3 |
| M13 | 1 | 1 |
| M14 | 4 | 4 |
| M15 | 4 | 4 |
| M16 | 6 | 7 |
| M17 | 1 | 6 |
| M18 | 6 | 6 |
| M19 | 7 | 7 |
| M20 | 3 | 3 |
| M21 | 1 | 2 |
| M22 | 11 | 17 |
| M23 | 1 | 2 |
| M24 | 2 | 2 |
| M25 | 3 | 3 |
| M26 | 4 | 5 |
| M27 | 3 | 3 |
| M28 | 1 | 4 |
| M29 | 2 | 5 |
| M30 | 2 | 2 |
| M31 | 0 | 19 |
| M32 | 0 | 8 |
| M33 | 0 | 1 |
| M34 | 0 | 1 |
| M35 | 0 | 4 |
| M36 | 0 | 2 |
| M37 | 0 | 1 |
| M38 | 0 | 2 |
| M39 | 0 | 1 |
suite as it is: good   fails 11 of 1828
with the prototypes: good   fails 0 of 1921
```

## 9. The perturbation battery (D71 to D73)

`battery\prototype_battery.py` is the prototype of the plan: 694 tool runs,
with no classes yet. `battery\prototype_out_reference.txt` and
`prototype_out_application.txt` are what it printed on the reference as it
stood and on the application, `prototype_pairs_*.csv` one line for each pair.

`verify\battery.py` is the tool. Its two lists are fixed in it (D73). Run on
the reference as changed, `battery\battery_reference.txt` (the whole output)
and `battery\battery_pairs_reference.csv`; its last lines:

```
(261 lines before these are in battery\battery_reference.txt)
LISTED AND HOLDS HERE signal_clf:number_sentinel_999999 (open: D73, pass 8)
LISTED AND HOLDS HERE binary_numeric:number_sentinel_999999 (open: D73, pass 8)
LISTED AND HOLDS HERE messy:number_sentinel_999999 (open: D73, pass 8)
794 tool runs in 226 s at 16 side by side (3435 s of tool time): 32 gate runs as they are and 762 pairs; 518 pairs left out (--list says why)
holds 766; stated limit 2 of 2 listed; open 26 of 30 listed (D73, pass 8); on neither list and failing 0
BATTERY: PASS (holds 766, stated limit 2, open 26)
```

`battery\battery_reference_before_the_message_fix.txt` is the run before the
last change to the reference: one pair failed on neither list,
`few_valued_leak:second_date_added`. The message of the new stop did not name
the date column that had been excluded with it, which spec 11 pins for every
message of `holdout_copies_in_training`. The reference was changed, not the
list.

Run on the application as it stands, `battery\battery_application.txt` and
`battery\battery_pairs_application.csv`; its last lines, and the failing
pairs by the task that owns them:

```
(1095 lines before these are in battery\battery_application.txt)
LISTED AND HOLDS HERE noise_reg:target_sentinel_-999 (open: D73, pass 8)
794 tool runs in 193 s at 16 side by side (2935 s of tool time): 32 gate runs as they are and 762 pairs; 518 pairs left out (--list says why)
holds 667; stated limit 2 of 2 listed; open 28 of 30 listed (D73, pass 8); on neither list and failing 97
BATTERY: FAIL (97 of 794 do not hold and are on neither list; holds 667, stated limit 2, open 28)
```

```
794 tool runs: fails 97, holds 667, open 28, stated limit 2
T55, the date rule: 33 pair(s)
      3  day_numbers: exit 0, no error: ymd_sentinel_99991231, ymd_sentinel_0, ymd_sentinel_20240230
     30  day_sentinel: exit 0, no error: base, ymd_sentinel_99991231, ymd_sentinel_0, ymd_sentinel_20240230, ymd_sentinel_in_dropped_row, number_sentinel_-999, number_sentinel_999999, number_sentinel_1e308, copy_two_valued, copy_distinct, constant_added, id_added_int, id_added_text, second_date_added, noise_added, rows_shuffled, row_repeated, blank_line_added, missing_cell_feature, missing_cel ...
T56, the stop under duplicates='group' after a removed leak: 20 pair(s)
      1  few_valued: exit 0, no error: copy_two_valued
     19  few_valued_leak: exit 0, no error: base, copy_two_valued, constant_added, id_added_int, id_added_text, second_date_added, noise_added, rows_shuffled, row_repeated, blank_line_added, missing_cell_feature, missing_cell_target, feature_renamed, feature_renamed_awkward, target_renamed, target_first, target_respelt, target_respelt_flipped, white_space_cells
T57, the fast profile's cap: 44 pair(s)
     22  large_entity: exit 0, no error: base, number_sentinel_-999, number_sentinel_999999, number_sentinel_1e308, target_sentinel_-999, copy_distinct, constant_added, id_added_int, id_added_text, second_date_added, noise_added, rows_shuffled, row_repeated, blank_line_added, missing_cell_feature, missing_cell_target, feature_renamed, feature_renamed_awkward, target_renamed, role_column_rename ...
     22  quarter_entity: exit 0, no error: base, number_sentinel_-999, number_sentinel_999999, number_sentinel_1e308, target_sentinel_-999, target_sentinel_max_double, copy_distinct, constant_added, id_added_int, id_added_text, second_date_added, noise_added, row_repeated, blank_line_added, missing_cell_feature, missing_cell_target, feature_renamed, feature_renamed_awkward, target_renamed, rol ...
open: 28: signal_clf:number_sentinel_-999, signal_clf:number_sentinel_999999, signal_reg:number_sentinel_-999, signal_reg:number_sentinel_999999, signal_reg:target_sentinel_-999, noise_reg:number_sentinel_999999, leak_reg:number_sentinel_-999, leak_reg:number_sentinel_999999, leak_reg:target_sentinel_-999, binary_numeric:number_sentinel_999999, messy:number_sentinel_-999, messy:number_sentinel ...
stated limit: 2: dup_noise:noise_added, int_stamps:missing_cell_date
```

What the battery sees of a small fault that one test holds (M13, for which
the plan therefore asks no second test; spec 8.1):

```
verify/battery.py on the reference carrying the one fault M13 (n_imputed counted over every loaded row):
  E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B verify/battery.py <a copy of the reference with BROKEN = 'M13'> --jobs 12 --work <a folder of its own>

FAIL messy:base [base]
FAIL messy:date_sentinel_9999 [same]
FAIL messy:date_sentinel_1900 [same]
FAIL messy:date_sentinel_zero [same]
FAIL messy:number_sentinel_1e308 [band]
FAIL messy:copy_two_valued [same]
FAIL messy:copy_distinct [same]
FAIL messy:constant_added [same]
FAIL messy:id_added_int [same]
FAIL messy:id_added_text [same]
FAIL messy:second_date_added [same]
FAIL messy:noise_added [band]
FAIL messy:blank_line_added [same]
FAIL messy:missing_cell_feature [band]
FAIL messy:missing_cell_date [same]
FAIL messy:missing_cell_target [band]
FAIL messy:feature_renamed [same]
FAIL messy:feature_renamed_awkward [same]
FAIL messy:target_renamed [same]
FAIL messy:role_column_renamed [same]
FAIL messy:target_first [same]
FAIL messy:target_respelt [same]
FAIL messy:target_respelt_flipped [band]
FAIL messy:white_space_cells [same]
FAIL messy:float_whole_numbers [same]
794 tool runs in 329 s at 12 side by side (3806 s of tool time): 32 gate runs as they are and 762 pairs; 518 pairs left out (--list says why)
holds 740; stated limit 2 of 2 listed; open 27 of 30 listed (D73, pass 8); on neither list and failing 25
BATTERY: FAIL (25 of 794 do not hold and are on neither list; holds 740, stated limit 2, open 27)
```

## 10. The generated tables (D56, D71 to D73)

`generated_tables\prototype_generated_tables.py` is the prototype of the
plan; `prototype_run_seed20261122.txt`, `prototype_run_seed1.txt` and
`prototype_run_seed2.txt` its three runs of 1,000 tables, the application
against the reference as both stood. `prototype_table296.txt`,
`row_order.py` and `row_order_output.txt` are the finding on forests under
time order (the application hands a fit its rows in file order, the reference
in time order), and `variants.py` with `variants_300.txt` what the test sees
of each broken copy on 300 tables.

`verify\generated_tables.py` is the tool: 1,000 tables of seed 20261122. The
reference against itself (`generated_tables\final_reference_against_itself.txt`
is the whole output):

```
E:\CODE\ml-professor\verify\reference\mlprofessor against E:\CODE\ml-professor\verify\reference\mlprofessor: 1000 tables from seed 20261122, 96.0 s at 24 workers
tables by class: {'agrees': 987, 'stated limit': 0, 'open': 13, 'fails': 0, 'harness': 0}

failing tables by owner
  T55, the date rule (D69 point 1, D72 point 1): 0 table(s): 
  T56, duplicates='group' after a removed leak (D69 point 2): 0 table(s): 
  T57, the fast profile's cap with an entity column (D70, D72 point 5): 0 table(s): 
  NONE OF THE THREE: 0 table(s): 

open: {'forests_under_time_order': 0, 'target_copy_kept': 13} | stated limits: {} | not compared: {}

what the 1000 tables of seed 20261122 reach
    17  a YYYYMMDD column with one sentinel cell
     9  a YYYYMMDD column with a sentinel in a third of the rows
     4  a YYYYMMDD column with two sentinels
    11  a YYYYMMDD time column with a value that is no date
     9  a two-valued copy of the target beside few-valued columns
     7  the same under duplicates='group'
   117  an exact copy of the target (any table)
    56  an entity of a quarter or more of the usable rows (grouped scheme)
    15  the same, past the fast cap under the fast profile
     4  an entity of more rows than the fast cap
```

The application as it stands against the reference
(`generated_tables\final_application_against_reference.txt`):

```
E:\CODE\ml-professor\mlprofessor against E:\CODE\ml-professor\verify\reference\mlprofessor: 1000 tables from seed 20261122, 96.9 s at 24 workers
tables by class: {'agrees': 940, 'stated limit': 0, 'open': 21, 'fails': 39, 'harness': 0}

failing tables by owner
  T55, the date rule (D69 point 1, D72 point 1): 27 table(s): 8, 37, 72, 103, 185, 284, 339, 351, 429, 470, 484, 538, 653, 701, 720, 728, 732, 737, 770, 808, 819, 864, 873, 918, 930, 950, 990
        21  [property] a column of dates written YYYYMMDD is excluded as a date, also when it holds a value that is no
        20  [exact] columns[daynum] role, reason, leak rule
        13  [exact] warnings[date_column_undeclared, daynum] (count)
         7  [fit] cv.chosen_mean
         7  [fit] cv.chosen_std
         7  [fit] models
         7  [fit] predictions.csv (y_pred)
         6  [property] a time column of dates written YYYYMMDD with a value that is no date stops the run (step 3; D69
         6  [outcome] a result on one side, a named error on the other
         6  [fit] headline.score
         6  [fit] headline.skill
         1  [fit] headline.model
  T56, duplicates='group' after a removed leak (D69 point 2): 5 table(s): 88, 152, 229, 400, 722
         5  [property] with duplicates='group' no combination of the features that reached the models is on both sides
  T57, the fast profile's cap with an entity column (D70, D72 point 5): 7 table(s): 301, 312, 330, 338, 740, 741, 795
         4  [property] an entity of a quarter or more of the usable rows is kept by the fast cap (step 6; D72)
         3  [property] the fast profile stops when an entity it must keep does not fit under its cap (step 6; D70, D72
         3  [outcome] a result on one side, a named error on the other
  NONE OF THE THREE: 0 table(s): 

open: {'forests_under_time_order': 8, 'target_copy_kept': 15} | stated limits: {} | not compared: {'rows': 44, 'two_errors': 1}

what the 1000 tables of seed 20261122 reach
    17  a YYYYMMDD column with one sentinel cell
     9  a YYYYMMDD column with a sentinel in a third of the rows
     4  a YYYYMMDD column with two sentinels
    11  a YYYYMMDD time column with a value that is no date
     9  a two-valued copy of the target beside few-valued columns
     7  the same under duplicates='group'
   117  an exact copy of the target (any table)
    56  an entity of a quarter or more of the usable rows (grouped scheme)
    15  the same, past the fast cap under the fast profile
     4  an entity of more rows than the fast cap
```

## 11. The harness after the change (proofs 8 and 9)

`loop/run_loop.py doctor`:

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
  ok   gate lists 47 check(s)
  ok   gate self-test: SELFTEST: PASS (good fixture passes twice with identical results; 31 failure mode(s) each caught by a check that names it; temp and project dirs agree)
  ok   gate timing: 12.6s per edit, 17.7s in full
  ok   plan valid: 60 task(s) (42 complex, 10 routine, 8 support)
  WARN 12 uncommitted change(s) in the harness; commit before an unattended run
Doctor: OK, 1 warning(s)
```

`verify\smoke_test.py --matrix` (`output\matrix.txt` is the whole output):

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
| F12b | duplicates='group' and a column the leak screen removes: feature combinations are on both sides and the run goes ahead | duplicates.group_leak_stops |
| F14b | one value that is no date, and a column of dates written YYYYMMDD is a numeric feature | time.date_sentinel_seen |
| F15 | the fast profile leaves out an entity larger than its cap, or one of a quarter or more of the rows, and reports on the rest | fast.large_entity_stops, fast.large_entity_kept |
| F16 | the positive class is inverted: skill that is there is reported as far below zero | model.signal_clf |
| F17 | squared error is read as higher-is-better: the worst configuration is chosen | model.signal_reg |
| F18 | a date column under --no-time-order gets no warning | time.undeclared_warns |
| F19 | the result object is not the file: ProfileResult.headline holds rounded numbers | result.matches_json |
| F20 | a part of the output is missing: the limitations statement is empty | output.complete |
| F21 | a clock value outside meta.timing: two identical runs differ | output.deterministic |
| F22 | the resamples depend on the number of cores | output.same_at_any_n_jobs |
```

`loop/run_loop.py build --dry-run` (`output\build_dry_run.txt` is the whole
output); the tasks of this pass:

```
(22 lines before these are in output\build_dry_run.txt)
[20:44:51]   T22    routine  sonnet x2 -> opus x3               gate=0 tests=2  Small defects: the output folder, process settings, the no-features message, the console summary
[20:44:51]   T23    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README's commands use a table kept in the repo
[20:44:51]   T24    complex  opus x3                            gate=0 tests=1  Guard of the third pass: a line pandas reads as the header. No code change expected
[20:44:51]   T25    complex  opus x3                            gate=0 tests=1  No number that is not finite is reported, or turned into null
[20:44:51]   T26    complex  opus x3                            gate=0 tests=1  White space round a target value; a target cell of only white space
[20:44:51]   T27    routine  sonnet x2 -> opus x3               gate=0 tests=1  The seed's range
[20:44:51]   T28    routine  sonnet x2 -> opus x3               gate=0 tests=1  The fast profile's error only when the full profile would run
[20:44:51]   T29    routine  sonnet x2 -> opus x3               gate=0 tests=1  The console summary prints four significant digits
[20:44:51]   T30    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: which rows the fast profile keeps, and what this pass changed
[20:44:51]   T31    complex  opus x3                            gate=0 tests=1  Guards of the fourth pass: six pinned behaviours the third review could break unseen. No code change expected
[20:44:51]   T32    complex  opus x3                            gate=1 tests=1  Copies of a holdout row among the training rows stop the run; the `duplicates` argument
[20:44:51]   T33    complex  opus x3                            gate=1 tests=1  Whole numbers are compared exactly: integer time keys, distinct values, and whole numbers too large for 64 bits
[20:44:51]   T34    complex  opus x3                            gate=0 tests=1  Number columns as a database driver returns them; a warning for every unsupported column
[20:44:51]   T35    complex  opus x3                            gate=0 tests=1  The type matrix: every kind of column, in every role, ends in a result or a named error
[20:44:51]   T36    routine  sonnet x2 -> opus x3               gate=0 tests=1  One to nine digits in a fraction of a second; a refusal that says what is wrong
[20:44:51]   T37    routine  sonnet x2 -> opus x3               gate=0 tests=1  The fast profile's cap leaving too few rows: its own error when the full profile would run
[20:44:51]   T38    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: copies of holdout rows, and what this pass changed
[20:44:51]   T39    complex  opus x3                            gate=0 tests=1  Guards of the fifth pass: what the fourth review pinned and the application already does. No code change expected
[20:44:51]   T40    complex  opus x3                            gate=0 tests=1  Each object column is read once, over all loaded rows (the fourth review, finding 1)
[20:44:51]   T41    complex  opus x3                            gate=0 tests=1  No part of the output can hold a number that is not finite; the boundary of a failed configuration; the three-class message (the fourth review, fi ...
[20:44:51]   T42    complex  opus x3                            gate=5 tests=1  A date column with nothing said about time order stops the run; `time_col=False` and `--no-time-order`; the README's commands
[20:44:51]   T43    complex  opus x3                            gate=2 tests=1  A declared entity column under the random scheme: `group_col`, the grouped split, the cap, the interval
[20:44:51]   T44    complex  opus x3                            gate=1 tests=2  Entities under time order: each entity's own previous value, the leak screen's baseline, and the stop for a panel with no entity declared
[20:44:51]   T45    complex  opus x3                            gate=0 tests=1  `duplicates="group"`: each set of copies on one side of the split and of every fold
[20:44:51]   T46    complex  opus x3                            gate=0 tests=1  The type matrix gains a role: every kind of column as the entity column
[20:44:51]   T47    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: entities, the two statements, `duplicates="group"`, and what is not detected
[20:44:51]   T48    complex  opus x3                            gate=0 tests=1  Guards of the sixth pass: the cut-off of the panel warning, and what a copy is. No code change expected
[20:44:51]   T49    complex  opus x3                            gate=2 tests=1  Both statements on every call: `time_col` and `group_col` left unsaid stop the run, whatever the table holds
[20:44:51]   T50    complex  opus x3                            gate=2 tests=1  `duplicates="group"` keeps together the rows that agree in every feature column; the count and the warning renamed; the copies stop says when each ...
[20:44:51]   T51    complex  opus x3                            gate=1 tests=1  The grouped holdout holds a quarter and no more than it must; a unit of a quarter or more is named
[20:44:51]   T52    complex  opus x3                            gate=1 tests=1  Dates written as whole numbers (YYYYMMDD) are a date form: excluded as a date, named, and accepted as the time column
[20:44:51]   T53    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: both statements in every command and example, what `group` means, the two warnings for a large entity, dates as whole numbers, and the ...
[20:44:51]   T54    complex  opus x3                            gate=0 tests=1  Guards of the seventh pass: a second test for eleven thin spots, a date beyond the nanosecond range, and two readings D72 pins. No code change exp ...
[20:44:51]   T55    complex  opus x3                            gate=1 tests=1  A column of dates written YYYYMMDD stays a date column when it holds a sentinel; as the time column, a value that is no date stops the run
[20:44:51]   T56    complex  opus x3                            gate=1 tests=1  `duplicates="group"` and a column the leak screen removes: one combination of the features that are left on both sides stops the run; the grouped  ...
[20:44:51]   T57    complex  opus x3                            gate=2 tests=1  The fast profile's cap with an entity column: an entity of a quarter or more of the rows is always kept, and the run stops with `fast_profile_grou ...
[20:44:51]   T58    routine  sonnet x2 -> opus x3               gate=0 tests=1  Two small things of the sixth review: the `date_column_undeclared` warning carries the caveat of its scheme, and a number that is not finite is ne ...
[20:44:51]   T59    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: a date column with a sentinel, the stop under `group` after a removed leak and what `group` measures, the fast profile with a large en ...
[20:44:51]   T60    complex  opus x3                            gate=0 tests=1  The judge's two other tools are clean on the build: the perturbation battery and the generated-table test
[20:44:51] Then: full gate (47 checks) + full suite, on failure an integration task at tier 'complex'.
```
