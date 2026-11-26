r"""spec.md for the ninth pass, part 3 of 3: section 10.11, the measurements of the pass,
and the rows of section 8.1 (from `--rows FILE`, one table row a line).

    python -B edit_spec10c.py <spec.md in> <spec.md out> --rows <8.1 rows file>
"""
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
ROWS = Path(sys.argv[sys.argv.index("--rows") + 1]).read_text(encoding="utf-8").strip()
t = SRC.read_bytes().decode("utf-8")


def sub(old, new, count=1):
    global t
    found = t.count(old)
    assert found == count, f"{found} matches, expected {count}, for:\n{old[:160]}"
    t = t.replace(old, new)


# ------------------------------------------------------------- 8.1 the rows --
sub('''### 8.1 Small faults the gate cannot see (D39, D44, D46, D52, D57, D62 to D66, D69 to D72, D74, D76)''',
    '''### 8.1 Small faults the gate cannot see (D39, D44, D46, D52, D57, D62 to D66, D69 to D72, D74, D76 to D78, D80)''')
sub('''| M51 | The panel warning names the training mean, whatever baseline the run used (D74). | `test_repeated_stamps.py::test_the_panel_warning_names_the_baseline_the_run_used` |
''', '''| M51 | The panel warning names the training mean, whatever baseline the run used (D74). | `test_repeated_stamps.py::test_the_panel_warning_names_the_baseline_the_run_used` |
''' + ROWS + "\n")

# ------------------------------------------------------------ 10.11 --
MEASURED = r'''### 10.11 Measured for the ninth pass (D77, D78, D80, D81)

Measured from 2026-11-25 to 2026-11-26 in the harness session of CP10. The
scripts, the seeds and a per-seed table for each set are in
`docs/measurements/CP10/` (D35), with a README that gives each command. Every
rule was measured first on a prototype, a copy of the reference as it stood
with one switch for each proposed rule, and then on the reference as changed:
on every set below the two give the same output, record by record (section 0
of that README). Data seed *k* of a gate table is the gate's base seed 1115 +
100 × *k*, *k* = 0 to 29; the tool's seed is 0.

**A. What the headline rests on (step 12a).**

- *The spread over shuffles.* On twenty gate tables with two columns that give
  the target only together, one shuffle's fall for the weaker of the two had a
  standard deviation of 0.016 to 0.049 in a classification and 0.04 to 0.24 in
  a regression, where the falls are 0.22 to 2.0. Five shuffles are the mean of
  five draws: on none of 6,000 draws per table did the weaker column's mean
  fall below 0.05 where its fall is above it (D78 point 7).
- *The cost,* at 100,000 rows by 100 columns, full profile, all 64 cores (D16):
  classification 194.6 seconds before and 218.2 after, regression 222.0 and
  243.5; the statement 21.8 and 21.5 seconds of it. A forest predicts on one
  thread, so a chosen forest's 495 predictions take 214 to 357 seconds one
  after another and 21 to 34 side by side: they are spread over the cores.
- *Two columns that give the target only together* (`a` = the target plus a
  noise of its spread, `b` = that noise), added to the gate's tables, 30 seeds
  each. Under the random and the grouped scheme, both tasks, the fast and the
  full profile: skill 1.000, the warning raised and both columns named on 30
  of 30 seeds of each of fifteen runs, and of `random_walk`, `int_stamps` and
  `panel` under the full profile. Under time order and the fast profile (D78
  point 2): skill 0.68 to 0.995, at 0.98 or more on 0 to 2 of 30, and both
  columns named in the list on 30 of 30 of `random_walk`, `int_stamps`,
  `panel` and `walk_repeated`. On `imbalanced`, 5% of one class, `a` alone
  ranks the rare class almost perfectly: it was excluded on 9 of 30 seeds,
  and on the other 21 skill was 1.000, the warning raised and only `a` named,
  `b`'s fall being 0.001 to 0.015. On `flat_tail` skill has no value (D36).
- *The gate runs that gain the warning* (30 seeds): `near_leak` on 27 and the
  undeclared `time_blocks` on 29; at the gate's own seed, `time_blocks_undeclared`
  alone. No noise table of the gate reaches 0.98 on any seed.
- *Copies (D78 point 3, D80 point 4).* Each copy's fall as a share of the
  column's fall alone, 30 seeds:

  | Copy | Linear | Forest | Boosting |
  |---|---|---|---|
  | exact, once or twice | 0.97 to 1.06 | 1.00 to 1.14 | 1.00 |
  | rescaled, 2 × f + 3 | 0.97 to 1.05 | 1.14 (1 table) | |
  | negated, −f | 0.97 to 1.05 | 1.00 to 1.11 | 0.99 to 1.00 |
  | a text column with its levels renamed | 0.78 to 1.63, mean 1.02 | | |

  In every case the copy and its column hold the same three numbers. Found by
  their loaded values, as first proposed, a rescaled copy was shuffled alone
  and each of the two showed 0.11 to 0.38 of the fall (mean 0.25).
  *A column and another that carries the same information,* a copy of `f1`
  with noise of 5% or 20% of its spread: `f1` fell on average by 0.37 to 0.98
  of its fall alone, the noisy copy by 0.01 to 0.28; at least one of the two
  named on 23 to 30 of 30, both on 0 to 27.
- *The floor of the list (D80 point 1).* Seeds of 30 on which the list names
  any column:

  | Gate run | at 0.05 alone | at the floor of D80 | the floor |
  |---|---|---|---|
  | `noise_clf` | 3 | 0 | 0.091 to 0.143 |
  | `noise_reg` | 0 | 0 | 0.118 to 0.182 |
  | `imbalanced` | 13 | 1 | 0.050 to 0.254 |
  | `time_blocks` | 20 | 0 | 0.157 to 0.261 |
  | `time_blocks_gap` | 21 | 1 | 0.177 to 0.251 |
  | `entities` | 17 | 1 | 0.170 to 0.292 |
  | `full_noise_small` | 18 | 0 | 0.211 to 0.242 |
  | `few_valued` (no interval) | 5 | 0 | 0.050 to 0.095 |
  | `day_numbers` | 13 | 2 | 0.082 to 0.117 |
  | `day_sentinel` | 10 | 1 | 0.078 to 0.115 |
  | `dominant_entity`, `quarter_entity` | 0 | 0 | 0.050 |

  The lag stays named on 30 of 30 seeds of `random_walk`, `int_stamps`,
  `panel`, `walk_repeated`, `walk_copy` and `walk_copy_cell`. On
  `walk_late_cell` it was named on 10 and is now named on none: the one
  holdout cell of 999999 makes the skill interval about 4e9 wide, and
  `extreme_value` and `largest_row_dominates` name that cell. *The stand-in*
  was measured on 12 kinds of noise table with no interval (20 entities of 25
  rows, fewer than 30 in the holdout; 100 timestamps of 5 rows, 25 in the
  holdout), both tasks, 30 seeds: a bootstrap of the holdout rows named a
  column on at most 1 of 30; a fixed 0.10 on up to 9 of 30, 0.15 on up to 5.
  With signal, and with the two columns of above, every candidate named them
  on 30 of 30.
- *The carrier pairs of the battery (8.2; D81).* Of 146: excluded 103, warned
  about by name 2, named by the near-perfect warning 20, named in the list 7,
  the headline within the unperturbed run's floor 8, lowered by more than it
  5, and `flat_tail:two_columns_together`, where skill has no value (D36).
  The five that lower it are coded copies under time order (text on
  `random_walk`, `walk_repeated`, `walk_copy`, `walk_copy_cell`; numbers on
  `walk_copy_cell`), by 1.46 to 907 in skill. Read with each pair's own floor,
  or in both directions, six pairs failed; read as D81 has it, none. The table
  of every pair, with its lift and both floors, is in the README.
- *The one-feature run of the gate* (`result.rests_on_recomputed`): the
  reported shuffled score of `level_prev` on `random_walk`, in standard
  deviations of the mean of five shuffles from its exact mean, −1.46 to +3.40
  over 30 seeds; over five one-feature runs of the gate, 150 in all, |z| at
  most 3.40 and above 3 once. F29, which shuffles nothing: −43.1 to −40.3.

**B. The reading by levels (step 8).**

- *Coded copies of a regression target,* as text with labels in random order,
  as text in the target's order, and as a numeric code in random order, 30
  seeds each, excluded / warned: under the random scheme (three runs) 8 bands
  0 / 270 and 16, 24, 30, 40 and 60 bands 270 / 0 each; beside a quarter
  entity, 8 bands warned on 90, 16 bands excluded on 87 and warned on 3, 24
  and up excluded on 90; beside a dominant entity, 8 bands warned on 78, 16 on
  33 excluded and 51 named, 24 and 30 bands 69 excluded and 21 named, 40 bands
  87 and 3, 60 bands 90 excluded. Under time order on the walks
  (`random_walk`, `int_stamps`, `panel`, 90 tables each count): 8 to 30
  bands excluded on none and warned about on 0 to 3, named in the list on 130
  to 150 of 270; 40 bands warned about on 24; 60 bands warned about on 87 and
  excluded on 3. Wherever a banded copy on a walk was named by nothing it
  lifted the headline by at most +0.019. On `flat_tail`, a time-ordered
  target that does not wander, 8 and 16 bands were warned about on 90 of 90
  and 24 and up excluded on 90 of 90. *Two to eight bands* (D80 point 3), text and numeric code
  together: under the random scheme (180 tables a count) 2 bands named in the
  list on 180, 3 on 134, 4 on 172, 5 warned on 58 and named on 122, 6 to 8
  warned on 180; level readings 0.61 to 0.66 at 2 bands, 0.84 to 0.88 at 4,
  0.90 to 0.96 at 6 to 8. Beside a quarter entity 5 bands warned on 16 of 60
  and 6 to 8 on 60; beside a dominant entity 6 to 8 bands warned on 22 to 52
  of 60, readings 0.39 to 0.97. A banded copy that nothing named lifted the
  headline by at most +0.037 over about 1,500 tables.
- *Codes of a class,* text or numeric, of which every level holds one class
  (10, 40 and 100 levels), under the three schemes: excluded on 1,260 of
  1,260 tables.
- *Must not be excluded,* 30 seeds under each of the three schemes (90): an
  honest column of 10, 24 and 60 levels that explains 50% or 70% of the
  target, as text or numbers: excluded on none and warned about on none; at
  90%, warned about on 1 to 37 of 90 and excluded on none. Noise of 10, 60
  and 300 levels; a column of two to four rows a level at 0%, 50% and 90%;
  every column of every clean gate dataset (10,110 columns); and
  `examples/customers.csv` (largest level reading 0.358): excluded and warned
  about on none. *Where the two meet,* by the rule as written: an honest column
  that explains 98% is excluded on 7 to 83 of 90 tables and warned about on
  the rest, and one of 99% on 90 of 90; a text column of two to four rows a
  level that explains 99% is excluded on 87 of 90.
- *The floor of the reading (D78 point 5).* With unseen levels not scored and
  no floor, regression noise columns never read 0.90 or more at any number of
  scored rows (at most 0.63, at 5 to 9 rows). Classification noise read 1.000
  when the rarer class had one to four scored rows, at most 0.867 at five to
  seven, and 0.74 or less from ten. With the floor of 20 rows and 10 of each
  class, none of 16,836 noise columns was excluded or warned about (largest
  reading 0.737). An unseen level given the baseline's prediction, as first
  worded, failed 24 battery pairs.
- *The smallest changes that defeat the reading:* time order, where the
  holdout's levels are not the training rows'; four bands or fewer, which read
  like an honest column of that strength; and a code that names an entity
  each of whose rows has one target value, which is excluded with the
  `group_col` sentence (D78 point 6) and is indistinguishable from a coded
  copy.

**C. The order reading without the tie fault (step 8).** Share against the
order, 30 tables each, of columns of 25, 40, 100 and 400 distinct values at
1,125, 20,000 and 100,000 training rows:

- *Noise, and a weak column (5% and 20% of the target):* as built (ties in the
  target's order) warned about on 30 of 30 wherever values repeat; the pairs
  form on none, shares 0.475 to 0.500. One point for each distinct value, the
  mean target, warned about an honest column that explains 5% on 23 of 30 at
  100,000 rows and 25 values, and on 18 of 30 one that explains 20% at 20,000
  rows: it was not adopted.
- *Quiet:* noise of 5, 10, 40, 100 and 400 values and noise rounded to one and
  two decimal places, beside a target with signal in other columns, on nine
  regression runs: 0 warnings in 1,890 runs; 10,110 clean gate columns, of
  which none has both conditions hold (the smallest share 0.457); the CP9
  small-table cells as before; `age` and `s6` of the diabetes table are no
  longer named.
- *Still warned about, under time order (`random_walk`, `int_stamps`,
  `panel`), 120 tables each, excluded / warned / named in the list / nothing:*
  a copy with 2% of its cells rotated 33 / 87 / 0 / 0; its square 30 / 90 / 0 /
  0; its exponential 30 / 90 / 0 / 0. Rounded to 40 distinct values: the
  square 30 / 90 / 0 / 0, the copy with 2% rotated 30 / 82 / 7 / 1, the
  exponential 30 / 84 / 5 / 1, the exact copy 30 / 89 / 1 / 0; to 100 values
  the exponential 33 / 86 / 0 / 1. Under the random and the grouped scheme
  every one of them is excluded. A copy with noise of 5% of the target's
  spread is the stated limit of D76, and is now named in the list on 89 of
  90.

**D. A copy with a handful of cells changed (step 8).** 1, 5 and 15 cells of
1,500 changed by 0.1, 1, 5, 20 and 1,000 times the target's typical step (time
order) or spread, by a factor of 1.5, and to another row's value, on the
gate's regressions and on long-tailed tables, 30 seeds each: under the random
and the grouped scheme excluded on every table before and after; under time
order, with 15 cells changed, excluded on 98 to 115 of 120 and warned about on
the rest, before and after. On long-tailed targets (log-normal, σ 1.0 and
1.5), where the tail marks more than 1% of the rows: the copy with 5 cells
changed by 20 spreads was excluded on 0 of 30 before and on 30 of 30 after,
under the random and the time-ordered scheme alike; with 15 cells, on 20 to 30
of 30. Without the median baseline, noise columns were excluded beside a
target with 1.2% to 1.5% of its cells set to −999 or 999999 (186 of 2,430
tables); with it, on none. No other column was excluded on any table, before
or after.

**E. A large time-ordered holdout (step 7).** On tables whose last timestamp
holds 0.2, 0.26, 0.3, 0.34, 0.4, 0.45, 0.5, 0.55, 0.6 and 0.7 of 1,000 rows
(5 seeds each), the holdout is that timestamp, and the warning was raised on
every table from 0.5 up and on none below. On 33 tables of 8 to 40 equal
timestamps of 40 rows the holdout holds 0.25 to 0.33 of the rows, and the
warning was never raised; nor on any run of the gate before this pass, over
30 seeds.

**The gate at its own seed.** No value of an older run moved: compared with
`docs/measurements/CP2/long_rows.py compare` on every recorded value of every
check, before and after (the README of CP10). The one run that gains a
warning at the gate's seed is `time_blocks_undeclared`
(`near_perfect_headline`).
'''
sub('''---

## 11. What is open''', MEASURED + '''
---

## 11. What is open''')

DST.write_bytes(t.encode("utf-8"))
print("wrote", DST)
