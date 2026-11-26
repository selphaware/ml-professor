r"""spec.md for the ninth pass, part 2 of 3: sections 8, 8.2, 11 and 12 (D77, D78, D80,
D81). Section 8.1's rows and section 10.11 are part 3.

    python -B edit_spec10b.py <spec.md in> <spec.md out>
"""
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
t = SRC.read_bytes().decode("utf-8")


def sub(old, new, count=1):
    global t
    found = t.count(old)
    assert found == count, f"{found} matches, expected {count}, for:\n{old[:160]}"
    t = t.replace(old, new)


# ------------------------------------------------------ 8: the failure modes --
sub('''| F28 | Under the grouped scheme the choice between configurations is made however few rows the folds outside one large unit validate on. | `groups.choice_not_made` | `dominant_entity`, as it is |
''', '''| F28 | Under the grouped scheme the choice between configurations is made however few rows the folds outside one large unit validate on. | `groups.choice_not_made` | `dominant_entity`, as it is |

**The seven failure modes of the ninth pass (D77, D78, D80, D81).** Each is the
reference with one rule of the pass taken out, with a gate run and one check
that names it; each new run was added last, so no seed of an older one moved.
Every check is structural but the first, which reads the gate's
`random_walk`: a run with one feature, where moving that feature's holdout
values among the holdout rows moves the predictions with them, whatever the
model. So the shuffled score is the squared error of the holdout targets
against a shuffle of `predictions.csv`, whose mean over every shuffle and
whose variance are exact, and the gate asserts the reported mean of five
shuffles within five of its standard deviations of that mean (measured, 30
data seeds: the reference −1.46 to +3.40, and over five one-feature runs,
150 in all, |z| at most 3.40; F29 −43.1 to −40.3; 10.11).

| Id | The silent failure | Gate check | Run |
|---|---|---|---|
| F29 | What the headline rests on is not of a shuffle of the holdout: every feature's holdout values are left where they are, so every fall is zero and the list names nothing. | `result.rests_on_recomputed` | `random_walk`, as it is |
| F30 | A headline skill of 0.98 or more carries no `near_perfect_headline` warning: two columns that give the target only together make a perfect headline in silence. | `rests_on.near_perfect_named` | `pair_together`: `signal_reg` with `pair_a` (the target plus a noise of its spread) and `pair_b` (that noise) |
| F31 | Columns that are copies of one another as the models receive them are shuffled one at a time: each shows a part of the fall they share. | `result.copies_share_a_fall` | `signal_reg_copied`: `signal_reg` with `f1_scaled` = 2 × `f1` + 3 |
| F32 | The leak screen does not read a column by its levels: the target cut into 30 bands, labels in random order, is a feature. | `leak.coded_copy_excluded` | `leak_reg_coded`: `signal_reg` with `amount_band` |
| F33 | The order reading puts rows of equal column value in the target's order: a column of noise with repeated values is warned about. | `leak.tied_noise_quiet` | `noise_tied`: `signal_reg` with `score` (a whole number from 1 to 40) and `noise_rounded` (to one decimal place) |
| F34 | Where more training rows are marked than may be set aside, rule 2 scores every row: a copy of a long-tailed target with five cells moved far off is kept. | `leak.changed_cells_excluded` | `skewed_copy`: a log-normal target and `amount_copy` = 1.07 × amount + 3, five cells moved up by 20 times the target's spread |
| F35 | Under time order a holdout of half of the rows or more is not named. | `time.large_holdout_named` | `last_stamp`: 60% of 1,000 rows on the last date |
''')

# ------------------------------------------------------- 8.2 the two tools --
sub('''### 8.2 The judge's other tools (D56, D71, D72, D73, D74, D76)''',
    '''### 8.2 The judge's other tools (D56, D71, D72, D73, D74, D76, D77, D78, D80, D81)''')
sub('''  flipped; white space round cells; whole numbers written as floats; and
  each other value of `--duplicates` where the run is about repeated rows;''',
    '''  flipped; white space round cells; whole numbers written as floats; and
  each other value of `--duplicates` where the run is about repeated rows.
  Since the ninth pass (D77 point 8) also: a column of noise rounded to one
  decimal place, and one of whole numbers from 1 to 40; the target as a
  code, as text and as numbers (a regression's target cut into bands with
  labels in random order; for a classification a code of 40 levels each of
  which holds one class); two columns that give the target only together; a
  copy of the target with five cells moved by 1, 20 and 1,000 times the
  target's spread, by a factor of 1.5, and to another row's value; an
  existing numeric feature put into 40 and into 10 bands; and a numeric
  feature written twice. Until then every numeric column of the gate and the
  battery was continuous, every altered-cell pair used one size of change,
  and no pair added two columns at once;''')
sub('''- on every run that finishes, the reconciliation rules of 5.3, and the
  largest row's share of the squared error as the judge counts it from
  `predictions.csv`.''',
    '''- **a carrier of the target (D77 point 9, D78 point 2, D80 point 2, D81).** A
  pair that plants one (a coded copy, two columns that give the target only
  together, a copy with cells changed, and the copies above) holds when a
  planted column ends in one of four outcomes: it is excluded as a suspected
  leak; a `strong_single_column` warning names it; the headline skill is 0.98
  or more and the `near_perfect_headline` warning names it; or
  `headline.rests_on` names it. Of two planted columns, either one named
  counts. Where none of the four holds, the pair holds when its headline
  skill is **no more than the unperturbed run's `headline.rests_on_floor`
  above the unperturbed run's headline skill** (D81): `within` that floor of
  it, either way, or `lowered` by more. The yardstick is the unperturbed
  run's, so a carrier cannot excuse itself by widening the interval it is
  measured against, and only a rise fails, since the claim is there to catch
  an inflated headline. The tool prints the count of each of the six
  outcomes, and of the pairs that end in none. A pair whose headline skill
  has no value is a stated limit (D36, below);
- a column of noise with repeated values: no new warning, and the run's
  conclusion; a feature put into bands: the conclusion holds; a feature
  written twice: the two share one fall (step 12a);
- on every run that finishes, the reconciliation rules of 5.3, the statement
  of step 12a among them (rules 17 to 21), and the largest row's share of the
  squared error as the judge counts it from `predictions.csv`.''')
sub('''generator's whole-number identifier `id_like` by rule 8b. **Agreement proves
neither side right; a disagreement is a finding.** The default is 1,000
tables from seed 20261122.''',
    '''generator's whole-number identifier `id_like` by rule 8b. Since the ninth
pass the properties include the statement of step 12a, on each side alone
(5.3, rules 17 to 21), and an exact copy of a feature sharing its fall: the
shuffled numbers themselves are not compared between the sides, since which
permutations are drawn is open (11). **Agreement proves neither side right;
a disagreement is a finding.** The default is 1,000 tables from seed
20261122 and 200 tables of the ninth pass's series: tables numbered from
1,000,000 up are the same recipe with columns of that pass's kinds drawn
from a stream of their own (noise with repeated values, the target as a code,
two columns that give the target only together, an exact copy of a feature,
a feature put into 10 or 40 bands), so that no table below 1,000,000 moves.''')
sub('''| battery | exactly four kinds, six pairs (D76 point 6). `dup_noise:noise_added` (D60: a copy of a row that differs by an added column is not a copy). `int_stamps:missing_cell_date` (D55: a whole-number time column of a CSV with an empty cell loads as floats, already rounded). `messy:sorted_by_measure` (D31: a distinct whole-number measure in a table sorted by it is called id-like). `copy_noise_5pct` on `random_walk`, `int_stamps` and `panel` (D76 point 2: under time order a copy of the target with noise above 1% of the target's spread) | none |''',
    '''| battery | the four kinds of D76 point 6, six pairs, and one pair of D36 (D80 point 2, D81), and nothing else. `dup_noise:noise_added` (D60: a copy of a row that differs by an added column is not a copy). `int_stamps:missing_cell_date` (D55: a whole-number time column of a CSV with an empty cell loads as floats, already rounded). `messy:sorted_by_measure` (D31: a distinct whole-number measure in a table sorted by it is called id-like). `copy_noise_5pct` on `random_walk`, `int_stamps` and `panel` (D76 point 2: under time order a copy of the target with noise above 1% of the target's spread). `flat_tail:two_columns_together` (D36: the headline skill has no value, so nothing can be said of what a carrier does to it, and no fall is computed) | none |''')
sub('''the failures the plan of the pass lists, which the application as built by
the seventh pass must show, and a strict test that asserts exit code 0 for
both tools is owned by the pass's last task (D74 point 7). The mutation
check leaves those files out: its''',
    '''the failures the plan of the pass lists, which the application as built by
the seventh pass must show, and a strict test that asserts exit code 0 for
both tools is owned by the pass's last task (D74 point 7). **Each pass keeps
that form (D78 point 8):** for the ninth pass the two older files,
`tests/test_judge_tools.py` and `tests/test_judge_tools_pass8.py`, accept
exactly the failures the ninth pass's plan lists, which the application as
built by the eighth pass shows, and the strict test is
`tests/test_judge_tools_pass9.py`, owned by the pass's last task. One test
that is strict for good was rejected: every such file is in the loop's
ratchet, and a strict one fails mid-build and reopens every finished task.
The mutation check leaves all of those files out: its''')

# --------------------------------------------- 11: what a message must name --
sub('''  folds`, and the number of every row that is the largest row of a
  configuration whose share is above 0.5 (step 11). The message of''',
    '''  folds`, and the number of every row that is the largest row of a
  configuration whose share is above 0.5 (step 11). And, since the ninth
  pass, three more (D77, D78). The message of `near_perfect_headline` holds
  `headline.skill` as a number and the name of each column of
  `headline.rests_on`, in quotes as `repr` writes it (step 12a). The message
  of `holdout_far_above_quarter` holds `H of the N rows`, H the holdout rows
  and N the rows split, as whole numbers in digits, and its fixed passage of
  5.4 (step 7). The message of `suspected_leak` for a column read by its
  levels and excluded by rule 2 holds the fixed sentence of 5.4 that names
  `group_col` (step 8). The message of''')
sub('''- how the one seed is turned into generator streams for the cap, the split, the
  folds and the resamples.''',
    '''- how the one seed is turned into generator streams for the cap, the split, the
  folds and the resamples, and so which permutations the five shuffles of
  step 12a draw and which resamples its stand-in draws: two correct builds
  give different shuffled scores, and the gate judges the one it recomputes
  by the distribution of a shuffle (8).''')

# ----------------------------------------------------- 11: the known limits --
sub('''**Known limits of this version, stated and not hidden (D56, D60, D66, D72,
D74, D76).** Nothing is open beside them: what D73 listed as open, and owned
by pass 8, is built (8.2).''',
    '''**Known limits of this version, stated and not hidden (D56, D60, D66, D72,
D74, D76, D77, D78, D80, D81).** Nothing is open beside them: what D73 listed
as open, and owned by pass 8, is built (8.2).

- **What is a defect of the leak screen, and what is a stated limit (D56,
  D77 point 9, D78 point 2, D80 point 2, D81).** Four passes each taught the
  screen one more disguise of a column that carries the target, and each
  review found the next; two columns that give the target only together
  cannot be seen by any reading of single columns. So the rule is this. **A
  column, or columns, that carry the target in a form the screen does not
  read count as a defect only when the output neither excludes them, nor
  warns about them by name, nor names them under the `near_perfect_headline`
  warning, nor names them in `headline.rests_on`,** and the headline rises
  with them by more than the floor of the list of the run without them. What
  those statements catch is a stated limit of the screen, and each kind
  measured is below. On the perturbation battery's 146 pairs that plant a
  carrier: excluded 103, warned about by name 2, named by the near-perfect
  warning 20, named in the list 7, the headline within the floor of the run
  without the carrier 8, lowered by more 5, and one where the headline skill
  has no value (D36; 8.2).
- **Coarse bands of the target (D78 point 4, D80 point 3).** A coded copy of
  a regression target with few bands is a weaker column than one with many,
  and the screen reads it as what it is. **Six to eight bands are warned
  about** (`strong_single_column`), at readings of 0.90 to 0.96; **five
  sometimes,** on 58 of 180 tables under the random scheme and 16 of 60 beside
  a quarter entity; **four or fewer read like an honest column of that
  strength,** 0.39 to 0.88, are not warned about, and are named in
  `headline.rests_on` when they lift the headline above the floor (on 134 to
  180 of 180 tables under the random scheme). **Under time order, on a target
  that wanders, no count of eight bands or fewer is warned about or
  excluded,** since the levels of the latest rows are not those of the
  earlier ones: on the gate's `random_walk`, `int_stamps` and `panel`, none
  of 2 to 30 bands was excluded and at most 3 of 90 warned about; at 40 and
  60 bands some are warned about (up to 51 of 90) and a few excluded (3 of
  90). On a time-ordered target that does not wander (`flat_tail`) the copy
  is read as under the random scheme: 8 and 16 bands warned about, 24 and up
  excluded. Wherever a banded copy was neither excluded, warned about nor
  named, it lifted the headline by less than 0.05: at most +0.037 over about
  1,500 tables (10.11). From 16 bands up such a copy is excluded on every
  table under the random scheme, and beside entities on most: 87 to 90 of 90
  beside a quarter entity, 33 to 90 of 90 beside a dominant one.
- **A coded copy whose levels the holdout rows mostly do not hold, under time
  order, lowers the headline (D81).** The latest rows of a trending target
  hold bands the earlier rows never did, so a model that leans on the code
  predicts the holdout badly: the headline falls far below the run without
  the column, and reads as no evidence of predictability (limitation 9 of
  5.4). In the battery, coded copies as text on `random_walk`,
  `walk_repeated`, `walk_copy` and `walk_copy_cell`, and as numbers on
  `walk_copy_cell`, lowered the headline skill by 1.5 to 907 (10.11). A
  lowered headline is not a confident wrong number: such a pair holds, and is
  counted apart (8.2).
- **Two columns that give the target only together, under time order and the
  fast profile (D78 point 2).** Neither is excluded or warned about, the
  headline stays below 0.98, and the list names them: on the battery's
  `random_walk`, `int_stamps`, `panel`, `walk_repeated` and `walk_copy` the
  pair lifted skill to 0.78 to 0.97 and `headline.rests_on` named it. Beside
  one extreme holdout cell (`walk_late_cell`) both headlines are about −1e9,
  which `extreme_value` and `largest_row_dominates` already name, and the
  carrier moves it by less than the floor.
- **What the statement cannot show (D77 point 1).** Two columns that carry the
  same information, and are not copies of one another as the models receive
  them, each show less than the fall they share, because the other is still
  there: beside `f1` and a copy of it with noise of 5% or 20% of its spread,
  `f1` fell on average by 0.37 to 0.98 of its fall alone and the noisy copy by
  0.01 to 0.28 of it; at least one of the two was named on 23 to 30 of 30
  tables, both on 0 to 27 (10.11). Exact copies as the models receive them are shuffled
  together and share one fall, so a table with two exact copies of a strong
  legitimate column shows both, each with the fall the column shows alone
  (0.97 to 1.14 times it under the linear model, a forest and boosting). The
  statement says what the number depends on, column by column; it says
  nothing of columns in combination.
- **A near-perfect headline on an easy table (D77, "Consequence").** A
  legitimately easy table, such as one that scores 0.995, carries the
  `near_perfect_headline` warning: so do the gate's `near_leak` (on 27 of 30
  seeds) and the undeclared `time_blocks` (29 of 30), whose headlines are
  near perfect for reasons the gate knows. The warning asks a question; the
  caller is the one who can answer it.''')
sub('''- **More than 1% of the training rows with a target wildly off.** The leak
  screen then sets none aside and scores every row (step 8): it is the
  screen as it was before D74 with the second reader added, and beside such
  a target a copy of it can stay a feature. With target cells of −999 beside
  a rescaled copy: on `leak_reg` the copy is excluded on 30 of 30 tables
  with 0.5% of the rows so changed, on 14 with 1% and on none with 2%; on
  `random_walk`, where the row after a wild cell is marked too, on 8 of 30
  with 0.5% and on none with 1% (10.10).''',
    '''- **More than 1% of the training rows with a target wildly off.** The leak
  screen then sets none aside (step 8). Rule 2 still leaves unscored the 1%
  of rows the column misses by most and the 1% the baseline misses by most,
  and is read against the fold's training median as well (D77 point 5, D78
  point 7), so a copy of a long-tailed target with up to 1% of its cells
  changed, by any amount, is excluded (10.11). Beside a target with more of
  its cells wild than the 1% left unscored, a copy of it can still stay a
  feature: what those cells do to the copy's error is then more than the
  rows left out can hold.''')

# ------------------------------------------------------------ 12: the index --
sub('''| D11 leak rule | 3 (step 8), 4.3; the screen is repaired by D74 and its warning tier has a second reading by D76 |''',
    '''| D11 leak rule | 3 (step 8), 4.3; the screen is repaired by D74, its warning tier has a second reading by D76, and it reads a column by its levels by D77 |''')
sub('''| D30 warning tier below the leak cut-off | 2.5, 3 (step 8), 4.3, 7, 8, 10.2; it reads every row (D74) and has a second reading, by order (D76) |''',
    '''| D30 warning tier below the leak cut-off | 2.5, 3 (step 8), 4.3, 7, 8, 10.2; it reads every row (D74) and has a second reading, by order (D76), which never puts rows of equal value in the target's order (D77); a coded copy of eight bands or fewer is in its tier (D78 point 4, D80 point 3) |''')
sub('''| D33 warning cut-off approved; three limits accepted | 2.5, 3 (steps 5, 8), 4.3, 10.1, 10.2 |''',
    '''| D33 warning cut-off approved; three limits accepted | 2.5, 3 (steps 5, 8), 4.3, 10.1, 10.2; amended by D77: a near-perfect headline always carries a warning of its own (step 12a) |''')
sub('''| D56 phase 1 is not shipped with known defects | 1, 3 (step 8a, "What this step does not see"), 11 ("Known limits"); `docs/BACKLOG.md` |''',
    '''| D56 phase 1 is not shipped with known defects | 1, 3 (step 8a, "What this step does not see"), 11 ("Known limits"); `docs/BACKLOG.md`. Its closing rule for the leak screen is amended by D77 point 9, D78 point 2, D80 point 2 and D81 (11, "What is a defect of the leak screen") |''')
sub('''| D76 second rulings on the plan of pass 8 | (1) the order reading of the warning tier: 2.5, 3 (step 8), 4.2, 4.3, 7, 8.1 (M43, M44), 10.10, 11.''',
    '''| D76 second rulings on the plan of pass 8 | (1) the order reading of the warning tier: 2.5, 3 (step 8), 4.2, 4.3, 7, 8.1 (M43, M44), 10.10, 11; read between rows of neighbouring distinct values since D77 point 4.''')
sub('''(2) the leak screen's repair: 2.5, 3 (step 8 and "Numbers that are not finite", rule 2), 4.2, 4.3, 7, 8 (F24, F25), 8.1 (M41, M42, M47), 9, 10.2, 10.10, 11; its fallback reading with the chosen model is replaced by D76.''',
    '''(2) the leak screen's repair: 2.5, 3 (step 8 and "Numbers that are not finite", rule 2), 4.2, 4.3, 7, 8 (F24, F25), 8.1 (M41, M42, M47), 9, 10.2, 10.10, 11; its fallback reading with the chosen model is replaced by D76; a categorical column's direct reading is replaced by the reading by levels, and rule 2 with too many rows marked is amended, by D77 points 3 and 5.''')
# the last row of the index, wherever it is: the new rows go after D76's
idx = t.index("| D76 second rulings on the plan of pass 8 |")
end = t.index("\n", idx)
t = t[:end + 1] + '''| D77 seventh review accepted; the headline says what it rests on (amends D11, D30, D33, D56, D74, D76) | (1) the statement of each feature: 2.2, 2.7, 3 (step 12a), 4.2, 5.2, 5.3 (rule 17), 7, 8 (F29, F31), 8.1, 10.11, 11. (2) the near-perfect warning: 2.5, 3 (step 12a), 4.3, 5.3 (rule 20), 7, 8 (F30), 8.1, 10.11, 11. (3) the reading by levels: 2.5, 3 (step 8), 4.2, 7, 8 (F32), 8.1, 10.11, 11. (4) the order reading without the tie fault: 2.5, 3 (step 8), 4.2, 4.3, 7, 8 (F33), 8.1, 10.11. (5) a copy with a handful of cells changed: 3 (step 8), 7, 8 (F34), 8.1, 10.11, 11. (6) a time-ordered holdout far above a quarter: 2.5, 3 (step 7), 4.3, 5.3 (rule 21), 5.4, 7, 8 (F35), 8.1, 10.11. (7) the judge-tools test: 8.2, in the form of D78 point 8. (8) the battery: 8.2, 10.11. (9) what is a limit of the screen: 11, amended by D78, D80 and D81 |
| D78 rulings on the plan of pass 9 (amends D77 points 1, 3, 7 and 9) | (1) the list in the headline and in the summary: 2.2, 3 (step 12a), 5.2, 5.3 (rule 19). (2) a carrier below 0.98 is named there: 8.2, 11. (3) copies shuffled together: 3 (step 12a), 4.3, 7, 8 (F31). (4) eight bands or fewer warned about: 11, in the words of D80 point 3. (5) unseen levels not scored, and the floor of the reading: 3 (step 8), 4.2. (6) a column of codes excluded, with the `group_col` sentence: 2.5, 3 (step 8), 5.4, 11. (7) the details: 2.5 (`shuffled_score_not_computed`, `holdout_far_above_quarter`), 3 (steps 7, 8, 12a), 4.2, 4.3. (8) each pass's judge-tools test: 8.2; `plan/test-plan.md` ("Ninth pass") |
| D79 the CP10 harness session finished on Opus | not in this spec: which model ran the session |
| D80 rulings on the CP10 prototype (amends D78; extends D79) | (1) the floor of the list and its stand-in: 2.2, 3 (step 12a), 4.3, 5.2, 5.3 (rule 18), 8.1, 10.11. (2) the carrier claim, and D36's stated limit: 8.2, 11. (3) the few-band case: 11. (4) copies up to sign: 3 (step 12a), 4.3, 7, 8.1. (5) `shuffled_score_not_computed`: 2.5. (6) who rules: not in this spec |
| D81 how the carrier claim reads its floor (clarifies D80 point 2) | 8.2 (the unperturbed run's floor, only a rise failing, and the six outcomes), 11 (a coded copy that lowers the headline), 10.11 |
''' + t[end + 1:]

DST.write_bytes(t.encode("utf-8"))
print("wrote", DST)
