r"""CLAUDE.md for the ninth pass (D77, D78, D80, D81), part 1: everything but the rows of
the small-faults table and the gate's timings, which part 2 adds.

    python -B edit_claude10.py <CLAUDE.md in> <CLAUDE.md out>
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


sub("`docs/DECISIONS.md` (D1 to D76).", "`docs/DECISIONS.md` (D1 to D81).")
sub('''generated-table test ("The judge's two other tools", below; D72). For the
eighth pass it accepts exactly the failures that pass's plan lists (D74), and
the strict test of the two tools belongs to the pass's last task: it is red
until the tasks before it are built.''', '''generated-table test ("The judge's two other tools", below; D72). For the
ninth pass it accepts exactly the failures that pass's plan lists, and so
does `tests/test_judge_tools_pass8.py`; the strict test of the two tools,
`tests/test_judge_tools_pass9.py`, belongs to the pass's last task: it is
red until the tasks before it are built (D78 point 8).''')

# ------------------------------------------------ the methodology paragraphs --
sub('''**The leak screen has two readers and two readings (D74, D76).** Beside its
tree a column is read, in a regression, by a straight line that a few rows
cannot move (the median of paired slopes), and in a classification directly,
with no minimum leaf.''', '''**The leak screen has its readers and two readings (D74, D76, D77).** Beside
its tree a column is read, in a regression, by a straight line that a few
rows cannot move (the median of paired slopes), and in a classification
directly, with no minimum leaf. **Since D77 it is also read by its levels,**
in both tasks: every categorical column, and a numeric column whose training
rows hold at least five rows for each distinct value. A validation row is
given the mean target of the fold's training rows of its level; **a row whose
level the fold's training rows do not hold is not scored** (D78 point 5),
and the reading has a value only with at least 20 rows scored and, in a
classification, 10 of each class or all of one. So the target cut into 16 or
more bands, with labels in any order, is excluded under the random scheme;
six to eight bands are warned about, and four or fewer read like an honest
column and are only named (D80 point 3). Under time order on a wandering
target the latest rows' levels are new, and a coded copy is mostly neither
excluded nor warned about (spec 11). The `suspected_leak` message of such a column says to
declare it as `group_col` if it names the entity each row belongs to (D78
point 6).''')
sub('''raises `strong_single_column` and excludes nothing.''',
    '''raises `strong_single_column` and excludes nothing. **Rows of equal column
value are never put in the target's order (D77 point 4):** the share is
taken over every pair of rows between neighbouring distinct values, so a
column of noise with repeated values is not named. **When more than 1% of
the training rows are marked as far off, none is set aside, but rule 2 still
leaves its 1% unscored and is read against each fold's training median as
well (D77 point 5, D78 point 7):** a copy of a long-tailed target with five
cells moved is excluded.''')
sub('''**Nothing is open (D73, D74, D76).**''', '''**The headline says what it rests on (D77, D78, D80, D81).** On every
finished run each feature's record holds `shuffled_score`, the chosen
configuration's score on the holdout with that feature's holdout values
shuffled (the mean of five shuffles, on a stream of their own; nothing
refitted), `shuffled_skill` and `skill_fall`. Copies of one another as the
models receive them (the same matrix column within 1e-9, or its negation;
the same block of level columns) are shuffled together and share one fall.
`headline.rests_on` lists the features whose fall is at least
`headline.rests_on_floor`, the larger of 0.05 and the half-width of the skill
interval (or of a bootstrap of the holdout rows where there is none), at
most five, largest first; the console summary prints it. **A headline skill
of 0.98 or more always carries one `near_perfect_headline` warning** that
names those columns and asks whether each is known before the outcome, in
both tasks and every scheme: an easy table carries it too. A model that
cannot predict a shuffled matrix leaves that feature's keys null with a
`shuffled_score_not_computed` warning; the run finishes. **Under time order a
holdout of half of the rows or more is named** by `holdout_far_above_quarter`
(D77 point 6). A carrier of the target that the screen does not read is a
defect only when nothing names it and the headline rises with it by more
than the unperturbed run's floor (D77 point 9, D80, D81; spec 11).

**Nothing is open (D73, D74, D76).**''')

# ------------------------------------------------------------- the output table --
sub('''| `columns` | one record per input column: role (`target`, `time`, `group`, `feature`, `excluded`), kind, reason, leak rule, four counts''',
    '''| `columns` | one record per input column: role (`target`, `time`, `group`, `feature`, `excluded`), kind, reason, leak rule, `shuffled_score`, `shuffled_skill` and `skill_fall` (null for a column that reached no model: D77), four counts''')
sub('''null for classification and where the error is zero | `.headline` |''',
    '''null for classification and where the error is zero; `rests_on`, the columns whose fall is at least `rests_on_floor` (D78, D80) | `.headline` |''')

# --------------------------------------------------------- the failure modes --
sub('''| F28 | under the grouped scheme the choice between configurations is made however few rows the folds validate on (D74) | groups.choice_not_made |
''', '''| F28 | under the grouped scheme the choice between configurations is made however few rows the folds validate on (D74) | groups.choice_not_made |
| F29 | what the headline rests on is not of a shuffle of the holdout: every feature's holdout values are left in place (D77) | result.rests_on_recomputed |
| F30 | a headline skill of 0.98 or more carries no `near_perfect_headline` warning (D77) | rests_on.near_perfect_named |
| F31 | columns that are copies of one another as the models receive them are shuffled one at a time (D78, D80) | result.copies_share_a_fall |
| F32 | the leak screen does not read a column by its levels: a coded copy of the target is a feature (D77, D78) | leak.coded_copy_excluded |
| F33 | the order reading puts rows of equal column value in the target's order: noise with repeated values is warned about (D77) | leak.tied_noise_quiet |
| F34 | where more training rows are marked than may be set aside, rule 2 scores every row: a copy with five cells moved is kept (D77, D78) | leak.changed_cells_excluded |
| F35 | under time order a holdout of half of the rows or more is not named (D77, D78) | time.large_holdout_named |
''')
sub('''F23 to F28 are the faults the judge's two other tools found on their first''',
    '''F29 to F35 are the rules of the ninth pass, each taken out of a broken copy
(D77, D78, D80). F23 to F28 are the faults the judge's two other tools found on their first''')
sub('''--selftest`: thirty-seven broken copies and 51 pairs of a check and its mode,
among 53 checks; it takes about twenty minutes), which''',
    '''--selftest`: forty-four broken copies and 58 pairs of a check and its mode,
among 60 checks; it takes about SELFTEST_MINUTES minutes), which''')
sub('''Six checks came with the eighth pass.''', '''Seven checks came with the ninth pass. `result.rests_on_recomputed` reads
`random_walk`, which has one feature: moving its holdout values moves the
predictions with them, so the gate computes the exact mean and spread of a
shuffle's squared error from `predictions.csv` and holds the reported
`shuffled_score` of `level_prev` within five standard deviations of it, and
checks every record's three keys, the floor, the list and the console line.
`rests_on.near_perfect_named` runs `pair_together`, `signal_reg` with
`pair_a` (the target plus a noise) and `pair_b` (that noise): skill 1.000 and
one `near_perfect_headline` warning that names both.
`result.copies_share_a_fall` runs `signal_reg_copied`, with `f1_scaled` = 2 ×
f1 + 3: the two records hold the same three numbers. `leak.coded_copy_excluded`
runs `leak_reg_coded`, with the target in 30 bands of random labels: excluded,
and the message names `group_col`. `leak.tied_noise_quiet` runs `noise_tied`,
with a whole number from 1 to 40 and noise rounded to one place: no warning
names either. `leak.changed_cells_excluded` runs `skewed_copy`, a log-normal
target and a copy with five cells moved by 20 spreads: excluded.
`time.large_holdout_named` runs `last_stamp`, 600 of 1,000 rows on the last
date: one `holdout_far_above_quarter` warning that says `600 of the 1000
rows`. All seven assert no band.

Six checks came with the eighth pass.''')

# ------------------------------------------------------ the judge's two tools --
sub('''## The judge's two other tools (D56, D71, D72, D73, D74, D76)''',
    '''## The judge's two other tools (D56, D71, D72, D73, D74, D76, D77, D78, D80, D81)''')
sub('''  the count, 0, is still printed. The battery's stated limits are exactly
  four kinds (D76): a copy of a row that differs by an added column (D60), a
  whole-number time column with an empty cell (D55), a distinct whole-number
  measure in a table sorted by it (D31), and under time order a copy of the
  target with noise above 1% of the target's spread (D76). The generated
  tables list none.''', '''  the count, 0, is still printed. The battery's stated limits are the four
  kinds of D76, a copy of a row that differs by an added column (D60), a
  whole-number time column with an empty cell (D55), a distinct whole-number
  measure in a table sorted by it (D31), and under time order a copy of the
  target with noise above 1% of the target's spread (D76), and one pair where
  the headline skill has no value (D36, D80). The generated tables list none.
- **A carrier of the target (D77 point 9, D78, D80, D81).** Since the ninth
  pass the battery also adds noise with repeated values, the target as a
  code, two columns that give the target only together, copies with five
  cells moved by several sizes, features put into bands and a feature written
  twice. A pair that plants a carrier holds when a planted column is
  excluded, warned about by name, named by the near-perfect warning or named
  in `headline.rests_on`; or when its headline skill is no more than the
  unperturbed run's `rests_on_floor` above the unperturbed run's (`within`,
  or `lowered` by more). It prints the six counts. The generated tables gain
  200 tables of the ninth pass's series (numbered from 1,000,000) with
  columns of those kinds.''')
sub('''  application, and from then on they are part of the full suite. For the
  eighth pass that file accepts exactly the failures the pass's plan lists,
  and a strict test of both tools is owned by the pass's last task (D74). The''',
    '''  application, and from then on they are part of the full suite. Each pass
  keeps its own file (D78 point 8): for the ninth pass `tests/test_judge_tools.py`
  and `tests/test_judge_tools_pass8.py` accept exactly the failures the pass's
  plan lists, and the strict test of both tools, `tests/test_judge_tools_pass9.py`,
  is owned by the pass's last task. Both older files are in the loop's
  ratchet, so neither can be strict while the build is under way. The''')

# ------------------------------------------------------- the mutation check --
sub('''each of eighty-eight variants of it: the thirty-seven of the table above and
M1 to M51. The good reference must fail nothing, and every variant at least
one test. It leaves out `tests/test_judge_tools.py` and the strict test of
the eighth pass, which run the judge's two other tools on the application.''',
    '''each of one hundred and thirteen variants of it: the forty-four of the table
above and M1 to M69. The good reference must fail nothing, and every variant
at least one test. It leaves out `tests/test_judge_tools.py` and each pass's
own file of that kind (`_pass8`, `_pass9`), which run the judge's two other
tools on the application. `--only` runs some cases alone.''')
sub('''The eighth pass loosens seven cases in the same way (D59, D74): those that
pinned the previous value as it was at repeated timestamps, where the
training mean was the only baseline and `stages.previous_value` refused
repeated keys.''', '''The eighth pass loosens seven cases in the same way (D59, D74): those that
pinned the previous value as it was at repeated timestamps, where the
training mean was the only baseline and `stages.previous_value` refused
repeated keys. The ninth pass loosens five (D59, D77): two that pinned the
holdout reaching the chosen model once, which the statement of what the
headline rests on now asks again, and three that pinned rule 2 scoring every
row when too many rows are marked (D77 point 5).''')
sub('''- `verify/reference/` is the judge's own fixture, with eighty-eight deliberate
  faults wired into it (thirty-seven the gate catches, fifty-one only the
  tests are relied on for).''', '''- `verify/reference/` is the judge's own fixture, with one hundred and
  thirteen deliberate faults wired into it (forty-four the gate catches,
  sixty-nine only the tests are relied on for).''')

DST.write_bytes(t.encode("utf-8"))
print("wrote", DST)
