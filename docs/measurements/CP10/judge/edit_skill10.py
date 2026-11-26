r"""The build skill (.claude/skills/mlprofessor-build/SKILL.md) for the ninth pass (D77,
D78, D80, D81).

    python -B edit_skill10.py <SKILL.md in> <SKILL.md out>
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


sub('''51. **The README, then the judge's two other tools.** Both must end with
    nothing failing and nothing open.
''', '''51. **The README, then the judge's two other tools.** Both must end with
    nothing failing and nothing open.

### Ninth pass, after the seventh review (tasks T69 to T76)

Four passes each taught the leak screen one more disguise of a column that
carries the target, and each review found the next; two columns that give
the target only together cannot be seen by any reading of single columns.
So the result now says what its headline rests on (D77), and the screen
gains three repairs. Five earlier cases were loosened by the tests stage
(D59): two pinned the holdout reaching the chosen model once, three pinned
rule 2 scoring every row when too many rows are marked.
`tests/test_judge_tools.py` and `tests/test_judge_tools_pass8.py` accept
exactly the failures the plan lists until the pass is built; the strict test
is `tests/test_judge_tools_pass9.py`, the last task's (D78 point 8). The
rules, in the order of the tasks:

52. **What the headline rests on.** After step 12, for each feature: the
    chosen model's score on the holdout matrix with that feature's part of it
    shuffled among the holdout rows, the mean of five shuffles, nothing
    refitted. Its skill, and the fall from the headline skill, in the
    column's record. Copies as the models receive them (the same matrix
    column within 1e-9, or its negation; the same block of level columns) are
    shuffled together. The list: falls of at least the floor, at most five,
    largest first, ties by position; the floor is max(0.05, the half-width of
    the skill interval, or of a bootstrap of the holdout rows where there is
    none). The summary prints it.
53. **The near-perfect warning.** At a headline skill of 0.98 or more, in
    either task, one warning that names the list.
54. **The leak screen reads a column by its levels.** Text columns, and
    numeric ones with five training rows for each distinct value. Unseen
    levels are not scored; the reading needs 20 rows scored, and 10 of each
    class. The message of such a column excluded names `group_col`.
55. **The order reading compares rows between neighbouring distinct values,**
    every row of one with every row of the other. Never sort equal values by
    the target.
56. **Too many rows marked as far off:** set none aside, but leave rule 2's
    1% unscored all the same, and read it against the fold's training median
    too.
57. **A time-ordered holdout of half of the rows or more is named.**
58. **The README, then the judge's two other tools.** A carrier of the target
    holds in the battery when something names it, or when the headline does
    not rise with it by more than the unperturbed run's floor (D81).
''')

sub('''- The warning tier reads every row; only the cut-off that excludes leaves the
  worst 1% unscored. The order reading warns and never excludes, and it is
  not taken of a column whose skill has no value (D36, D76).''',
    '''- The warning tier reads every row; only the cut-off that excludes leaves the
  worst 1% unscored, and it does so even when too many rows are far off to
  set any aside, against the fold's median as well as its mean (D77, D78).
  The order reading warns and never excludes, and it is not taken of a column
  whose skill has no value (D36, D76).
- Never put rows of equal value in the target's order (D77). Sorting ties by
  the target makes the target rise inside every run of equal values, and a
  column of rounded noise reads as one the target follows. Compare the rows
  of neighbouring values with each other instead.
- A column of labels is read by its levels, not by the order of their names
  (D77). The target cut into 30 bands with random labels is noise to a tree
  on sorted codes and exact to a mean per level. Do not score a level the
  fold never saw: give it the baseline and a coded copy is diluted below the
  cut-off; score it on too few rows and noise reads as perfect.
- Say what the number rests on, and do not judge it (D77, D78). A shuffled
  column's fall is a description: the tool cannot know whether a column is
  known before the outcome, and it can always say what the headline depends
  on. Compare copies on the matrix the models receive, not on the values as
  loaded: a rescaled copy is the same column to a scaled model, and shuffled
  apart each shows a quarter of the fall.
- A fall that the headline's own uncertainty could give is no statement
  (D80). Floor the list at the half-width of the skill interval: at 0.05
  alone it named a noise column on two thirds of the seeds of a noise table.''')

sub('''- Both end with a line of three counts. Only a failure outside their two fixed
  lists fails the tool; that failure is yours. Since the eighth pass nothing
  is listed as open; the battery lists four kinds of stated limit, and the
  generated tables none (spec 8.2).''',
    '''- Both end with a line of three counts. Only a failure outside their two fixed
  lists fails the tool; that failure is yours. Since the eighth pass nothing
  is listed as open; the battery lists the four kinds of stated limit of D76
  and one pair of D36, and the generated tables none (spec 8.2). The battery
  also prints six counts for the pairs that plant a carrier of the target:
  excluded, warned, named by the near-perfect warning, named in the list,
  within the unperturbed run's floor, and lowered (D81).''')
sub('''  the eighth pass). For the eighth pass that file accepts exactly the failures the plan
  lists, and the strict test of both tools belongs to the pass's last task.''',
    '''  the eighth pass). For the ninth pass that file and
  `tests/test_judge_tools_pass8.py` accept exactly the failures the plan
  lists, and the strict test of both tools, `tests/test_judge_tools_pass9.py`,
  belongs to the pass's last task (D78 point 8).''')

# a few rows of the table of common failures
sub('''| A real integer measure (price, volume) is excluded as id-like |''',
    '''| A coded copy of the target (bands with labels in random order) is kept | The screen read a categorical column of a regression by a tree on codes of its sorted level names: random labels make it noise. | Read it by its levels, the mean target of each level in the fold's training rows, unseen levels not scored (D77, D78). |
| A column of rounded noise, or a score from 1 to 40, gets `strong_single_column` | The order reading sorted rows of equal value by the target, so the target rose inside each run of ties. | Compare every row of one value with every row of the next value; never order ties by the target (D77). |
| A copy of a long-tailed target with a few cells changed is kept | More than 1% of the rows were marked as far off, none could be set aside, and rule 2 then scored every row: the few changed cells were most of the copy's error. | Leave rule 2's 1% unscored all the same, and read it against the fold's training median too (D77, D78). |
| `near_perfect_headline` on a table that is genuinely easy | Not a fault: the warning is raised at 0.98 whatever the table. | Nothing to fix; it names what the headline rests on, and the caller decides (D77). |
| A copy of a feature shows a quarter of the fall the feature shows alone | The copies were found by their loaded values, so a rescaled copy was shuffled on its own. | Find copies on the final fit's matrix, up to sign (D78, D80). |
| A real integer measure (price, volume) is excluded as id-like |''')

DST.write_bytes(t.encode("utf-8"))
print("wrote", DST)
