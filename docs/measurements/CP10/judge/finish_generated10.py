"""The last step of the ninth pass's generated-table test: its usage text, applied to
the output of patch_generated10.py.

    python -B finish_generated10.py <generated_tables.py made by the patch> <out file>
"""
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
t = SRC.read_bytes().decode("utf-8")


def sub(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:120])
    t = t.replace(old, new)


sub('''           [--seed S] [--only i,j] [--jobs N] [--json FILE] [--keep] [--describe]''',
    '''           [--ninth N] [--seed S] [--only i,j] [--jobs N] [--json FILE] [--keep]
           [--describe]''')
sub('''verify/reference/mlprofessor. Without options: tables 0 to 999 of seed 20261122,
about 190 seconds at 24 workers on this machine (2026-11-25, with some other
work running; about 100 until the eighth pass). Before a review the lead runs
more, by hand: `--tables 10000`, or other seeds.''',
    '''verify/reference/mlprofessor. Without options: tables 0 to 999 of seed 20261122,
and the first 200 of the ninth pass's series, tables 1,000,000 to 1,000,199: about
110 seconds at 24 workers on this machine (2026-11-26, idle). Before a review the
lead runs more, by hand: `--tables 10000 --ninth 2000`, or other seeds.''')
sub('''among them dates written YYYYMMDD with a sentinel in one cell or in a third of
the rows, and a two-valued copy of the target beside few-valued columns.''',
    '''among them dates written YYYYMMDD with a sentinel in one cell or in a third of
the rows, and a two-valued copy of the target beside few-valued columns. The ninth
pass's series (D77, D78, D80) is the same recipe, each table numbered from 1,000,000
up, with columns of that pass's kinds drawn from a stream of their own, so that no
table below 1,000,000 moves: noise with repeated values (rounded to one decimal place,
or a whole number from 1 to 40), the target as a code (text or numbers, labels in
random order), two columns that give the target only together, an exact copy of a
feature, and a feature put into 10 or 40 bands.

What the headline rests on is a property of each side alone (5.2, step 12; D77,
D78, D80): every column record holds `shuffled_score`, `shuffled_skill` and
`skill_fall`, null where the column reached no model or a warning says its score
could not be computed, and their arithmetic holds; `headline.rests_on_floor` is
0.05 or the half-width of the skill interval, the larger, at least 0.05 where
there is no interval, and null with no skill; `headline.rests_on` is the columns
whose fall is at least the floor, at most five, largest first, ties in file order;
one `near_perfect_headline` record exactly when the skill is 0.98 or more; an
exact copy of a feature shares its fall; and one `holdout_far_above_quarter`
record exactly when a time-ordered holdout holds half of the rows or more. The
shuffled numbers are not compared between the sides: they are of the fitted
models.''')
DST.write_bytes(t.encode("utf-8"))
print("wrote", DST)
