"""Write the ninth pass's `_variant.py`: the reference's own, with the new variants.

    python -B make_variant10.py <reference folder as it stood> <out folder>
"""
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
text = (SRC / "_variant.py").read_bytes().decode("utf-8")


def sub(old, new):
    global text
    assert text.count(old) == 1, old[:80]
    text = text.replace(old, new)


sub('''Eighty-eight faults in all: thirty-seven the gate's self-test proves a check
catches (F1 to F28, with the forms F6b, F6c, F6d, F7b, F7c, F7d, F8b, F12b and
F14b), and fifty-one that only the test suite is relied on for (M1 to M51).''',
    '''One hundred and thirteen faults in all: forty-four the gate's self-test proves a
check catches (F1 to F35, with the forms F6b, F6c, F6d, F7b, F7c, F7d, F8b, F12b and
F14b), and sixty-nine that only the test suite is relied on for (M1 to M69).''')
sub('''    F28  under the grouped scheme the choice between configurations is made
         however few rows the folds outside one large unit validate on
''', '''    F28  under the grouped scheme the choice between configurations is made
         however few rows the folds outside one large unit validate on
"F29" .. "F35" the failure modes of the ninth pass (D77, D78, D80), each with a gate
              run and a check that names it.
    F29  the statement of what the headline rests on is not of a shuffle of the
         holdout: each column's holdout values are left in place, so every fall
         is zero and the list names nothing
    F30  a headline skill of 0.98 or more carries no `near_perfect_headline`
         warning
    F31  columns that are copies of one another are shuffled one at a time:
         each shows a part of the fall they share
    F32  the leak screen does not read a column by its levels: a coded copy of
         the target, its labels in random order, is kept
    F33  the order reading puts rows of equal column value in the target's
         order, as D76 built it: a column of noise with repeated values is
         warned about
    F34  where more training rows are marked than may be set aside, rule 2
         scores every row, as D74 had it: a copy of the target with five cells
         changed by a large amount is kept
    F35  under time order a holdout of half of the rows or more is not named
''')
sub('''    M51 the panel warning names the training mean whatever baseline the run
        used (D74)
"""''', '''    M51 the panel warning names the training mean whatever baseline the run
        used (D74)
"M52" .. "M69" small faults of the ninth pass (D77, D78, D80).
    M52 the shuffles of the statement are drawn from a stream that depends on
        the number of cores
    M53 a column that reached no model is given a statement as well
    M54 the list takes the largest falls whatever their size: it has no floor
    M55 among equal falls the list goes by name, not by file order
    M56 copies are columns whose values are the same as loaded: a rescaled
        copy, the same column as the models receive it, is shuffled alone
    M57 the near-perfect warning is raised in a classification only: a
        regression's near-perfect headline carries none
    M58 in the reading by levels a level that a fold's training rows do not
        hold is given the baseline's prediction and scored (CP10 as first worded)
    M59 the reading by levels has a value on any number of scored rows
    M60 whether a numeric column is read by its levels is counted over every
        row split, not over the training rows
    M61 the `suspected_leak` message of a column read by its levels does not
        say to declare it as `group_col`
    M62 the order reading takes one point for each distinct value, the mean
        target of its rows, in place of every pair of rows
    M63 where more training rows are marked than may be set aside, rule 2
        leaves its rows unscored and is not read against the fold's training
        median
    M64 only a time-ordered holdout above one half of the rows is named, not
        one of one half
    M65 the console summary prints nothing for the list when it is empty
    M66 a model that cannot predict a shuffled matrix stops the run
    M67 the floor of the list is 0.05 alone, whatever the headline's
        uncertainty (D80 point 1)
    M68 where the headline skill has no interval, the floor of the list is
        0.05 alone: no stand-in (D80 point 1)
    M69 a negated copy of a column is not a copy of it, and is shuffled alone
        (D80 point 4)
"""''')
DST.mkdir(parents=True, exist_ok=True)
(DST / "_variant.py").write_bytes(text.encode("utf-8"))
print("wrote", DST / "_variant.py")
