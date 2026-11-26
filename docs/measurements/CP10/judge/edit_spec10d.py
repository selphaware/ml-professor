"""spec.md for the ninth pass, part 4: the counts of section 8 and 8.1, and the paragraph
on the variants of the ninth pass. Applied to the spec in place.

    python -B edit_spec10d.py <spec.md>
"""
import sys
from pathlib import Path

P = Path(sys.argv[1])
t = P.read_bytes().decode("utf-8")


def sub(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:100])
    t = t.replace(old, new)


sub('''that proves it and fails if one pair is missing: 51 pairs of a check and the
mode it names, over 37 broken copies, among 53 checks.''',
    '''that proves it and fails if one pair is missing: 58 pairs of a check and the
mode it names, over 44 broken copies, among 60 checks.''')
sub('''the table: until it has, the names for M48 to M50 are those the plan gives.
''', '''the table: until it has, the names for M48 to M50 are those the plan gives.

M52 to M69 were added with the ninth pass (D77, D78, D80). Each is the
plausible half-built form of one new rule: a stream that is not the
statement's own (M52); a statement for a column that never reached a model
(M53); a list with no floor (M54), the wrong order among ties (M55), the
floor as first proposed (M67) or with no stand-in (M68); copies found by
their values (M56) or not up to sign (M69); the warning for one task only
(M57); an unseen level scored as CP10 first worded it (M58), a reading on too
few rows (M59), "few" counted over the wrong rows (M60) and a message that
leaves out its advice (M61); the order reading by the form measured and not
adopted (M62); rule 2 with half of its repair (M63); the cut-off on the wrong
side (M64); a summary that says nothing of an empty list (M65); and a run
that stops where it must go on (M66). The seven gate forms of this pass each
fail tests of the same files: F29 and F31 `test_rests_on.py`, F30
`test_near_perfect.py`, F32 `test_screen_levels.py`, F33 `test_order_ties.py`,
F34 `test_screen_marked.py`, F35 `test_large_holdout.py`. The tests of this
pass are written by its tests stage, and the mutation check then proves each
row of the table, the names being those the plan gives (each was proved on a
prototype of the test in the harness session).
''')
sub('''through the suite in the same way, and each must fail at least one test: 88
variants in all, the 37 of the gate and M1 to M51.''',
    '''through the suite in the same way, and each must fail at least one test: 113
variants in all, the 44 of the gate and M1 to M69.''')
P.write_bytes(t.encode("utf-8"))
print("edited", P)
