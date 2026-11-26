r"""The last step of the ninth pass's battery: applied to the output of patch_battery10.py.
It mends four lines the patch joined, brings the words up to D80 and D81, and leaves out
the six gate runs of the ninth pass, each with its reason.

    python -B finish_battery10.py <battery.py made by patch_battery10.py> <out battery.py>
"""
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
t = SRC.read_bytes().decode("utf-8")


def sub(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:120])
    t = t.replace(old, new)


# ---- lines the patch joined (a backslash and a line end inside its strings)
sub('''    spread = 0.5 if classes else         math.sqrt(sum((v - mean) ** 2 for v in values.values()) / len(values)) or 1.0''',
    '''    spread = 0.5 if classes else \\
        math.sqrt(sum((v - mean) ** 2 for v in values.values()) / len(values)) or 1.0''')
sub('''        base = self.base["doc"]["headline"].get("skill")             if self.base and self.base.get("finished") else None''',
    '''        finished = bool(self.base and self.base.get("finished"))
        base = self.base["doc"]["headline"].get("skill") if finished else None''')
sub('''        lift = skill - base if isinstance(skill, (int, float))             and isinstance(base, (int, float)) else None''',
    '''        lift = skill - base if isinstance(skill, (int, float)) \\
            and isinstance(base, (int, float)) else None''')
sub('''        base_floor = self.base["doc"]["headline"].get("rests_on_floor")             if self.base and self.base.get("finished") else None''',
    '''        base_floor = self.base["doc"]["headline"].get("rests_on_floor") if finished else None''')

# ---- the words
sub('''**Nothing is open.** D73 gave every fault of the first run to pass 8, and the
rules of D74 and D76 make each of those pairs hold: OPEN is empty, and the
last lines still give its count, 0. The stated limits are four kinds, six
pairs (D76 point 6): a copy that differs by an added column (D60), a
whole-number time column with an empty cell (D55), a distinct whole-number
measure in a table sorted by it (D31), and under time order a copy of the
target with noise above 1% of the target's spread (D76). **A pair asserts
what the spec intends, never an outcome known to be wrong:** each of the six
is asserted as it should come out, fails, and is counted as a stated limit,
not as holding.''', '''**Nothing is open.** D73 gave every fault of the first run to pass 8, and the
rules of D74 and D76 make each of those pairs hold: OPEN is empty, and the
last lines still give its count, 0. The stated limits are the four kinds of
D76, six pairs: a copy that differs by an added column (D60), a whole-number
time column with an empty cell (D55), a distinct whole-number measure in a
table sorted by it (D31), and under time order a copy of the target with
noise above 1% of the target's spread (D76); and one pair where the headline
skill has no value, so that nothing can be said of what a carrier does to it
(D36, D80 point 2). **A pair asserts what the spec intends, never an outcome
known to be wrong:** each of the seven is asserted as it should come out,
fails, and is counted as a stated limit, not as holding.

**A pair that plants a carrier of the target** (a coded copy, two columns
that give the target only together, a copy with cells altered, and the
others: D77 point 9, D78, D80, D81) holds when each planted column ends in
one of four outcomes, or the headline does not rise: the column is excluded
as a suspected leak; it is named by a `strong_single_column` warning; the
headline skill is 0.98 or more and the `near_perfect_headline` warning names
it; or `headline.rests_on` names it. Of two planted columns, either one named
counts. Where none of the four holds, the pair holds when its headline skill
is no more than the unperturbed run's `headline.rests_on_floor` above the
unperturbed run's headline skill (D81): `within` that floor of it either way,
or `lowered` by more than the floor. The last lines count each of the six
outcomes, and the pairs that end in none.''')
sub('''# (run, change) -> the decision that states the limit. The pair is asserted as the
# spec's intent has it, fails, and is reported as a stated limit. Exactly four
# kinds, each with its decision (D76 point 6), and no other.''',
    '''# (run, change) -> the decision that states the limit. The pair is asserted as the
# spec's intent has it, fails, and is reported as a stated limit. Exactly the four
# kinds of D76 (point 6), each with its decision, and D36 where the headline skill has
# no value (D80 point 2, D81), and no other.''')
sub('''def _carrier(case, names, what):
    """The change plants a carrier of the target. It holds when each planted column is
    excluded as a suspected leak, or warned about by name, or the headline is at or
    above 0.98 and the `near_perfect_headline` warning names it (D77 point 9)."""''',
    '''def _carrier(case, names, what):
    """The change plants a carrier of the target. It holds when a planted column is
    excluded as a suspected leak, or warned about by name, or the headline is at or
    above 0.98 and the `near_perfect_headline` warning names it, or `headline.rests_on`
    names it (D77 point 9, D78 point 2); or, where none of these holds, when the
    headline skill is no more than the unperturbed run's rests-on floor above the
    unperturbed run's (D80 point 2, D81)."""''')
sub('''    case.must = ("each of " + ", ".join(f"`{n}`" for n in names) + " is excluded as a suspected "
                 "leak, or named by a `strong_single_column` warning, or named by the "
                 "`near_perfect_headline` warning, or named in `headline.rests_on`: " + what)''',
    '''    case.must = ("one of " + ", ".join(f"`{n}`" for n in names) + " is excluded as a suspected "
                 "leak, or named by a `strong_single_column` warning, or named by the "
                 "`near_perfect_headline` warning, or named in `headline.rests_on`; or the "
                 "headline skill is no more than the unperturbed run's rests-on floor above "
                 "the unperturbed run's (D81): " + what)''')

# ---- the six gate runs of the ninth pass
sub('''    "full_signal_reg_j4": "full profile; about the core count, not about the table",
}''', '''    "full_signal_reg_j4": "full profile; about the core count, not about the table",
    # The six runs of the ninth pass (D77, D78, D80, D81). Each table is a gate table with
    # one of the battery's own changes of that pass already made, or a table whose
    # conclusion no change of the battery bears on.
    "pair_together": "`signal_reg` with two columns that give the target only together: "
                     "the pair `signal_reg:two_columns_together`, and that change on every "
                     "other run, carry its conclusion",
    "signal_reg_copied": "`signal_reg` with `f1` written again on another scale: the pair "
                         "`signal_reg:feature_written_twice`, and that change on every other "
                         "run, carry its conclusion",
    "leak_reg_coded": "`signal_reg` with the target as a code of 30 levels: the pairs "
                      "`signal_reg:coded_copy_text` and `:coded_copy_number`, and those "
                      "changes on every other run, carry its conclusion",
    "noise_tied": "`signal_reg` with two columns of noise whose values repeat: the pairs "
                  "`signal_reg:noise_rounded_1dp` and `:noise_score_1_to_40`, and those "
                  "changes on every other run, carry its conclusion",
    "skewed_copy": "a copy of a long-tailed target with five cells moved: the pairs "
                   "`copy_5_cells_*` on every run carry the conclusion, at several sizes of "
                   "change; the gate check reads the long tail itself",
    "last_stamp": "its conclusion is the size of a holdout of whole timestamps, which no "
                  "change of the battery moves: a change that adds or alters a column leaves "
                  "the timestamps as they are",
}''')
DST.write_bytes(t.encode("utf-8"))
print("wrote", DST)
