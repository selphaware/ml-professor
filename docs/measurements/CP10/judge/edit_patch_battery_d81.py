r"""patch_battery10.py brought up to D81: the carrier claim reads the unperturbed run's
floor, and only a rise above it fails.

A carrier pair that ends in none of the four outcomes of D78:
  within    its headline skill is within the unperturbed run's rests-on floor of the
            unperturbed run's headline skill, either way;
  lowered   its headline skill is below the unperturbed run's by more than that floor
            (the sixth outcome, counted and printed apart);
  fails     its headline skill is above the unperturbed run's by more than that floor,
            or either headline skill has no value (and the pair is not a stated limit).
"""
from pathlib import Path

p = Path(__file__).resolve().parent / "patch_battery10.py"
t = p.read_text(encoding="utf-8")


def sub(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:100])
    t = t.replace(old, new)


sub('''        if len(silent) < len(names):
            return                  # of two planted columns, either one named counts (D80)
        within = lift is not None and isinstance(floor, (int, float)) and abs(lift) <= floor
        if within:
            for name in names:
                self.c.outcome[name] = "within"
            return
        self.require(not silent,
                     f"the headline moves {lift!r} from the run without it (floor {floor!r}), "''',
    '''        if len(silent) < len(names):
            return                  # of two planted columns, either one named counts (D80)
        # D81: the yardstick is the unperturbed run's floor, and only a rise above it fails
        if lift is not None and isinstance(base_floor, (int, float)) and lift <= base_floor:
            for name in names:
                self.c.outcome[name] = "within" if lift >= -base_floor else "lowered"
            return
        self.require(not silent,
                     f"the headline skill is {skill!r}, {lift!r} from the unperturbed run's "
                     f"{base!r}, whose rests-on floor is {base_floor!r}; "''')
sub('''    strongest = ("excluded", "warned", "named", "listed", "within", "silent")''',
    '''    strongest = ("excluded", "warned", "named", "listed", "within", "lowered", "silent")''')
sub('''          f"{tally['listed']}; the headline within the run's floor of the run without it: "
          f"{tally['within']}; none of the five: {tally['silent']}")''',
    '''          f"{tally['listed']}; the headline within the unperturbed run's floor of it: "
          f"{tally['within']}; the headline lowered by more than that floor: "
          f"{tally['lowered']}; none of these: {tally['silent']}")''')
p.write_text(t, encoding="utf-8", newline="\n")
print("edited")
