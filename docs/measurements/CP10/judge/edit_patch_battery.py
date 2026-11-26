"""Three corrections to patch_battery10.py, found on its first run: each was a fault of
the pair as the prototype made it, not of the package."""
from pathlib import Path

p = Path(__file__).resolve().parent / "patch_battery10.py"
t = p.read_text(encoding="utf-8")


def sub(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:90])
    t = t.replace(old, new)


# 1. two added columns make every row differ: on `dup_noise` that is the stated limit of
#    `noise_added` (D60) again, and on `few_valued_leak` it has its own words there
sub('''def ch_two_together(case, rng):
    if case.info.get("malformed"):
        return "the table cannot be read, so no column of it is judged"
''', '''def ch_two_together(case, rng):
    if case.info.get("malformed"):
        return "the table cannot be read, so no column of it is judged"
    if _left_to_other_pairs(case):
        return _left_to_other_pairs(case)
''')
# 2. for a class the noise has a spread of one half whatever the share of the rarer
#    class, so that `pair_a` alone never separates the classes
sub('''    mean = sum(values.values()) / len(values)
    spread = math.sqrt(sum((v - mean) ** 2 for v in values.values()) / len(values)) or 1.0
    a, b = [""] * len(case.rows), [""] * len(case.rows)''',
    '''    mean = sum(values.values()) / len(values)
    spread = 0.5 if classes else \\
        math.sqrt(sum((v - mean) ** 2 for v in values.values()) / len(values)) or 1.0
    a, b = [""] * len(case.rows), [""] * len(case.rows)''')
# 3. a feature that holds one cell far outside the rest has one band for all its other
#    values: nothing is left of it
sub('''        name = case.info.get("num")
        if not name or case.info.get("malformed"):
            return "no numeric feature"
        at = case.col(name)
        held = {}''', '''        name = case.info.get("num")
        if not name or case.info.get("malformed"):
            return "no numeric feature"
        if case.info.get("cell") == "feature":
            return (f"`{name}` holds one cell far outside its other values: bands of equal "
                    "width would put every other value into one")
        at = case.col(name)
        held = {}''')
p.write_text(t, encoding="utf-8", newline="\n")
print("edited", p)
