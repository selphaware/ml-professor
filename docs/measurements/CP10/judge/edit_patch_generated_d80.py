"""patch_generated10.py brought up to D80 point 1: the list's floor."""
from pathlib import Path

p = Path(__file__).resolve().parent / "patch_generated10.py"
t = p.read_text(encoding="utf-8")


def sub(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:100])
    t = t.replace(old, new)


sub('''        want_list = [(name, f) for _, _, name, f in sorted(falls9) if f >= REST_FALL][:REST_MOST]''',
    '''        floor9 = at(doc, "headline.rests_on_floor")
        low9, high9 = at(doc, "headline.skill_low"), at(doc, "headline.skill_high")
        if skill9 is None:
            if floor9 is not None:
                bad("with no headline skill the floor of the list is null (D80 point 1)",
                    repr(floor9))
        elif is_number(low9) and is_number(high9):
            if not (is_number(floor9) and close(floor9, max(REST_FALL, (high9 - low9) / 2), 1e-9)):
                bad("the floor of the list is 0.05 or the half-width of the headline skill's "
                    "interval, the larger (D80 point 1)",
                    f"{floor9!r}; the interval {low9!r} to {high9!r}")
        elif not (is_number(floor9) and floor9 >= REST_FALL):
            bad("where the headline skill has no interval the floor of the list is a number of "
                "at least 0.05 (D80 point 1)", repr(floor9))
        cut9 = floor9 if is_number(floor9) else REST_FALL
        want_list = [(name, f) for _, _, name, f in sorted(falls9) if f >= cut9][:REST_MOST]''')
sub('''            bad("headline.rests_on lists the columns whose fall is at least 0.05, at most five, "
                "largest first, ties in file order (D78 point 1)",''',
    '''            bad("headline.rests_on lists the columns whose fall is at least the floor, at most "
                "five, largest first, ties in file order (D78 point 1, D80 point 1)",''')
p.write_text(t, encoding="utf-8", newline="\n")
print("edited")
