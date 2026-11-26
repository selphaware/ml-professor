"""edit_make_proto5.py: the floor of R4 for a classification, as measured: at least
`B_MIN` rows scored, and of each class at least `B_MIN_CLASS` scored rows, or every
validation row of that class scored (so a class of fewer than ten validation rows is
read as it was before D78 when none of its rows is left unscored)."""
from pathlib import Path

p = Path(__file__).resolve().parent / "edit_make_proto5.py"
t = p.read_text(encoding="utf-8")
old = r'''                rarer = int(min(truth[seen].sum(), seen.sum() - truth[seen].sum()))
                third = raw if int(seen.sum()) >= P.B_MIN and rarer >= P.B_MIN_CLASS else None'''
new = r'''                rarer = int(min(truth[seen].sum(), seen.sum() - truth[seen].sum()))
                enough = int(seen.sum()) >= P.B_MIN
                for one in (0, 1):
                    scored, held = int((truth[seen] == one).sum()), int((truth == one).sum())
                    enough = enough and (scored >= P.B_MIN_CLASS or scored == held)
                third = raw if enough else None'''
assert t.count(old) == 1
p.write_text(t.replace(old, new), encoding="utf-8", newline="\n")
print("ok")
