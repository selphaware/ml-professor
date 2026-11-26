"""edit_make_proto5.py: for a classification the floor of R4 also counts the scored rows
of each class, and the log says how many the rarer class has."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = HERE / "edit_make_proto5.py"
t = p.read_text(encoding="utf-8")
old = r'''                raw = _auc(truth[seen], by_level[seen]) if seen.any() else None
                third = raw if int(seen.sum()) >= P.B_MIN else None
                if log is not None:
                    log["levels_raw"] = (None if raw is None else 2 * raw - 1, int(seen.sum()),
                                         int(len(seen)))'''
new = r'''                raw = _auc(truth[seen], by_level[seen]) if seen.any() else None
                rarer = int(min(truth[seen].sum(), seen.sum() - truth[seen].sum()))
                third = raw if int(seen.sum()) >= P.B_MIN and rarer >= P.B_MIN_CLASS else None
                if log is not None:
                    log["levels_raw"] = (None if raw is None else 2 * raw - 1, int(seen.sum()),
                                         int(len(seen)), rarer)'''
assert t.count(old) == 1
p.write_text(t.replace(old, new), encoding="utf-8", newline="\n")
p = HERE / "_proto.py"
s = p.read_text(encoding="utf-8")
if "B_MIN_CLASS" not in s:
    s = s.replace('B_MIN = int(_get("MLP_B_MIN", "20"))',
                  'B_MIN = int(_get("MLP_B_MIN", "20"))\n'
                  '# ... and, for a classification, the fewest of them of each class\n'
                  'B_MIN_CLASS = int(_get("MLP_B_MIN_CLASS", "10"))')
    assert "B_MIN_CLASS" in s
    p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
