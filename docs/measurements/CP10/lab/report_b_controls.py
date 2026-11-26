"""CP10, R4 (D78 point 5): every must-not control of section B, with unseen levels not
scored, against the same tables with unseen levels given the baseline's prediction.

    python -B report_b_controls.py <b_honest under R4> <b_honest as first proposed>

Counts are excluded / warned, of 30 tables, summed over the three schemes (90), text and
numeric columns apart.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

new, old = lab10.load(sys.argv[1]), lab10.load(sys.argv[2])


def key_of(r):
    p = dict(r["params"])
    p.pop("scheme")
    kind = "numeric" if p.pop("numeric") else "text"
    return (r["make"], kind, " ".join(f"{k}={v}" for k, v in sorted(p.items())))


def tally(recs):
    out = defaultdict(Counter)
    for r in recs:
        out[key_of(r)][lab10.fate(r, "level")] += 1
    return out


a, b = tally(new), tally(old)
print("| table | the column | parameters | under R4: excluded / warned / listed / none | "
      "as first proposed: excluded / warned / listed / none |")
print("|---|---|---|---|---|")
for key in sorted(a):
    x, y = a[key], b.get(key, Counter())
    print(f"| {key[0]} | {key[1]} | {key[2]} | {x['excluded']} / {x['warned']} / {x['listed']} / "
          f"{x['silent'] + x['named']} | {y['excluded']} / {y['warned']} / {y['listed']} / "
          f"{y['silent'] + y['named']} |")
moved = sum(1 for r1, r2 in zip(new, old) if lab10.fate(r1, "level") != lab10.fate(r2, "level")
            and {lab10.fate(r1, "level"), lab10.fate(r2, "level")} & {"excluded", "warned"})
print(f"\ntables on which R4 moves the column into or out of a tier: {moved} of {len(new)}")
