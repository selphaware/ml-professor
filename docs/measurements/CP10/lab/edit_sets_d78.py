"""Add the measurement sets the D78 rulings ask for: R2 (a rescaled and a relabelled
copy beside a plain one), R3 (two to eight bands) and the candidate gate runs, old and
new."""
from pathlib import Path

HERE = Path(__file__).resolve().parent

p = HERE / "lab10.py"
t = p.read_text(encoding="utf-8")
if "def duplicated_scaled" not in t:
    old = '''def rounded_feature(column, distinct):'''
    new = '''def duplicated_scaled(column):
    """A numeric column written again on another scale: 2 x the column + 3."""
    def change(header, rows, target, rng, info):
        c = header.index(column)
        values = {}
        for i, row in enumerate(rows):
            try:
                values[i] = repr(2.0 * float(row[c]) + 3.0)
            except ValueError:
                values[i] = row[c]
        _add(header, rows, f"{column}_scaled", values)
    return change


def duplicated_relabelled(column):
    """A text column written again with every level renamed."""
    def change(header, rows, target, rng, info):
        c = header.index(column)
        _add(header, rows, f"{column}_renamed",
             {i: ("" if row[c].strip() == "" else "L_" + row[c].strip().upper())
              for i, row in enumerate(rows)})
    return change


def rounded_feature(column, distinct):'''
    assert t.count(old) == 1
    t = t.replace(old, new)
    old = '''    if kind == "roundf":                # roundf:f1:40'''
    new = '''    if kind == "dupscaled":             # dupscaled:f1
        return duplicated_scaled(rest[0])
    if kind == "duprelabel":            # duprelabel:region
        return duplicated_relabelled(rest[0])
    if kind == "roundf":                # roundf:f1:40'''
    assert t.count(old) == 1
    t = t.replace(old, new)
    p.write_text(t, encoding="utf-8", newline="\n")

p = HERE / "measure10.py"
t = p.read_text(encoding="utf-8")
if '"r2_copies"' not in t:
    old = '''    if what == "b_bands":'''
    new = '''    if what == "r2_copies":    # R2: copies as values, rescaled, relabelled
        out = []
        for run in ("signal_reg", "signal_clf", "full_signal_reg_j1", "full_signal_clf_j1"):
            for change in ("dup:f1", "dupscaled:f1", "duprelabel:region" if "reg" in run
                           else "duprelabel:seg"):
                out += [dict(run=run, k=k, change=change) for k in ks]
        for copies in (0, 1, 2):
            out += [dict(make="curved", k=k, params=dict(copies=copies, n=600),
                         args={"profile": "full"}, tag="full") for k in ks]
        return out
    if what == "r3_bands":     # R3: two to eight bands
        return [dict(run=run, k=k, change=f"bands:{q}:{kind}", tag=scheme)
                for scheme, runs in (("random", ["signal_reg", "noise_reg", "leak_reg"]),
                                     ("grouped", ["quarter_entity", "dominant_entity"]),
                                     ("time", ["random_walk", "int_stamps", "panel"]))
                for run in runs for q in (2, 3, 4, 5, 6, 7, 8)
                for kind in ("text_random", "num_random") for k in ks]
    if what == "g_cand":       # the candidate runs of the new gate checks
        out = []
        for change in ("bands:30:text_random", "bands:30:num_random",
                       "tied:levels:40+tied:round:1", "two", "dup:f1"):
            out += [dict(run="signal_reg", k=k, change=change) for k in ks]
        out += [dict(run="signal_clf", k=k, change="codes:40:text") for k in ks]
        out += [dict(make="skewed_reg", k=k, tag="random",
                     params=dict(scheme="random", sigma=1.0, cells=5, size=20.0)) for k in ks]
        out += [dict(make="last_stamp", k=k, params=dict(share=0.6)) for k in ks]
        return out
    if what == "b_bands":'''
    assert t.count(old) == 1
    t = t.replace(old, new)
    p.write_text(t, encoding="utf-8", newline="\n")
print("ok")
