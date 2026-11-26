"""The lab for D80: the floor in each record, a negated copy, and tables whose headline
has no interval (few holdout entities, few holdout timestamps), noise and carriers."""
from pathlib import Path

HERE = Path(__file__).resolve().parent

p = HERE / "lab10.py"
t = p.read_text(encoding="utf-8")
if "rests_on_floor" not in t:
    t = t.replace('''        rests_on=h.get("rests_on"),''', '''        rests_on=h.get("rests_on"),
        floor=h.get("rests_on_floor"),''')
    t = t.replace('''def duplicated_relabelled(column):''', '''def duplicated_negated(column):
    """A numeric column written again with its sign changed."""
    def change(header, rows, target, rng, info):
        c = header.index(column)
        values = {}
        for i, row in enumerate(rows):
            try:
                values[i] = repr(-float(row[c]))
            except ValueError:
                values[i] = row[c]
        _add(header, rows, f"{column}_negated", values)
    return change


def duplicated_relabelled(column):''')
    t = t.replace('''    if kind == "duprelabel":            # duprelabel:region''', '''    if kind == "dupneg":                # dupneg:f1
        return duplicated_negated(rest[0])
    if kind == "duprelabel":            # duprelabel:region''')
    assert "rests_on_floor" in t and "duplicated_negated(rest[0])" in t
    p.write_text(t, encoding="utf-8", newline="\n")

p = HERE / "tables10.py"
t = p.read_text(encoding="utf-8")
if "def no_interval" not in t:
    t += '''

def no_interval(seed, kind="grouped", classification=False, signal=False, carrier="none"):
    """A table whose headline has no interval: under the grouped scheme 20 entities of 25
    rows (about five in the holdout, fewer than 30), or under time order 100 timestamps
    of 5 rows (25 holdout timestamps, fewer than 30). Two features that carry the target
    when `signal`, two of noise; and a carrier of the target when asked: `code` (30
    equal-count bands as text, labels in random order; for a class, a code of 40 levels
    each of one class) or `two` (the target plus a noise, and that noise). Tag 11400."""
    rng = np.random.default_rng([11400, seed, ["grouped", "stamps"].index(kind),
                                 int(classification), int(signal),
                                 ["none", "code", "two"].index(carrier)])
    n = 500
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    z = (1.6 * f1 - 1.2 * f2) if signal else np.zeros(n)
    if kind == "grouped":
        who = np.repeat(np.arange(20), 25)
        z = z + (0.0 if signal else 0.0)
        y = z + rng.normal(size=n) + rng.normal(size=20)[who]
    else:
        who = None
        y = z + rng.normal(size=n)
    if classification:
        y = (y > np.median(y)).astype(int)
    frame = pd.DataFrame({"f1": f1, "f2": f2, "n1": rng.normal(size=n), "n2": rng.normal(size=n)})
    if carrier == "code":
        if classification:
            labels, half = rng.permutation(40), 20
            code = np.where(y == 1, labels[half + rng.integers(0, 20, n)], labels[rng.integers(0, 20, n)])
        else:
            ranks = np.argsort(np.argsort(y, kind="stable"), kind="stable")
            code = rng.permutation(30)[np.minimum(29, ranks * 30 // n)]
        frame["target_code"] = np.array([f"code_{int(v):03d}" for v in code])
    elif carrier == "two":
        noise = rng.normal(0, float(np.std(y)) or 1.0, size=n)
        frame["pair_a"], frame["pair_b"] = y + noise, noise
    frame["y"] = np.where(y == 1, "yes", "no") if classification else y
    if kind == "grouped":
        frame.insert(0, "who", np.array([f"E{v:02d}" for v in who])[rng.permutation(n)]
                     if False else np.array([f"E{v:02d}" for v in who]))
        return frame, "y", dict(time_col=False, group_col="who"), {}
    frame.insert(0, "t", np.repeat(np.arange(100), 5))
    return frame, "y", dict(time_col="t", group_col=False), {}
'''
    p.write_text(t, encoding="utf-8", newline="\n")

p = HERE / "measure10.py"
t = p.read_text(encoding="utf-8")
if '"d80_noint"' not in t:
    old = '''    if what == "b_bands":'''
    new = '''    if what == "d80_noint":
        return [dict(make="no_interval", k=k, tag=kind,
                     params=dict(kind=kind, classification=clf, signal=signal, carrier=carrier))
                for kind in ("grouped", "stamps") for clf in (False, True)
                for signal in (False, True) for carrier in ("none", "code", "two") for k in ks]
    if what == "d80_copies":
        out = []
        for run in ("signal_reg", "signal_clf", "full_signal_reg_j1", "full_signal_clf_j1"):
            for change in ("dup:f1", "dupscaled:f1", "dupneg:f1"):
                out += [dict(run=run, k=k, change=change) for k in ks]
            out += [dict(run=run, k=k) for k in ks]
        return out
    if what == "b_bands":'''
    assert t.count(old) == 1
    p.write_text(t.replace(old, new), encoding="utf-8", newline="\n")
print("ok")
