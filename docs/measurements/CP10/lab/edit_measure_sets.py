"""One edit of measure10.py: two more sets."""
from pathlib import Path

p = Path(__file__).resolve().parent / "measure10.py"
t = p.read_text(encoding="utf-8")
old = '''    if what == "b_bands":'''
new = '''    if what == "b_few":     # numeric codes of many levels: where "few" ends
        out = []
        for run in ("signal_reg", "noise_reg"):
            out += [dict(run=run, k=k, change=f"bands:{q}:num_random", tag="bands")
                    for q in (100, 150, 225, 300, 450) for k in ks]
        for run in ("signal_clf", "noise_clf"):
            out += [dict(run=run, k=k, change=f"codes:{q}:num", tag="codes")
                    for q in (100, 150, 224, 300, 450) for k in ks]
        for levels in (100, 150, 225, 300, 450):
            for r2 in (0.9, 0.99):
                out += [dict(make="honest_levels", k=k, tag="honest",
                             params=dict(scheme="random", levels=levels, r2=r2, numeric=True))
                        for k in ks]
            out += [dict(make="noise_levels", k=k, tag="noise",
                         params=dict(scheme="random", levels=levels, numeric=True)) for k in ks]
        return out
    if what == "a_two_full":
        return [dict(run=run, k=k, change="two", args={"profile": "full"})
                for run in ("random_walk", "int_stamps", "panel") for k in ks]
    if what == "b_bands":'''
assert t.count(old) == 1
p.write_text(t.replace(old, new), encoding="utf-8", newline="\n")
print("edited", p)
