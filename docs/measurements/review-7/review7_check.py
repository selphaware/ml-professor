"""Review 7, findings 1 and 2: the lead's reproduction on the pass 8 build, full profile, real public tables."""
import sys, warnings; sys.path.insert(0, "/home/claude/realdata/app9")
import numpy as np, pandas as pd, mlprofessor
from sklearn.datasets import load_diabetes
warnings.simplefilter("ignore")
NO = dict(time_col=False, group_col=False, profile="full", n_jobs=2)
def go(d, y="target"):
    r = mlprofessor.profile(d, y, **NO).to_dict()
    ex = sorted(c["name"] for c in r["columns"] if c["reason"] == "suspected_leak")
    ws = sorted({(w["code"], w["column"]) for w in r["warnings"] if w["code"] in ("strong_single_column", "suspected_leak")})
    return f"skill {r['headline']['skill']:.3f} ({r['headline']['model']}), excluded {ex}, warned {ws}"
db = load_diabetes(as_frame=True).frame
mpg = pd.read_csv("data/mpg.csv").drop(columns=["name"]).dropna().rename(columns={"mpg": "target"})
for name, d in (("diabetes", db), ("mpg", mpg)):
    print(name, "clean:", go(d))
    for seed in (1, 2, 3):
        g = np.random.default_rng(seed); e = d.copy()
        band = pd.qcut(e["target"], 24, labels=False, duplicates="drop")
        names = np.array([f"tier_{i:02d}" for i in g.permutation(int(band.max()) + 1)])
        e["tier"] = names[band.to_numpy()]
        print(f"  + 24 bands of the target, labels in random order, seed {seed}:", go(e))
    e = d.copy(); z = np.random.default_rng(9).normal(0, e["target"].std(), len(e)); e["a"] = e["target"] + z; e["b"] = z
    print("  + two columns that give the target together (a = y + z, b = z):", go(e))
