"""Review 6, findings 1 and 2: the lead's reproduction on the pass 6 build (commit e917ba8), fast profile."""
import sys, warnings; sys.path.insert(0, "/home/claude/realdata/app7")
import numpy as np, pandas as pd, mlprofessor
from mlprofessor import errors
warnings.simplefilter("ignore")
NO = dict(time_col=False, group_col=False, profile="fast", n_jobs=2)
def go(d, y="y", **kw):
    try:
        r = mlprofessor.profile(d, y, **{**NO, **kw}).to_dict()
        ex = {c["name"]: c["reason"] for c in r["columns"] if c["role"] == "excluded"}
        return f"skill {r['headline']['skill']:+.3f}, excluded {ex}, warnings {sorted({w['code'] for w in r['warnings']})}"
    except errors.ProfileError as e: return f"STOPS {e.code}"
print("FINDING 1: a day column written YYYYMMDD, three rows a day, the label random per block of five days")
for seed in (0, 1, 2):
    g = np.random.default_rng(seed); days = pd.bdate_range("2021-01-04", periods=600)
    d = pd.DataFrame({"day": np.repeat(days.strftime("%Y%m%d").astype(int), 3)})
    block = np.repeat(np.arange(120), 15); d["y"] = g.integers(0, 2, 120)[block]; d["x1"] = g.normal(size=len(d)); d["x2"] = g.normal(size=len(d))
    print(f"  seed {seed}  clean:               {go(d)}")
    for name, v in (("one cell 99991231", 99991231), ("one cell 0", 0)):
        e = d.copy(); e.loc[7, "day"] = v; print(f"  seed {seed}  {name:19s}: {go(e)}")
    e = d.copy(); e.loc[g.choice(len(e), 90, replace=False), "day"] = 99991231; print(f"  seed {seed}  5% of cells 99991231: {go(e)}")
print("FINDING 2: noise in 16 feature combinations, plus a `status` column that copies the target, duplicates='group'")
for seed in (0, 1, 2):
    g = np.random.default_rng(seed); n = 2000
    d = pd.DataFrame({f"c{i}": g.integers(0, k, n).astype(str) for i, k in enumerate((2, 2, 4))}); d["y"] = g.integers(0, 2, n)
    e = d.copy(); e["status"] = np.where(e["y"] == 1, "closed", "open")
    print(f"  seed {seed}  without status: {go(d, duplicates='group')}")
    print(f"  seed {seed}  with status:    {go(e, duplicates='group')}")
