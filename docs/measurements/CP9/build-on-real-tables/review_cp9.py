"""Lead's check of the pass 8 build (application and reference side by side): two things the real tables showed."""
import sys, subprocess, json, warnings, os
import numpy as np, pandas as pd
which = sys.argv[1]; sys.path.insert(0, f"/home/claude/realdata/{which}")
warnings.simplefilter("ignore")
import mlprofessor
from mlprofessor import errors
from cases import lagged
from sklearn.datasets import load_diabetes
D = "/home/claude/realdata/data/"; NO = dict(time_col=False, group_col=False)
def go(d, y, **kw):
    try:
        r = mlprofessor.profile(d, y, profile="fast", n_jobs=2, **kw).to_dict()
        ex = sorted(c["name"] for c in r["columns"] if c["reason"] == "suspected_leak")
        ws = sorted(w["column"] for w in r["warnings"] if w["code"] == "strong_single_column")
        return dict(skill=round(r["headline"]["skill"], 3), excluded=ex, strong=ws)
    except errors.ProfileError as e: return dict(error=e.code)
out = {}
# 1. the order reading on columns with repeated values
out["diabetes clean"] = go(load_diabetes(as_frame=True).frame, "target", **NO)
for seed in (1, 2, 3):
    g = np.random.default_rng(seed); n = 1200
    d = pd.DataFrame({"y": g.normal(size=n), "cont": g.normal(size=n), "rounded_1dp": g.normal(size=n).round(1), "score_1_to_40": g.integers(1, 41, n), "score_1_to_10": g.integers(1, 11, n).astype(float)})
    out[f"pure noise, tied columns, seed {seed}"] = go(d, "y", **NO)
out["diamonds"] = go(D + "diamonds.csv", "price", **NO)
out["mpg"] = go(pd.read_csv(D + "mpg.csv"), "mpg", **NO)
tx = pd.read_csv(D + "taxis.csv"); tx = tx[tx["payment"].notna()].drop(columns=["pickup", "dropoff", "total"])
out["taxis tip"] = go(tx, "tip", **NO)
# 2. a copy of the target with 5 cells changed, under time order
vix = pd.read_csv(D + "vix.csv"); vix.columns = [c.lower() for c in vix.columns]
base = lagged(vix, ["close", "high", "low"], [1, 2, 5]); lv = base[["date", "close"] + [c for c in base.columns if "_lag" in c]].dropna().reset_index(drop=True)
for label, bump in (("+7.0", lambda v, g: v + 7.0), ("x1.5", lambda v, g: v * 1.5), ("another row's value", None), ("+0.5", lambda v, g: v + 0.5)):
    res = []
    for seed in range(8):
        g = np.random.default_rng(100 + seed); c = lv.copy(); c["copy"] = c["close"].to_numpy()
        idx = g.choice(np.arange(len(c) - 2000, len(c) - 500), 5, replace=False)   # all five among the training rows the cap keeps
        c.loc[idx, "copy"] = (c["close"].to_numpy()[g.choice(len(c), 5)] if bump is None else bump(c.loc[idx, "copy"].to_numpy(), g))
        r = go(c, "close", time_col="date", group_col=False)
        res.append("excluded" if "copy" in r.get("excluded", []) else ("warned %.2f" % r["skill"] if "copy" in r.get("strong", []) else "SILENT %.2f" % r["skill"]))
    out[f"VIX + copy, 5 training cells changed ({label})"] = res
json.dump(out, sys.stdout, indent=1)
