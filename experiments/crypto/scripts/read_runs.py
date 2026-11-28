"""Combine the up_7d and down_7d holdout predictions and read them."""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE = sys.argv[1]
src = pd.read_csv(f"{BASE}/data/profiler/a_up_7d.csv")
full = pd.read_csv(f"{BASE}/data/built/btc_daily_full.csv")
pu = pd.read_csv(f"{BASE}/runs/a_up_7d/predictions.csv").rename(columns={"y_true": "up", "y_pred": "p_up"})
pdn = pd.read_csv(f"{BASE}/runs/a_down_7d/predictions.csv").rename(columns={"y_true": "down", "y_pred": "p_down"})

assert (pu.row.values == pdn.row.values).all(), "holdout rows differ"
h = pu.merge(pdn, on="row")
h["date"] = src.date.iloc[h.row].values
h = h.merge(full[["date"] + [c for c in full.columns if c in ("close", "z", "z_t", "fwd_ret_7", "class")]], on="date", how="left")
print("holdout rows", len(h), h.date.iloc[0], "to", h.date.iloc[-1])
print("full columns available:", [c for c in full.columns if not c.startswith(("ret_", "vol_", "oc_"))][:12])

# 1. direction or magnitude
print("\nSpearman p_up vs p_down:", round(spearmanr(h.p_up, h.p_down).statistic, 3))

# 2. three-class view
train = src.iloc[:3783]
base_up, base_dn = 0.387, 0.278          # class shares on the earliest 75% (reports/dataset.md)
h["p_range"] = (1 - h.p_up - h.p_down).clip(lower=0)
s = h[["p_up", "p_range", "p_down"]].sum(axis=1)
for c in ["p_up", "p_range", "p_down"]:
    h[c + "_n"] = h[c] / s
h["actual"] = np.select([h.up == 1, h.down == 1], ["UP", "DOWN"], "RANGE")
base = {"UP": base_up, "RANGE": 1 - base_up - base_dn, "DOWN": base_dn}
lift = pd.DataFrame({k: h[f"p_{k.lower()}_n"] / v for k, v in base.items()})
h["pred"] = lift.idxmax(axis=1)
print("\npredicted class shares:", h.pred.value_counts(normalize=True).round(3).to_dict())
print("actual class shares:   ", h.actual.value_counts(normalize=True).round(3).to_dict())
cm = pd.crosstab(h.pred, h.actual, margins=True)
print(cm)
print("hit rate (all):", round((h.pred == h.actual).mean(), 3), " chance with these shares:",
      round(sum(h.pred.value_counts(normalize=True)[k] * h.actual.value_counts(normalize=True)[k] for k in base), 3))
auc_range = roc_auc_score((h.actual == "RANGE").astype(int), h.p_range_n)
auc_up = roc_auc_score(h.up, h.p_up)
auc_dn = roc_auc_score(h.down, h.p_down)
print(f"AUC up {auc_up:.3f}  down {auc_dn:.3f}  range(from 1-p_up-p_down) {auc_range:.3f}  macro {np.mean([auc_up, auc_dn, auc_range]):.3f}")

# 3. stability by half-year
h["half"] = pd.to_datetime(h.date).dt.year.astype(str) + np.where(pd.to_datetime(h.date).dt.month <= 6, "H1", "H2")
rows = []
for k, g in h.groupby("half"):
    rows.append((k, len(g), round(g.down.mean(), 3),
                 round(roc_auc_score(g.down, g.p_down), 3) if g.down.nunique() == 2 else None,
                 round(roc_auc_score(g.up, g.p_up), 3) if g.up.nunique() == 2 else None))
print("\nby half-year: period, rows, down share, AUC down, AUC up")
for r_ in rows:
    print(r_)

# 4. deciles of p_down: realised down share
h["dec_dn"] = pd.qcut(h.p_down, 5, labels=False)
h["dec_up"] = pd.qcut(h.p_up, 5, labels=False)
print("\np_down quintile -> mean p_down, realised DOWN share, realised UP share")
print(h.groupby("dec_dn").agg(p=("p_down", "mean"), down=("down", "mean"), up=("up", "mean"), n=("row", "size")).round(3))
print("\np_up quintile -> mean p_up, realised UP share, realised DOWN share")
print(h.groupby("dec_up").agg(p=("p_up", "mean"), up=("up", "mean"), down=("down", "mean"), n=("row", "size")).round(3))

# 5. what the down model tracks, and simple benchmarks on the same holdout
feat = src.set_index(src.index).iloc[h.row]
feat.index = h.index
cors = {c: spearmanr(feat[c], h.p_down).statistic for c in src.columns[1:-1]}
top = sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:10]
print("\nfeatures most correlated with p_down (holdout):")
for c, v in top:
    print(f"  {c:<22} {v:+.3f}   holdout AUC alone for down: {roc_auc_score(h.down, feat[c]):.3f}")
cors_u = {c: spearmanr(feat[c], h.p_up).statistic for c in src.columns[1:-1]}
top_u = sorted(cors_u.items(), key=lambda kv: -abs(kv[1]))[:6]
print("features most correlated with p_up (holdout):")
for c, v in top_u:
    print(f"  {c:<22} {v:+.3f}   holdout AUC alone for up: {roc_auc_score(h.up, feat[c]):.3f}")
for c in ["zret_14", "zret_30", "past_z_7", "vol_ratio_7_30", "dd_ath", "mvrv_z365"]:
    print(f"benchmark {c:<15} holdout AUC up {roc_auc_score(h.up, feat[c]):.3f}  down {roc_auc_score(h.down, feat[c]):.3f}")

h.to_csv("holdout_combined.csv", index=False)
