"""Table B against table A cut to B's dates: same holdout, paired comparison."""
import json
import math
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE = sys.argv[1]
R = f"{BASE}/runs"
P = f"{BASE}/data/profiler"
rng = np.random.default_rng(0)


def load(run):
    return pd.read_csv(f"{R}/{run}/predictions.csv")


def cv_info(run):
    p = json.load(open(f"{R}/{run}/profile.json"))
    ch = [m for m in p["models"] if m["chosen"]][0]
    return ch["config"], round(ch["cv_mean"], 3), [round(x, 3) for x in ch["fold_scores"]], p["validation"]["holdout_start"][:10], p["validation"]["holdout_end"][:10]


def block_boot_diff(y, s1, s2, n_boot=2000):
    n = len(y)
    L = math.ceil(n ** (1 / 3))
    starts = np.arange(n - L + 1)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + L) for s in rng.choice(starts, math.ceil(n / L))])[:n]
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        d.append(roc_auc_score(yy, s2[idx]) - roc_auc_score(yy, s1[idx]))
    d = np.array(d)
    return L, np.percentile(d, [2.5, 97.5]), (d <= 0).mean()


src_b = pd.read_csv(f"{P}/b_up_7d.csv")
src_a_full = pd.read_csv(f"{P}/a_up_7d.csv")
full_up = pd.read_csv(f"{R}/a_up_7d/predictions.csv")
full_dn = pd.read_csv(f"{R}/a_down_7d/predictions.csv")
full_up["date"] = src_a_full.date.iloc[full_up.row].values
full_dn["date"] = src_a_full.date.iloc[full_dn.row].values

for t in ["up_7d", "down_7d"]:
    a = load(f"a_short_{t}")
    b = load(f"b_{t}")
    assert (a.row.values == b.row.values).all() and (a.y_true.values == b.y_true.values).all()
    y = a.y_true.values.astype(int)
    auc_a = roc_auc_score(y, a.y_pred)
    auc_b = roc_auc_score(y, b.y_pred)
    L, ci, p_le0 = block_boot_diff(y, a.y_pred.values, b.y_pred.values)
    print(f"\n=== {t}: holdout {len(y)} days, positive share {y.mean():.3f}")
    print("  A-short chosen, cv mean, folds, holdout:", cv_info(f"a_short_{t}"))
    print("  B       chosen, cv mean, folds, holdout:", cv_info(f"b_{t}"))
    print(f"  AUC A-short {auc_a:.3f}  B {auc_b:.3f}  diff B-A {auc_b-auc_a:+.3f}  "
          f"paired block-bootstrap 95% CI on diff [{ci[0]:+.3f}, {ci[1]:+.3f}] (block {L}), share of resamples with diff <= 0: {p_le0:.2f}")
    print(f"  rank corr of the two models' predictions: {spearmanr(a.y_pred, b.y_pred).statistic:.3f}")
    # full-history model on the same dates
    dates = src_b.date.iloc[a.row].values
    fp = (full_up if t == "up_7d" else full_dn).set_index("date").reindex(dates)
    m = fp.y_pred.notna().values
    print(f"  full-history table-A model on these same dates: AUC {roc_auc_score(y[m], fp.y_pred.values[m]):.3f} ({m.sum()} days)")
    # what the B model tracks
    feat = src_b.iloc[a.row].reset_index(drop=True)
    cols = src_b.columns[1:-1]
    cors = sorted(((c, spearmanr(feat[c], b.y_pred).statistic) for c in cols), key=lambda kv: -abs(kv[1]))
    print("  B model's predictions most correlated with:")
    for c, v in cors[:8]:
        print(f"     {c:<24} {v:+.3f}   alone AUC {roc_auc_score(y, feat[c]):.3f}")
    bn = [c for c in cols if c.startswith(("fund", "basis", "usdt", "perp", "taker", "trades", "avg_trade", "bn_"))]
    best_bn = sorted(((c, roc_auc_score(y, feat[c])) for c in bn), key=lambda kv: -abs(kv[1] - 0.5))[:4]
    print("  strongest Binance columns alone:", [(c, round(v, 3)) for c, v in best_bn])

# direction vs magnitude in B
bu, bd = load("b_up_7d"), load("b_down_7d")
au, ad = load("a_short_up_7d"), load("a_short_down_7d")
print("\nrank corr p_up vs p_down: B", round(spearmanr(bu.y_pred, bd.y_pred).statistic, 3),
      " A-short", round(spearmanr(au.y_pred, ad.y_pred).statistic, 3))
q = pd.qcut(bd.y_pred, 5, labels=False)
print("B p_down quintile -> realised DOWN share, UP share")
print(pd.DataFrame({"down": bd.y_true, "up": bu.y_true, "q": q}).groupby("q").mean().round(3))
