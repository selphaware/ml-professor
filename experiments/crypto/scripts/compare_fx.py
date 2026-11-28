"""FX tables against their counterparts: C vs A and D vs B, paired on the same holdout days."""
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


def chosen(run):
    p = json.load(open(f"{R}/{run}/profile.json"))
    m = [x for x in p["models"] if x["chosen"]][0]
    return m["config"], round(m["cv_mean"], 3)


def paired(y, s_base, s_new, n_boot=2000):
    n = len(y)
    L = math.ceil(n ** (1 / 3))
    starts = np.arange(n - L + 1)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + L) for s in rng.choice(starts, math.ceil(n / L))])[:n]
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], s_new[idx]) - roc_auc_score(y[idx], s_base[idx]))
    d = np.array(d)
    return np.percentile(d, [2.5, 97.5]), (d <= 0).mean()


print(f"{'pair':<22}{'base AUC':>9}{'FX AUC':>8}{'diff':>8}   95% CI on diff      share<=0  base model / FX model (cv mean)")
for new, base in (("c", "a"), ("d", "b")):
    for t in ("up_7d", "down_7d"):
        a = pd.read_csv(f"{R}/{base}_{t}/predictions.csv")
        b = pd.read_csv(f"{R}/{new}_{t}/predictions.csv")
        assert (a.row.values == b.row.values).all() and (a.y_true.values == b.y_true.values).all()
        y = a.y_true.values.astype(int)
        A, B = roc_auc_score(y, a.y_pred), roc_auc_score(y, b.y_pred)
        ci, share = paired(y, a.y_pred.values, b.y_pred.values)
        print(f"{new.upper()} vs {base.upper()} {t:<14}{A:>9.3f}{B:>8.3f}{B-A:>+8.3f}   [{ci[0]:+.3f}, {ci[1]:+.3f}]   {share:>6.2f}   "
              f"{chosen(f'{base}_{t}')} / {chosen(f'{new}_{t}')}   rank corr {spearmanr(a.y_pred, b.y_pred).statistic:.2f}")

# what the FX-table down models lean on, among the FX columns
for new in ("c", "d"):
    src = pd.read_csv(f"{P}/{new}_down_7d.csv")
    pr = pd.read_csv(f"{R}/{new}_down_7d/predictions.csv")
    f = src.iloc[pr.row].reset_index(drop=True)
    fx = [c for c in src.columns if c.startswith(("fx_", "btc_dxy"))]
    cors = sorted(((c, spearmanr(f[c], pr.y_pred).statistic) for c in fx), key=lambda kv: -abs(kv[1]))[:4]
    print(f"\n{new.upper()} down model: FX columns most correlated with its P(down) on the holdout:",
          [(c, round(v, 2)) for c, v in cors])
