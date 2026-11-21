import sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/realdata/app")
import mlprofessor
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_diabetes
D = "/home/claude/realdata/data/"
warnings.simplefilter("ignore")

print("== 1. Titanic: recompute the headline independently")
tit = pd.read_csv(D + "titanic.csv").drop(columns=["alive"])
r = mlprofessor.profile(tit, "survived", profile="fast", duplicates="keep")
p = r.predictions
src = tit["survived"].to_numpy()[p["row"].to_numpy()]
print("  y_true equals the source rows:", bool((src == p["y_true"].to_numpy()).all()))
auc = roc_auc_score(p["y_true"], p["y_pred"])
print(f"  AUC by scikit-learn {auc:.6f} vs headline {r.headline['score']:.6f}; holdout {len(p)} of {len(tit)} = {len(p)/len(tit):.3f}")
print("  holdout class share", round(p["y_true"].mean(), 4), "vs table", round(tit["survived"].mean(), 4))
# effect of the copies on the titanic headline
d = r.to_dict()
feat = [c["name"] for c in d["columns"] if c["role"] == "feature"]
hold = set(p["row"]); train = tit.drop(index=list(hold))
key = lambda df: df[feat + ["survived"]].astype(object).where(df[feat + ["survived"]].notna(), "NA").apply(lambda row: "|".join(map(str, row)), axis=1)
tk = set(key(train)); hk = key(tit.loc[p["row"]])
has = hk.isin(tk).to_numpy()
print(f"  my own count of holdout rows with a copy: {has.sum()} (tool says {d['validation']['n_holdout_copies']})")
print(f"  AUC on holdout rows with a copy {roc_auc_score(p['y_true'][has], p['y_pred'][has]):.3f}, without {roc_auc_score(p['y_true'][~has], p['y_pred'][~has]):.3f}")

print("== 2. VIX level: check the time split and both baselines independently")
vix = pd.read_csv(D + "vix.csv"); vix.columns = [c.lower() for c in vix.columns]
for c in ["close", "high", "low"]:
    for k in [1, 2, 5]:
        vix[f"{c}_lag{k}"] = vix[c].shift(k)
lv = vix[["date", "close"] + [c for c in vix.columns if "_lag" in c]].dropna().reset_index(drop=True)
r = mlprofessor.profile(lv, "close", time_col="date", profile="fast")
p = r.predictions; d = r.to_dict(); v = d["validation"]
rows = p["row"].to_numpy()
print("  holdout rows are the latest rows of the table:", bool((np.sort(rows) == np.arange(len(lv) - len(rows), len(lv))).all()))
print("  validation:", {k: v[k] for k in ("train_start", "train_end", "holdout_start", "holdout_end") if k in v})
kept = d["rows"]; print("  rows:", {x["item"]: x["count"] for x in kept})
y = lv["close"].to_numpy()
prev = y[rows - 1]
print(f"  previous-value MSE by hand {np.mean((y[rows]-prev)**2):.6f} vs reported {[b for b in d['baselines'] if b['name']=='previous_value'][0]['score']:.6f}")
model_mse = np.mean((p["y_true"] - p["y_pred"])**2)
print(f"  model MSE by hand {model_mse:.6f} vs headline {r.headline['score']:.6f}")

print("== 3. Near-copies on real continuous features (review 4, finding 2)")
db = load_diabetes(as_frame=True).frame
rng = np.random.default_rng(3)
noise = db.copy(); noise["target"] = rng.permutation(noise["target"].to_numpy())
def skill(df, **kw):
    try:
        rr = mlprofessor.profile(df, "target", profile="fast", **kw)
        h = rr.headline
        return f"skill {h['skill']:.3f} (p {h['p_value']}, model {h['model']}, copies {rr.to_dict()['validation'].get('n_holdout_copies')}, warnings {sorted({w['code'] for w in rr.to_dict()['warnings']})})"
    except mlprofessor.errors.ProfileError as e:
        return f"STOPS: {e.code}"
print("  diabetes features, shuffled target (pure noise):      ", skill(noise))
twice = pd.concat([noise, noise], ignore_index=True)
print("  the same, every row twice:                            ", skill(twice))
b = twice.copy(); b["batch"] = np.repeat([1, 2], len(noise))
print("  every row twice + a batch column:                     ", skill(b))
j = twice.copy(); j.loc[len(noise):, "bmi"] += 1e-9
print("  every row twice, second copy's bmi changed by 1e-9:   ", skill(j))
