"""Pass 8 (CP9) build on real public tables: what was open after pass 7, the new statements and warnings, and unchanged tables."""
import json, sys, time, traceback, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/realdata/app9")
import mlprofessor
from mlprofessor import errors
from cases import lagged
D = "/home/claude/realdata/data/"; OUT = []
NO = dict(time_col=False, group_col=False)
SHOW = ("strong_single_column", "extreme_value", "largest_row_dominates", "selection_not_made", "autocorrelated_panel_target", "suspected_leak")

def run(name, data, target, expect, **kw):
    kw.setdefault("profile", "fast"); kw.setdefault("n_jobs", 2)
    t0 = time.time(); rec = {"case": name, "expect": expect, "args": {k: str(v) for k, v in kw.items()}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            d = mlprofessor.profile(data, target, **kw).to_dict(); h, v = d["headline"], d["validation"]
            rec.update(outcome="ran", scheme=v["scheme"], baseline=h["baseline"], skill=h["skill"], model=h["model"],
                       ci=[h.get("skill_low"), h.get("skill_high")],
                       baselines=[b["name"] for b in d.get("baselines", [])],
                       share=h.get("largest_row_share"), share_base=h.get("baseline_largest_row_share"),
                       warnings=sorted({w["code"] for w in d["warnings"]}),
                       texts=[(w["code"], w.get("column"), w["message"][:200]) for w in d["warnings"] if w["code"] in SHOW],
                       excluded={c["name"]: c["reason"] for c in d["columns"] if c["role"] == "excluded"},
                       headline_keys=sorted(k for k in h if "largest" in k))
        except errors.ProfileError as exc:
            rec.update(outcome="named error", code=exc.code, message=str(exc))
        except Exception as exc:
            rec.update(outcome="RAW EXCEPTION", code=type(exc).__name__, message=str(exc)[:300], trace=traceback.format_exc()[-600:])
    rec["seconds"] = round(time.time() - t0, 1); rec["py_warnings"] = sorted({str(w.message)[:80] for w in caught}); OUT.append(rec)
    f = lambda x: "None" if x is None else f"{x:.3f}"
    print("-" * 110); print(f"{name}  [{rec['seconds']}s]\n   expect: {expect}")
    if rec["outcome"] == "ran":
        print(f"   RAN  {rec['scheme']}, {rec['model']}, baseline {rec['baseline']} of {rec['baselines']}, skill {f(rec['skill'])} [{f(rec['ci'][0])}, {f(rec['ci'][1])}], largest row share: model {f(rec['share'])}, baseline {f(rec['share_base'])}")
        print(f"   warnings {rec['warnings']}; excluded {rec['excluded']}")
        for c, col, m in rec["texts"]: print(f"   >> {c} [{col}]: {m}")
    else:
        print(f"   {rec['outcome']}: {rec['code']}: {rec['message'][:420]}"); print(rec.get("trace", ""), end="")
    if rec["py_warnings"]: print("   PYTHON WARNINGS:", rec["py_warnings"])
    return rec

from sklearn.datasets import load_breast_cancer, load_diabetes
vix = pd.read_csv(D + "vix.csv"); vix.columns = [c.lower() for c in vix.columns]
base = lagged(vix, ["close", "high", "low"], [1, 2, 5]); lv = base[["date", "close"] + [c for c in base.columns if "_lag" in c]].dropna().reset_index(drop=True)
T = dict(time_col="date", group_col=False)
print("\n##### A. What was open after pass 7 (D73)")
run("VIX level, clean", lv, "close", "0.023 as before", **T)
run("VIX level, ONE row repeated", pd.concat([lv, lv.iloc[[len(lv) - 900]]], ignore_index=True), "close", "was 0.843; now about the clean figure, previous value still the baseline", **T)
rng = np.random.default_rng(4); ten = pd.concat([lv, lv.iloc[rng.choice(np.arange(len(lv) - 2000, len(lv)), 200, replace=False)]], ignore_index=True)
run("VIX level, 10% of the kept rows repeated", ten, "close", "about the clean figure", **T)
cp = lv.copy(); cp["close_in_other_units"] = cp["close"] * 1.07 + 3.0
run("VIX level + exact rescaled copy of the target", cp, "close", "was 1.000; now the copy is excluded as a suspected leak", **T)
c2 = lv.copy(); c2["copy_few_changed"] = c2["close"].to_numpy(); c2.loc[rng.choice(np.arange(len(c2) - 2000, len(c2)), 5, replace=False), "copy_few_changed"] += 7.0
run("VIX level + copy with 5 cells changed", c2, "close", "excluded", **T)
c3 = lv.copy(); c3["close_squared"] = c3["close"] ** 2
run("VIX level + the square of the target", c3, "close", "at least warned (order reading)", **T)
c4 = lv.copy(); c4["noisy_copy"] = c4["close"] + rng.normal(0, 0.05 * c4["close"].iloc[-2000:].std(), len(c4))
run("VIX level + copy with 5% noise", c4, "close", "STATED LIMIT (D76): may pass in silence", **T)
db = load_diabetes(as_frame=True).frame
ex = db.copy(); ex.loc[10, "bmi"] = 999999.0
run("diabetes, one cell 999999 in a feature", ex, "target", "was -0.111 with no word; now the value is named", **NO)
ey = db.copy(); ey.loc[10, "target"] = 999999.0
run("diabetes, one cell 999999 in the target", ey, "target", "the value is named; the largest row's share says the rest", **NO)
run("diabetes, clean", db, "target", "0.490034; shares small; no new warning", **NO)

print("\n##### B. The new readings on real columns that are NOT copies (false alarms would show here)")
same = vix[["date", "open", "high", "low", "close"]].copy()
run("VIX close from the same day's open, high, low (time order)", same, "close", "legitimate-looking strong features; what does the tool say?", **T)
run("diamonds price (carat, x, y, z are strong)", D + "diamonds.csv", "price", "0.933 as before; any strong_single_column is a question, not an exclusion", **NO)
run("breast cancer", load_breast_cancer(as_frame=True).frame, "target", "0.994549; nothing new", **NO)
tx = pd.read_csv(D + "taxis.csv"); tx = tx[tx["payment"].notna()].drop(columns=["pickup", "dropoff"])
run("taxis fare (total and distance are near-copies of fare)", tx, "fare", "`total` strong; excluded or warned, never silent", **NO)
run("taxis tip (heavy-tailed, many zeros)", tx.drop(columns=["total"]), "tip", "runs; extreme_value quiet unless a value truly stands apart", **NO)
mpg = pd.read_csv(D + "mpg.csv")
run("mpg", mpg, "mpg", "0.822 as before", **NO)

print("\n##### C. Identifier and no-choice rules")
tit = pd.read_csv(D + "titanic.csv").drop(columns=["alive"]); t3 = tit.copy(); t3.insert(0, "passenger_id", np.arange(1, len(t3) + 1)); t3 = t3.sample(frac=1.0, random_state=3).reset_index(drop=True)
run("titanic + a consecutive id, rows SHUFFLED, keep", t3, "survived", "passenger_id excluded as id_like (was a feature when shuffled); 0.745", duplicates="keep", **NO)
m2 = mpg.copy(); m2["maker"] = np.where(m2["origin"] == "usa", "usa", m2["name"]); m2 = m2.drop(columns=["name", "origin"])
run("mpg, every US car one entity (62.6%)", m2, "mpg", "as before (0.822) unless the folds cannot choose, then selection_not_made", time_col=False, group_col="maker")

print("\n##### D. Unchanged tables")
st = pd.read_csv(D + "stocks.csv"); st["date"] = pd.to_datetime(st["date"], format="%b %d %Y"); st = st.sort_values(["symbol", "date"])
st["price_lag1"] = st.groupby("symbol")["price"].shift(1); lvl = st[["date", "symbol", "price", "price_lag1"]].dropna()
run("penguins sex", pd.read_csv(D + "penguins.csv"), "sex", "0.937642", **NO)
run("titanic keep", tit, "survived", "0.744865", duplicates="keep", **NO)
run("titanic group", tit, "survived", "0.645030", duplicates="group", **NO)
run("stocks level, group_col=symbol", lvl, "price", "-0.138855", time_col="date", group_col="symbol")
run("stocks level, group_col=False (a panel, on the caller's word)", lvl, "price", "was 0.979 against the training mean; now both baselines are scored", time_col="date", group_col=False)
json.dump(OUT, open("/home/claude/realdata/out/results_cp9.json", "w"), indent=1, default=str)
print("\nSUMMARY:", {k: sum(1 for r in OUT if r["outcome"] == k) for k in {r["outcome"] for r in OUT}}, "| python warnings in", sum(1 for r in OUT if r["py_warnings"]), "runs")
