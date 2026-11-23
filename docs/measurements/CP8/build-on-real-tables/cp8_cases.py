"""Pass 7 (CP8) build on real public tables: sentinel dates, group after a removed leak, the fast cap, and the items left open for pass 8."""
import json, sys, time, traceback, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/realdata/app8")
import mlprofessor
from mlprofessor import errors
from cases import lagged
D = "/home/claude/realdata/data/"; OUT = []
NO = dict(time_col=False, group_col=False)

def run(name, data, target, expect, **kw):
    kw.setdefault("profile", "fast"); kw.setdefault("n_jobs", 2)
    t0 = time.time(); rec = {"case": name, "expect": expect, "args": {k: str(v) for k, v in kw.items()}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            d = mlprofessor.profile(data, target, **kw).to_dict(); h, v = d["headline"], d["validation"]
            n = {r["item"]: r["count"] for r in d["rows"]}
            rec.update(outcome="ran", scheme=v["scheme"], baseline=h["baseline"], skill=h["skill"],
                       ci=[h.get("skill_low"), h.get("skill_high")], interval=h.get("interval_method"), p=h.get("p_value"),
                       groups=[v.get("n_groups_training"), v.get("n_groups_holdout")], rows=n,
                       warnings=sorted({w["code"] for w in d["warnings"]}),
                       texts={w["code"]: w["message"][:230] for w in d["warnings"] if w["code"] in ("date_column_undeclared", "same_features_grouped") or w["code"].startswith("large_group")},
                       excluded={c["name"]: c["reason"] for c in d["columns"] if c["role"] == "excluded"})
        except errors.ProfileError as exc:
            rec.update(outcome="named error", code=exc.code, message=str(exc))
        except Exception as exc:
            rec.update(outcome="RAW EXCEPTION", code=type(exc).__name__, message=str(exc)[:300], trace=traceback.format_exc()[-500:])
    rec["seconds"] = round(time.time() - t0, 1); rec["py_warnings"] = sorted({str(w.message)[:80] for w in caught}); OUT.append(rec)
    f = lambda x: "None" if x is None else f"{x:.3f}"
    print("-" * 110); print(f"{name}  [{rec['seconds']}s]\n   expect: {expect}")
    if rec["outcome"] == "ran":
        hold = {k: v for k, v in rec["rows"].items() if k in ("training", "holdout", "dropped_row_cap")}
        print(f"   RAN  {rec['scheme']}, baseline {rec['baseline']}, skill {f(rec['skill'])} [{f(rec['ci'][0])}, {f(rec['ci'][1])}] via {rec['interval']}, units {rec['groups']}, {hold}")
        print(f"   warnings {rec['warnings']}; excluded {rec['excluded']}")
        for k, m in rec["texts"].items(): print(f"   >> {k}: {m}")
    else:
        print(f"   {rec['outcome']}: {rec['code']}: {rec['message'][:520]}"); print(rec.get("trace", ""), end="")
    if rec["py_warnings"]: print("   PYTHON WARNINGS:", rec["py_warnings"])
    return rec

vix = pd.read_csv(D + "vix.csv"); vix.columns = [c.lower() for c in vix.columns]
base = lagged(vix, ["close", "high", "low"], [1, 2, 5]); lv = base[["date", "close"] + [c for c in base.columns if "_lag" in c]].dropna().reset_index(drop=True)
w = pd.read_csv(D + "weather.csv"); w["rain_tomorrow"] = (w["precipitation"].shift(-1) > 0).astype(int); w = w.iloc[:-1].copy()
w["day"] = pd.to_datetime(w["date"]).dt.strftime("%Y%m%d").astype(int); wd = w.drop(columns=["date"])
print("\n##### A. A date column with a sentinel (D69, D72)")
run("weather, day as YYYYMMDD, clean, both False", wd, "rain_tomorrow", "`day` excluded as a date, with the warning", **NO)
for name, val, share in (("one cell 99991231", 99991231, None), ("one cell 0", 0, None), ("30% of rows 99991231", 99991231, 0.30)):
    e = wd.copy(); idx = [7] if share is None else np.random.default_rng(1).choice(len(e), int(len(e) * share), replace=False)
    e.loc[idx, "day"] = val
    run(f"weather, day with {name}, both False", e, "rain_tomorrow", "still excluded as a date; skill as the clean table's", **NO)
e = wd.copy(); e.loc[7, "day"] = 99991231
run("weather, day with one cell 99991231, time_col=day", e, "rain_tomorrow", "stops: time_column_invalid, naming 99991231 and its one row", time_col="day", group_col=False)
run("weather, clean day, time_col=day", wd, "rain_tomorrow", "runs under time order", time_col="day", group_col=False)
q = pd.DataFrame({"period": np.repeat([20230331, 20230630, 20230930, 20231231, 20240331, 20240630, 20240930, 20241231], 60)})
g = np.random.default_rng(3); q["x"] = g.normal(size=len(q)); q["y"] = (g.random(len(q)) < 0.5).astype(int); q.loc[5, "period"] = 99991231
run("eight quarter-ends with one sentinel (the case my second clause was for)", q, "y", "`period` excluded as a date", **NO)

print("\n##### B. duplicates='group' and a column the leak screen removes (D69, D72)")
tit = pd.read_csv(D + "titanic.csv")
run("titanic WITH `alive` (a copy of the target), duplicates=group", tit, "survived", "stops, naming `alive`, and says to remove it and run again", duplicates="group", **NO)
run("titanic with `alive`, duplicates=keep", tit, "survived", "`alive` excluded as a suspected leak; runs", duplicates="keep", **NO)
run("titanic without `alive`, duplicates=group", tit.drop(columns=["alive"]), "survived", "0.645 as before; the warning says what group measures", duplicates="group", **NO)

print("\n##### C. The fast profile and a large entity (D70, D72)")
tx = pd.read_csv(D + "taxis.csv"); tx = tx[tx["payment"].notna() & tx["pickup_zone"].notna() & tx["pickup_borough"].notna()].drop(columns=["pickup", "dropoff"])
t2 = tx.copy(); t2["area"] = np.where(t2["pickup_borough"] == "Manhattan", "Manhattan", t2["pickup_zone"]); t2 = t2.drop(columns=["pickup_zone", "pickup_borough", "total", "tip"])
run("taxis, Manhattan one entity of 5,236 rows, FAST", t2, "fare", "stops: fast_profile_group_too_large, pointing to the full profile (was: skill 0.878 on the other rows)", time_col=False, group_col="area")
run("the same, FULL", t2, "fare", "as before: Manhattan is the holdout, warned", time_col=False, group_col="area", profile="full")
mpg = pd.read_csv(D + "mpg.csv"); m2 = mpg.copy(); m2["maker"] = np.where(m2["origin"] == "usa", "usa", m2["name"]); m2 = m2.drop(columns=["name", "origin"])
run("mpg, every US car one entity (62.6%), fast", m2, "mpg", "0.822 as before; the large entity kept and named", time_col=False, group_col="maker")
t3 = tx.copy(); t3 = t3.drop(columns=["total", "tip"])
run("taxis by pickup zone (largest zone well under a quarter), fast", t3.drop(columns=["pickup_borough"]), "fare", "runs on a sample of whole zones", time_col=False, group_col="pickup_zone")

print("\n##### D. Unchanged tables: must be identical to the previous build")
from sklearn.datasets import load_breast_cancer, load_diabetes
st = pd.read_csv(D + "stocks.csv"); st["date"] = pd.to_datetime(st["date"], format="%b %d %Y"); st = st.sort_values(["symbol", "date"])
st["price_lag1"] = st.groupby("symbol")["price"].shift(1); lvl = st[["date", "symbol", "price", "price_lag1"]].dropna()
run("breast cancer, both said", load_breast_cancer(as_frame=True).frame, "target", "0.994549", **NO)
run("diabetes, both said", load_diabetes(as_frame=True).frame, "target", "0.490034", **NO)
run("penguins sex, both said", pd.read_csv(D + "penguins.csv"), "sex", "0.937642", **NO)
run("diamonds, both said", D + "diamonds.csv", "price", "0.933327", **NO)
run("titanic keep, both said", tit.drop(columns=["alive"]), "survived", "0.744865", duplicates="keep", **NO)
run("VIX level, time_col=date, group_col=False", lv, "close", "0.023366", time_col="date", group_col=False)
run("stocks price level, time_col=date, group_col=symbol", lvl, "price", "-0.138855", time_col="date", group_col="symbol")
run("stocks price level, time_col=date, group_col=False", lvl, "price", "0.978920", time_col="date", group_col=False)

print("\n##### E. Open for pass 8 (D73): what today's build still does on real data")
one = pd.concat([lv, lv.iloc[[len(lv) - 900]]], ignore_index=True)  # a row inside the 2,000 the fast cap keeps
run("OPEN: VIX level with ONE row repeated, time_col=date, group_col=False", one, "close", "today: the previous-value baseline is lost and skill jumps (0.023 on the clean table)", time_col="date", group_col=False)
cp = lv.copy(); cp["close_in_other_units"] = cp["close"] * 1.07 + 3.0
run("OPEN: VIX level plus an exact rescaled copy of the target, time order", cp, "close", "today: is the copy excluded as a leak?", time_col="date", group_col=False)
ex = load_diabetes(as_frame=True).frame.copy(); ex.loc[10, "bmi"] = 999999.0
run("OPEN: diabetes with one cell of 999999 in a feature", ex, "target", "today: does one cell move the skill (0.490 clean)?", **NO)
json.dump(OUT, open("/home/claude/realdata/out/results_cp8.json", "w"), indent=1, default=str)
print("\nSUMMARY:", {k: sum(1 for r in OUT if r["outcome"] == k) for k in {r["outcome"] for r in OUT}}, "| python warnings in", sum(1 for r in OUT if r["py_warnings"]), "runs")
