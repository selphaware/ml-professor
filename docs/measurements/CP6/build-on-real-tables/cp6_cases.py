"""Pass 5 (CP6) build on real public tables: the two new stops, entities, duplicates="group"."""
import json, sys, time, traceback, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/realdata/app6")
import mlprofessor
from mlprofessor import errors
from cases import lagged
D = "/home/claude/realdata/data/"; OUT = []

def run(name, data, target, expect, **kw):
    kw.setdefault("profile", "fast"); kw.setdefault("n_jobs", 2)
    t0 = time.time(); rec = {"case": name, "expect": expect, "args": {k: str(v) for k, v in kw.items()}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            d = mlprofessor.profile(data, target, **kw).to_dict(); h, v = d["headline"], d["validation"]
            rec.update(outcome="ran", scheme=v["scheme"], metric=h["metric"], baseline=h["baseline"], skill=h["skill"],
                       ci=[h.get("skill_low"), h.get("skill_high")], interval=h.get("interval_method"), p=h.get("p_value"),
                       model=h["model"], groups=[v.get("n_groups_training"), v.get("n_groups_holdout")],
                       copies=[v.get("n_holdout_copies"), v.get("n_rows_with_copies")],
                       input={k: d["input"].get(k) for k in ("time_col", "group_col", "duplicates")},
                       warnings=sorted({w["code"] for w in d["warnings"]}),
                       excluded={c["name"]: c["reason"] for c in d["columns"] if c["role"] == "excluded"})
        except errors.ProfileError as exc:
            rec.update(outcome="named error", code=exc.code, message=str(exc)[:420])
        except Exception as exc:
            rec.update(outcome="RAW EXCEPTION", code=type(exc).__name__, message=str(exc)[:300], trace=traceback.format_exc()[-500:])
    rec["seconds"] = round(time.time() - t0, 1); rec["py_warnings"] = sorted({str(w.message)[:80] for w in caught}); OUT.append(rec)
    f = lambda x: "None" if x is None else f"{x:.3f}"
    print("-" * 110); print(f"{name}  [{rec['seconds']}s]\n   expect: {expect}")
    if rec["outcome"] == "ran":
        print(f"   RAN  {rec['scheme']}, baseline {rec['baseline']}, skill {f(rec['skill'])} [{f(rec['ci'][0])}, {f(rec['ci'][1])}] via {rec['interval']}, p {f(rec['p'])}, groups {rec['groups']}, copies {rec['copies']}")
        print(f"   input {rec['input']}; warnings {rec['warnings']}; excluded {rec['excluded']}")
    else:
        print(f"   {rec['outcome']}: {rec['code']}: {rec['message']}"); print(rec.get("trace", ""), end="")
    if rec["py_warnings"]: print("   PYTHON WARNINGS:", rec["py_warnings"])
    return rec

vix = pd.read_csv(D + "vix.csv"); vix.columns = [c.lower() for c in vix.columns]
base = lagged(vix, ["close", "high", "low"], [1, 2, 5]); lv = base[["date", "close"] + [c for c in base.columns if "_lag" in c]].dropna()
print("\n##### A. A date column and nothing said about time order (D58 part 1)")
run("VIX level, lag features, nothing said", lv, "close", "stops: time_order_undeclared (was skill 0.939 with a warning)")
run("VIX level, time_col=date", lv, "close", "previous_value baseline, skill small (was 0.023)", time_col="date")
run("VIX level, time_col=False", lv, "close", "runs on the caller's word, with the warning and the caveat", time_col=False)
run("weather.csv as shipped, target temp_max, nothing said", D + "weather.csv", "temp_max", "stops: time_order_undeclared, naming `date`")

st = pd.read_csv(D + "stocks.csv"); st["date"] = pd.to_datetime(st["date"], format="%b %d %Y"); st = st.sort_values(["symbol", "date"])
st["price_lag1"] = st.groupby("symbol")["price"].shift(1); st["ret1"] = st.groupby("symbol")["price"].pct_change()
st["ret_next"] = st.groupby("symbol")["ret1"].shift(-1)
lvl = st[["date", "symbol", "price", "price_lag1"]].dropna(); ret = st[["date", "symbol", "ret_next", "ret1"]].dropna()
print("\n##### B. A panel under time order (D57 part 1, D58 part 3)")
run("stocks price level, time_col=date, nothing said about entities", lvl, "price", "stops: entities_undeclared (was skill 0.979)", time_col="date")
run("stocks price level, time_col=date, group_col=symbol", lvl, "price", "each stock's own previous value is the baseline; skill about 0", time_col="date", group_col="symbol")
run("stocks price level, time_col=date, group_col=False", lvl, "price", "runs on the caller's word, with autocorrelated_panel_target", time_col="date", group_col=False)
run("stocks next-month return, time_col=date, nothing said", ret, "ret_next", "runs: returns are not persistent, so nothing to ask", time_col="date")
run("stocks next-month return, time_col=date, group_col=symbol", ret, "ret_next", "runs; skill about 0", time_col="date", group_col="symbol")

print("\n##### C. A declared entity column under the random scheme (D57 part 1)")
tx = pd.read_csv(D + "taxis.csv"); tx = tx[tx["payment"].notna()]
run("taxis payment, nothing said (two timestamp columns)", tx, "payment", "stops: time_order_undeclared, naming pickup and dropoff")
a = run("taxis payment, time_col=False", tx, "payment", "random scheme", time_col=False)
b = run("taxis payment, time_col=False, group_col=pickup_zone", tx, "payment", "grouped scheme; interval from whole zones; no p-value", time_col=False, group_col="pickup_zone")
mpg = pd.read_csv(D + "mpg.csv")
run("mpg, group_col=name (a model name can repeat across years)", mpg, "mpg", "grouped scheme; `name` is not a feature", group_col="name")
run("mpg, nothing said", mpg, "mpg", "random; `name` excluded as free text or id-like")
run("stocks return as a plain table, time_col=False, group_col=symbol", ret, "ret_next", "named error: too_few_groups (five stocks)", time_col=False, group_col="symbol")
pen = pd.read_csv(D + "penguins.csv")
run("penguins sex, group_col=island (three islands)", pen, "sex", "named error: too_few_groups", group_col="island")
run("penguins sex, group_col=nobody", pen, "sex", "named error: column not found", group_col="nobody")

print("\n##### D. duplicates (D49, D57 part 3, D60)")
tit = pd.read_csv(D + "titanic.csv").drop(columns=["alive"])
run("titanic, default", tit, "survived", "stops: holdout_copies_in_training, naming keep and group")
run("titanic, duplicates=keep", tit, "survived", "runs; random scheme; warning", duplicates="keep")
run("titanic, duplicates=group", tit, "survived", "grouped; copies counted; no p-value; skill near the keep figure", duplicates="group")
sh = tit.copy(); sh["survived"] = np.random.default_rng(5).permutation(sh["survived"].to_numpy()); tw = pd.concat([sh, sh], ignore_index=True)
run("titanic, target shuffled, every row twice, duplicates=keep", tw, "survived", "inflated: this is what keep permits", duplicates="keep")
run("titanic, target shuffled, every row twice, duplicates=group", tw, "survived", "skill about 0: the copies cannot be on both sides", duplicates="group")
tb = tw.copy(); tb["batch"] = np.repeat([1, 2], len(sh))
run("titanic, target shuffled, every row twice + batch column, default", tb, "survived", "KNOWN OPEN DEFECT (D60): not detected; stops only on Titanic's natural repeats")
bc = __import__("sklearn.datasets", fromlist=["x"]).load_breast_cancer(as_frame=True).frame
bn = bc.copy(); bn["target"] = np.random.default_rng(5).permutation(bn["target"].to_numpy()); b2 = pd.concat([bn, bn], ignore_index=True); b2["batch"] = np.repeat([1, 2], len(bn))
run("breast cancer, target shuffled, every row twice + batch column, default", b2, "target", "KNOWN OPEN DEFECT (D60): runs with an inflated skill and no stop")
run("breast cancer, target shuffled, every row twice, default", pd.concat([bn, bn], ignore_index=True), "target", "stops: exact copies")
run("breast cancer, target shuffled, every row twice, duplicates=group", pd.concat([bn, bn], ignore_index=True), "target", "skill about 0", duplicates="group")

print("\n##### E. Tables with nothing new to say: should be as before")
run("breast cancer", bc, "target", "as before: skill about 0.98")
run("diabetes", __import__("sklearn.datasets", fromlist=["x"]).load_diabetes(as_frame=True).frame, "target", "as before")
run("penguins sex", pen, "sex", "as before")
run("customers example CSV", D + "customers.csv", "churned", "stops or runs? it has signup_date: expect time_order_undeclared")
run("diamonds 53,940 rows", D + "diamonds.csv", "price", "as before: skill about 0.97")
json.dump(OUT, open("/home/claude/realdata/out/results_cp6.json", "w"), indent=1, default=str)
print("\nSUMMARY:", {k: sum(1 for r in OUT if r["outcome"] == k) for k in {r["outcome"] for r in OUT}}, "| python warnings in", sum(1 for r in OUT if r["py_warnings"]), "runs")
