"""Pass 6 (CP7) build on real public tables: both statements, group on the features, the grouped holdout, dates as numbers."""
import json, os, subprocess, sys, time, traceback, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/realdata/app7")
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
                       same=v.get("n_rows_same_features"), copies=v.get("n_holdout_copies"),
                       input={k: d["input"].get(k) for k in ("time_col", "group_col", "duplicates")},
                       warnings=sorted({w["code"] for w in d["warnings"]}),
                       large=[w["message"][:150] for w in d["warnings"] if w["code"].startswith("large_group")],
                       excluded={c["name"]: c["reason"] for c in d["columns"] if c["role"] == "excluded"})
        except errors.ProfileError as exc:
            rec.update(outcome="named error", code=exc.code, message=str(exc))
        except Exception as exc:
            rec.update(outcome="RAW EXCEPTION", code=type(exc).__name__, message=str(exc)[:300], trace=traceback.format_exc()[-500:])
    rec["seconds"] = round(time.time() - t0, 1); rec["py_warnings"] = sorted({str(w.message)[:80] for w in caught}); OUT.append(rec)
    f = lambda x: "None" if x is None else f"{x:.3f}"
    print("-" * 110); print(f"{name}  [{rec['seconds']}s]\n   expect: {expect}")
    if rec["outcome"] == "ran":
        hold = {k: v for k, v in rec["rows"].items() if "holdout" in k or "training" in k}
        print(f"   RAN  {rec['scheme']}, baseline {rec['baseline']}, skill {f(rec['skill'])} [{f(rec['ci'][0])}, {f(rec['ci'][1])}] via {rec['interval']}, p {f(rec['p'])}, units {rec['groups']}, same-feature rows {rec['same']}, {hold}")
        print(f"   said {rec['input']}; warnings {rec['warnings']}; excluded {rec['excluded']}")
        for m in rec["large"]: print("   >>", m)
    else:
        print(f"   {rec['outcome']}: {rec['code']}: {rec['message'][:700]}"); print(rec.get("trace", ""), end="")
    if rec["py_warnings"]: print("   PYTHON WARNINGS:", rec["py_warnings"])
    return rec

from sklearn.datasets import load_breast_cancer, load_diabetes
bc = load_breast_cancer(as_frame=True).frame
print("\n##### A. Both statements on every call (D63)")
run("breast cancer, nothing said", bc, "target", "stops: time_order_undeclared, and the message asks for the entities too")
run("breast cancer, time_col=False only", bc, "target", "stops: entities_undeclared", time_col=False)
run("breast cancer, group_col=False only", bc, "target", "stops: time_order_undeclared", group_col=False)
run("breast cancer, both said", bc, "target", "runs: 0.994549 as before", **NO)
vix = pd.read_csv(D + "vix.csv"); vix.columns = [c.lower() for c in vix.columns]
base = lagged(vix, ["close", "high", "low"], [1, 2, 5]); lv = base[["date", "close"] + [c for c in base.columns if "_lag" in c]].dropna()
run("VIX level, time_col=date, nothing said about entities", lv, "close", "stops: entities_undeclared, although it is one series", time_col="date")
run("VIX level, time_col=date, group_col=False", lv, "close", "0.023 as before", time_col="date", group_col=False)
st = pd.read_csv(D + "stocks.csv"); st["date"] = pd.to_datetime(st["date"], format="%b %d %Y"); st = st.sort_values(["symbol", "date"])
st["price_lag1"] = st.groupby("symbol")["price"].shift(1); lvl = st[["date", "symbol", "price", "price_lag1"]].dropna()
run("stocks price level, time_col=date, group_col=symbol", lvl, "price", "-0.139 as before", time_col="date", group_col="symbol")
run("stocks price level, time_col=date, group_col=False", lvl, "price", "0.979 on the caller's word, with the panel warning", time_col="date", group_col=False)
pen = pd.read_csv(D + "penguins.csv"); tit = pd.read_csv(D + "titanic.csv").drop(columns=["alive"])
run("diabetes, both said", load_diabetes(as_frame=True).frame, "target", "0.490034 as before", **NO)
run("penguins sex, both said", pen, "sex", "0.937642 as before", **NO)
run("diamonds, both said", D + "diamonds.csv", "price", "0.933327 as before", **NO)
run("titanic keep, both said", tit, "survived", "0.744865 as before", duplicates="keep", **NO)

print("\n##### B. duplicates='group' on the features alone (D62)")
run("titanic, default, both said", tit, "survived", "stops; the message names keep first and says when each fits", **NO)
run("titanic, duplicates=group", tit, "survived", "about 0.645 (the lead's patched run); was 0.741", duplicates="group", **NO)
sh = tit.copy(); sh["survived"] = np.random.default_rng(5).permutation(sh["survived"].to_numpy()); tw = pd.concat([sh, sh], ignore_index=True)
run("titanic shuffled, every row twice, group", tw, "survived", "about 0", duplicates="group", **NO)
for seed in (1, 2, 3):
    g = np.random.default_rng(seed); n = 4000
    d = pd.DataFrame({f"c{i}": g.integers(0, k, n).astype(str) for i, k in enumerate((2, 3, 4))}); d["y"] = (g.random(n) < 0.5).astype(int)
    run(f"noise, 24 feature combinations, seed {seed}, group", d, "y", "about 0; was -0.28, -0.27, -0.56; no interval (24 units)", duplicates="group", **NO)

print("\n##### C. The grouped holdout (D64, D66)")
mpg = pd.read_csv(D + "mpg.csv")
run("mpg, group_col=name", mpg, "mpg", "0.783 as before, holdout about a quarter", time_col=False, group_col="name")
m2 = mpg.copy(); m2["maker"] = np.where(m2["origin"] == "usa", "usa", m2["name"]); m2 = m2.drop(columns=["name", "origin"])
run("mpg, one entity of 62% (every US car), the rest by name", m2, "mpg", "holdout a quarter; large_group_kept_in_training names the 62%", time_col=False, group_col="maker")
tx = pd.read_csv(D + "taxis.csv"); tx = tx[tx["payment"].notna() & tx["pickup_zone"].notna() & tx["pickup_borough"].notna()].drop(columns=["pickup", "dropoff"])
t2 = tx.copy(); t2["area"] = np.where(t2["pickup_borough"] == "Manhattan", "Manhattan", t2["pickup_zone"]); t2 = t2.drop(columns=["pickup_zone", "pickup_borough"])
print("   (Manhattan share of the taxi rows: %.1f%%)" % (100 * (t2["area"] == "Manhattan").mean()))
run("taxis fare (regression), Manhattan as one entity, the rest by zone", t2.drop(columns=["total", "tip"]), "fare", "FAST profile: what does the 2,000-row cap do with an entity of 5,236 rows?", time_col=False, group_col="area")
run("the same, FULL profile", t2.drop(columns=["total", "tip"]), "fare", "Manhattan is the holdout and large_group_is_holdout says so", time_col=False, group_col="area", profile="full")
run("taxis payment (classification), same entities", t2, "payment", "a share per class; whichever case applies is named", time_col=False, group_col="area")

print("\n##### D. Dates written as whole numbers (D66)")
vd = lv.copy(); vd["day"] = pd.to_datetime(vd["date"]).dt.strftime("%Y%m%d").astype(int); vd = vd.drop(columns=["date"])
run("VIX level, day as 20240105, nothing said", vd, "close", "stops, naming `day` as a column that holds dates")
run("VIX level, day as a number, both False", vd, "close", "`day` excluded as a date, date_column_undeclared; 0.939 on the caller's word", **NO)
run("VIX level, time_col=day", vd, "close", "0.023, as with a date-typed column", time_col="day", group_col=False)
b3 = bc.copy(); b3["account"] = np.random.default_rng(1).integers(10_000_000, 99_999_999, len(b3))
run("breast cancer + a random 8-digit account number", b3, "target", "`account` is not a date; skill near 0.99", **NO)
b4 = bc.copy(); b4["yyyymm"] = np.random.default_rng(1).choice([202401, 202402, 202403], len(b4))
run("breast cancer + a YYYYMM column", b4, "target", "not seen as a date (stated limit): stays a feature", **NO)

print("\n##### E. The command line")
env = dict(os.environ, PYTHONPATH="/home/claude/realdata/app7")
import tempfile
for flags in ([], ["--no-time-order"], ["--no-entities"], ["--no-time-order", "--no-entities"], ["--time-col", "signup_date", "--no-entities"]):
    out = tempfile.mkdtemp()
    p = subprocess.run([sys.executable, "-m", "mlprofessor", "profile", "--data", D + "customers.csv", "--target", "churned", "--out", out, "--profile", "fast", "--n-jobs", "1", *flags], capture_output=True, text=True, env=env)
    msg = (p.stderr.strip().splitlines() or [""])[-1][:200] if p.returncode else " / ".join(p.stdout.strip().splitlines()[:2])[:200]
    print(f"   {flags}: exit {p.returncode}; wrote {sorted(os.listdir(out))}; {msg}")
json.dump(OUT, open("/home/claude/realdata/out/results_cp7.json", "w"), indent=1, default=str)
print("\nSUMMARY:", {k: sum(1 for r in OUT if r["outcome"] == k) for k in {r["outcome"] for r in OUT}}, "| python warnings in", sum(1 for r in OUT if r["py_warnings"]), "runs")
