"""Run the three repeat rules on real public tables: clean, and with copies added."""
import os, sys, time, warnings
import numpy as np, pandas as pd
import repeat_rules as rr
warnings.simplefilter("ignore")
D = os.environ.get("REAL_DATA", "data").rstrip("/\\") + os.sep

def tables():
    out = {}
    t = pd.read_csv(D + "titanic.csv").drop(columns=["alive"]); out["titanic (891)"] = (t, "survived", "classification")
    d = pd.read_csv(D + "diamonds.csv"); out["diamonds (53,940)"] = (d, "price", "regression")
    from sklearn.datasets import load_breast_cancer, load_diabetes
    out["breast cancer (569)"] = (load_breast_cancer(as_frame=True).frame, "target", "classification")
    out["diabetes (442)"] = (load_diabetes(as_frame=True).frame, "target", "regression")
    p = pd.read_csv(D + "penguins.csv").dropna(subset=["sex"]); out["penguins (333)"] = (p, "sex", "classification")
    c = pd.read_csv(D + "customers.csv").drop(columns=["customer_id", "signup_date"]); out["customers example (400)"] = (c, "churned", "classification")
    w = pd.read_csv(D + "weather.csv"); w["rain_tomorrow"] = (w["precipitation"].shift(-1) > 0).astype(int); w = w.iloc[:-1].drop(columns=["date"])
    out["Seattle weather (1,460)"] = (w, "rain_tomorrow", "classification")
    v = pd.read_csv(D + "vix.csv"); v.columns = [x.lower() for x in v.columns]
    v["change_next"] = v["close"].shift(-1) - v["close"]; v = v.dropna().drop(columns=["date"])
    out["VIX daily OHLC (9,277)"] = (v, "change_next", "regression")
    x = pd.read_csv(D + "taxis.csv").drop(columns=["pickup", "dropoff"]); x = x[x["payment"].notna()]
    out["NYC taxis (6,389)"] = (x, "payment", "classification")
    tp = pd.read_csv(D + "tips.csv"); out["tips (244)"] = (tp, "smoker", "classification")
    m = pd.read_csv(D + "mpg.csv").drop(columns=["name"]); out["mpg (398)"] = (m, "mpg", "regression")
    return out

def line(name, frame, target, task, **kw):
    t0 = time.time()
    try:
        f = rr.count_repeats(frame, target, task, seed=0, **kw)
    except Exception as e:
        print(f"{name:58s} ERROR {type(e).__name__}: {str(e)[:90]}"); return None
    c = f["calibrated"]; h = f["rows"]["holdout"] if isinstance(f["rows"], dict) else None
    s = lambda r: f"{f[r]['count']:5d} {'STOP' if f[r]['stop'] else 'run '}"
    print(f"{name:58s} holdout {h!s:>6}  d49 {s('d49')}  first {s('first_rule')}  calibrated {s('calibrated')}  (counted {c['counted']}, chance {c['chance']:.1f})  {time.time()-t0:.1f}s")
    return f

def twice(frame, extra):
    n = len(frame); out = pd.concat([frame, frame], ignore_index=True); rng = np.random.default_rng(1)
    if extra == "unique": out["load_id"] = rng.random(len(out))
    if extra == "batch": out["batch"] = np.repeat([1, 2], n)
    return out

def some_twice(frame, share):
    rng = np.random.default_rng(2); pick = rng.choice(len(frame), int(len(frame) * share), replace=False)
    out = pd.concat([frame, frame.iloc[pick]], ignore_index=True); out["load_id"] = rng.random(len(out)); return out

def oversample(frame, target, times):
    counts = frame[target].value_counts(); minority = counts.idxmin()
    extra = pd.concat([frame[frame[target] == minority]] * (times - 1), ignore_index=True)
    return pd.concat([frame, extra], ignore_index=True)

if __name__ == "__main__":
    which = sys.argv[1]
    T = tables()
    if which == "clean":
        print("CLEAN REAL TABLES, as they are. A stop here is a false stop unless the table truly holds repeats.")
        for k, (f, y, task) in T.items(): line(k, f, y, task)
    if which == "copies":
        print("REAL TABLES WITH COPIES ADDED. Every line here should stop.")
        for k, (f, y, task) in T.items():
            if len(f) > 20000: f = f.sample(8000, random_state=0).reset_index(drop=True)
            line(k + " | every row twice + unique col", twice(f, "unique"), y, task)
            line(k + " | every row twice + batch col", twice(f, "batch"), y, task)
            line(k + " | 5% of rows twice + unique col", some_twice(f, 0.05), y, task)
    if which == "over":
        print("OVERSAMPLING BEFORE THE SPLIT (the minority class copied k times). Every line here should stop.")
        for k in ["titanic (891)", "breast cancer (569)", "Seattle weather (1,460)", "NYC taxis (6,389)", "customers example (400)"]:
            f, y, task = T[k]
            for times in (2, 5, 10, 25, 50):
                line(f"{k} | minority x{times}", oversample(f, y, times), y, task)
        print("WHOLE TABLE REPEATED k TIMES")
        f, y, task = T["breast cancer (569)"]
        for times in (2, 10, 20, 21, 30):
            line(f"breast cancer | whole table x{times}", pd.concat([f] * times, ignore_index=True), y, task)
