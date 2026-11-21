"""Real-data cases. Each states what a correct tool should conclude."""
import numpy as np
import pandas as pd


def lagged(df, cols, lags):
    out = df.copy()
    for c in cols:
        for k in lags:
            out[f"{c}_lag{k}"] = out[c].shift(k)
    return out


def main(run, D, which):
    groups = {
        "tabular": tabular, "time": time_series, "panel": panel, "edges": edges, "full": full_profile,
    }
    for name, fn in groups.items():
        if which in ("all", name):
            fn(run, D)


def tabular(run, D):
    tit = pd.read_csv(D + "titanic.csv")
    run("titanic CSV as shipped, target survived", D + "titanic.csv", "survived",
        "`alive` is a copy of the target: excluded as a leak. Many repeated rows: the copies rule may stop it")
    run("titanic CSV, duplicates=keep", D + "titanic.csv", "survived",
        "`alive` excluded as a leak; skill around 0.6 to 0.75 (AUC about 0.85)", duplicates="keep")
    run("titanic without `alive`, duplicates=keep", tit.drop(columns=["alive"]), "survived",
        "skill around 0.6 to 0.75; no leak flagged", duplicates="keep")
    rng = np.random.default_rng(7)
    shuf = tit.drop(columns=["alive"]).copy()
    shuf["survived"] = rng.permutation(shuf["survived"].to_numpy())
    run("titanic, target shuffled (control), duplicates=keep", shuf, "survived",
        "skill about zero, p not small", duplicates="keep")
    run("titanic, target `alive` (yes/no text)", tit.drop(columns=["survived"]), "alive",
        "same answer as survived; classes yes/no", duplicates="keep")

    from sklearn.datasets import load_breast_cancer, load_diabetes
    bc = load_breast_cancer(as_frame=True).frame
    run("breast cancer (sklearn), target", bc, "target",
        "legitimately very predictable (AUC about 0.99); nothing excluded as a leak; strong columns warned at most")
    db = load_diabetes(as_frame=True).frame
    run("diabetes (sklearn), regression target", db, "target",
        "moderate skill, about 0.4 to 0.5 over the mean")

    dia = pd.read_csv(D + "diamonds.csv")
    run("diamonds 53,940 rows, target price (fast)", D + "diamonds.csv", "price",
        "high skill (about 0.95+); carat strong; fast profile caps at 2,000 rows")
    pen = pd.read_csv(D + "penguins.csv")
    run("penguins, target species (3 classes)", pen, "species", "named error: three classes are not supported")
    run("penguins, target sex", pen, "sex", "rows with no sex dropped and counted; good skill")
    run("example customers.csv", D + "customers.csv", "churned", "runs; modest skill")


def time_series(run, D):
    vix = pd.read_csv(D + "vix.csv")
    vix.columns = [c.lower() for c in vix.columns]
    base = lagged(vix, ["close", "high", "low"], [1, 2, 5])
    lv = base[["date", "close"] + [c for c in base.columns if "_lag" in c]].dropna()
    run("VIX daily level, lag features, time_col=date", lv, "close",
        "headline baseline is previous_value; skill small (level is persistent)", time_col="date")
    run("VIX daily level, lag features, NO time_col", lv, "close",
        "random split with the date-undeclared warning; skill near 1 against the mean (the caveat case)")
    ch = base.copy()
    ch["change_next"] = ch["close"].shift(-1) - ch["close"]
    ch["chg1"] = ch["close"] - ch["close_lag1"]
    ch["dist_ma20"] = ch["close"] - ch["close"].rolling(20).mean()
    chd = ch[["date", "change_next", "chg1", "dist_ma20", "close"]].dropna()
    run("VIX next-day change, time_col=date", chd, "change_next",
        "skill about zero, perhaps slightly positive (mean reversion)", time_col="date")
    run("VIX next-day change, time_col=date, gap=1", chd, "change_next",
        "as above; gap rows counted", time_col="date", gap=1)

    run("sp500.csv as shipped (dates like 'Jan 1 2000'), time_col=date", D + "sp500.csv", "price",
        "named error: the time column is not year-first ISO", time_col="date")
    sp = pd.read_csv(D + "sp500.csv")
    sp["date"] = pd.to_datetime(sp["date"], format="%b %d %Y")
    sp["ret_next"] = sp["price"].pct_change().shift(-1)
    sp["ret1"] = sp["price"].pct_change()
    sp["ret3"] = sp["price"].pct_change(3)
    spd = sp[["date", "ret_next", "ret1", "ret3"]].dropna()
    run("S&P 500 monthly next return (123 rows), time_col=date", spd, "ret_next",
        "skill about zero; small-holdout warnings; possibly no interval (few timestamps)", time_col="date")

    w = pd.read_csv(D + "weather.csv")
    wl = lagged(w, ["temp_max", "temp_min", "precipitation", "wind"], [1, 2, 7])
    wl = wl[["date", "temp_max"] + [c for c in wl.columns if "_lag" in c]].dropna()
    run("Seattle weather, temp_max from lags, time_col=date (ISO text)", wl, "temp_max",
        "previous_value baseline; modest positive skill over persistence", time_col="date")
    wr = w.copy()
    wr["rain_tomorrow"] = (wr["precipitation"].shift(-1) > 0).astype("float")
    wr = wr.iloc[:-1]
    wr["rain_tomorrow"] = wr["rain_tomorrow"].astype(int)
    run("Seattle weather, rain tomorrow (binary), time_col=date", wr.drop(columns=[]), "rain_tomorrow",
        "moderate skill (AUC about 0.7 to 0.8)", time_col="date")


def panel(run, D):
    st = pd.read_csv(D + "stocks.csv")
    st["date"] = pd.to_datetime(st["date"], format="%b %d %Y")
    st = st.sort_values(["symbol", "date"])
    st["price_lag1"] = st.groupby("symbol")["price"].shift(1)
    st["ret_next"] = st.groupby("symbol")["price"].pct_change().shift(-1)
    st["ret1"] = st.groupby("symbol")["price"].pct_change()
    lvl = st[["date", "symbol", "price", "price_lag1"]].dropna()
    run("stocks panel (5 symbols), price level with own lag, time_col=date", lvl, "price",
        "panel: baseline is the training mean and the autocorrelation warning fires; "
        "the lag may be flagged as a leak (known limit D31); skill would be inflated otherwise", time_col="date")
    ret = st[["date", "symbol", "ret_next", "ret1"]].dropna()
    run("stocks panel, next-month return, time_col=date", ret, "ret_next",
        "skill about zero", time_col="date")


def edges(run, D):
    tit = pd.read_csv(D + "titanic.csv")
    t2 = tit.drop(columns=["alive"]).copy()
    t2["fare_text"] = t2["fare"].map(lambda v: f"{v * 1000:,.2f}")
    run("titanic, fare written with thousands separators", t2, "survived",
        "fare_text excluded as numeric_with_separators, with a warning", duplicates="keep")
    t3 = tit.drop(columns=["alive"]).copy()
    t3["passenger_id"] = np.arange(1, len(t3) + 1)
    run("titanic plus a consecutive id column", t3, "survived", "passenger_id excluded as id_like", duplicates="keep")
    twice = pd.concat([tit.drop(columns=["alive"]), tit.drop(columns=["alive"])], ignore_index=True)
    run("titanic stacked twice", twice, "survived", "stops: holdout copies in training")
    twice["batch"] = np.repeat([1, 2], len(tit))
    run("titanic stacked twice with a batch column (review 4, finding 2)", twice, "survived",
        "SHOULD stop; today the batch column hides the copies")
    vix = pd.read_csv(D + "vix.csv")
    vix.columns = [c.lower() for c in vix.columns]
    run("VIX same-day: close from open/high/low, time_col=date", vix, "close",
        "very high skill, legitimately (same-day range); high/low may be warned as strong single columns",
        time_col="date")


def full_profile(run, D):
    tit = pd.read_csv(D + "titanic.csv").drop(columns=["alive"])
    a = run("titanic, FULL profile, n_jobs=2", tit, "survived", "skill near the fast figure", duplicates="keep",
            profile="full", n_jobs=2)
    b = run("titanic, FULL profile, n_jobs=1", tit, "survived", "identical to n_jobs=2", duplicates="keep",
            profile="full", n_jobs=1)
    if a["outcome"] == b["outcome"] == "ran":
        da, dbb = a["_res"].to_dict(), b["_res"].to_dict()
        same = (da["headline"] == dbb["headline"]) and a["_res"].predictions.equals(b["_res"].predictions)
        print(f"  >>> n_jobs 1 vs 2: headline and predictions identical: {same}")
    dia = pd.read_csv(D + "diamonds.csv").sample(8000, random_state=1).reset_index(drop=True)
    run("diamonds 8,000-row sample, FULL profile", dia, "price", "skill about 0.97+", profile="full", n_jobs=2)
