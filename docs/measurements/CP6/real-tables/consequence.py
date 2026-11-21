"""Prototype, lead's own: instead of guessing whether matching rows are copies, measure whether
the score depends on them. Twins = holdout rows with a training row equal in the target and in
every column but at most one (the same column for the whole table), numbers compared to 6
significant digits. Then refit with the matched training rows removed and compare the skill."""
import sys, warnings, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from real_tables import tables, twice, some_twice, oversample
warnings.simplefilter("ignore")

def keys(frame, cols):
    parts = []
    for c in cols:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            v = s.astype(float).to_numpy()
            parts.append(pd.Series(np.char.mod("%.6g", v), index=frame.index))
        else:
            parts.append(s.astype(object).where(s.notna(), "<NA>").astype(str))
    return pd.util.hash_pandas_object(pd.concat(parts, axis=1), index=False).to_numpy()

def twins(train, hold, target):
    cols = list(train.columns); best = (None, np.zeros(len(hold), bool), np.zeros(len(train), bool))
    exact = None
    for drop in [None] + [c for c in cols if c != target]:
        use = [c for c in cols if c != drop]
        kt, kh = keys(train, use), keys(hold, use)
        mh = np.isin(kh, kt); mt = np.isin(kt, kh[mh])
        if drop is None: exact = int(mh.sum())
        if mh.sum() > best[1].sum(): best = (drop, mh, mt)
    return exact, best

def skill(train, hold, target, task, seed=0):
    X = pd.get_dummies(pd.concat([train, hold]).drop(columns=[target]), dummy_na=True).astype(float)
    X = X.fillna(X.iloc[:len(train)].median()).fillna(0.0)
    Xt, Xh = X.iloc[:len(train)], X.iloc[len(train):]
    yt, yh = train[target], hold[target]
    if task == "classification":
        if yt.nunique() < 2: return None, None
        m = RandomForestClassifier(200, n_jobs=2, random_state=seed).fit(Xt, yt)
        p = m.predict_proba(Xh)[:, list(m.classes_).index(sorted(yh.unique())[-1])] if sorted(yh.unique())[-1] in m.classes_ else np.zeros(len(yh))
        pos = (yh == sorted(yh.unique())[-1]).to_numpy()
        return p, lambda idx: (roc_auc_score(pos[idx], p[idx]) - 0.5) / 0.5 if 0 < pos[idx].sum() < len(idx) else np.nan
    m = RandomForestRegressor(200, n_jobs=2, random_state=seed).fit(Xt, yt)
    p = m.predict(Xh); y = yh.to_numpy(float); base = float(yt.mean())
    return p, lambda idx: 1 - mean_squared_error(y[idx], p[idx]) / mean_squared_error(y[idx], np.full(len(idx), base))

def case(name, frame, target, task, seed=0):
    t0 = time.time()
    frame = frame.dropna(subset=[target]).reset_index(drop=True)
    strat = frame[target] if task == "classification" else None
    tr, ho = train_test_split(frame, test_size=0.25, random_state=seed, stratify=strat)
    exact, (drop, mh, mt) = twins(tr, ho, target)
    _, s0 = skill(tr, ho, target, task)
    tr2 = tr[~mt]
    _, s1 = skill(tr2, ho, target, task) if len(tr2) >= 50 else (None, None)
    idx = np.arange(len(ho)); a = s0(idx); b = s1(idx) if s1 else np.nan
    rng = np.random.default_rng(0); d = []
    if s1:
        for _ in range(300):
            i = rng.integers(0, len(ho), len(ho)); d.append(s0(i) - s1(i))
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
    else: lo = hi = np.nan
    print(f"{name:52s} holdout {len(ho):6d} exact {exact:5d} twins {int(mh.sum()):5d} ({mh.mean():6.1%}) col left out {str(drop):12.12s} train removed {mt.mean():6.1%} | skill {a:6.3f} -> {b:6.3f}  drop {a-b:6.3f} [{lo:6.3f},{hi:6.3f}]  {time.time()-t0:.0f}s", flush=True)

def noise(frame, target, seed=5):
    out = frame.copy(); out[target] = np.random.default_rng(seed).permutation(out[target].to_numpy()); return out

def jitter(frame, target):
    out = pd.concat([frame, frame], ignore_index=True); rng = np.random.default_rng(3)
    for c in out.columns:
        if c != target and pd.api.types.is_float_dtype(out[c]): out[c] = out[c] * (1 + 1e-9 * rng.standard_normal(len(out)))
    return out

if __name__ == "__main__":
    T = tables(); which = sys.argv[1]
    small = lambda k: (T[k][0].sample(8000, random_state=0).reset_index(drop=True) if len(T[k][0]) > 20000 else T[k][0])
    if which == "clean":
        print("CLEAN REAL TABLES: the drop should be about nothing")
        for k, (f, y, task) in T.items(): case(k, f, y, task)
        rng = np.random.default_rng(0); n = 8000
        d = pd.DataFrame({f"c{i}": rng.integers(0, 3, n) for i in range(8)}).astype(str)
        d["y"] = ((d.c0 == "1") ^ (d.c1 == "2") ^ (rng.random(n) < 0.2)).astype(int)
        case("made-up: 8 three-level columns, 8,000 rows", d, "y", "classification")
    if which == "copies":
        print("COPIES ON A TARGET WITH NO SIGNAL (the target shuffled first): the true skill is 0, so any skill before is memory")
        for k in ["titanic (891)", "diabetes (442)", "Seattle weather (1,460)", "NYC taxis (6,389)", "VIX daily OHLC (9,277)", "diamonds (53,940)", "breast cancer (569)"]:
            f, y, task = small(k), T[k][1], T[k][2]; z = noise(f, y)
            case(k + " | noise, as is", z, y, task)
            case(k + " | noise, twice + unique col", twice(z, "unique"), y, task)
            case(k + " | noise, twice + batch col", twice(z, "batch"), y, task)
            case(k + " | noise, twice, numbers moved 1e-9", jitter(z, y), y, task)
            case(k + " | noise, 5% twice + unique col", some_twice(z, 0.05), y, task)
            if task == "classification": case(k + " | noise, minority x25", oversample(z, y, 25), y, task)
    if which == "real":
        print("COPIES ON THE REAL TARGET")
        for k in ["titanic (891)", "diabetes (442)", "Seattle weather (1,460)", "NYC taxis (6,389)", "breast cancer (569)", "customers example (400)"]:
            f, y, task = small(k), T[k][1], T[k][2]
            case(k + " | twice + unique col", twice(f, "unique"), y, task)
            if task == "classification": case(k + " | minority x25", oversample(f, y, 25), y, task)
