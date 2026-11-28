"""The FX Phase 0 measurements of reports/fx_sources.md, repeatable.

    python inspect_ecb.py

Reads data/raw/ (the ECB files, the manifest and Bitstamp) and prints what
it finds. It writes nothing (ruling 26). It does not import the build's
modules: the FX columns are modelled here in their own code, so its section
on first rows is a model of the build, not the build.
"""
from __future__ import annotations

import io
import json
import math
import zipfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "raw"
CUR = ["USD", "JPY", "GBP", "CHF", "AUD", "CAD", "SEK"]
A_FIRST, B_FIRST = pd.Timestamp("2012-12-17"), pd.Timestamp("2019-12-09")
SIGNS = (("usdjpy", 1), ("usdchf", 1), ("usdcad", 1), ("eurusd", -1), ("gbpusd", -1),
         ("audusd", -1))


def easter(y: int) -> pd.Timestamp:
    """Gregorian Easter Sunday (the anonymous algorithm)."""
    a = y % 19
    b, c = divmod(y, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return pd.Timestamp(y, month, day + 1)


def holiday_name(d: pd.Timestamp) -> str | None:
    e = easter(d.year)
    return {pd.Timestamp(d.year, 1, 1): "1 Jan", e - timedelta(days=2): "Good Friday",
            e + timedelta(days=1): "Easter Monday", pd.Timestamp(d.year, 5, 1): "1 May",
            pd.Timestamp(d.year, 12, 25): "25 Dec", pd.Timestamp(d.year, 12, 26): "26 Dec",
            pd.Timestamp(d.year, 12, 31): "31 Dec"}.get(d)


def load() -> dict:
    api = pd.read_csv(RAW / "ecb" / "exr_daily.csv", dtype=str, keep_default_na=False)
    has = api["OBS_VALUE"].str.strip() != ""
    wide = api[has].pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE")[CUR]
    with zipfile.ZipFile(RAW / "ecb" / "eurofxref-hist.zip") as z:
        hist = pd.read_csv(io.BytesIO(z.read("eurofxref-hist.csv")), dtype=str,
                           keep_default_na=False)
    hist.columns = [c.strip() for c in hist.columns]
    candles = {}
    for p in sorted((RAW / "bitstamp").glob("ohlc_*.json")):
        for c in json.loads(p.read_bytes())["data"]["ohlc"]:
            candles[int(c["timestamp"])] = c
    bs = pd.DataFrame([candles[k] for k in sorted(candles)])
    bs.index = pd.to_datetime(bs["timestamp"].astype("int64"), unit="s")
    man = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    last = pd.Timestamp(man["bitstamp"]["last_complete_day"])
    bs = bs[bs.index <= last]
    return {"api": api, "has": has, "wide": wide, "hist": hist.set_index("Date"), "bs": bs,
            "manifest": man, "last": last}


def currencies(d: dict) -> None:
    api, has, wide = d["api"], d["has"], d["wide"]
    print("== Each currency (the API file)")
    for c in CUR:
        s = wide[c].dropna()
        print(f"  {c}: {s.index.min()}..{s.index.max()}, {len(s)} fixings")
    print("  statuses of rows with a value:", api[has]["OBS_STATUS"].value_counts().to_dict(),
          "| of rows without:", api[~has]["OBS_STATUS"].value_counts().to_dict())
    h = sorted(api[~has]["TIME_PERIOD"].unique())
    print(f"  holiday rows: {int((~has).sum())} on {len(h)} dates, {h[0]}..{h[-1]}")
    print("  dates with some but not all seven:", int(wide.isna().any(axis=1).sum()))
    z = d["hist"][CUR].apply(lambda col: col.str.strip())
    common = z.index.intersection(wide.index)
    same = int((z.loc[common] == wide.loc[common]).sum().sum())
    print(f"  the zip: {len(z)} rows; {len(common)} dates in common; {same} of {len(common) * 7} "
          f"values identical as text; API dates not in the zip: "
          f"{len(wide.index.difference(z.index)) + len(h)} (holiday rows included)")


def weekdays(d: dict) -> None:
    print("== Weekdays with no fixing, by year (a = API marks the date as a holiday)")
    fix = pd.DatetimeIndex(sorted(d["wide"].index))
    hdates = set(pd.DatetimeIndex(d["api"][~d["has"]]["TIME_PERIOD"].unique()))
    gaps = pd.bdate_range(fix.min(), fix.max()).difference(fix)
    for y in sorted(set(gaps.year)):
        days = [g for g in gaps if g.year == y]
        print(f"  {y}: {len(days)}: " + ", ".join(
            f"{g.strftime('%m-%d')} {holiday_name(g) or 'NOT A HOLIDAY'}{' a' if g in hdates else ''}"
            for g in days))
    print(f"  {len(gaps)} in all; not one of the listed holidays: "
          f"{[str(g.date()) for g in gaps if holiday_name(g) is None] or 'none'}")
    print(f"  fixings on a weekend: {int((fix.day_of_week >= 5).sum())}")


def publication(d: dict) -> None:
    print("== Publication")
    files = d["manifest"]["ecb"]["files"]
    for f in files[:2]:
        print(f"  {f['file']}: Last-Modified {f.get('response_headers', {}).get('Last-Modified')}")
    print("  TITLE_COMPL:", sorted(set(d["api"]["TITLE_COMPL"].str.extract(r", ([^,]*)$")[0])))
    by_day: dict[str, list] = {}
    for f in files[2:]:
        moment = f["updated_after"][:16].replace("T", " ")
        by_day.setdefault(f["period"], []).append((moment, f["rows"]))
    for day, answers in by_day.items():
        print(f"  {day}: " + "; ".join(f"after {t} {n}" for t, n in answers))


def series(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ln of the six pairs and dxy on the fixing dates, and the 43 modelled columns."""
    v = d["wide"].astype(float)
    v.index = pd.DatetimeIndex(v.index)
    L = pd.DataFrame({"eurusd": np.log(v.USD), "usdjpy": np.log(v.JPY / v.USD),
                      "gbpusd": np.log(v.USD / v.GBP), "usdchf": np.log(v.CHF / v.USD),
                      "audusd": np.log(v.USD / v.AUD), "usdcad": np.log(v.CAD / v.USD)})
    L["dxy"] = (-0.576 * L.eurusd + 0.136 * L.usdjpy - 0.119 * L.gbpusd + 0.091 * L.usdcad
                + 0.042 * np.log(v.SEK / v.USD) + 0.036 * L.usdchf)
    cols = {}
    for s in L.columns:
        x = L[s]
        for n in (1, 5, 20, 60):
            cols[f"fx_{s}_ret_{n}"] = x - x.shift(n)
        vol = x.diff().rolling(20).std(ddof=1)
        cols[f"fx_{s}_vol_20"] = vol
        lv = np.log(vol)
        cols[f"fx_{s}_vol_z_250"] = (lv - lv.rolling(250).mean()) / lv.rolling(250).std(ddof=1)
    bs = d["bs"]
    close = bs["close"].astype(float).where(bs["volume"].astype(float) > 0)
    cf = close.reindex(L.index)
    br = np.log(cf / cf.shift(1))
    dd = L["dxy"].diff()
    ok = dd.notna() & br.notna()
    count = ok.rolling(60).sum()
    corr = pd.Series(np.nan, index=L.index)
    for k in range(60, len(L)):
        if L.index[k - 60] < bs.index.min() or count.iloc[k] < math.ceil(0.9 * 60):
            continue
        m = ok.iloc[k - 59:k + 1].to_numpy()
        corr.iloc[k] = np.corrcoef(dd.iloc[k - 59:k + 1].to_numpy()[m],
                                   br.iloc[k - 59:k + 1].to_numpy()[m])[0, 1]
    cols["btc_dxy_corr_60"] = corr
    return L, pd.DataFrame(cols)


def moves(L: pd.DataFrame) -> None:
    print("== The ten largest daily log moves (F' > F)")
    dl = L.diff()
    prev = pd.Series(L.index[:-1], index=L.index[1:])
    for col in dl.columns:
        top = dl[col].dropna().abs().sort_values(ascending=False).head(10)
        print(f"  {col}: " + "; ".join(f"{prev[t].date()}>{t.date()} {dl.at[t, col]:+.4f}"
                                      for t in top.index))
    print("== Sign check of dxy (share of moves the expected way, both moving)")
    for label, start in (("all", L.index[0]), ("from 2011-08-18", pd.Timestamp("2011-08-18"))):
        part = dl[dl.index >= start]
        out = []
        for col, sign in SIGNS:
            a, b = part["dxy"], part[col]
            m = (a != 0) & (b != 0) & a.notna() & b.notna()
            out.append(f"{col} {float((np.sign(a[m]) == sign * np.sign(b[m])).mean()):.1%}")
        print(f"  {label} ({int(part['dxy'].notna().sum())} moves): " + ", ".join(out))


def first_rows(d: dict, fx: pd.DataFrame) -> pd.DataFrame:
    print("== First rows (model)")
    bs = d["bs"]
    cal = pd.date_range(bs.index.min(), d["last"], freq="D")
    present = (bs["volume"].astype(float) > 0).reindex(cal, fill_value=False).to_numpy()
    run, inside = 0, np.zeros(len(cal), dtype=bool)
    for i, p in enumerate(present):
        run = 0 if p else run + 1
        inside[i] = run >= 3
    pos = fx.index.searchsorted(cal, side="right") - 1
    rows = fx.iloc[pos].set_axis(cal)
    rows.loc[inside, "btc_dxy_corr_60"] = np.nan
    every = rows.notna().all(axis=1)
    print(f"  first fixing date with all 42 series columns: "
          f"{fx.drop(columns='btc_dxy_corr_60').notna().all(axis=1).idxmax().date()}; "
          f"first BTC day with all 43: {every.idxmax().date()}")
    for name, first in (("C", A_FIRST), ("D", B_FIRST)):
        part = rows[rows.index >= first]
        blank = part.isna().sum()
        lag = pd.Series((part.index - fx.index[pos[cal >= first]]).days)
        print(f"  table {name}: {part.index[0].date()}..{part.index[-1].date()}, {len(part)} rows; "
              f"blank FX cells {blank[blank > 0].to_dict() or 'none'} on "
              f"{[str(t.date()) for t in part.index[part.isna().any(axis=1)]]}; "
              f"rows by days since their fixing {lag.value_counts().sort_index().to_dict()}")
    return rows


def level_preview(rows: pd.DataFrame) -> None:
    print("== Level screen of the 43 modelled columns over table A's rows")
    part = rows[rows.index >= A_FIRST]
    n = int(math.floor(0.75 * len(part)))
    early, late = part.iloc[:n], part.iloc[n:]
    share = {c: float(((late[c] < early[c].min()) | (late[c] > early[c].max())).mean())
             for c in part.columns}
    top = sorted(share.items(), key=lambda kv: -kv[1])
    print(f"  above 10%: {[c for c, s in top if s > 0.10] or 'none'}; highest: "
          + ", ".join(f"{c} {s:.1%}" for c, s in top[:6]))


def main() -> int:
    d = load()
    currencies(d)
    weekdays(d)
    publication(d)
    L, fx = series(d)
    moves(L)
    rows = first_rows(d, fx)
    level_preview(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
