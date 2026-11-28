"""Build the BTC daily tables A to D from the raw downloads.

    python build_dataset.py [--raw FOLDER] [--out FOLDER] [--k VALUE] [--k-unchecked]

With no options it reads data/raw/, writes data/built/ and data/profiler/,
and computes k by rule 5. --raw reads another raw folder of the same layout
(with its own manifest.json); --out writes built/ and profiler/ under
another folder; --k fixes k. Rule 5 is computed and recorded either way,
and with --k the build stops if rule 5 gives another value (ruling 22).
--k-unchecked turns that stop off. It exists for check 2's copies of the raw
data cut at earlier dates, whose shorter history gives rule 5 another value,
and it is refused when --raw is data/raw itself. Paths resolve from this
file, so it runs from any folder. It never touches the network.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True  # leave no __pycache__ beside the scripts

import features  # noqa: E402
import labels  # noqa: E402

DATA = HERE.parent / "data"
REAL_RAW = DATA / "raw"
ECB_FILE = ("ecb", "exr_daily.csv")   # ruling 21: the only ECB file the build reads
DAY = pd.Timedelta(days=1)
TABLE_A_FIRST = pd.Timestamp("2012-12-17")   # ruling 12
TABLE_B_FIRST = pd.Timestamp("2019-12-09")   # ruling 13
CM_LIVE_FROM = pd.Timestamp("2021-02-24")    # ruling 15
CM_METRICS = features.ONCHAIN + ("CapMVRVCur", "AssetEODCompletionTime")
FUNDING_HOURS = (0, 8, 16)
TARGETS = ("up_7d", "down_7d")
LABEL_COLUMNS = ["cm_day", "fwd_ret_7", "z_7", "class_7", "up_7d", "down_7d"]


class BuildError(RuntimeError):
    """The raw data or the result breaks a rule."""


# ------------------------------------------------------------------ raw --

def read_manifest(raw: Path) -> dict:
    return json.loads((raw / "manifest.json").read_text(encoding="utf-8"))


def load_bitstamp(raw: Path, last_day: pd.Timestamp) -> pd.DataFrame:
    """Daily candles by date; zero-volume days are missing days (ruling 10)."""
    candles: dict[int, dict] = {}
    for path in sorted((raw / "bitstamp").glob("ohlc_*.json")):
        for candle in json.loads(path.read_bytes())["data"]["ohlc"]:
            stamp = int(candle["timestamp"])
            if stamp % 86400:
                raise BuildError(f"Bitstamp candle not at 00:00 UTC: {stamp}")
            if stamp in candles and candles[stamp] != candle:
                raise BuildError(f"Bitstamp pages disagree on {stamp}")
            candles[stamp] = candle
    if not candles:
        raise BuildError("no Bitstamp candles")
    frame = pd.DataFrame([candles[s] for s in sorted(candles)])
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"].astype("int64"), unit="s"),
                                   name="date").as_unit("ns")
    frame = frame[["open", "high", "low", "close", "volume"]].astype(float)
    frame = frame.loc[frame.index <= last_day]
    frame.loc[frame["volume"] == 0, :] = np.nan
    return frame


def load_coinmetrics(raw: Path, last_day: pd.Timestamp) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for path in sorted((raw / "coinmetrics").glob("series_*.json")):
        for row in json.loads(path.read_bytes())["data"]:
            rows.setdefault(row["time"], {}).update(row)
    frame = pd.DataFrame([rows[k] for k in sorted(rows)])
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame["time"].str[:10]), name="date").as_unit("ns")
    frame = frame.reindex(columns=list(CM_METRICS))
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame.loc[frame.index <= last_day - DAY]


def load_klines(raw: Path, stem: str, last_day: pd.Timestamp) -> pd.DataFrame:
    rows: dict[int, list] = {}
    for path in sorted((raw / "binance").glob(f"{stem}_*.json")):
        for row in json.loads(path.read_bytes()):
            if row[0] in rows and rows[row[0]] != row:
                raise BuildError(f"Binance {stem} pages disagree on {row[0]}")
            rows[row[0]] = row
    columns = ["close", "quote_volume", "trades", "taker_buy_quote"]
    if not rows:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"), dtype=float)
    keys = sorted(rows)
    frame = pd.DataFrame({
        "close": [float(rows[k][4]) for k in keys],
        "quote_volume": [float(rows[k][7]) for k in keys],
        "trades": [float(rows[k][8]) for k in keys],
        "taker_buy_quote": [float(rows[k][10]) for k in keys],
    }, index=pd.DatetimeIndex(pd.to_datetime(np.array(keys, dtype="int64"), unit="ms"),
                              name="date").as_unit("ns"))
    return frame.loc[frame.index <= last_day]


def load_funding(raw: Path, last_day: pd.Timestamp) -> pd.Series:
    """fund_1d: the sum of the prints at 00, 08 and 16 UTC of each day.

    Each print's time is floored to the hour first. A day has a value only
    with exactly one print at each of the three hours.
    """
    prints: dict[int, float] = {}
    for path in sorted((raw / "binance").glob("funding_*.json")):
        for row in json.loads(path.read_bytes()):
            prints[int(row["fundingTime"])] = float(row["fundingRate"])
    if not prints:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="date"))
    times = np.array(sorted(prints), dtype="int64")
    hours = pd.to_datetime((times // 3_600_000) * 3_600_000, unit="ms")
    frame = pd.DataFrame({"rate": [prints[t] for t in times], "day": hours.normalize(),
                          "hour": hours.hour})
    frame = frame[frame["day"] <= last_day]
    regular = frame[frame["hour"].isin(FUNDING_HOURS)]
    counts = regular.groupby(["day", "hour"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(FUNDING_HOURS), fill_value=0)
    complete = counts.index[(counts == 1).all(axis=1)]
    sums = regular[regular["day"].isin(complete)].groupby("day")["rate"].sum()
    sums.index = pd.DatetimeIndex(sums.index, name="date").as_unit("ns")
    return sums


def load_ecb(raw: Path, last_day: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """The ECB reference rates: the seven currencies per EUR, by fixing date.

    Rows with no value are the API's holiday rows: they are dropped, since
    they are not fixing dates (ruling 21). A fixing date that lacks one of
    the seven keeps a blank there, so it has no value in any series.
    """
    path = raw.joinpath(*ECB_FILE)
    if not path.exists():
        raise BuildError(f"{path} is missing: tables C and D need the ECB rates")
    rows = pd.read_csv(path, dtype=str, keep_default_na=False,
                       usecols=["CURRENCY", "CURRENCY_DENOM", "TIME_PERIOD", "OBS_VALUE"])
    if set(rows["CURRENCY_DENOM"]) - {"EUR"}:
        raise BuildError("an ECB rate is not against EUR")
    empty = rows["OBS_VALUE"].str.strip() == ""
    holidays = rows[empty]
    rows = rows[~empty & rows["CURRENCY"].isin(features.ECB_CURRENCIES)]
    if rows.duplicated(["CURRENCY", "TIME_PERIOD"]).any():
        raise BuildError("an ECB currency has two values for one date")
    wide = rows.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE")
    wide = wide.reindex(columns=list(features.ECB_CURRENCIES)).astype(float)
    wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index), name="date").as_unit("ns")
    wide = wide.sort_index()
    wide = wide.loc[wide.index <= last_day]
    if (wide <= 0).any().any():
        raise BuildError("an ECB rate is zero or negative")
    kept = holidays[pd.to_datetime(holidays["TIME_PERIOD"]) <= last_day]
    info = {"file": "/".join(ECB_FILE),
            "fixing_dates": int(len(wide)),
            "first_fixing": wide.index.min().strftime("%Y-%m-%d") if len(wide) else None,
            "last_fixing": wide.index.max().strftime("%Y-%m-%d") if len(wide) else None,
            "holiday_rows_dropped": int(len(kept)),
            "holiday_dates_dropped": int(kept["TIME_PERIOD"].nunique())}
    return wide, info


def cm_asof(cm: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    """Ruling 15: on row t, the latest day D <= t-1 known at the close of t.

    From 2021-02-24 day D is known at the close of the first day whose close
    (00:00 UTC of the next day) is at or after AssetEODCompletionTime; before
    that it is known at the close of D+1.
    """
    days = cm.index
    known = days + DAY
    done = pd.to_datetime(cm["AssetEODCompletionTime"], unit="s")
    live = days >= CM_LIVE_FROM
    by_completion = pd.DatetimeIndex(done.dt.ceil("D")) - DAY
    known = known.where(~live, np.maximum(known, by_completion))
    ok = ~(live & done.isna().to_numpy())
    start = calendar[0]
    asof = np.full(len(calendar), -1, dtype=np.int64)
    day_pos = ((days - start) // DAY).to_numpy()
    known_pos = ((known - start) // DAY).to_numpy()
    for d, kpos, good in zip(day_pos, known_pos, ok):
        if not good or kpos >= len(calendar):
            continue
        kpos = max(int(kpos), 0)
        asof[kpos] = max(asof[kpos], int(d))
    asof = np.maximum.accumulate(asof)
    by_pos = cm.copy()
    by_pos.index = day_pos
    values = by_pos.reindex(np.where(asof >= 0, asof, -10 ** 9))
    values.index = calendar
    asof_day = pd.Series(pd.NaT, index=calendar, dtype="datetime64[ns]")
    asof_day[asof >= 0] = start + pd.to_timedelta(asof[asof >= 0], unit="D")
    return values, asof_day


# ---------------------------------------------------------------- build --

def build(raw: Path, k_given: float | None, k_unchecked: bool = False) -> dict:
    manifest = read_manifest(raw)
    last_day = pd.Timestamp(manifest["bitstamp"]["last_complete_day"])
    bs = load_bitstamp(raw, last_day)
    cm = load_coinmetrics(raw, last_day)
    spot = load_klines(raw, "spot_klines", last_day)
    perp = load_klines(raw, "perp_klines", last_day)
    funding = load_funding(raw, last_day)

    start = min(bs.index.min(), cm.index.min())
    calendar = pd.date_range(start, last_day, freq="D", name="date").as_unit("ns")
    beyond = calendar[-1] + DAY
    firsts = {"spot": spot.index.min() if len(spot) else beyond,
              "perp": perp.index.min() if len(perp) else beyond,
              "funding": funding.index.min() if len(funding) else beyond}

    ohlcv = bs.reindex(calendar)
    table = features.Table(calendar)
    info = features.bitstamp_columns(table, ohlcv, bs.index.min())
    sigma30 = pd.Series(info["sigma30"], index=calendar)
    fwd = labels.forward(ohlcv["close"], sigma30)

    a_rows = (calendar >= TABLE_A_FIRST) & (calendar <= last_day)
    b_rows = (calendar >= TABLE_B_FIRST) & (calendar <= last_day)
    # Rule 5 is computed and recorded whether or not k is given (ruling 22).
    with_z = int(fwd["z_7"][a_rows].notna().sum())
    k_rule = labels.rule5_k(fwd["z_7"][a_rows]) if math.floor(0.75 * with_z) >= 1 else None
    if k_given is None:
        if k_rule is None:
            raise BuildError("table A has no rows with a z, so k cannot be computed: pass --k")
        k = k_rule["k"]
    else:
        k = float(k_given)
        if k_rule is not None and k_rule["k"] != k and not k_unchecked:
            raise BuildError(f"rule 5 gives k = {k_rule['k']} on this data, not the k given "
                             f"({k}). k is fixed (ruling 22): check the raw data and --k")
    classes = labels.classify(fwd["z_7"], k)

    features.past_class_column(table, k)
    features.calendar_columns(table)
    asof, asof_day = cm_asof(cm, calendar)
    features.onchain_columns(table, asof)
    features.binance_columns(table, spot.reindex(calendar), perp.reindex(calendar),
                             funding.reindex(calendar), info["source"],
                             ohlcv["close"].to_numpy(dtype=float), firsts)
    rates, ecb_info = load_ecb(raw, last_day)
    fix_dates = rates.index
    fx_table = features.Table(fix_dates)
    logs, fx_present = features.fx_logs(rates)
    btc_close = ohlcv["close"].reindex(fix_dates).to_numpy(dtype=float)
    features.fx_columns(fx_table, logs, fx_present, btc_close,
                        np.asarray(fix_dates >= bs.index.min()))
    fx_day = features.carry_fx(table, fx_table, fix_dates, info["source"].inside)
    ecb_info["dates_missing_a_currency"] = [d.strftime("%Y-%m-%d")
                                            for d in fix_dates[~fx_present]]
    problems = table.problems + fx_table.problems
    if problems:
        raise BuildError("values that are not finite where their inputs exist:\n  "
                         + "\n  ".join(problems))

    cols = features.D_COLUMNS
    frame = pd.DataFrame({c: table.data[c] for c in cols}, index=calendar)
    for c in cols:
        if np.isinf(frame[c].to_numpy()).any():
            raise BuildError(f"{c} holds an infinite value")
    for c in features.INTEGER_COLUMNS:
        frame[c] = frame[c].round().astype("Int64")
    frame["cm_day"] = asof_day.dt.strftime("%Y-%m-%d")
    frame["fx_day"] = fx_day.dt.strftime("%Y-%m-%d")
    frame = pd.concat([frame, fwd, classes], axis=1)

    for name, rows, columns, first in (("A", a_rows, features.A_COLUMNS, TABLE_A_FIRST),
                                       ("B", b_rows, features.B_COLUMNS, TABLE_B_FIRST),
                                       ("C", a_rows, features.C_COLUMNS, TABLE_A_FIRST),
                                       ("D", b_rows, features.D_COLUMNS, TABLE_B_FIRST)):
        if rows.any():
            missing = [c for c in columns if pd.isna(frame.at[first, c])]
            if missing:
                raise BuildError(f"table {name}'s first row {first.date()} lacks "
                                 f"{', '.join(missing)}")
    return {"frame": frame, "a_rows": a_rows, "b_rows": b_rows, "k": k, "k_rule": k_rule,
            "last_day": last_day, "firsts": firsts, "calendar": calendar,
            "asof_day": asof_day, "fx_day": fx_day, "ecb_info": ecb_info}


# --------------------------------------------------------------- output --

def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.to_csv(index=False, lineterminator="\n", na_rep="")
    path.write_bytes(text.encode("utf-8"))


def blank_runs(frame: pd.DataFrame, columns) -> list[dict]:
    """Runs of rows with at least one blank feature, and the columns blank there."""
    blank = frame[columns].isna()
    rows = blank.index[blank.any(axis=1)]
    out, start, prev = [], None, None
    for day in list(rows) + [None]:
        if day is not None and prev is not None and day == prev + DAY:
            prev = day
            continue
        if start is not None:
            block = blank.loc[start:prev]
            out.append({"first": start.strftime("%Y-%m-%d"), "last": prev.strftime("%Y-%m-%d"),
                        "rows": len(block),
                        "columns": [c for c in columns if block[c].any()]})
        start = prev = day
    return out


def rule8_first(frame: pd.DataFrame, columns) -> str | None:
    full = frame[columns].notna().all(axis=1)
    return full.index[full][0].strftime("%Y-%m-%d") if full.any() else None


def table_info(t: pd.DataFrame, columns) -> dict:
    if not len(t):
        return {"rows": 0}
    target = t["up_7d"].notna()
    blanks = t[columns].isna().sum()
    return {"first": t.index[0].strftime("%Y-%m-%d"), "last": t.index[-1].strftime("%Y-%m-%d"),
            "rows": len(t), "rows_with_target": int(target.sum()),
            "rows_without_target": int((~target).sum()),
            "blank_cells": {c: int(n) for c, n in blanks.items() if n},
            "blank_runs": blank_runs(t, columns)}


def summary(result: dict) -> dict:
    """build_info.json: tables A and B, k and rule 5 (unchanged by tables C and D)."""
    frame, a_rows, b_rows = result["frame"], result["a_rows"], result["b_rows"]
    a, b = frame[a_rows], frame[b_rows]
    info = {
        "k": result["k"],
        "k_rule": result["k_rule"],
        "last_day": result["last_day"].strftime("%Y-%m-%d"),
        "source_first_days": {k: v.strftime("%Y-%m-%d") for k, v in result["firsts"].items()},
        "table_a": table_info(a, features.A_COLUMNS),
        "table_b": table_info(b, features.B_COLUMNS),
        "rule8_first_day": {"a": rule8_first(frame, features.A_COLUMNS),
                            "b": rule8_first(frame, features.B_COLUMNS)},
    }
    if result["k_rule"] is not None and len(a):
        cls = a["class_7"]
        have = cls.dropna()
        n_early = result["k_rule"]["rows"]
        by_year = {str(y): labels.shares(g) for y, g in have.groupby(have.index.year)}
        info["class_shares"] = {"earliest_75": labels.shares(have.iloc[:n_early]),
                                "latest_25": labels.shares(have.iloc[n_early:]),
                                "all": labels.shares(have), "by_year": by_year}
    lagged = (a.index - pd.to_datetime(a["cm_day"])) != DAY
    info["cm_rows_not_t_minus_1"] = [
        {"row": d.strftime("%Y-%m-%d"), "cm_day": a.at[d, "cm_day"]} for d in a.index[lagged]]
    return info


def summary_cd(result: dict) -> dict:
    """build_info_cd.json: tables C and D and the ECB rates (ruling 22)."""
    frame, a_rows, b_rows = result["frame"], result["a_rows"], result["b_rows"]
    fx_day = result["fx_day"]

    def days_back(rows) -> dict:
        lag = pd.Series((frame.index[rows] - pd.DatetimeIndex(fx_day[rows])).days)
        return {str(int(n)): int(c) for n, c in lag.value_counts().sort_index().items()}

    return {
        "k": result["k"],
        "last_day": result["last_day"].strftime("%Y-%m-%d"),
        "ecb": result["ecb_info"],
        "fx_columns": list(features.FX_COLUMNS),
        "table_c": table_info(frame[a_rows], features.C_COLUMNS),
        "table_d": table_info(frame[b_rows], features.D_COLUMNS),
        "rule8_first_day": {"c": rule8_first(frame, features.C_COLUMNS),
                            "d": rule8_first(frame, features.D_COLUMNS)},
        "rows_by_days_since_their_fixing": {"c": days_back(a_rows), "d": days_back(b_rows)},
    }


def write_outputs(result: dict, out: Path) -> dict:
    frame = result["frame"]
    built, profiler = out / "built", out / "profiler"
    full = frame[result["a_rows"]]
    dates = full.index.strftime("%Y-%m-%d")
    full_out = pd.concat([pd.DataFrame({"date": dates}, index=full.index),
                          full[features.B_COLUMNS + LABEL_COLUMNS]], axis=1)
    write_csv(full_out, built / "btc_daily_full.csv")
    fx_out = pd.concat([pd.DataFrame({"date": dates}, index=full.index),
                        full[["fx_day"] + features.FX_COLUMNS]], axis=1)
    write_csv(fx_out, built / "fx_daily.csv")
    for name, rows, columns in (("a", result["a_rows"], features.A_COLUMNS),
                                ("b", result["b_rows"], features.B_COLUMNS),
                                ("c", result["a_rows"], features.C_COLUMNS),
                                ("d", result["b_rows"], features.D_COLUMNS)):
        part = frame[rows]
        for target in TARGETS:
            out_frame = pd.concat([pd.DataFrame({"date": part.index.strftime("%Y-%m-%d")},
                                                index=part.index),
                                   part[columns + [target]]], axis=1)
            write_csv(out_frame, profiler / f"{name}_{target}.csv")
    info = summary(result)
    text = json.dumps(info, indent=2, sort_keys=True) + "\n"
    (built / "build_info.json").write_bytes(text.encode("utf-8"))
    info_cd = summary_cd(result)
    text = json.dumps(info_cd, indent=2, sort_keys=True) + "\n"
    (built / "build_info_cd.json").write_bytes(text.encode("utf-8"))
    return {**info, "cd": info_cd}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=DATA / "raw")
    parser.add_argument("--out", type=Path, default=DATA)
    parser.add_argument("--k", type=float, default=None)
    parser.add_argument("--k-unchecked", action="store_true",
                        help="with --k: do not stop when rule 5 gives another k (check 2's "
                             "cut copies only; refused on data/raw)")
    args = parser.parse_args(argv)
    raw = args.raw.resolve()
    if args.k_unchecked:
        real = REAL_RAW.resolve()
        same = raw == real or (raw.exists() and real.exists() and os.path.samefile(raw, real))
        if args.k is None:
            print("BUILD FAILED: --k-unchecked needs --k", file=sys.stderr)
            return 1
        if same:
            print(f"BUILD FAILED: --k-unchecked is refused on {real}: no build of the real "
                  f"data may skip the k check (ruling 22)", file=sys.stderr)
            return 1
    try:
        result = build(raw, args.k, args.k_unchecked)
        info = write_outputs(result, args.out.resolve())
    except BuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 1
    rule = info["k_rule"]
    text = f"k = {info['k']} ({'given' if args.k is not None else 'rule 5'}"
    if rule:
        text += (f"; rule 5: 1/3 quantile {rule['quantile']:.6f} over {rule['rows']} rows, "
                 f"{rule['first']}..{rule['last']}, rounded {rule['k']}")
    print(text + ")")
    cd = info["cd"]
    a, b, c, d = info["table_a"], info["table_b"], cd["table_c"], cd["table_d"]
    for name, t in (("A", a), ("B", b), ("C", c), ("D", d)):
        if t["rows"]:
            print(f"table {name}: {t['first']}..{t['last']}, {t['rows']} rows, "
                  f"{t['rows_with_target']} with a target, "
                  f"{sum(t['blank_cells'].values())} blank feature cells on "
                  f"{sum(r['rows'] for r in t['blank_runs'])} rows")
        else:
            print(f"table {name}: no rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())