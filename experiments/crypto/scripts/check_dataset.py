"""Independent checks of the built BTC daily dataset: tables A to D.

    python check_dataset.py

Prints one line per check, PASS or FAIL with the reason (some checks add
indented detail lines under theirs). Exits 0 only when checks 1 to 6 and 8
pass; check 7 is a report. It does not import labels.py, features.py or
build_dataset.py: it reads the raw and built files, recomputes what it
checks in its own code, and runs build_dataset.py only as a separate
process. Its temporary folders are made under data/scratch/ and removed.
"""
from __future__ import annotations

import csv
import functools
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RAW = DATA / "raw"
BUILT = DATA / "built"
PROFILER = DATA / "profiler"
SCRATCH = DATA / "scratch"
BUILD = HERE / "build_dataset.py"
DAY = pd.Timedelta(days=1)

A_FIRST = pd.Timestamp("2012-12-17")   # ruling 12; table C's first row too
B_FIRST = pd.Timestamp("2019-12-09")   # ruling 13; table D's first row too
CM_LIVE_FROM = pd.Timestamp("2021-02-24")  # ruling 15
SEED = 20261128
N_DATES = 40
TOL = 1e-9
TABLES = ("a", "b", "c", "d")
FIRSTS = {"a": A_FIRST, "b": B_FIRST, "c": A_FIRST, "d": B_FIRST}
PROFILER_FILES = {f"{t}_{target}.csv": target for t in TABLES for target in ("up_7d", "down_7d")}
LABEL_NAMES = ("fwd_ret_7", "z_7", "class_7")
# Dates where a rule bites, always among the 40 (besides the four required):
# the 2015 Bitstamp gap, Coin Metrics rows that use an earlier day, and the
# Easter weekend of 2026, whose Saturday, Sunday and Monday (a TARGET
# closing day) all carry the fixing of Thursday 2026-04-02 (ruling 25).
RULE_DATES = ("2015-01-06", "2015-01-07", "2015-01-08", "2015-01-09",
              "2021-02-25", "2024-08-16", "2025-05-03",
              "2026-04-04", "2026-04-05", "2026-04-06")
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# The FX columns of tables C and D, in catalogue order (02-add-fx.md), and
# how many fixings before the row's own fixing each one reads.
FX_SERIES = ("eurusd", "usdjpy", "gbpusd", "usdchf", "audusd", "usdcad", "dxy")
FX_COLS = ([f"fx_{s}_{p}" for s in FX_SERIES
            for p in ("ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "vol_z_250")]
           + ["btc_dxy_corr_60"])
FX_REACH = {**{f"fx_{s}_ret_{n}": n for s in FX_SERIES for n in (1, 5, 20, 60)},
            **{f"fx_{s}_vol_20": 20 for s in FX_SERIES},
            **{f"fx_{s}_vol_z_250": 249 + 20 for s in FX_SERIES},
            "btc_dxy_corr_60": 60}
ECB_API = Path("ecb") / "exr_daily.csv"
ECB_CURRENCIES = ("USD", "JPY", "GBP", "CHF", "AUD", "CAD", "SEK")

# Check 8. Tables A and B as recorded before the FX session changed any
# script (reports/fx_sources.md, section 1), and the SHA-256 of
# manifest.json as it was before the ECB entry was added: the entries of
# Bitstamp, Coin Metrics and Binance re-serialise to exactly that file.
RECORDED = {
    "built/btc_daily_full.csv": "c2f44ab747f6e403c0a802c44ac677bb23a5080bbc62be604b0741e3b1abc72e",
    "built/build_info.json": "fecb623d730ffabf641212ad57984e70cce4835b3e1be3edce632f6bdfd4d72a",
    "profiler/a_down_7d.csv": "f962cdfc2063d1e1a12ee45b5c44a8cfe9f32a9390722e1e61b885e6a0483383",
    "profiler/a_short_down_7d.csv": "7d8d34e00a2f8db4b8a1122e71756c04d27e8017a8e2af39a5fd5ab4b00bfa60",
    "profiler/a_short_up_7d.csv": "d067c2285c422f291a6b34a7b59b4474b914c89b12662c79c1a972c26dee7aae",
    "profiler/a_up_7d.csv": "64b652325c3c50c75223d53b89f2863b03db3e892c74e1f8f2d5dc9c6875903a",
    "profiler/b_down_7d.csv": "b3016a45489c765f96d0202c91a97c70c84364b5ffa3dd225d797edfe424a555",
    "profiler/b_up_7d.csv": "8c76019717bcd3ba9e585d203ed4b5056610548673bfbabb5d6e289e1c5f9d21",
}
PRE_FX_MANIFEST_SHA256 = "78c02d3c6641856f245bf7af3526187d1984743556bbfc27551d561ae9a9f187"


# ----------------------------------------------------------------- read --

def manifest(raw: Path = RAW) -> dict:
    return json.loads((raw / "manifest.json").read_text(encoding="utf-8"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bitstamp_frame(raw: Path = RAW) -> pd.DataFrame:
    candles = {}
    for path in sorted((raw / "bitstamp").glob("ohlc_*.json")):
        for c in json.loads(path.read_bytes())["data"]["ohlc"]:
            candles[int(c["timestamp"])] = c
    frame = pd.DataFrame([candles[k] for k in sorted(candles)])
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"].astype("int64"), unit="s"))
    return frame[["open", "high", "low", "close", "volume"]].astype(float)


@functools.lru_cache(maxsize=None)
def ecb_rows(raw: str = str(RAW)) -> tuple[list, list]:
    """The header and rows of the ECB API file, parsed once."""
    text = (Path(raw) / ECB_API).read_bytes().decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    return rows[0], rows[1:]


def ecb_fixings(raw: Path = RAW) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Fixing dates, read in this file's own code: the dates with a value.

    Returns them, and those among them that lack one of the seven
    currencies (which have no value in any FX series).
    """
    header, rows = ecb_rows(str(raw))
    at = {name: header.index(name) for name in ("CURRENCY", "TIME_PERIOD", "OBS_VALUE")}
    seen: dict[str, set] = {}
    for r in rows:
        if r and r[at["OBS_VALUE"]].strip() and r[at["CURRENCY"]] in ECB_CURRENCIES:
            seen.setdefault(r[at["TIME_PERIOD"]], set()).add(r[at["CURRENCY"]])
    dates = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in seen))
    partial = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d, c in seen.items() if len(c) < 7))
    return dates, partial


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"date": str, "class_7": str, "cm_day": str, "fx_day": str},
                       keep_default_na=False, na_values=[""])


def numeric_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(all="ignore"):
        diff = np.abs(a - b)
        close = (diff <= TOL) | (diff <= TOL * np.maximum(np.abs(a), np.abs(b)))
    return both_nan | (close & ~np.isnan(a) & ~np.isnan(b))


def feature_columns() -> dict[str, list[str]]:
    """Each table's feature columns, from the header of its up_7d file."""
    return {t: pd.read_csv(PROFILER / f"{t}_up_7d.csv", nrows=0).columns.tolist()[1:-1]
            for t in TABLES}


def runs_text(dates) -> str:
    days = sorted(pd.Timestamp(d) for d in dates)
    parts, start, prev = [], None, None
    for day in days + [None]:
        if day is not None and prev is not None and day == prev + DAY:
            prev = day
            continue
        if start is not None:
            n = (prev - start).days + 1
            parts.append(f"{start.date()}" if n == 1 else f"{start.date()}..{prev.date()} ({n})")
        start = prev = day
    return ", ".join(parts)


# --------------------------------------------------------- own labels --

def own_labels(k: float) -> pd.DataFrame:
    """z_t, the class and the targets, recomputed from raw Bitstamp."""
    bs = bitstamp_frame()
    last = pd.Timestamp(manifest()["bitstamp"]["last_complete_day"])
    bs = bs[bs.index <= last]
    first = bs.index.min()
    cal = pd.date_range(first, last, freq="D")
    close = bs["close"].where(bs["volume"] > 0).reindex(cal)
    present = close.notna().to_numpy()
    run, inside = 0, np.zeros(len(cal), dtype=bool)
    for i, p in enumerate(present):
        run = 0 if p else run + 1
        inside[i] = run >= 3
    c = close.to_numpy()
    r = np.full(len(c), np.nan)
    r[1:] = np.log(c[1:] / c[:-1])
    sigma = np.full(len(c), np.nan)
    for i in range(30, len(c)):          # returns i-29..i need closes from i-30
        if inside[i]:
            continue
        window = r[i - 29:i + 1]
        ok = window[~np.isnan(window)]
        if len(ok) >= 27:
            sigma[i] = float(np.std(ok, ddof=1))
    fwd = np.full(len(c), np.nan)
    fwd[:-7] = np.log(c[7:] / c[:-7])
    z = fwd / (sigma * math.sqrt(7))
    cls = np.where(np.isnan(z), "", np.where(z > k, "UP", np.where(z < -k, "DOWN", "RANGE")))
    return pd.DataFrame({"fwd": fwd, "z": z, "cls": cls, "sigma": sigma}, index=cal)


def own_k(z: pd.Series) -> float:
    rows = z[(z.index >= A_FIRST)].dropna()
    early = rows.iloc[: int(math.floor(0.75 * len(rows)))]
    return round(float(np.quantile(np.abs(early.to_numpy()), 1 / 3)), 2)


# --------------------------------------------------------------- checks --

def check_labels(state: dict):
    k = state["info"]["k"]
    lab = own_labels(k)
    state["labels"] = lab
    k_own = own_k(lab["z"])
    problems = []
    if k_own != k:
        problems.append(f"k by rule 5 is {k_own}, build_info says {k}")
    if state["info_cd"].get("k") != k:
        problems.append(f"build_info_cd.json says k = {state['info_cd'].get('k')}")
    full = state["full"]
    idx = pd.DatetimeIndex(full["date"])
    mine = lab.reindex(idx)
    if not numeric_equal(full["z_7"], mine["z"]).all():
        problems.append(f"z_7 differs on {int((~numeric_equal(full['z_7'], mine['z'])).sum())} rows")
    if not numeric_equal(full["fwd_ret_7"], mine["fwd"]).all():
        problems.append("fwd_ret_7 differs")
    cls = full["class_7"].fillna("").to_numpy()
    if (cls != mine["cls"].to_numpy()).any():
        problems.append(f"class_7 differs on {int((cls != mine['cls'].to_numpy()).sum())} rows")
    expected = {"up_7d": np.where(mine["cls"] == "", np.nan, (mine["cls"] == "UP").astype(float)),
                "down_7d": np.where(mine["cls"] == "", np.nan, (mine["cls"] == "DOWN").astype(float))}
    for name, frame in [("btc_daily_full.csv", full)] + list(state["profilers"].items()):
        for target in ("up_7d", "down_7d"):
            if target not in frame.columns:
                continue
            idx_f = pd.DatetimeIndex(frame["date"])
            want = pd.Series(expected[target], index=idx).reindex(idx_f).to_numpy()
            got = frame[target].to_numpy(dtype=float)
            if not numeric_equal(got, want).all():
                problems.append(f"{name} {target} differs on "
                                f"{int((~numeric_equal(got, want)).sum())} rows")
            raw_text = (PROFILER / name if name != "btc_daily_full.csv" else BUILT / name)
            values = set(pd.read_csv(raw_text, usecols=[target], dtype=str,
                                     keep_default_na=False)[target])
            if not values <= {"0", "1", ""}:
                problems.append(f"{name} {target} holds {sorted(values - {'0', '1', ''})[:3]}")
    # C's targets are A's and D's are B's, row for row.
    for mine_t, theirs in (("c", "a"), ("d", "b")):
        for target in ("up_7d", "down_7d"):
            x = state["profilers"][f"{mine_t}_{target}.csv"]
            y = state["profilers"][f"{theirs}_{target}.csv"]
            if x["date"].tolist() != y["date"].tolist():
                problems.append(f"{mine_t}_{target}.csv's dates differ from {theirs}_{target}.csv's")
            elif not numeric_equal(x[target], y[target]).all():
                problems.append(f"{mine_t}_{target}.csv's targets differ from {theirs}_{target}.csv's")
    blank = int(np.isnan(mine["z"]).sum())
    reason = (f"k = {k} (rule 5 recomputed: {k_own}); z, class, up_7d, down_7d recomputed from "
              f"raw Bitstamp agree on {len(full)} rows of btc_daily_full.csv and every row of the "
              f"8 profiler files; C's targets equal A's and D's equal B's, row for row; blank on "
              f"{blank} rows, exactly where C_t, C_t+7 or sigma30_t is missing")
    return (not problems, "; ".join(problems) if problems else reason, [])


def cut_raw(day: pd.Timestamp, src: Path, dst: Path) -> None:
    """Copy the raw folder as it was known at the close of ``day``."""
    close = day + DAY
    (dst / "bitstamp").mkdir(parents=True)
    (dst / "coinmetrics").mkdir()
    (dst / "binance").mkdir()
    (dst / "ecb").mkdir()
    limit_s = int(day.timestamp())
    for path in sorted((src / "bitstamp").glob("*.json")):
        doc = json.loads(path.read_bytes())
        doc["data"]["ohlc"] = [c for c in doc["data"]["ohlc"] if int(c["timestamp"]) <= limit_s]
        (dst / "bitstamp" / path.name).write_bytes(json.dumps(doc).encode("utf-8"))
    # A day's completion time is one metric among the series files: read it
    # from all of them before cutting any.
    completion: dict[str, float] = {}
    for path in sorted((src / "coinmetrics").glob("series_*.json")):
        for row in json.loads(path.read_bytes())["data"]:
            if row.get("AssetEODCompletionTime") is not None:
                completion[row["time"]] = float(row["AssetEODCompletionTime"])
    for path in sorted((src / "coinmetrics").glob("*.json")):
        if not path.name.startswith("series_"):
            shutil.copyfile(path, dst / "coinmetrics" / path.name)
            continue
        doc = json.loads(path.read_bytes())
        kept = []
        for row in doc["data"]:
            d = pd.Timestamp(row["time"][:10])
            if d > day - DAY:
                continue
            if d >= CM_LIVE_FROM:
                done = completion.get(row["time"])
                if done is None or pd.Timestamp(int(done), unit="s") > close:
                    continue
            kept.append(row)
        doc["data"] = kept
        (dst / "coinmetrics" / path.name).write_bytes(json.dumps(doc).encode("utf-8"))
    limit_ms = int(day.timestamp()) * 1000
    close_ms = int(close.timestamp()) * 1000
    for path in sorted((src / "binance").glob("*.json")):
        rows = json.loads(path.read_bytes())
        if path.name.startswith("funding_"):
            rows = [r for r in rows if (r["fundingTime"] // 3_600_000) * 3_600_000 < close_ms]
        else:
            rows = [r for r in rows if r[0] <= limit_ms]
        (dst / "binance" / path.name).write_bytes(json.dumps(rows).encode("utf-8"))
    # Ruling 25: the API file cut to fixing dates on or before the day. The
    # fixing of F is published by about 16:00 CET on F, before F's close.
    # The zip and the probes are not read by the build (ruling 21).
    header, rows = ecb_rows(str(src))
    at = header.index("TIME_PERIOD")
    limit = day.strftime("%Y-%m-%d")
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(r for r in rows if r and r[at] <= limit)
    (dst / ECB_API).write_bytes(out.getvalue().encode("utf-8"))
    man = manifest(src)
    for entry in man.values():
        entry["last_complete_day"] = day.strftime("%Y-%m-%d")
        entry["cut_for_check"] = True
    (dst / "manifest.json").write_bytes(json.dumps(man, indent=2).encode("utf-8"))


def run_build(raw: Path, out: Path, k: float | None,
              unchecked: bool = False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-B", str(BUILD), "--raw", str(raw), "--out", str(out)]
    if k is not None:
        cmd += ["--k", repr(k)]
    if unchecked:
        cmd.append("--k-unchecked")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def check_point_in_time(state: dict):
    full, fxd = state["full"], state["fx"]
    k = state["info"]["k"]
    cols = state["columns"]
    dates = pd.DatetimeIndex(full["date"])
    date_set = set(dates)
    last_target = dates[full["up_7d"].notna().to_numpy()].max()
    chosen = [A_FIRST, B_FIRST, last_target, dates.max()]
    chosen += [pd.Timestamp(d) for d in RULE_DATES]
    chosen = list(dict.fromkeys(d for d in chosen if d in date_set))
    rng = np.random.default_rng(SEED)
    taken = set(chosen)
    rest = [d for d in dates if d not in taken]
    picks = rng.choice(len(rest), size=N_DATES - len(chosen), replace=False)
    chosen += [rest[i] for i in sorted(picks)]
    chosen = sorted(chosen)
    fixings, _ = ecb_fixings()
    fixing_set = set(fixings)
    kinds = {"Saturday": [d for d in chosen if d.day_of_week == 5],
             "Sunday": [d for d in chosen if d.day_of_week == 6],
             "weekday with no fixing (a TARGET closing day)":
                 [d for d in chosen if d.day_of_week < 5 and d not in fixing_set]}
    lacking = [name for name, days in kinds.items() if not days]
    details = [f"    dates: {', '.join(d.strftime('%Y-%m-%d') for d in chosen)}",
               "    among them: " + "; ".join(f"{name} {', '.join(str(d.date()) for d in days)}"
                                             for name, days in kinds.items())]
    if lacking:
        return False, f"the {len(chosen)} dates hold no {', '.join(lacking)}", details

    problems = []
    # Every row takes the latest fixing date on or before it.
    pos = fixings.searchsorted(dates, side="right") - 1
    want = pd.DatetimeIndex(fixings[np.clip(pos, 0, None)])
    got = pd.DatetimeIndex(fxd["fx_day"])
    if (pos < 0).any() or not (pd.DatetimeIndex(fxd["date"]) == dates).all() \
            or not (got == want).all():
        problems.append("fx_daily.csv: fx_day is not the latest fixing date on or before the row")

    refs = {"built/btc_daily_full.csv": (full, list(dict.fromkeys(cols["a"] + cols["b"])), "cm_day"),
            "built/fx_daily.csv": (fxd, FX_COLS, "fx_day"),
            "profiler/c_up_7d.csv": (state["profilers"]["c_up_7d.csv"], cols["c"], None),
            "profiler/d_up_7d.csv": (state["profilers"]["d_up_7d.csv"], cols["d"], None)}
    refs = {n: (f.set_index(pd.DatetimeIndex(f["date"])), c, t) for n, (f, c, t) in refs.items()}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="pit_", dir=SCRATCH))

    def one(day: pd.Timestamp):
        folder = base / day.strftime("%Y%m%d")
        cut_raw(day, RAW, folder / "raw")
        done = run_build(folder / "raw", folder / "out", k, unchecked=True)
        if done.returncode != 0:
            return day, None, f"build failed: {done.stderr.strip()[-300:]}"
        compared, bad_cells = {}, []
        for name, (ref_all, columns, text_col) in refs.items():
            cut = read_table(folder / "out" / name)
            if not len(cut):
                if name.endswith("d_up_7d.csv") and day < B_FIRST:
                    continue
                return day, None, f"{name} has no rows"
            cut = cut.set_index(pd.DatetimeIndex(cut["date"]))
            if cut.index.max() != day:
                return day, None, f"{name}: last row {cut.index.max().date()}, not {day.date()}"
            ref = ref_all.loc[cut.index]
            for c in columns:
                ok = numeric_equal(cut[c].to_numpy(dtype=float), ref[c].to_numpy(dtype=float))
                if not ok.all():
                    bad_cells += [(d, f"{name}:{c}") for d in cut.index[~ok]]
            if text_col and (cut[text_col].fillna("") != ref[text_col].fillna("")).any():
                bad_cells.append((None, f"{name}:{text_col}"))
            compared[name] = len(cut)
        info = json.loads((folder / "out" / "built" / "build_info.json").read_text(encoding="utf-8"))
        rule5 = (info.get("k_rule") or {}).get("k")
        on_day = [c for d, c in bad_cells if d == day]
        return day, (compared, len(bad_cells), on_day, rule5), None

    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
        results = list(pool.map(one, chosen))
    shutil.rmtree(base, ignore_errors=True)
    rows_compared: dict[str, int] = {}
    other_k = 0
    for day, res, err in results:
        if err:
            problems.append(f"{day.date()}: {err}")
            continue
        compared, n_bad, on_day, rule5 = res
        for name, n in compared.items():
            rows_compared[name] = rows_compared.get(name, 0) + n
        other_k += rule5 is not None and rule5 != k
        if on_day:
            problems.append(f"{day.date()}: its own row differs in {', '.join(on_day[:6])}")
        elif n_bad:
            problems.append(f"{day.date()}: {n_bad} cells of earlier rows differ")
    details.append("    rows compared: " + ", ".join(f"{n} {v}" for n, v in rows_compared.items()))
    if problems:
        return False, "; ".join(problems[:6]), details
    n_cols = len(set(cols["c"]) | set(cols["d"]))
    return True, (f"{len(chosen)} dates (seed {SEED}; the first rows of tables A to D, the last row "
                  f"with a target, the last row, and {', '.join(RULE_DATES)} among them): each "
                  f"rebuilt from raw data cut at its close, the ECB file cut to fixing dates on or "
                  f"before it, with k = {k} (--k-unchecked: rule 5 on the cut history gives another "
                  f"k on {other_k} of them); its row and every earlier row agree in all {n_cols} "
                  f"feature columns of tables A to D within {TOL}, and in cm_day and fx_day; every "
                  f"row of fx_daily.csv takes the latest fixing date on or before it"), details


def check_one_target(state: dict):
    problems = []
    cols = state["columns"]
    full = state["full"]
    expected = {"a": cols["a"], "b": cols["b"], "c": cols["a"] + FX_COLS, "d": cols["b"] + FX_COLS}
    for name, target in PROFILER_FILES.items():
        frame = state["profilers"][name]
        header = frame.columns.tolist()
        if header[0] != "date":
            problems.append(f"{name}: first column is {header[0]}")
        if header[-1] != target:
            problems.append(f"{name}: last column is {header[-1]}, not {target}")
        if sum(h in ("up_7d", "down_7d") for h in header) != 1:
            problems.append(f"{name}: not exactly one target")
        for bad in LABEL_NAMES + ("cm_day", "fx_day", "sigma30"):
            if bad in header:
                problems.append(f"{name} holds {bad}")
        if header[1:-1] != expected[name[0]]:
            problems.append(f"{name}: feature columns are not table {name[0].upper()}'s "
                            f"(A's, or B's, followed by the 43 FX columns for C and D)")
        ref = full.set_index("date").reindex(frame["date"])
        label_values = {"fwd_ret_7": ref["fwd_ret_7"].to_numpy(dtype=float),
                        "z_7": ref["z_7"].to_numpy(dtype=float),
                        "class_7": ref["class_7"].map({"UP": 1.0, "RANGE": 0.0, "DOWN": -1.0})
                        .to_numpy(dtype=float)}
        for col in header[1:-1]:
            values = frame[col].to_numpy(dtype=float)
            for label, lv in label_values.items():
                both = ~np.isnan(values) & ~np.isnan(lv)
                if both.sum() > 10 and numeric_equal(values[both], lv[both]).all():
                    problems.append(f"{name}: {col} equals {label}")
    texts = {name: (PROFILER / name).read_text(encoding="utf-8").splitlines()
             for name in PROFILER_FILES}
    for t in TABLES:
        strip = [[line.rsplit(",", 1)[0] for line in texts[f"{t}_{x}.csv"]]
                 for x in ("up_7d", "down_7d")]
        if strip[0] != strip[1]:
            problems.append(f"{t}_up_7d.csv and {t}_down_7d.csv differ apart from the target")
    # C's lines begin with A's features as written, and D's with B's.
    for mine_t, theirs in (("c", "a"), ("d", "b")):
        mine_lines, their_lines = texts[f"{mine_t}_up_7d.csv"], texts[f"{theirs}_up_7d.csv"]
        if len(mine_lines) != len(their_lines) or any(
                not m.startswith(t.rsplit(",", 1)[0] + ",") for m, t in zip(mine_lines, their_lines)):
            problems.append(f"{mine_t}_up_7d.csv's lines do not begin with "
                            f"{theirs}_up_7d.csv's features, as written")
    up = {t: state["profilers"][f"{t}_up_7d.csv"].set_index("date") for t in TABLES}
    missing = [c for c in cols["a"] if c not in cols["b"]]
    if missing:
        problems.append(f"table A columns not in table B: {missing[:5]}")
    common = up["b"].index.intersection(up["a"].index)
    if len(common) != len(up["b"]):
        problems.append("table B has dates that table A lacks")
    for c in cols["a"]:
        if c in up["b"].columns and not numeric_equal(up["a"].loc[common, c],
                                                      up["b"].loc[common, c]).all():
            problems.append(f"{c} differs between tables A and B")
    for mine_t, theirs in (("c", "a"), ("d", "b")):
        if up[mine_t].index.tolist() != up[theirs].index.tolist():
            problems.append(f"table {mine_t.upper()}'s dates are not table {theirs.upper()}'s")
            continue
        for c in cols[theirs]:
            if not numeric_equal(up[mine_t][c], up[theirs][c]).all():
                problems.append(f"{c} differs between tables {theirs.upper()} and {mine_t.upper()}")
    cd = up["d"].index.intersection(up["c"].index)
    for c in FX_COLS:
        if not numeric_equal(up["c"].loc[cd, c], up["d"].loc[cd, c]).all():
            problems.append(f"{c} differs between tables C and D")
    fxd = state["fx"]
    if fxd.columns.tolist() != ["date", "fx_day"] + FX_COLS:
        problems.append("fx_daily.csv is not date, fx_day and the 43 FX columns in order")
    else:
        f = fxd.set_index("date")
        if f.index.tolist() != up["c"].index.tolist():
            problems.append("fx_daily.csv's dates are not table C's")
        else:
            for c in FX_COLS:
                if not numeric_equal(f[c], up["c"][c]).all():
                    problems.append(f"{c} differs between fx_daily.csv and table C")
    if problems:
        return False, "; ".join(problems[:6]), []
    return True, (f"eight profiler files: date, then {len(cols['a'])} (A), {len(cols['b'])} (B), "
                  f"{len(cols['c'])} (C) or {len(cols['d'])} (D) features, then one target; C is "
                  f"A's columns followed by the {len(FX_COLS)} FX columns, D is B's followed by "
                  f"the same, in the same order in the up and down files; no column holds or "
                  f"equals the forward return, z or the class; up and down files identical apart "
                  f"from the target; A's columns all in B with equal values on {len(common)} "
                  f"common dates; C's lines begin with A's features as written, D's with B's; "
                  f"the FX columns equal in C, D and fx_daily.csv on {len(cd)} common dates"), []


def check_rows(state: dict):
    problems = []
    man = manifest()
    last = pd.Timestamp(man["bitstamp"]["last_complete_day"])
    others = {s: e["last_complete_day"] for s, e in man.items()}
    if len(set(others.values())) != 1:
        problems.append(f"sources fetched for different last days: {others}")
    n_files = 0
    for source, entry in man.items():
        for f in entry["files"]:
            n_files += 1
            if sha256((RAW / f["file"]).read_bytes()) != f["sha256"]:
                problems.append(f"{f['file']} does not match its SHA-256 in the manifest")
    fixings, _ = ecb_fixings()
    if len(fixings) and fixings.max() > last:
        problems.append(f"the ECB file holds a fixing for {fixings.max().date()}, after {last.date()}")
    frames = {"btc_daily_full.csv": (state["full"], A_FIRST), "fx_daily.csv": (state["fx"], A_FIRST),
              **{n: (state["profilers"][n], FIRSTS[n[0]]) for n in PROFILER_FILES}}
    counts = {}
    for name, (frame, first) in frames.items():
        text = frame["date"].astype(str)
        if not text.map(lambda s: bool(ISO.match(s))).all():
            problems.append(f"{name}: a date is not ISO YYYY-MM-DD text")
        dates = pd.DatetimeIndex(text)
        if len(dates) > 1 and ((dates[1:] - dates[:-1]) != DAY).any():
            problems.append(f"{name}: dates do not rise by exactly one day")
        if dates[0] != first:
            problems.append(f"{name}: first row {dates[0].date()}, not {first.date()}")
        if dates[-1] != last:
            problems.append(f"{name}: last row {dates[-1].date()}, not {last.date()} (manifest)")
        for target in ("up_7d", "down_7d"):
            if target in frame.columns and frame[target].iloc[-7:].notna().any():
                problems.append(f"{name}: {target} not blank on the last seven rows")
        counts[name] = len(frame)
    if problems:
        return False, "; ".join(problems[:6]), []
    return True, (f"dates strictly rising by one day, ISO text, no day missing; tables A and C "
                  f"{A_FIRST.date()}..{last.date()} ({counts['a_up_7d.csv']} and "
                  f"{counts['c_up_7d.csv']} rows), tables B and D {B_FIRST.date()}..{last.date()} "
                  f"({counts['b_up_7d.csv']} and {counts['d_up_7d.csv']} rows), fx_daily.csv as "
                  f"table A; last row is the manifest's last completed day, and no ECB fixing is "
                  f"later; targets blank on the last seven rows (and elsewhere exactly as rule 6, "
                  f"check 1); all {n_files} raw files match their SHA-256"), []


def lookbacks(a_cols, b_cols) -> dict:
    """Each A or B column: the sources it is built from and how far back it reads."""
    B, S, P, F, CM = "bitstamp", "spot", "perp", "funding", "cm"
    spec: dict[str, tuple] = {}
    for n in (1, 2, 3, 5, 7, 14, 21, 30, 60, 90, 180, 365):
        spec[f"ret_{n}"] = ((B,), n)
    for n in (7, 14, 30):
        spec[f"zret_{n}"] = ((B,), max(n, 30))
    spec["past_z_7"] = spec["past_class_7"] = ((B,), 37)
    for n in (7, 14, 30, 90, 365):
        spec[f"vol_cc_{n}"] = ((B,), n)
    for n in (7, 30):
        for e in ("park", "gk", "rs"):
            spec[f"vol_{e}_{n}"] = ((B,), n - 1)
    spec.update({"vol_yz_30": ((B,), 30), "vol_ratio_7_30": ((B,), 30),
                 "vol_ratio_30_90": ((B,), 90), "vol_ratio_30_365": ((B,), 365),
                 "vol_of_vol_30": ((B,), 36), "vol_z_365": ((B,), 394),
                 "hl_range": ((B,), 0), "hl_range_mean_7": ((B,), 6), "body": ((B,), 0),
                 "clv": ((B,), 0), "upper_wick": ((B,), 0), "lower_wick": ((B,), 0),
                 "body_mean_7": ((B,), 6), "clv_mean_7": ((B,), 6),
                 "sma_50_200": ((B,), 199), "macd": ((B,), 78), "macd_signal": ((B,), 105),
                 "macd_hist": ((B,), 105), "dd_ath": ((B,), 0), "days_since_ath": ((B,), 0),
                 "dist_high_365": ((B,), 364), "dist_low_365": ((B,), 364),
                 "adx_14": ((B,), 85), "aroon_osc_25": ((B,), 25), "rsi_14": ((B,), 43),
                 "rsi_30": ((B,), 91), "stoch_k_14": ((B,), 13), "stoch_d_14_3": ((B,), 15),
                 "bb_pctb_20": ((B,), 19), "bb_width_20": ((B,), 19), "cci_20": ((B,), 19),
                 "lvol_z_30": ((B,), 29), "lvol_z_90": ((B,), 89),
                 "lvol_ratio_7_30": ((B,), 29), "amihud_z_365": ((B,), 394),
                 "obv_slope_30": ((B,), 30), "cmf_20": ((B,), 19),
                 "up_vol_share_30": ((B,), 30), "maxdd_30": ((B,), 29),
                 "dow": ((), 0), "month": ((), 0), "days_since_halving": ((), 0),
                 "halving_phase": ((), 0), "mvrv": ((CM,), 0), "mvrv_z365": ((CM,), 364),
                 "fund_1d": ((F,), 0), "fund_7": ((F,), 6), "fund_30": ((F,), 29),
                 "fund_z_90": ((F,), 89), "basis": ((S, P), 0), "basis_mean_7": ((S, P), 6),
                 "basis_z_90": ((S, P), 89), "usdt_prem": ((S, B), 0),
                 "usdt_prem_z_90": ((S, B), 89), "perp_spot_lvol": ((S, P), 0),
                 "perp_spot_lvol_z_90": ((S, P), 89), "trades_z_90": ((S,), 89),
                 "avg_trade_z_90": ((S,), 89), "bn_lvol_z_30": ((S,), 29)})
    for n in (7, 20, 50, 100, 200):
        spec[f"dist_sma_{n}"] = ((B,), n - 1)
    for n in (30, 90):
        spec[f"trend_t_{n}"] = ((B,), n - 1)
        for e in ("skew", "kurt", "acf1"):
            spec[f"{e}_{n}"] = ((B,), n)
    for n in (14, 30):
        spec[f"up_share_{n}"] = ((B,), n)
    for m in ("adractcnt", "txcnt", "hashrate", "feetotntv"):
        spec.update({f"oc_{m}_g7": ((CM,), 7), f"oc_{m}_g30": ((CM,), 30),
                     f"oc_{m}_z365": ((CM,), 364)})
    for side, source in (("spot", S), ("perp", P)):
        spec.update({f"taker_share_{side}": ((source,), 0),
                     f"taker_share_{side}_mean_7": ((source,), 6),
                     f"taker_share_{side}_z_90": ((source,), 89)})
    unknown = [c for c in dict.fromkeys(a_cols + b_cols) if c not in spec]
    if unknown:
        raise RuntimeError(f"no lookback for {unknown}")
    return spec


def source_gaps() -> dict:
    """Each source's first day and its missing days, from the raw files."""
    last = pd.Timestamp(manifest()["bitstamp"]["last_complete_day"])
    out = {}
    bs = bitstamp_frame()
    bs = bs[bs.index <= last]
    have = bs.index[bs["volume"] > 0]
    out["bitstamp"] = (bs.index.min(), pd.date_range(bs.index.min(), last).difference(have))
    for side, stem in (("spot", "spot_klines"), ("perp", "perp_klines")):
        days = set()
        for path in (RAW / "binance").glob(f"{stem}_*.json"):
            days |= {pd.Timestamp(r[0], unit="ms") for r in json.loads(path.read_bytes())}
        days = pd.DatetimeIndex(sorted(d for d in days if d <= last))
        out[side] = (days.min(), pd.date_range(days.min(), last).difference(days))
    per_day: dict[pd.Timestamp, list] = {}
    for path in (RAW / "binance").glob("funding_*.json"):
        for r in json.loads(path.read_bytes()):
            hour = pd.Timestamp((r["fundingTime"] // 3_600_000) * 3_600_000, unit="ms")
            per_day.setdefault(hour.normalize(), []).append(hour.hour)
    full = sorted(d for d, h in per_day.items() if sorted(h) == [0, 8, 16] and d <= last)
    out["funding"] = (full[0], pd.date_range(full[0], last).difference(pd.DatetimeIndex(full)))
    out["cm"] = (pd.Timestamp("2009-01-04"), pd.DatetimeIndex([]))
    return out


def check_finite(state: dict):
    problems, details = [], []
    cols = state["columns"]
    spec = lookbacks(cols["a"], cols["b"])
    gaps = source_gaps()
    fixings, partial = ecb_fixings()
    frames = {"btc_daily_full.csv": state["full"], "fx_daily.csv": state["fx"],
              **state["profilers"]}
    for name, frame in frames.items():
        numeric = frame.select_dtypes(include=[np.number])
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            problems.append(f"{name} holds an infinite value")

    def fx_allowed(col: str, day: pd.Timestamp) -> bool:
        """A blank FX cell needs a gap of its sources inside its window of fixings."""
        j = int(fixings.searchsorted(day, side="right")) - 1
        reach = FX_REACH[col]
        if j - reach < 0:
            return True                  # the window starts before the first fixing
        start = fixings[j - reach]
        if len(partial) and ((partial >= start) & (partial <= fixings[j])).any():
            return True
        if col == "btc_dxy_corr_60":
            first, missing = gaps["bitstamp"]
            if start < first:
                return True
            if len(missing) and ((missing >= start) & (missing <= day)).any():
                return True
        return False

    def allowed(col: str, day: pd.Timestamp) -> bool:
        if col in FX_REACH:
            return fx_allowed(col, day)
        sources, back = spec[col]
        for s in sources:
            first, missing = gaps[s]
            if day - back * DAY < first:
                return True
            if len(missing) and ((missing >= day - back * DAY) & (missing <= day)).any():
                return True
        return False

    full = state["full"]
    listings = (("table A", state["profilers"]["a_up_7d.csv"], cols["a"], True),
                ("table B", state["profilers"]["b_up_7d.csv"], cols["b"], True),
                ("table C, its FX columns (its A columns are table A's, check 3)",
                 state["profilers"]["c_up_7d.csv"], FX_COLS, True),
                ("table D, its FX columns (its B columns are table B's, check 3)",
                 state["profilers"]["d_up_7d.csv"], FX_COLS, True),
                ("full file, table B columns before table B",
                 full[pd.DatetimeIndex(full["date"]) < B_FIRST],
                 [c for c in cols["b"] if c not in cols["a"]], False))
    for label, frame, columns, itemise in listings:
        dates = pd.DatetimeIndex(frame["date"])
        total = 0
        lines = []
        for col in columns:
            blank = dates[frame[col].isna().to_numpy()]
            if not len(blank):
                continue
            total += len(blank)
            bad = [d for d in blank if not allowed(col, d)]
            if bad:
                problems.append(f"{label}: {col} blank on {runs_text(bad)} with no gap "
                                f"of its sources in reach")
            lines.append(f"      {col}: {runs_text(blank)}")
        details.append(f"    {label}: {total} blank cells" + (":" if lines else ""))
        if itemise:
            details.extend(lines)
        else:
            details.append(f"      every one of the {len(columns)} columns is blank only before "
                           f"its window lies inside its sources' history"
                           if not any(p.startswith(label) for p in problems) else "")
    if problems:
        return False, "; ".join(problems[:6]), details
    gap_text = "; ".join(f"{s}: {runs_text(m)}" for s, (f, m) in gaps.items()
                         if len(m) and s != "cm")
    return True, (f"no infinite value in any file; every blank feature cell lies within its "
                  f"column's reach of a missing source day, or before its window lies inside "
                  f"its sources' history; an FX column's reach is counted in fixings, and "
                  f"{len(partial)} fixing dates lack a currency (missing days: {gap_text})"), details


def check_deterministic(state: dict):
    k = state["info"]["k"]
    SCRATCH.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="det_", dir=SCRATCH))
    other_k = round(k + 0.01, 2)
    jobs = {"one": (RAW, base / "one", k, False), "two": (RAW, base / "two", k, False),
            "other_k": (RAW, base / "other_k", other_k, False),
            "refused": (RAW, base / "refused", k, True)}
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        runs = dict(zip(jobs, pool.map(lambda job: run_build(*job), jobs.values())))
    problems = [f"build failed: {runs[n].stderr.strip()[-300:]}" for n in ("one", "two")
                if runs[n].returncode]
    names = ["built/btc_daily_full.csv", "built/build_info.json", "built/fx_daily.csv",
             "built/build_info_cd.json"] + [f"profiler/{n}" for n in PROFILER_FILES]
    if not problems:
        for n in names:
            one, two = (base / "one" / n).read_bytes(), (base / "two" / n).read_bytes()
            if one != two:
                problems.append(f"{n} differs between the two builds")
            elif (DATA / n).read_bytes() != one:
                problems.append(f"data/{n} is not what the build writes now")
    # Ruling 22: the build stops when rule 5 gives another k than --k, and
    # refuses --k-unchecked on the real raw folder.
    stop = runs["other_k"]
    if stop.returncode == 0 or "rule 5 gives k" not in stop.stderr:
        problems.append(f"a build with --k {other_k} did not stop on rule 5's k")
    refused = runs["refused"]
    if refused.returncode == 0 or "refused" not in refused.stderr or (base / "refused").exists():
        problems.append("the build did not refuse --k-unchecked on data/raw")
    shutil.rmtree(base, ignore_errors=True)
    if problems:
        return False, "; ".join(problems), []
    return True, (f"two builds with --k {k} into two temporary folders wrote byte-identical "
                  f"files ({len(names)} files), identical to those in data/; a build with "
                  f"--k {other_k} stops, since rule 5 gives {k}; --k-unchecked on data/raw is "
                  f"refused and writes nothing (ruling 22)"), []


def check_level_screen(state: dict):
    full, fxd = state["full"], state["fx"]
    cols = state["columns"]
    if full["date"].tolist() != fxd["date"].tolist():
        return None, "btc_daily_full.csv and fx_daily.csv have different dates", []
    frame = pd.concat([full.reset_index(drop=True), fxd[FX_COLS].reset_index(drop=True)], axis=1)
    features = list(dict.fromkeys(cols["a"] + cols["b"] + cols["c"] + cols["d"]))
    n_early = int(math.floor(0.75 * len(frame)))
    early, late = frame.iloc[:n_early], frame.iloc[n_early:]
    rows = []
    for c in features:
        e = early[c].dropna()
        l = late[c].dropna()
        if not len(e) or not len(l):
            rows.append((c, float("nan"), len(l)))
            continue
        out = ((l < e.min()) | (l > e.max())).mean()
        rows.append((c, float(out), len(l)))
    rows.sort(key=lambda r: -1 if math.isnan(r[1]) else -r[1])
    above = [r for r in rows if r[1] > 0.10]
    state["level_screen"] = rows
    details = [f"    over table A's rows; earliest 75%: {frame['date'].iloc[0]}.."
               f"{frame['date'].iloc[n_early - 1]}; latest 25%: {frame['date'].iloc[n_early]}.."
               f"{frame['date'].iloc[-1]}"]
    details += [f"    {c}: {share:.1%} of {n} late values outside the early range"
                for c, share, n in rows if share > 0]
    fx_above = [r for r in above if r[0] in FX_REACH]
    return None, (f"{len(above)} of {len(features)} features above 10% ({len(fx_above)} of the "
                  f"{len(FX_COLS)} FX columns): "
                  + (", ".join(f"{c} {s:.1%}" for c, s, _ in above) or "none")), details


def check_ab_unchanged(state: dict):
    problems = []
    for name, want in RECORDED.items():
        path = DATA / name
        if not path.exists():
            problems.append(f"data/{name} is missing")
        elif sha256(path.read_bytes()) != want:
            problems.append(f"data/{name} has changed")
    before = {s: e for s, e in manifest().items() if s != "ecb"}
    text = (json.dumps(before, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if sha256(text) != PRE_FX_MANIFEST_SHA256:
        problems.append("the manifest's entries other than ecb are not those it had before")
    listed = {f["file"]: f["sha256"] for e in before.values() for f in e["files"]}
    on_disk = [p for p in sorted(RAW.rglob("*")) if p.is_file() and p.name != "manifest.json"
               and p.relative_to(RAW).parts[0] != "ecb"]
    for path in on_disk:
        rel = path.relative_to(RAW).as_posix()
        if rel not in listed:
            problems.append(f"raw/{rel} is not in the manifest as it was")
        elif sha256(path.read_bytes()) != listed[rel]:
            problems.append(f"raw/{rel} has changed")
    for rel in listed:
        if not (RAW / rel).exists():
            problems.append(f"raw/{rel} is missing")
    if problems:
        return False, "; ".join(problems[:6]), []
    return True, (f"the {len(RECORDED)} files of data/built/ and data/profiler/ recorded in Phase "
                  f"0 keep their SHA-256; the manifest's Bitstamp, Coin Metrics and Binance "
                  f"entries are those it had before the FX session, and the {len(on_disk)} raw "
                  f"files outside data/raw/ecb/ are exactly those entries' files, unchanged"), []


def main() -> int:
    needed = [BUILT / "btc_daily_full.csv", BUILT / "build_info.json", BUILT / "fx_daily.csv",
              BUILT / "build_info_cd.json"] + [PROFILER / n for n in PROFILER_FILES]
    for path in needed:
        if not path.exists():
            print(f"FAIL setup: {path} is missing; run build_dataset.py --k 0.36 first")
            return 1
    state = {"info": json.loads((BUILT / "build_info.json").read_text(encoding="utf-8")),
             "info_cd": json.loads((BUILT / "build_info_cd.json").read_text(encoding="utf-8")),
             "full": read_table(BUILT / "btc_daily_full.csv"),
             "fx": read_table(BUILT / "fx_daily.csv"),
             "profilers": {n: read_table(PROFILER / n) for n in PROFILER_FILES},
             "columns": feature_columns()}
    checks = [("1 labels", check_labels), ("2 point_in_time", check_point_in_time),
              ("3 one_target", check_one_target), ("4 rows", check_rows),
              ("5 finite", check_finite), ("6 deterministic", check_deterministic),
              ("7 level_screen", check_level_screen), ("8 ab_unchanged", check_ab_unchanged)]
    passed = True
    for name, fn in checks:
        try:
            ok, reason, details = fn(state)
        except Exception as exc:  # a check that cannot run fails, and says why
            ok, reason, details = False, f"{type(exc).__name__}: {exc}", []
        word = "REPORT" if ok is None else ("PASS" if ok else "FAIL")
        print(f"{word} {name}: {reason}", flush=True)
        for line in details:
            if line:
                print(line)
        if ok is False:
            passed = False
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
