"""Fetch Binance BTCUSDT daily klines (spot and USD-M perpetual) and every
funding print of the perpetual, each page saved as received.

Paths resolve from this file, so the script runs from any folder. It empties
and rewrites data/raw/binance/, and writes the "binance" entry of
data/raw/manifest.json. No key is used.

Every request carries endTime = 00:00 UTC of the run's UTC date minus 1 ms,
so no kline of a day that has not ended, and no funding print of such a day
(the 00:00 print of today belongs to today's row), is ever received.
Pages go forward from startTime until a page comes back short.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "raw"
OUT = RAW / "binance"
MANIFEST = RAW / "manifest.json"
SOURCE = "binance"

SPOT_KLINES = "https://api.binance.com/api/v3/klines"
PERP_KLINES = "https://fapi.binance.com/fapi/v1/klines"
FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
SYMBOL = "BTCUSDT"
LIMIT = 1000
DAY_MS = 86_400_000
SPOT_FROM = date(2017, 1, 1)     # before the spot listing
PERP_FROM = date(2019, 1, 1)     # before the perpetual listing
USER_AGENT = "crypto-dataset-fetch/1.0"
PAUSE_SECONDS = 0.5


class FetchError(RuntimeError):
    """The source did not answer as expected."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def day_ms(day: date) -> int:
    """Unix milliseconds of 00:00 UTC on ``day``."""
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()) * 1000


def ms_text(stamp: int) -> str:
    return datetime.fromtimestamp(stamp / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def http_get(url: str, params: dict) -> tuple[str, datetime, bytes]:
    """GET with a few retries on rate limits and server errors."""
    full = url + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        full, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(1, 5):
        fetched = utc_now()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return full, fetched, response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()[:400]
            if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(10 * attempt)
                continue
            raise FetchError(f"{full} answered HTTP {exc.code}: {body!r}") from None
        except urllib.error.URLError as exc:
            if attempt < 4:
                time.sleep(5 * attempt)
                continue
            raise FetchError(f"{full} could not be reached: {exc.reason}") from None
    raise AssertionError("unreachable")


def parse_list(body: bytes, url: str) -> list:
    try:
        rows = json.loads(body)
    except ValueError:
        raise FetchError(f"{url} did not return JSON: {body[:400]!r}") from None
    if not isinstance(rows, list):
        raise FetchError(f"{url} did not return a list: {body[:400]!r}")
    return rows


def save(name: str, url: str, fetched: datetime, body: bytes, rows: int,
         first: str, last: str, files: list) -> None:
    (OUT / name).write_bytes(body)
    files.append({
        "file": f"binance/{name}",
        "url": url,
        "fetched_utc": iso(fetched),
        "rows": rows,
        "sha256": hashlib.sha256(body).hexdigest(),
        "first": first,
        "last": last,
    })


def fetch_klines(url: str, stem: str, start_day: date, end_ms: int,
                 files: list) -> int:
    """Daily klines from start_day forward; returns the number received."""
    start = day_ms(start_day)
    number = total = 0
    while True:
        params = {"symbol": SYMBOL, "interval": "1d", "limit": LIMIT,
                  "startTime": start, "endTime": end_ms}
        full, fetched, body = http_get(url, params)
        rows = parse_list(body, full)
        for row in rows:
            if not isinstance(row, list) or len(row) != 12:
                raise FetchError(f"{full}: unexpected kline {row!r}")
            # A kline may close early (an exchange outage); it may never run
            # past its own UTC day.
            if row[0] % DAY_MS or not row[0] < row[6] <= row[0] + DAY_MS - 1:
                raise FetchError(f"{full}: kline not within one UTC day: {row[:1] + row[6:7]}")
            if row[6] > end_ms:
                raise FetchError(f"{full}: kline of a day not ended: {ms_text(row[0])}")
        if not rows:
            if number == 0:
                raise FetchError(f"{full}: no klines at all")
            break
        number += 1
        total += len(rows)
        save(f"{stem}_{number:03d}.json", full, fetched, body, len(rows),
             ms_text(rows[0][0])[:10], ms_text(rows[-1][0])[:10], files)
        print(f"{stem}_{number:03d}.json: {len(rows)} klines "
              f"{ms_text(rows[0][0])[:10]}..{ms_text(rows[-1][0])[:10]}")
        if len(rows) < LIMIT:
            break
        start = rows[-1][0] + DAY_MS
        time.sleep(PAUSE_SECONDS)
    return total


def fetch_funding(start_day: date, end_ms: int, files: list) -> int:
    """Every funding print from start_day forward; returns the number received."""
    start = day_ms(start_day)
    number = total = 0
    while True:
        params = {"symbol": SYMBOL, "limit": LIMIT, "startTime": start, "endTime": end_ms}
        full, fetched, body = http_get(FUNDING, params)
        rows = parse_list(body, full)
        for row in rows:
            if not isinstance(row, dict) or not {"fundingTime", "fundingRate"} <= set(row):
                raise FetchError(f"{full}: unexpected print {row!r}")
            if row["fundingTime"] > end_ms:
                raise FetchError(f"{full}: print after the cut: {ms_text(row['fundingTime'])}")
        if not rows:
            if number == 0:
                raise FetchError(f"{full}: no funding prints at all")
            break
        number += 1
        total += len(rows)
        first, last = ms_text(rows[0]["fundingTime"]), ms_text(rows[-1]["fundingTime"])
        save(f"funding_{number:03d}.json", full, fetched, body, len(rows),
             first, last, files)
        print(f"funding_{number:03d}.json: {len(rows)} prints {first}..{last}")
        if len(rows) < LIMIT:
            break
        start = rows[-1]["fundingTime"] + 1
        time.sleep(PAUSE_SECONDS)
    return total


def write_manifest(entry: dict) -> None:
    """Replace this source's entry in manifest.json, keeping the others."""
    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest[SOURCE] = entry
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary = MANIFEST.with_name(MANIFEST.name + ".tmp")
    temporary.write_bytes(text.encode("utf-8"))
    os.replace(temporary, MANIFEST)


def main() -> int:
    started = utc_now()
    today = started.date()
    last_day = today - timedelta(days=1)
    end_ms = day_ms(today) - 1
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    files: list[dict] = []
    spot = fetch_klines(SPOT_KLINES, "spot_klines", SPOT_FROM, end_ms, files)
    perp = fetch_klines(PERP_KLINES, "perp_klines", PERP_FROM, end_ms, files)
    funding = fetch_funding(PERP_FROM, end_ms, files)
    print(f"spot {spot} klines, perpetual {perp} klines, {funding} funding prints; "
          f"last completed UTC day {last_day}")
    write_manifest({
        "script": "scripts/fetch_binance.py",
        "run_started_utc": iso(started),
        "last_complete_day": last_day.isoformat(),
        "end_time_ms": end_ms,
        "storage": "one JSON file per request: the response body, byte for byte",
        "files": files,
    })
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FetchError as exc:
        print(f"FETCH FAILED: {exc}", file=sys.stderr)
        sys.exit(1)