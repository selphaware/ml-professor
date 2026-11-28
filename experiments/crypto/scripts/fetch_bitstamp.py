"""Fetch Bitstamp's daily BTC/USD candles, every page saved as received.

Paths resolve from this file, so the script runs from any folder. It empties
and rewrites data/raw/bitstamp/, and writes the "bitstamp" entry of
data/raw/manifest.json. Only completed UTC days are asked for: the last one is
the day before the UTC date at the start of the run.

Paging: each request asks for one window of at most 1000 days, with start,
end and limit=1000, so the window is the same whichever of start or end the
endpoint honours. Overlapping candles across pages must agree exactly.
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
OUT = RAW / "bitstamp"
MANIFEST = RAW / "manifest.json"
SOURCE = "bitstamp"

URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
STEP = 86400                  # one candle per UTC day, in seconds
LIMIT = 1000                  # candles per request
FIRST_DAY = date(2011, 1, 1)  # before Bitstamp's first trade
FIELDS = {"timestamp", "open", "high", "low", "close", "volume"}
USER_AGENT = "crypto-dataset-fetch/1.0"
PAUSE_SECONDS = 1.0


class FetchError(RuntimeError):
    """The source did not answer as expected."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def day_ts(day: date) -> int:
    """Unix seconds of 00:00 UTC on ``day``."""
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def ts_day(stamp: int) -> str:
    return datetime.fromtimestamp(stamp, timezone.utc).date().isoformat()


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
                time.sleep(5 * attempt)
                continue
            raise FetchError(f"{full} answered HTTP {exc.code}: {body!r}") from None
        except urllib.error.URLError as exc:
            if attempt < 4:
                time.sleep(5 * attempt)
                continue
            raise FetchError(f"{full} could not be reached: {exc.reason}") from None
    raise AssertionError("unreachable")


def candles_of(body: bytes, url: str) -> list[dict]:
    """The list of candles in one page, after checking its shape."""
    try:
        doc = json.loads(body)
        candles = doc["data"]["ohlc"]
    except (ValueError, KeyError, TypeError):
        raise FetchError(f"{url} did not return data.ohlc: {body[:400]!r}") from None
    if not isinstance(candles, list):
        raise FetchError(f"{url}: data.ohlc is not a list")
    for candle in candles:
        if not isinstance(candle, dict) or set(candle) != FIELDS:
            raise FetchError(f"{url}: unexpected candle {candle!r}")
    return candles


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
    cutoff = day_ts(today)  # the open of the first day that has not ended
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()

    files = []
    seen: dict[int, dict] = {}
    window_start = FIRST_DAY
    number = 0
    while window_start <= last_day:
        window_end = min(window_start + timedelta(days=LIMIT - 1), last_day)
        params = {"step": STEP, "limit": LIMIT,
                  "start": day_ts(window_start), "end": day_ts(window_end)}
        url, fetched, body = http_get(URL, params)
        candles = candles_of(body, url)
        stamps = [int(candle["timestamp"]) for candle in candles]
        off_grid = [s for s in stamps if s % STEP]
        if off_grid:
            raise FetchError(f"{url}: candles not at 00:00 UTC, e.g. {off_grid[:3]}")
        unfinished = [s for s in stamps if s >= cutoff]
        if unfinished:
            raise FetchError(f"{url}: candles for a day not ended: {unfinished}")
        if stamps != sorted(set(stamps)):
            raise FetchError(f"{url}: timestamps not strictly rising")
        for candle in candles:
            stamp = int(candle["timestamp"])
            if stamp in seen and seen[stamp] != candle:
                raise FetchError(f"pages disagree on the candle of {ts_day(stamp)}")
            seen[stamp] = candle
        number += 1
        name = f"ohlc_{number:03d}.json"
        (OUT / name).write_bytes(body)
        record = {
            "file": f"bitstamp/{name}",
            "url": url,
            "fetched_utc": iso(fetched),
            "rows": len(candles),
            "sha256": hashlib.sha256(body).hexdigest(),
            "first_day": ts_day(stamps[0]) if stamps else None,
            "last_day": ts_day(stamps[-1]) if stamps else None,
        }
        files.append(record)
        print(f"{name}: asked {window_start}..{window_end}, got {len(candles)} "
              f"candles {record['first_day']}..{record['last_day']}")
        window_start = window_end + timedelta(days=1)
        time.sleep(PAUSE_SECONDS)

    if not seen:
        raise FetchError("no candles at all")
    first_seen, last_seen = ts_day(min(seen)), ts_day(max(seen))
    print(f"{len(seen)} distinct days, {first_seen}..{last_seen}; "
          f"last completed UTC day {last_day}")
    if last_seen != last_day.isoformat():
        print(f"NOTE: the last candle is {last_seen}, not {last_day}")
    write_manifest({
        "script": "scripts/fetch_bitstamp.py",
        "run_started_utc": iso(started),
        "last_complete_day": last_day.isoformat(),
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