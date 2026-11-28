"""Fetch Coin Metrics Community data for btc, every page saved as received.

Paths resolve from this file, so the script runs from any folder. It empties
and rewrites data/raw/coinmetrics/, and writes the "coinmetrics" entry of
data/raw/manifest.json. No key is used. Requests are spaced so that no more
than 10 fall in any 6 seconds.

What it fetches:
  catalog_free_*.json  catalog-v2/asset-metrics for btc: what the community
                       API serves without a key
  catalog_all_*.json   catalog-all-v2/asset-metrics for btc: every metric,
                       free or not, to say which concepts are not free
  series_gNN_*.json    timeseries/asset-metrics for every free metric at
                       frequency 1d, in groups, through the last completed day
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "raw"
OUT = RAW / "coinmetrics"
MANIFEST = RAW / "manifest.json"
SOURCE = "coinmetrics"

ROOT_URL = "https://community-api.coinmetrics.io/v4"
CATALOG_FREE = ROOT_URL + "/catalog-v2/asset-metrics"
CATALOG_ALL = ROOT_URL + "/catalog-all-v2/asset-metrics"
SERIES = ROOT_URL + "/timeseries/asset-metrics"
ASSET = "btc"
FREQUENCY = "1d"
FIRST_TIME = "2009-01-03"
GROUP_SIZE = 25
PAGE_SIZE = 10000
USER_AGENT = "crypto-dataset-fetch/1.0"
WINDOW_SECONDS = 6.0
WINDOW_REQUESTS = 9  # one below the stated limit of 10 per 6 seconds


class FetchError(RuntimeError):
    """The source did not answer as expected."""


_sent: collections.deque = collections.deque()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def pace() -> None:
    """Wait until one more request keeps 6-second windows under the limit."""
    while len(_sent) >= WINDOW_REQUESTS:
        wait = WINDOW_SECONDS - (time.monotonic() - _sent[0])
        if wait > 0:
            time.sleep(wait + 0.05)
        _sent.popleft()
    _sent.append(time.monotonic())


def http_get(full: str) -> tuple[datetime, bytes]:
    """GET one URL, with retries on rate limits and server errors."""
    request = urllib.request.Request(
        full, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(1, 5):
        pace()
        fetched = utc_now()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return fetched, response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()[:600]
            if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(WINDOW_SECONDS * attempt)
                continue
            raise FetchError(f"{full} answered HTTP {exc.code}: {body!r}") from None
        except urllib.error.URLError as exc:
            if attempt < 4:
                time.sleep(WINDOW_SECONDS * attempt)
                continue
            raise FetchError(f"{full} could not be reached: {exc.reason}") from None
    raise AssertionError("unreachable")


def fetch_pages(url: str, params: dict, stem: str, files: list) -> list[dict]:
    """Fetch every page of one query; save each; return the parsed pages."""
    full = url + "?" + urllib.parse.urlencode(params)
    pages = []
    number = 0
    while full:
        fetched, body = http_get(full)
        try:
            doc = json.loads(body)
        except ValueError:
            raise FetchError(f"{full} did not return JSON: {body[:400]!r}") from None
        if not isinstance(doc, dict) or not isinstance(doc.get("data"), list):
            raise FetchError(f"{full} has no data list: {body[:400]!r}")
        number += 1
        name = f"{stem}_p{number:02d}.json"
        (OUT / name).write_bytes(body)
        files.append({
            "file": f"coinmetrics/{name}",
            "url": full,
            "fetched_utc": iso(fetched),
            "rows": rows_of(stem, doc),
            "sha256": hashlib.sha256(body).hexdigest(),
        })
        pages.append(doc)
        full = doc.get("next_page_url") or ""
    return pages


def rows_of(stem: str, doc: dict) -> int:
    """Rows in a page: metric entries for a catalog, timestamps for a series."""
    if stem.startswith("catalog"):
        return sum(len(item.get("metrics") or []) for item in doc["data"])
    return len(doc["data"])


def daily_metrics(pages: list[dict]) -> dict[str, dict]:
    """metric -> its 1d frequency record, from catalog pages."""
    found = {}
    for doc in pages:
        for item in doc["data"]:
            if item.get("asset") != ASSET:
                raise FetchError(f"catalog item for another asset: {item.get('asset')!r}")
            for metric in item.get("metrics") or []:
                for freq in metric.get("frequencies") or []:
                    if freq.get("frequency") == FREQUENCY:
                        found[metric["metric"]] = freq
    return found


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
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    files: list[dict] = []

    free_pages = fetch_pages(CATALOG_FREE, {"assets": ASSET}, "catalog_free", files)
    all_pages = fetch_pages(CATALOG_ALL, {"assets": ASSET}, "catalog_all", files)
    free = daily_metrics(free_pages)
    every = daily_metrics(all_pages)
    print(f"catalog: {len(free)} free metrics at {FREQUENCY}, "
          f"{len(every)} in the full catalog at {FREQUENCY}")
    if "PriceUSD" not in free:
        raise FetchError("PriceUSD is not among the free daily metrics")

    names = sorted(free)
    groups = [names[i:i + GROUP_SIZE] for i in range(0, len(names), GROUP_SIZE)]
    seen_rows: dict[str, int] = {}
    latest = ""
    for index, group in enumerate(groups, start=1):
        params = {
            "assets": ASSET,
            "metrics": ",".join(group),
            "frequency": FREQUENCY,
            "start_time": FIRST_TIME,
            "end_time": last_day.isoformat(),
            "page_size": PAGE_SIZE,
            "paging_from": "start",
        }
        pages = fetch_pages(SERIES, params, f"series_g{index:02d}", files)
        for doc in pages:
            for row in doc["data"]:
                day = str(row.get("time", ""))[:10]
                if not day or day > last_day.isoformat():
                    raise FetchError(f"row for a day not ended, or no time: {row!r}"[:300])
                latest = max(latest, day)
                for metric in group:
                    if row.get(metric) is not None:
                        seen_rows[metric] = seen_rows.get(metric, 0) + 1
        print(f"group {index}: {len(group)} metrics, "
              f"{sum(len(d['data']) for d in pages)} rows in {len(pages)} page(s)")

    empty = [m for m in names if m not in seen_rows]
    if empty:
        print(f"NOTE: {len(empty)} free metrics returned no value: {', '.join(empty)}")
    print(f"latest row {latest}; last completed UTC day {last_day}")
    write_manifest({
        "script": "scripts/fetch_coinmetrics.py",
        "run_started_utc": iso(started),
        "last_complete_day": last_day.isoformat(),
        "storage": "one JSON file per request: the response body, byte for byte; "
                   "a query of several pages has one file per page",
        "files": files,
    })
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FetchError as exc:
        print(f"FETCH FAILED: {exc}", file=sys.stderr)
        sys.exit(1)