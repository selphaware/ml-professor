"""Fetch the ECB euro foreign exchange reference rates, as received.

Paths resolve from this file, so the script runs from any folder. It empties
and rewrites data/raw/ecb/, and writes the "ecb" entry of
data/raw/manifest.json, leaving every other entry as it was. The last day
asked for is the Bitstamp entry's last completed day, already in the
manifest (the last day of tables A and B). It must be a day that has ended.

Requests, all to the ECB, with no key:
1. the Data Portal API: the daily reference rates of USD, JPY, GBP, CHF,
   AUD, CAD and SEK against EUR, as SDMX-CSV (format=csvdata), from the
   first fixing to the last day: ecb/exr_daily.csv;
2. the historical file, saved as received: ecb/eurofxref-hist.zip. It cannot
   be cut by date, so the run stops if it holds a day after the last day;
3. publication probes. The API's updatedAfter parameter returns only the
   observations updated after a given instant. For each probed fixing date
   D the API is asked for D at 12:00 UTC on D and at 00:00 UTC on D+1. The
   probed dates are the last 20 fixing dates and 40 earlier ones drawn with
   a fixed seed from Bitstamp's first day on; the last 5 are also asked at
   13:30 to 22:00 UTC on D. The answers go to ecb/probes/. An empty answer
   or HTTP 404 (no results) is saved as received like any other.

Everything is fetched and checked before anything is written.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "raw"
OUT = RAW / "ecb"
PROBES = OUT / "probes"
MANIFEST = RAW / "manifest.json"
SOURCE = "ecb"

CURRENCIES = ("USD", "JPY", "GBP", "CHF", "AUD", "CAD", "SEK")
API = ("https://data-api.ecb.europa.eu/service/data/EXR/D."
       + "+".join(CURRENCIES) + ".EUR.SP00.A")
HIST = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
API_FIELDS = {"KEY", "CURRENCY", "CURRENCY_DENOM", "TIME_PERIOD", "OBS_VALUE"}
KEEP_HEADERS = ("Date", "Last-Modified", "ETag", "Content-Type", "Content-Length",
                "Content-Disposition")
USER_AGENT = "crypto-dataset-fetch/1.0"
PAUSE_SECONDS = 0.5
PROBE_DAYS = 20
FINE_DAYS = 5
FINE_TIMES = ("13:30", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30", "17:00",
              "18:00", "20:00", "22:00")
SAMPLE_DAYS = 40
SAMPLE_FROM = "2011-08-18"   # Bitstamp's first day: the history the tables can read
SAMPLE_SEED = 20261128


class FetchError(RuntimeError):
    """The source did not answer as expected."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def http_get(url: str, params: dict | None, allow_404: bool = False):
    """GET with a few retries on rate limits and server errors.

    Returns the full URL, the UTC time of the request, the HTTP status, the
    headers kept for the manifest, and the body as received.
    """
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    request = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, 5):
        fetched = utc_now()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                headers = {k: response.headers[k] for k in KEEP_HEADERS if k in response.headers}
                return full, fetched, response.status, headers, response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
            if exc.code == 404 and allow_404:
                headers = {k: exc.headers[k] for k in KEEP_HEADERS if k in exc.headers}
                return full, fetched, 404, headers, body
            if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(5 * attempt)
                continue
            raise FetchError(f"{full} answered HTTP {exc.code}: {body[:400]!r}") from None
        except urllib.error.URLError as exc:
            if attempt < 4:
                time.sleep(5 * attempt)
                continue
            raise FetchError(f"{full} could not be reached: {exc.reason}") from None
    raise AssertionError("unreachable")


def api_rows(body: bytes, url: str) -> list[dict]:
    """The observations of one SDMX-CSV answer, after checking its shape."""
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise FetchError(f"{url}: the answer is not UTF-8 text") from None
    reader = csv.DictReader(io.StringIO(text))
    fields = set(reader.fieldnames or [])
    if not API_FIELDS <= fields:
        raise FetchError(f"{url}: missing fields {sorted(API_FIELDS - fields)}: {body[:300]!r}")
    return list(reader)


def check_api(rows: list[dict], url: str, last_day: date) -> dict:
    """The main answer: the seven currencies against EUR, no day after the last."""
    currencies = {r["CURRENCY"] for r in rows}
    if currencies != set(CURRENCIES):
        raise FetchError(f"{url}: currencies {sorted(currencies)}")
    if {r["CURRENCY_DENOM"] for r in rows} != {"EUR"}:
        raise FetchError(f"{url}: a rate not against EUR")
    days = sorted({r["TIME_PERIOD"] for r in rows})
    for d in days:
        try:
            date.fromisoformat(d)
        except ValueError:
            raise FetchError(f"{url}: TIME_PERIOD {d!r} is not a date") from None
    if days[-1] > last_day.isoformat():
        raise FetchError(f"{url}: holds {days[-1]}, after {last_day}")
    keys = [(r["CURRENCY"], r["TIME_PERIOD"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise FetchError(f"{url}: a currency has two rows for one day")
    return {"first": days[0], "last": days[-1], "days": days}


def check_hist(body: bytes, last_day: date) -> dict:
    """The historical file: one CSV holding a Date column and the seven currencies."""
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            names = archive.namelist()
            if len(names) != 1:
                raise FetchError(f"{HIST}: the zip holds {names}")
            text = archive.read(names[0]).decode("utf-8-sig")
    except zipfile.BadZipFile:
        raise FetchError(f"{HIST}: not a zip file: {body[:200]!r}") from None
    reader = csv.reader(io.StringIO(text))
    header = [h.strip() for h in next(reader)]
    missing = [c for c in ("Date",) + CURRENCIES if c not in header]
    if missing:
        raise FetchError(f"{HIST}: {names[0]} lacks columns {missing}")
    rows = [r for r in reader if r and any(c.strip() for c in r)]
    days = sorted(r[header.index("Date")].strip() for r in rows)
    if days[-1] > last_day.isoformat():
        raise FetchError(f"{HIST} holds {days[-1]}, after {last_day}: it cannot be cut "
                         f"by date, so it is not kept")
    return {"member": names[0], "rows": len(rows), "first": days[0], "last": days[-1]}


def record(name: str, url: str, fetched: datetime, status: int, headers: dict | None,
           body: bytes, rows: int, **extra) -> dict:
    out = {"file": f"ecb/{name}", "url": url, "fetched_utc": iso(fetched), "http_status": status,
           "rows": rows, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
    if headers is not None:
        out["response_headers"] = headers
    out.update(extra)
    return out


def write_manifest(entry: dict) -> None:
    """Set this source's entry in manifest.json; every other entry keeps its bytes."""
    before = MANIFEST.read_bytes()
    manifest = json.loads(before)
    if (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8") != before:
        raise FetchError("manifest.json does not re-serialise to its own bytes; not touched")
    others = {k: v for k, v in manifest.items() if k != SOURCE}
    manifest[SOURCE] = entry
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary = MANIFEST.with_name(MANIFEST.name + ".tmp")
    temporary.write_bytes(text.encode("utf-8"))
    os.replace(temporary, MANIFEST)
    after = json.loads(MANIFEST.read_bytes())
    if {k: v for k, v in after.items() if k != SOURCE} != others:
        raise FetchError("an entry of another source changed in manifest.json")


def probe_times(days: list[str]) -> list[tuple[str, datetime]]:
    """(fixing date, instant) pairs for the updatedAfter probes.

    ``days`` are the dates that have a value, in order.
    """
    pairs = []
    recent = days[-PROBE_DAYS:]
    older = [d for d in days[:-PROBE_DAYS] if d >= SAMPLE_FROM]
    sample = sorted(random.Random(SAMPLE_SEED).sample(older, SAMPLE_DAYS))
    for d in sample + recent:
        day = date.fromisoformat(d)
        noon = datetime(day.year, day.month, day.day, 12, 0, tzinfo=timezone.utc)
        times = [noon]
        if d in recent[-FINE_DAYS:]:
            for hm in FINE_TIMES:
                h, m = (int(x) for x in hm.split(":"))
                times.append(datetime(day.year, day.month, day.day, h, m, tzinfo=timezone.utc))
        times.append(datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(days=1))
        pairs += [(d, t) for t in times]
    return pairs


def main() -> int:
    started = utc_now()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    last_day = date.fromisoformat(manifest["bitstamp"]["last_complete_day"])
    if last_day >= started.date():
        raise FetchError(f"the last day {last_day} has not ended")
    print(f"last day: {last_day} (the Bitstamp entry's last completed day)")

    url, fetched, status, headers, body = http_get(
        API, {"format": "csvdata", "endPeriod": last_day.isoformat()})
    rows = api_rows(body, url)
    api = check_api(rows, url, last_day)
    files = [(OUT / "exr_daily.csv", body,
              record("exr_daily.csv", url, fetched, status, headers, body, len(rows),
                     first=api["first"], last=api["last"]))]
    counts = {c: sum(r["CURRENCY"] == c for r in rows) for c in CURRENCIES}
    print(f"exr_daily.csv: {len(rows)} rows, {len(api['days'])} dates "
          f"{api['first']}..{api['last']}, {len(body)} bytes; per currency {counts}")
    time.sleep(PAUSE_SECONDS)

    url, fetched, status, headers, body = http_get(HIST, None)
    hist = check_hist(body, last_day)
    files.append((OUT / "eurofxref-hist.zip", body,
                  record("eurofxref-hist.zip", url, fetched, status, headers, body,
                         hist["rows"], member=hist["member"], first=hist["first"],
                         last=hist["last"])))
    print(f"eurofxref-hist.zip: {hist['member']}, {hist['rows']} rows "
          f"{hist['first']}..{hist['last']}, {len(body)} bytes; "
          f"Last-Modified {headers.get('Last-Modified')}")

    fixing_days = sorted({r["TIME_PERIOD"] for r in rows if r["OBS_VALUE"].strip()})
    answers: dict[str, list[str]] = {}
    for d, moment in probe_times(fixing_days):
        time.sleep(PAUSE_SECONDS)
        after = moment.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        url, fetched, status, headers, body = http_get(
            API, {"format": "csvdata", "startPeriod": d, "endPeriod": d, "updatedAfter": after},
            allow_404=True)
        n = len(api_rows(body, url)) if status == 200 and body.strip() else 0
        name = f"probes/{d}_after_{moment.strftime('%Y%m%dT%H%MZ')}.csv"
        files.append((OUT / name, body,
                      record(name, url, fetched, status, None, body, n,
                             period=d, updated_after=after)))
        answers.setdefault(d, []).append(f"{moment.strftime('%d %H:%M')} {status}/{n}")
    for d, parts in answers.items():
        print(f"probe {d} (updatedAfter: HTTP/rows): {'; '.join(parts)}")

    OUT.mkdir(parents=True, exist_ok=True)
    for old in sorted(OUT.rglob("*"), reverse=True):
        if old.is_file():
            old.unlink()
        elif old.is_dir():
            old.rmdir()
    PROBES.mkdir(parents=True, exist_ok=True)
    for path, body, _ in files:
        path.write_bytes(body)
    write_manifest({
        "script": "scripts/fetch_ecb.py",
        "run_started_utc": iso(started),
        "last_complete_day": last_day.isoformat(),
        "storage": ("one file per request, the response body byte for byte: the API's "
                    "SDMX-CSV, the historical zip, and the updatedAfter probes under probes/"),
        "files": [rec for _, _, rec in files],
    })
    print(f"wrote {len(files)} files under data/raw/ecb/ and the manifest's ecb entry")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FetchError as exc:
        print(f"FETCH FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
