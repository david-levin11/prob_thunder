"""
Find thunder/lightning reports in ASOS/AWOS METARs over the NWS Juneau CWA
using the Synoptic Weather API.

Example:
    python find_ajk_thunder_metars.py \
        --start 2020-01-01 \
        --end 2025-12-31 \
        --output ajk_thunder_metars.csv

Token:
    Set your Synoptic API token as an environment variable:

    Windows PowerShell:
        $env:SYNOPTIC_TOKEN="your_token_here"

    macOS/Linux:
        export SYNOPTIC_TOKEN="your_token_here"
"""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://api.synopticdata.com/v2/stations/timeseries"

# METAR present weather thunderstorm / lightning indicators.
#
# TS    = thunderstorm present weather group
# VCTS  = thunderstorm in vicinity
# LTG   = lightning, often in remarks
# TSB/ TSE = thunderstorm began / ended, often in remarks
#
# Examples this should catch:
#   - TSRA
#   - -TSRA
#   - VCTS
#   - +TSGR
#   - LTG DSNT W
#   - TSB12
#   - TSE34
THUNDER_REGEX = re.compile(
    r"""
    (?:
        (?<![A-Z])
        [-+]?
        (?:VC)?
        TS
        [A-Z]{0,6}
        (?![A-Z])
    )
    |
    (?:
        (?<![A-Z])
        LTG
        (?![A-Z])
    )
    |
    (?:
        (?<![A-Z])
        TS[BE]\d{0,4}
        (?![A-Z])
    )
    """,
    re.VERBOSE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find thunder/lightning METAR reports at ASOS/AWOS sites in the NWS Juneau CWA."
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Start date/time in UTC. Accepts YYYY-MM-DD or YYYY-MM-DDTHH:MM.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date/time in UTC. Accepts YYYY-MM-DD or YYYY-MM-DDTHH:MM.",
    )
    parser.add_argument(
        "--output",
        default="ajk_thunder_metars.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=30,
        help="Number of days per API request chunk. Default: 30.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Seconds to sleep between API requests. Default: 0.25.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Synoptic API token. If omitted, uses SYNOPTIC_TOKEN environment variable.",
    )
    parser.add_argument(
        "--include-hfmetars",
        action="store_true",
        help="Include high-frequency 5-minute HF-METAR observations. Default is traditional METAR/SPECI only.",
    )

    return parser.parse_args()


def parse_datetime_utc(value: str) -> datetime:
    """
    Parse YYYY-MM-DD or YYYY-MM-DDTHH:MM into timezone-aware UTC datetime.
    """
    value = value.strip()

    if len(value) == 10:
        dt = datetime.strptime(value, "%Y-%m-%d")
    else:
        value = value.replace("Z", "")
        dt = datetime.strptime(value, "%Y-%m-%dT%H:%M")

    return dt.replace(tzinfo=timezone.utc)


def synoptic_time(dt: datetime) -> str:
    """
    Convert datetime to Synoptic API time format: YYYYmmddHHMM.
    """
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M")


def date_chunks(
    start: datetime,
    end: datetime,
    chunk_days: int,
) -> list[tuple[datetime, datetime]]:
    """
    Split a long request period into smaller chunks.
    """
    chunks = []
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((current, chunk_end))
        current = chunk_end

    return chunks


def request_timeseries(
    token: str,
    start: datetime,
    end: datetime,
    include_hfmetars: bool = False,
) -> dict[str, Any]:
    """
    Query Synoptic Time Series for ASOS/AWOS METARs in the AJK CWA.
    """
    params = {
        "token": token,
        "cwa": "AJK",
        "network": "1",  # ASOS/AWOS network
        "vars": "metar",
        "start": synoptic_time(start),
        "end": synoptic_time(end),
        "obtimezone": "UTC",
        "hfmetars": "1" if include_hfmetars else "0",
        "output": "json",
    }

    response = requests.get(BASE_URL, params=params, timeout=60)

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Synoptic request failed: {response.status_code}\n"
            f"URL: {response.url}\n"
            f"Response: {response.text[:1000]}"
        ) from exc

    data = response.json()

    summary = data.get("SUMMARY", {})
    response_code = str(summary.get("RESPONSE_CODE", ""))

    # Synoptic often uses RESPONSE_CODE=1 for OK.
    # Other codes may still include useful text, so expose the message.
    if response_code not in {"1", "OK"}:
        message = summary.get("RESPONSE_MESSAGE", "Unknown Synoptic API response")
        raise RuntimeError(
            f"Synoptic API returned RESPONSE_CODE={response_code}: {message}\n"
            f"URL: {response.url}"
        )

    return data


def extract_station_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract METAR rows containing thunder/lightning text from one Synoptic response.
    """
    events: list[dict[str, Any]] = []

    for station in data.get("STATION", []):
        stid = station.get("STID")
        name = station.get("NAME")
        latitude = station.get("LATITUDE")
        longitude = station.get("LONGITUDE")
        elevation = station.get("ELEVATION")
        mnet_id = station.get("MNET_ID")
        status = station.get("STATUS")

        observations = station.get("OBSERVATIONS", {})

        dates = observations.get("date_time", [])
        metars = observations.get("metar", [])

        # Synoptic variable arrays are typically parallel arrays.
        # Some responses may omit metar or return None.
        if not dates or not metars:
            continue

        for valid_time, metar in zip(dates, metars):
            if not metar:
                continue

            metar = str(metar).strip()
            matches = [m.group(0) for m in THUNDER_REGEX.finditer(metar)]

            if not matches:
                continue

            events.append(
                {
                    "stid": stid,
                    "name": name,
                    "valid_time_utc": valid_time,
                    "lat": latitude,
                    "lon": longitude,
                    "elevation": elevation,
                    "mnet_id": mnet_id,
                    "status": status,
                    "matched_codes": ",".join(sorted(set(matches))),
                    "metar": metar,
                }
            )

    return events


def main() -> None:
    args = parse_args()

    token = args.token or os.getenv("SYNOPTIC_TOKEN")
    if not token:
        raise SystemExit(
            "No Synoptic token provided. Use --token or set SYNOPTIC_TOKEN."
        )

    start = parse_datetime_utc(args.start)
    end = parse_datetime_utc(args.end)

    if end <= start:
        raise SystemExit("--end must be after --start")

    chunks = date_chunks(start, end, args.chunk_days)

    all_events: list[dict[str, Any]] = []

    print(f"Searching AJK ASOS/AWOS METARs from {start} to {end}")
    print(f"Using {len(chunks)} request chunk(s) of up to {args.chunk_days} days")

    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(
            f"[{i}/{len(chunks)}] "
            f"{chunk_start:%Y-%m-%d %H:%M} to {chunk_end:%Y-%m-%d %H:%M} UTC"
        )

        data = request_timeseries(
            token=token,
            start=chunk_start,
            end=chunk_end,
            include_hfmetars=args.include_hfmetars,
        )

        events = extract_station_events(data)
        print(f"    found {len(events)} thunder/lightning METAR(s)")

        all_events.extend(events)

        if args.sleep > 0:
            time.sleep(args.sleep)

    if not all_events:
        print("No thunder/lightning METARs found.")
        pd.DataFrame(
            columns=[
                "stid",
                "name",
                "valid_time_utc",
                "lat",
                "lon",
                "elevation",
                "mnet_id",
                "status",
                "matched_codes",
                "metar",
            ]
        ).to_csv(args.output, index=False)
        return

    df = pd.DataFrame(all_events)

    # Remove duplicate station/time/METAR combinations in case chunk boundaries overlap
    # or Synoptic returns repeated reports.
    df = df.drop_duplicates(subset=["stid", "valid_time_utc", "metar"])

    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True, errors="coerce")
    df = df.sort_values(["valid_time_utc", "stid"])

    df.to_csv(args.output, index=False)

    print()
    print(f"Wrote {len(df):,} thunder/lightning METAR(s) to: {args.output}")
    print()
    print("Counts by station:")
    print(df.groupby(["stid", "name"]).size().sort_values(ascending=False))


if __name__ == "__main__":
    main()