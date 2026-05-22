#!/usr/bin/env python

from pathlib import Path
import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


GEOJSON_FILE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\aicc_lightning\TOALightningPoints_2012_2025_WGS84.geojson"
)

OUT_PARQUET_DIR = Path(
    r"C:\Users\David.Levin\NBMLightningVer\aicc_lightning\parquet"
)
OUT_PARQUET_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 250_000


def feature_to_row(feature):
    props = feature.get("properties", {}) or {}
    geom = feature.get("geometry", {}) or {}
    coords = geom.get("coordinates", [None, None])

    lon = coords[0] if len(coords) >= 1 else props.get("LONGITUDE")
    lat = coords[1] if len(coords) >= 2 else props.get("LATITUDE")

    amplitude_a = props.get("AMPLITUDE")

    return {
        "utc_datetime_ms": props.get("UTCDATETIME"),
        "lat": lat,
        "lon": lon,
        "amplitude_a": amplitude_a,
        "peak_current_ka": amplitude_a / 1000.0 if amplitude_a is not None else None,
        "polarity": props.get("POLARITY"),
        "stroke_type": props.get("STROKETYPE"),
        "is_ground": str(props.get("STROKETYPE")).upper() == "GROUND_STROKE",
        "is_cloud": str(props.get("STROKETYPE")).upper() == "CLOUD_STROKE",
        "err_semi_major": props.get("ERRSEMIMAJOR"),
        "err_semi_minor": props.get("ERRSEMIMINOR"),
        "err_ellipse_angle": props.get("ERRELIPSEANGLE"),
        "gdop": props.get("GDOP"),
        "network_code": props.get("NETWORKCODE"),
        "milliseconds": props.get("MILLISECONDS"),
        "strike_seq_number": props.get("STRIKESEQNUMBER"),
        "objectid": props.get("OBJECTID"),
    }


def write_chunk(rows, chunk_number):
    df = pd.DataFrame(rows)

    df["datetime"] = pd.to_datetime(
        df["utc_datetime_ms"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(subset=["datetime", "lat", "lon"])

    df["year"] = df["datetime"].dt.year.astype("int16")
    df["month"] = df["datetime"].dt.month.astype("int8")
    df["day"] = df["datetime"].dt.day.astype("int8")
    df["hour"] = df["datetime"].dt.hour.astype("int8")

    numeric_cols = [
        "utc_datetime_ms",
        "amplitude_a",
        "peak_current_ka",
        "err_semi_major",
        "err_semi_minor",
        "err_ellipse_angle",
        "gdop",
        "network_code",
        "milliseconds",
        "strike_seq_number",
        "objectid",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    table = pa.Table.from_pandas(df, preserve_index=False)

    pq.write_to_dataset(
        table,
        root_path=str(OUT_PARQUET_DIR),
        partition_cols=["year", "month"],
    )

    print(f"Wrote chunk {chunk_number:,}: {len(df):,} rows")


def main():
    rows = []
    total = 0
    chunk_number = 0

    with GEOJSON_FILE.open("rb") as f:
        for feature in ijson.items(f, "features.item"):
            rows.append(feature_to_row(feature))
            total += 1

            if len(rows) >= CHUNK_SIZE:
                chunk_number += 1
                write_chunk(rows, chunk_number)
                rows = []

            if total % 500_000 == 0:
                print(f"Processed {total:,} features")

    if rows:
        chunk_number += 1
        write_chunk(rows, chunk_number)

    print(f"Done. Processed {total:,} features.")
    print(f"Parquet written to: {OUT_PARQUET_DIR}")


if __name__ == "__main__":
    main()