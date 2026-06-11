#!/usr/bin/env python

"""Inspect a large GeoJSON lightning dataset without loading it fully into memory.

This script:
- Prints top-level GeoJSON structure.
- Prints the first few feature geometries/properties.
- Summarizes property names and Python types.
- Counts geometry types.
- Reports approximate coordinate bounds.
- Optionally writes a small sample GeoJSON/CSV for easier inspection.

Requires:
    pip install ijson pandas
"""

from pathlib import Path
from collections import Counter, defaultdict
import json
import ijson
import pandas as pd


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

GEOJSON_FILE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\aicc_lightning\TOALightningPoints_2012_2025_WGS84.geojson"
)

N_SAMPLE = 20

OUT_SAMPLE_CSV = GEOJSON_FILE.with_name("lightning_geojson_sample.csv")
OUT_SUMMARY_TXT = GEOJSON_FILE.with_name("lightning_geojson_summary.txt")


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def safe_type_name(value):
    """Return a simple type name for a JSON value."""
    if value is None:
        return "None"
    return type(value).__name__


def extract_point_coords(geometry):
    """
    Return lon, lat from Point geometry if possible.

    GeoJSON coordinates are normally [lon, lat] or [lon, lat, z].
    """
    if not geometry:
        return None, None

    if geometry.get("type") != "Point":
        return None, None

    coords = geometry.get("coordinates")

    if not isinstance(coords, list) or len(coords) < 2:
        return None, None

    return coords[0], coords[1]


def inspect_geojson(path, n_sample=20):
    """Stream through a large GeoJSON FeatureCollection and summarize structure."""

    if not path.exists():
        raise FileNotFoundError(path)

    feature_count = 0
    geometry_types = Counter()
    property_names = Counter()
    property_types = defaultdict(Counter)
    sample_rows = []

    lon_min = lon_max = None
    lat_min = lat_max = None

    with path.open("rb") as f:
        # This assumes normal GeoJSON:
        # {
        #   "type": "FeatureCollection",
        #   "features": [
        #      {"type": "Feature", "geometry": ..., "properties": ...}
        #   ]
        # }
        features = ijson.items(f, "features.item")

        for feature in features:
            feature_count += 1

            geometry = feature.get("geometry", {})
            properties = feature.get("properties", {}) or {}

            geom_type = geometry.get("type", "UNKNOWN")
            geometry_types[geom_type] += 1

            lon, lat = extract_point_coords(geometry)

            if lon is not None and lat is not None:
                lon_min = lon if lon_min is None else min(lon_min, lon)
                lon_max = lon if lon_max is None else max(lon_max, lon)
                lat_min = lat if lat_min is None else min(lat_min, lat)
                lat_max = lat if lat_max is None else max(lat_max, lat)

            for key, value in properties.items():
                property_names[key] += 1
                property_types[key][safe_type_name(value)] += 1

            if len(sample_rows) < n_sample:
                row = {
                    "_feature_index": feature_count,
                    "_geometry_type": geom_type,
                    "_lon": lon,
                    "_lat": lat,
                }
                row.update(properties)
                sample_rows.append(row)

            if feature_count % 500_000 == 0:
                print(f"Scanned {feature_count:,} features...")

    sample_df = pd.DataFrame(sample_rows)

    summary_lines = []
    summary_lines.append(f"File: {path}")
    summary_lines.append(f"Total features: {feature_count:,}")
    summary_lines.append("")
    summary_lines.append("Geometry types:")
    for k, v in geometry_types.most_common():
        summary_lines.append(f"  {k}: {v:,}")

    summary_lines.append("")
    summary_lines.append("Approximate Point coordinate bounds:")
    summary_lines.append(f"  lon_min: {lon_min}")
    summary_lines.append(f"  lon_max: {lon_max}")
    summary_lines.append(f"  lat_min: {lat_min}")
    summary_lines.append(f"  lat_max: {lat_max}")

    summary_lines.append("")
    summary_lines.append("Property fields:")
    for key in sorted(property_names):
        type_summary = ", ".join(
            f"{typ}={cnt:,}" for typ, cnt in property_types[key].most_common()
        )
        summary_lines.append(
            f"  {key}: present in {property_names[key]:,} features; types: {type_summary}"
        )

    summary_text = "\n".join(summary_lines)

    print(summary_text)

    sample_df.to_csv(OUT_SAMPLE_CSV, index=False)
    OUT_SUMMARY_TXT.write_text(summary_text, encoding="utf-8")

    print(f"\nWrote sample CSV: {OUT_SAMPLE_CSV}")
    print(f"Wrote summary TXT: {OUT_SUMMARY_TXT}")

    print("\nSample rows:")
    print(sample_df.head(n_sample).to_string(index=False))


if __name__ == "__main__":
    inspect_geojson(GEOJSON_FILE, n_sample=N_SAMPLE)