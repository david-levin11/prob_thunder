from pathlib import Path
from collections import Counter
import ijson
import pandas as pd

GEOJSON_FILE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\aicc_lightning\TOALightningPoints_2012_2025_WGS84.geojson"
)

rows = []
stroke_types = Counter()
polarities = Counter()

with GEOJSON_FILE.open("rb") as f:
    for i, feature in enumerate(ijson.items(f, "features.item"), start=1):
        props = feature["properties"]
        geom = feature["geometry"]
        coords = geom["coordinates"]

        stroke_types[props.get("STROKETYPE")] += 1
        polarities[props.get("POLARITY")] += 1

        if len(rows) < 20:
            utc_raw = props.get("UTCDATETIME")
            rows.append({
                "lon_geom": coords[0],
                "lat_geom": coords[1],
                "lon_attr": props.get("LONGITUDE"),
                "lat_attr": props.get("LATITUDE"),
                "UTCDATETIME": utc_raw,
                "datetime_from_ms": pd.to_datetime(utc_raw, unit="ms", utc=True, errors="coerce"),
                "LOCALDATETIME": props.get("LOCALDATETIME"),
                "MILLISECONDS": props.get("MILLISECONDS"),
                "STRIKETIME": props.get("STRIKETIME"),
                "AMPLITUDE": props.get("AMPLITUDE"),
                "peak_current_ka": props.get("AMPLITUDE") / 1000.0 if props.get("AMPLITUDE") is not None else None,
                "POLARITY": props.get("POLARITY"),
                "STROKETYPE": props.get("STROKETYPE"),
                "GDOP": props.get("GDOP"),
                "ERRSEMIMAJOR": props.get("ERRSEMIMAJOR"),
                "ERRSEMIMINOR": props.get("ERRSEMIMINOR"),
            })

        if i >= 100_000:
            break

sample = pd.DataFrame(rows)

print(sample.to_string(index=False))
print("\nSTROKETYPE counts:")
print(stroke_types)
print("\nPOLARITY counts:")
print(polarities)