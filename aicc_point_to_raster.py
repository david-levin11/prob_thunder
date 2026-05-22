#!/usr/bin/env python

"""Create 6-hour and 12-hour binary lightning truth rasters from AICC Parquet data.

Optimized approach:
- Load one month of lightning points from partitioned Parquet.
- Also load a small amount of prior-month data needed for windows ending early
  on the first day of the month.
- For each valid time ending at 00Z, 06Z, 12Z, and 18Z:
    - subset points in [valid_time - window, valid_time)
    - project lon/lat points to the NBM raster CRS
    - map points to raster row/column
    - write binary raster on the NBM grid

Output:
    aicc_total_06_20km/aicc_total_06h_YYYYMMDD_HH00Z.tif
    aicc_total_12_20km/aicc_total_12h_YYYYMMDD_HH00Z.tif

Binary raster values:
    1 = one or more lightning strokes in cell/window
    0 = no lightning in cell/window
    NaN = outside Alaska mask, if APPLY_ALASKA_MASK_TO_OUTPUT=True
"""

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
import rioxarray as rxr
import geopandas as gpd

from pyproj import Transformer
import rasterio
from rasterio.transform import rowcol
from rasterio.features import geometry_mask


# =============================================================================
# CONFIG
# =============================================================================

PARQUET_DIR = Path(
    r"C:\Users\David.Levin\NBMLightningVer\aicc_lightning\parquet"
)

OUT_ROOT = Path(
    r"C:\Users\David.Levin\NBMLightningVer\aicc_lightning\rasters_total"
)

# Representative NBM raster. All output rasters will match this grid exactly.
TEMPLATE_RASTER = Path(
    r"C:\Users\David.Levin\NBMLightningVer\nbm_data\2023\07\22\0100\tstm12\blendv4.1_alaska_tstm12_2023-07-22T0100_F077.tif"
)

ALASKA_BOUNDARY_FILE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\reference\cb_2018_us_state_5m.shp"
)

# Full AICC period
YEARS = list(range(2012, 2026))
#YEARS = [2024]

# Use all months for climatology creation.
# For only thunder season, change to range(3, 11).
MONTHS = range(1, 13)
#MONTHS = [6]

WINDOWS = [6, 12]
VALID_HOURS = [0, 6, 12, 18]

# "total"  = GROUND_STROKE + CLOUD_STROKE
# "ground" = GROUND_STROKE only
# "cloud"  = CLOUD_STROKE only
LIGHTNING_MODE = "total"

# Used in output folder/file names.
OUTPUT_PREFIX = f"aicc_{LIGHTNING_MODE}"

APPLY_ALASKA_MASK_TO_OUTPUT = True
ALASKA_MASK_ALL_TOUCHED = True

SKIP_EXISTING = True

# Set to a small number for testing, e.g. 20.
MAX_FILES_TO_WRITE = None

# Columns expected from your Parquet conversion.
PARQUET_COLUMNS = [
    "datetime",
    "lat",
    "lon",
    "stroke_type",
    "is_ground",
    "is_cloud",
]

OUTPUT_NODATA = -9999.0
# =============================================================================
# GRID / MASK HELPERS
# =============================================================================

def load_template():
    """Load template raster and grid metadata."""

    if not TEMPLATE_RASTER.exists():
        raise FileNotFoundError(f"Template raster not found: {TEMPLATE_RASTER}")

    template = rxr.open_rasterio(TEMPLATE_RASTER, mask_and_scale=True)

    shape = template.values[0].shape
    transform = template.rio.transform()
    crs = template.rio.crs

    if crs is None:
        raise ValueError("Template raster has no CRS.")

    print("Template grid:")
    print(f"  Shape: {shape}")
    print(f"  CRS: {crs}")
    print(f"  Bounds: {template.rio.bounds()}")

    return template, shape, transform, crs


def load_alaska_boundary(target_crs):
    """Load Alaska boundary and reproject it to the raster CRS."""

    if not ALASKA_BOUNDARY_FILE.exists():
        raise FileNotFoundError(f"Alaska boundary not found: {ALASKA_BOUNDARY_FILE}")

    gdf = gpd.read_file(ALASKA_BOUNDARY_FILE)

    if gdf.empty:
        raise ValueError(f"Boundary file is empty: {ALASKA_BOUNDARY_FILE}")

    if gdf.crs is None:
        # Your boundary .prj indicated NAD83 geographic lat/lon.
        gdf = gdf.set_crs("EPSG:4269")

    # If using a full US state boundary file, filter to Alaska.
    if "STUSPS" in gdf.columns:
        gdf = gdf[gdf["STUSPS"].astype(str).str.upper().eq("AK")]
    elif "STATEFP" in gdf.columns:
        gdf = gdf[gdf["STATEFP"].astype(str).eq("02")]
    elif "NAME" in gdf.columns:
        gdf = gdf[gdf["NAME"].astype(str).str.lower().eq("alaska")]

    if gdf.empty:
        raise ValueError(
            "Could not identify Alaska polygon in boundary file. "
            "Check STUSPS, STATEFP, or NAME fields."
        )

    return gdf.to_crs(target_crs)


def build_alaska_mask(template):
    """Rasterize Alaska boundary to the template grid."""

    alaska = load_alaska_boundary(template.rio.crs)

    mask = geometry_mask(
        alaska.geometry,
        out_shape=template.values[0].shape,
        transform=template.rio.transform(),
        invert=True,  # True inside Alaska
        all_touched=ALASKA_MASK_ALL_TOUCHED,
    )

    print(
        f"Alaska mask: {mask.sum():,} of {mask.size:,} cells "
        f"({mask.sum() / mask.size:.2%})"
    )

    return mask


# =============================================================================
# PARQUET LOADING
# =============================================================================

def month_start_end(year, month):
    """Return start/end datetimes for one month."""

    start = datetime(year, month, 1)

    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)

    return start, end


def previous_month(year, month):
    """Return previous year/month."""

    if month == 1:
        return year - 1, 12

    return year, month - 1


def load_partition_month(dataset, year, month, start_dt=None, end_dt=None):
    """Load one year/month partition with optional datetime filtering."""

    # Timestamp objects for pyarrow filtering
    filters = [
        ds.field("year") == year,
        ds.field("month") == month,
    ]

    if start_dt is not None:
        filters.append(ds.field("datetime") >= pd.Timestamp(start_dt, tz="UTC"))

    if end_dt is not None:
        filters.append(ds.field("datetime") < pd.Timestamp(end_dt, tz="UTC"))

    filt = filters[0]
    for f in filters[1:]:
        filt = filt & f

    try:
        table = dataset.to_table(columns=PARQUET_COLUMNS, filter=filt)
    except Exception:
        # In case is_ground/is_cloud are absent, fall back to required columns.
        cols = ["datetime", "lat", "lon", "stroke_type"]
        table = dataset.to_table(columns=cols, filter=filt)

    if table.num_rows == 0:
        return pd.DataFrame(columns=["datetime", "lat", "lon", "stroke_type"])

    df = table.to_pandas()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lon"])

    return df


def load_month_with_lookback(dataset, year, month, max_window_hours):
    """
    Load current month plus enough prior-month data for early-month windows.

    Example:
      Valid time 2023-07-01 00Z with a 12h window needs data back to
      2023-06-30 12Z.
    """

    start, end = month_start_end(year, month)
    lookback_start = start - timedelta(hours=max_window_hours)

    print(
        f"  Loading points for {year}-{month:02d} "
        f"plus lookback from {lookback_start:%Y-%m-%d %H:%MZ}"
    )

    frames = []

    # Prior-month lookback
    prev_y, prev_m = previous_month(year, month)
    if lookback_start.year == prev_y and lookback_start.month == prev_m:
        prev_df = load_partition_month(
            dataset,
            prev_y,
            prev_m,
            start_dt=lookback_start,
            end_dt=start,
        )
        if not prev_df.empty:
            frames.append(prev_df)

    # Current month
    cur_df = load_partition_month(
        dataset,
        year,
        month,
        start_dt=start,
        end_dt=end,
    )
    if not cur_df.empty:
        frames.append(cur_df)

    if not frames:
        return pd.DataFrame(columns=["datetime", "lat", "lon", "stroke_type"])

    df = pd.concat(frames, ignore_index=True)

    # Ensure only needed time span remains.
    df = df[
        (df["datetime"] >= pd.Timestamp(lookback_start, tz="UTC")) &
        (df["datetime"] < pd.Timestamp(end, tz="UTC"))
    ].copy()

    print(f"  Loaded {len(df):,} points before mode filter")

    return df


def filter_lightning_mode(df):
    """Filter to total, ground-only, or cloud-only lightning."""

    if df.empty:
        return df

    stroke = df["stroke_type"].astype(str).str.upper()

    if LIGHTNING_MODE == "total":
        return df[stroke.isin(["GROUND_STROKE", "CLOUD_STROKE"])].copy()

    if LIGHTNING_MODE == "ground":
        return df[stroke.eq("GROUND_STROKE")].copy()

    if LIGHTNING_MODE == "cloud":
        return df[stroke.eq("CLOUD_STROKE")].copy()

    raise ValueError(f"Unknown LIGHTNING_MODE: {LIGHTNING_MODE}")


# =============================================================================
# RASTERIZATION
# =============================================================================

def build_valid_times(year, month):
    """Build valid times ending at 00, 06, 12, and 18Z for one month."""

    start, end = month_start_end(year, month)

    valid_times = []
    current = start

    while current < end:
        if current.hour in VALID_HOURS:
            valid_times.append(current)

        current += timedelta(hours=1)

    return valid_times


def output_path(window, valid_dt):
    """Return output path for one window/valid time."""

    window_str = f"{window:02d}"

    out_dir = OUT_ROOT / f"{OUTPUT_PREFIX}_{window_str}_20km"
    filename = (
        f"{OUTPUT_PREFIX}_{window_str}h_"
        f"{valid_dt.strftime('%Y%m%d_%H00Z')}.tif"
    )

    return out_dir / filename


def project_points_to_grid(df, transformer, transform, shape):
    """Project lon/lat points and return valid raster rows/cols."""

    if df.empty:
        return np.array([], dtype=int), np.array([], dtype=int)

    x, y = transformer.transform(
        df["lon"].to_numpy(dtype=float),
        df["lat"].to_numpy(dtype=float),
    )

    rows, cols = rowcol(transform, x, y)

    rows = np.asarray(rows)
    cols = np.asarray(cols)

    in_grid = (
        (rows >= 0) &
        (rows < shape[0]) &
        (cols >= 0) &
        (cols < shape[1])
    )

    return rows[in_grid], cols[in_grid]


def points_to_binary_grid(df, transformer, shape, transform, alaska_mask=None):
    """Rasterize point presence to a binary grid."""

    out = np.zeros(shape, dtype="float32")

    if not df.empty:
        rows, cols = project_points_to_grid(
            df=df,
            transformer=transformer,
            transform=transform,
            shape=shape,
        )

        if rows.size > 0:
            if alaska_mask is not None:
                inside_ak = alaska_mask[rows, cols]
                rows = rows[inside_ak]
                cols = cols[inside_ak]

            if rows.size > 0:
                out[rows, cols] = 1.0

    if APPLY_ALASKA_MASK_TO_OUTPUT and alaska_mask is not None:
        out[~alaska_mask] = OUTPUT_NODATA

    return out


def write_raster(template, arr, out_path):
    """Write a 2D array as a GeoTIFF using template raster metadata."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": template.rio.crs,
        "transform": template.rio.transform(),
        "nodata": OUTPUT_NODATA if APPLY_ALASKA_MASK_TO_OUTPUT else None,
        "compress": "deflate",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def create_truth_rasters():
    """Create binary lightning truth rasters from monthly Parquet loads."""

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    dataset = ds.dataset(PARQUET_DIR, format="parquet", partitioning="hive")

    template, shape, transform, raster_crs = load_template()

    alaska_mask = None
    if APPLY_ALASKA_MASK_TO_OUTPUT:
        alaska_mask = build_alaska_mask(template)

    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)

    files_written = 0
    max_window = max(WINDOWS)

    for year in YEARS:
        for month in MONTHS:
            print(f"\nProcessing {year}-{month:02d}")

            month_df = load_month_with_lookback(
                dataset=dataset,
                year=year,
                month=month,
                max_window_hours=max_window,
            )

            month_df = filter_lightning_mode(month_df)

            print(
                f"  Points after {LIGHTNING_MODE!r} filter: "
                f"{len(month_df):,}"
            )

            valid_times = build_valid_times(year, month)

            for valid_dt in valid_times:
                valid_ts = pd.Timestamp(valid_dt, tz="UTC")

                for window in WINDOWS:
                    out_path = output_path(window, valid_dt)

                    if SKIP_EXISTING and out_path.exists():
                        continue

                    start_ts = valid_ts - pd.Timedelta(hours=window)

                    subset = month_df[
                        (month_df["datetime"] >= start_ts) &
                        (month_df["datetime"] < valid_ts)
                    ]

                    arr = points_to_binary_grid(
                        df=subset,
                        transformer=transformer,
                        shape=shape,
                        transform=transform,
                        alaska_mask=alaska_mask,
                    )

                    lightning_pixels = int(np.count_nonzero(arr == 1.0))

                    write_raster(template, arr, out_path)
                    files_written += 1

                    lightning_pixels = int(np.count_nonzero(arr == 1.0))
                    valid_zero_pixels = int(np.count_nonzero(arr == 0.0))
                    nodata_pixels = int(np.count_nonzero(arr == OUTPUT_NODATA))

                    print(
                        f"    {window:02d}h {valid_dt:%Y%m%d_%H00Z}: "
                        f"{len(subset):,} points, "
                        f"{lightning_pixels:,} lightning pixels, "
                        f"{valid_zero_pixels:,} no-lightning pixels, "
                        f"{nodata_pixels:,} nodata pixels"
)

                    if MAX_FILES_TO_WRITE is not None and files_written >= MAX_FILES_TO_WRITE:
                        print(f"Reached MAX_FILES_TO_WRITE={MAX_FILES_TO_WRITE}. Stopping.")
                        template.close()
                        return

    template.close()
    print(f"\nDone. Files written: {files_written:,}")


if __name__ == "__main__":
    create_truth_rasters()