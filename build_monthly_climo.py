#!/usr/bin/env python

"""Build monthly lightning climatology rasters for Alaska.

This script can build monthly climatology from different truth-raster datasets,
including GLD, AICC, or a union dataset, as long as the filenames contain:

    <prefix>_06h_YYYYMMDD_HHMMZ.tif
    <prefix>_12h_YYYYMMDD_HHMMZ.tif

Examples:
    gld_06h_20230725_0600Z.tif
    aicc_total_06h_20230725_0600Z.tif
    union_06h_20230725_0600Z.tif

For each interval/month combination:

    climo_prob = number of yes/lightning rasters / number of valid samples

Output examples:
    gld_climo_06h_07_prob.tif
    aicc_total_climo_06h_07_prob.tif
    union_climo_12h_08_prob.tif
"""

import re
from pathlib import Path
from datetime import datetime

import numpy as np
import rioxarray as rxr
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATASET_NAME = "aicc_total"
# Options/examples:
#   "gld"
#   "aicc_total"
#   "union"

# Root folder containing the interval subfolders.
# For GLD:
# BASE_TRUTH_ROOT = Path(r"C:\Users\David.Levin\NBMLightningVer\gld_rasters")
#
# For AICC:
BASE_TRUTH_ROOT = Path(
    r"C:\Users\David.Levin\NBMLightningVer\aicc_lightning\rasters_total_dilated"
)

# Output folder for monthly climatology rasters.
CLIMO_OUT_DIR = Path(
    rf"C:\Users\David.Levin\NBMLightningVer\{DATASET_NAME}_climatology"
)
CLIMO_OUT_DIR.mkdir(parents=True, exist_ok=True)

ALASKA_BOUNDARY_FILE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\reference\cb_2018_us_state_5m.shp"
)

# Years/months used to build climatology.
# For AICC you can use the longer record:
CLIMO_YEARS = list(range(2012, 2026))

# For GLD you may want:
# CLIMO_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

MONTHS = range(3, 11)  # March through October
INTERVAL_HOURS = [6, 12]

# Subfolder and filename settings by dataset.
DATASET_CONFIGS = {
    "gld": {
        "subdir_template": "gld_{interval_str}_20km",
        "file_glob_template": "gld_{interval_str}h_*.tif",
        "output_prefix": "gld_climo",
        "truth_yes_values": [1],
        "input_nodata_values": [255, -9999],
    },
    "aicc_total": {
        "subdir_template": "aicc_total_{interval_str}_20km_dilated",
        "file_glob_template": "aicc_total_{interval_str}h_*.tif",
        "output_prefix": "aicc_total_climo",
        "truth_yes_values": [1],
        "input_nodata_values": [255, -9999],
    },
    "union": {
        "subdir_template": "union_{interval_str}_20km",
        "file_glob_template": "union_{interval_str}h_*.tif",
        "output_prefix": "union_climo",
        "truth_yes_values": [1],
        "input_nodata_values": [255, -9999],
    },
}

# If True, use an Alaska state mask in addition to raster nodata.
USE_ALASKA_MASK = True
ALASKA_MASK_ALL_TOUCHED = True
ALASKA_MASK_CACHE = {}

# Output climatology rasters are continuous probabilities/counts, so float32 is appropriate.
OUTPUT_NODATA = -9999.0


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def get_dataset_config():
    """Return configuration for the selected dataset."""

    if DATASET_NAME not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown DATASET_NAME={DATASET_NAME!r}. "
            f"Known options: {list(DATASET_CONFIGS)}"
        )

    return DATASET_CONFIGS[DATASET_NAME]


def parse_truth_valid_time(path):
    """Parse valid datetime from a truth-raster filename.

    Handles:
        gld_06h_20230725_0600Z.tif
        aicc_total_06h_20230725_0600Z.tif
        union_12h_20230725_1800Z.tif
    """

    match = re.search(r"_(\d{2})h_(\d{8})_(\d{4})Z\.tif$", path.name)

    if not match:
        return None

    _, ymd, hm = match.groups()

    return datetime.strptime(ymd + hm, "%Y%m%d%H%M")


def safe_divide(num, den):
    """Return num / den, with NaN where denominator is zero."""

    out = np.full_like(num, np.nan, dtype="float32")
    valid = den > 0
    out[valid] = num[valid] / den[valid]

    return out


def load_alaska_boundary(target_crs):
    """Load Alaska state boundary and reproject to raster CRS."""

    if not ALASKA_BOUNDARY_FILE.exists():
        raise FileNotFoundError(f"Missing Alaska boundary: {ALASKA_BOUNDARY_FILE}")

    alaska = gpd.read_file(ALASKA_BOUNDARY_FILE)

    if alaska.empty:
        raise ValueError(f"Alaska boundary file is empty: {ALASKA_BOUNDARY_FILE}")

    if alaska.crs is None:
        alaska = alaska.set_crs("EPSG:4269")

    # If this is a full US state file, filter to Alaska.
    if "STUSPS" in alaska.columns:
        alaska = alaska[alaska["STUSPS"].astype(str).str.upper().eq("AK")]
    elif "STATEFP" in alaska.columns:
        alaska = alaska[alaska["STATEFP"].astype(str).eq("02")]
    elif "NAME" in alaska.columns:
        alaska = alaska[alaska["NAME"].astype(str).str.lower().eq("alaska")]

    if alaska.empty:
        raise ValueError("Could not find Alaska polygon in boundary file.")

    return alaska.to_crs(target_crs)


def build_alaska_mask(ds_template):
    """Rasterize Alaska boundary onto a raster grid."""

    raster_shape = ds_template.values[0].shape
    raster_transform = ds_template.rio.transform()
    raster_crs = ds_template.rio.crs

    cache_key = (
        raster_shape,
        tuple(round(x, 6) for x in raster_transform),
        str(raster_crs),
        ALASKA_MASK_ALL_TOUCHED,
    )

    if cache_key in ALASKA_MASK_CACHE:
        return ALASKA_MASK_CACHE[cache_key]

    alaska = load_alaska_boundary(raster_crs)

    alaska_mask = geometry_mask(
        alaska.geometry,
        out_shape=raster_shape,
        transform=raster_transform,
        invert=True,
        all_touched=ALASKA_MASK_ALL_TOUCHED,
    )

    ALASKA_MASK_CACHE[cache_key] = alaska_mask

    print(f"Built Alaska mask: {alaska_mask.sum():,} grid cells inside Alaska")

    return alaska_mask


def find_truth_rasters(interval):
    """Find all truth rasters for one interval."""

    cfg = get_dataset_config()
    interval_str = f"{interval:02d}"

    truth_dir = BASE_TRUTH_ROOT / cfg["subdir_template"].format(
        interval_str=interval_str
    )

    if not truth_dir.exists():
        raise FileNotFoundError(f"Truth directory not found: {truth_dir}")

    pattern = cfg["file_glob_template"].format(interval_str=interval_str)

    return sorted(truth_dir.glob(pattern))


def read_truth_array(ds):
    """Read one binary truth raster and return yes/no plus valid mask.

    Handles rasters read as:
      - uint8 with 255 nodata
      - float with NaN nodata
      - float/int with -9999 nodata

    Returns:
      yes: bool array where lightning is observed
      valid: bool array where raster cell should count in climatology
    """

    cfg = get_dataset_config()

    arr = ds.values[0].astype(float)

    valid = np.isfinite(arr)

    for nodata_val in cfg["input_nodata_values"]:
        valid &= arr != nodata_val

    yes = np.zeros(arr.shape, dtype=bool)

    for yes_val in cfg["truth_yes_values"]:
        yes |= arr == yes_val

    if USE_ALASKA_MASK:
        alaska_mask = build_alaska_mask(ds)
        valid &= alaska_mask

    return yes, valid


def write_float_raster_like(template_ds, arr, out_path, nodata=OUTPUT_NODATA):
    """Write a float32 GeoTIFF using metadata from template raster."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_arr = arr.astype("float32").copy()
    out_arr[~np.isfinite(out_arr)] = nodata

    profile = {
        "driver": "GTiff",
        "height": out_arr.shape[0],
        "width": out_arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": template_ds.rio.crs,
        "transform": template_ds.rio.transform(),
        "nodata": nodata,
        "compress": "deflate",
    }

    if out_path.exists():
        try:
            out_path.unlink()
        except PermissionError as e:
            raise PermissionError(
                f"Cannot overwrite {out_path}. Close ArcPro/QGIS/Python viewers "
                "or write to a new output directory."
            ) from e

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_arr, 1)

    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------
# MAIN CLIMO BUILD
# ---------------------------------------------------------------------

def build_monthly_climatology():
    """Build and save monthly climatology rasters without day/night split."""

    cfg = get_dataset_config()
    output_prefix = cfg["output_prefix"]

    print(f"Building monthly climatology for DATASET_NAME={DATASET_NAME!r}")
    print(f"Input root:  {BASE_TRUTH_ROOT}")
    print(f"Output dir:  {CLIMO_OUT_DIR}")
    print(f"Years:       {min(CLIMO_YEARS)}-{max(CLIMO_YEARS)}")
    print(f"Months:      {list(MONTHS)}")

    for interval in INTERVAL_HOURS:
        interval_str = f"{interval:02d}"
        truth_files = find_truth_rasters(interval)

        print(f"\nInterval {interval_str}: found {len(truth_files):,} truth rasters")

        accum = {}
        template_by_key = {}

        used_files = 0

        for t_path in truth_files:
            valid_dt = parse_truth_valid_time(t_path)

            if valid_dt is None:
                print(f"Could not parse valid time, skipping: {t_path.name}")
                continue

            if valid_dt.year not in CLIMO_YEARS:
                continue

            if valid_dt.month not in MONTHS:
                continue

            key = valid_dt.month

            with rxr.open_rasterio(t_path, mask_and_scale=True) as ds:
                yes, valid = read_truth_array(ds)

                if key not in accum:
                    shape = yes.shape
                    accum[key] = {
                        "sum_obs": np.zeros(shape, dtype="float64"),
                        "count": np.zeros(shape, dtype="float64"),
                    }
                    template_by_key[key] = ds.copy(deep=True)

                accum[key]["sum_obs"][valid] += yes[valid].astype(float)
                accum[key]["count"][valid] += 1.0

            used_files += 1

        print(f"Interval {interval_str}: used {used_files:,} files after year/month filtering")

        for month, vals in sorted(accum.items()):
            climo_prob = safe_divide(vals["sum_obs"], vals["count"])
            climo_prob[vals["count"] == 0] = np.nan

            count_arr = vals["count"].astype("float32")
            count_arr[count_arr == 0] = np.nan

            template = template_by_key[month]

            prob_path = (
                CLIMO_OUT_DIR /
                f"{output_prefix}_{interval_str}h_{month:02d}_prob.tif"
            )

            count_path = (
                CLIMO_OUT_DIR /
                f"{output_prefix}_{interval_str}h_{month:02d}_count.tif"
            )

            write_float_raster_like(template, climo_prob, prob_path)
            write_float_raster_like(template, count_arr, count_path)

            mean_climo = np.nanmean(climo_prob)
            max_count = np.nanmax(count_arr)

            print(
                f"Interval {interval_str}, month={month:02d}: "
                f"mean climo={mean_climo:.5f}, max samples={max_count:.0f}"
            )


if __name__ == "__main__":
    build_monthly_climatology()