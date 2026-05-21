#!/usr/bin/env python

"""Build monthly GLD lightning climatology rasters for Alaska.

For each interval/month combination, this script computes:

    climo_prob = number of GLD yes events / number of valid GLD samples

The climatology is gridded and Alaska-masked.

Example output:
    gld_climo_06h_07_prob.tif
    gld_climo_06h_07_count.tif
    gld_climo_12h_07_prob.tif
    gld_climo_12h_07_count.tif
"""

import re
from pathlib import Path
from datetime import datetime

import numpy as np
import rioxarray as rxr
import geopandas as gpd
from rasterio.features import geometry_mask


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_TRUTH_ROOT = Path(
    r"C:\Users\David.Levin\NBMLightningVer\gld_rasters"
)

CLIMO_OUT_DIR = Path(
    r"C:\Users\David.Levin\NBMLightningVer\gld_climatology"
)
CLIMO_OUT_DIR.mkdir(parents=True, exist_ok=True)

ALASKA_BOUNDARY_FILE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\reference\cb_2018_us_state_5m.shp"
)

CLIMO_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
MONTHS = range(3, 11)  # March through October
INTERVAL_HOURS = [6, 12]

TRUTH_YES_THRESHOLD = 0
ALASKA_MASK_CACHE = {}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def parse_truth_valid_time(path):
    """Parse valid datetime from GLD raster filename.

    Expected:
        gld_06h_20230725_0600Z.tif
        gld_12h_20230725_1800Z.tif
    """

    match = re.search(r"gld_(\d{2})h_(\d{8})_(\d{4})Z", path.name)

    if not match:
        return None

    _, ymd, hm = match.groups()

    return datetime.strptime(ymd + hm, "%Y%m%d%H%M")


def safe_divide(num, den):
    """Return num / den, with NaN where denominator is zero."""

    out = np.full_like(num, np.nan, dtype=float)
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
        # Your .prj indicates NAD83 geographic lat/lon.
        alaska = alaska.set_crs("EPSG:4269")

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
    )

    if cache_key in ALASKA_MASK_CACHE:
        return ALASKA_MASK_CACHE[cache_key]

    alaska = load_alaska_boundary(raster_crs)

    alaska_mask = geometry_mask(
        alaska.geometry,
        out_shape=raster_shape,
        transform=raster_transform,
        invert=True,          # True inside Alaska
        all_touched=False,
    )

    ALASKA_MASK_CACHE[cache_key] = alaska_mask

    print(f"Built Alaska mask: {alaska_mask.sum():,} grid cells inside Alaska")

    return alaska_mask


def find_truth_rasters(interval):
    """Find all GLD truth rasters for one interval."""

    interval_str = f"{interval:02d}"
    truth_dir = BASE_TRUTH_ROOT / f"gld_{interval_str}_20km"

    if not truth_dir.exists():
        raise FileNotFoundError(f"Truth directory not found: {truth_dir}")

    return sorted(truth_dir.glob(f"gld_{interval_str}h_*.tif"))


def write_raster_like(template_ds, arr, out_path, nodata=np.nan):
    """Write a 2D array as a GeoTIFF using template raster metadata."""

    out = template_ds.copy(deep=True)
    out.values[0] = arr.astype("float32")
    out = out.rio.write_nodata(nodata, encoded=True)

    out.rio.to_raster(out_path, compress="deflate")
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------
# MAIN CLIMO BUILD
# ---------------------------------------------------------------------

def build_monthly_climatology():
    """Build and save monthly GLD climatology rasters without day/night split."""

    for interval in INTERVAL_HOURS:
        interval_str = f"{interval:02d}"
        truth_files = find_truth_rasters(interval)

        print(f"\nInterval {interval_str}: found {len(truth_files):,} truth rasters")

        # key = month
        accum = {}
        template_by_key = {}

        for t_path in truth_files:
            valid_dt = parse_truth_valid_time(t_path)

            if valid_dt is None:
                continue

            if valid_dt.year not in CLIMO_YEARS:
                continue

            if valid_dt.month not in MONTHS:
                continue

            key = valid_dt.month

            with rxr.open_rasterio(t_path, mask_and_scale=True) as ds:
                o_raw = ds.values[0].astype(float)
                alaska_mask = build_alaska_mask(ds)

                valid = ~np.isnan(o_raw) & alaska_mask
                o_yes = o_raw > TRUTH_YES_THRESHOLD

                if key not in accum:
                    accum[key] = {
                        "sum_obs": np.zeros(o_raw.shape, dtype=float),
                        "count": np.zeros(o_raw.shape, dtype=float),
                    }
                    template_by_key[key] = ds.copy(deep=True)

                accum[key]["sum_obs"][valid] += o_yes[valid].astype(float)
                accum[key]["count"][valid] += 1.0

        for month, vals in accum.items():
            climo_prob = safe_divide(vals["sum_obs"], vals["count"])

            # Keep cells outside Alaska as NaN.
            climo_prob[vals["count"] == 0] = np.nan

            count_arr = vals["count"].astype(float)
            count_arr[count_arr == 0] = np.nan

            template = template_by_key[month]

            prob_path = (
                CLIMO_OUT_DIR /
                f"gld_climo_{interval_str}h_{month:02d}_prob.tif"
            )

            count_path = (
                CLIMO_OUT_DIR /
                f"gld_climo_{interval_str}h_{month:02d}_count.tif"
            )

            write_raster_like(template, climo_prob, prob_path)
            write_raster_like(template, count_arr, count_path)

            mean_climo = np.nanmean(climo_prob)
            max_count = np.nanmax(count_arr)

            print(
                f"Interval {interval_str}, month={month:02d}: "
                f"mean climo={mean_climo:.5f}, max samples={max_count:.0f}"
            )


if __name__ == "__main__":
    build_monthly_climatology()