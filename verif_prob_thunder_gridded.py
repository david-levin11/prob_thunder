#!/home/chad.kahler/anaconda3/envs/verif/bin/python

"""Verify NBM thunder probabilities against NLDN lightning truth by region.

Script purpose:
- Pair each forecast thunder-probability raster with the matching valid-time
    NLDN truth raster.
- Evaluate skill separately by region, by day/night period, and by forecast
    lead hour.
- Aggregate two verification views in one monthly output table:
    1) Probability-bin statistics for reliability and Brier Score components.
    2) Threshold contingency-table counts (hits, misses, false alarms, correct
         negatives) for thresholds from 0.1 to 0.9.

High-level flow:
1. Load region polygons and validate required name field.
2. For each configured interval and year/month, scan matching forecast files
    (for example, tstm06 or tstm12).
3. Parse valid time from forecast filename and build matching truth filename.
4. On first valid file pair, rasterize region masks onto the forecast grid.
5. For each region, accumulate probabilistic-bin and threshold statistics.
6. Write one monthly CSV per interval containing all
    regions/periods/forecast-hours.

Inputs:
- BASE_FCST: Root directory with forecast GeoTIFF files.
- BASE_TRUTH_ROOT: Root directory containing truth subfolders such as
    nldn_06_20km and nldn_12_20km.
- REGION_FILE: Polygon file defining regions (reprojected if needed).

Output:
- OUT_DIR/verif_II_YYYY_MM.csv where II is the configured interval (for
    example 06 or 12), with one row per probability bin for each
    region/period/forecast-hour key, plus appended threshold metrics and an
    interval_hour column.
"""

import os
import re
import numpy as np
import pandas as pd
import rioxarray as rxr
import geopandas as gpd
from rasterio.features import geometry_mask
from pyproj import CRS
from pathlib import Path
from datetime import datetime, timedelta

# --- CONFIG ---
BASE_TRUTH_ROOT = Path(r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\prob_thunder\nldn\nldn_rasters")
BASE_FCST = Path(r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\prob_thunder\geotiff")
OUT_DIR = Path(r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\prob_thunder\nldn\prob_thunder_verif\monthly_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# YEARS = [2023, 2024, 2025]
YEARS = [2023,2024,2025]
MONTHS = range(3, 11) 
# Process one or more thunder-probability accumulation intervals in one run.
PROB_THUNDER_INTERVALS = [6, 12]
PROB_BINS = np.linspace(0, 1, 11) # 0.0, 0.1 ... 1.0
THRESHOLDS = np.round(np.arange(0.1, 1.0, 0.1), 2).tolist()
VALID_HOUR_TO_PERIOD = {0: "day", 6: "day", 12: "night", 18: "night"}

# Region polygons (shapefile or GeoJSON) already in the same projection as data.
# Update this to your region boundary file (e.g., NWS regions).
REGION_FILE = Path(r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\reference\nws_regions_lcc.shp")
REGION_NAME_COL = "NWS_REG"
CONUS_REGION_NAME = "CONUS"

def get_file_times(filename):
    """Extract valid datetime and lead hour from a forecast filename.

    Expected fragment format: YYYY-MM-DDTHHMM_FXXX where XXX is lead hour.

    Args:
        filename: Forecast filename to parse.

    Returns:
        tuple[datetime | None, int | None]: (valid_dt, forecast_hour). Returns
        (None, None) when the expected pattern is not present.
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{4})_F(\d{3})", filename)
    if match:
        init_str, f_hour = match.groups()
        init_dt = datetime.strptime(init_str, "%Y-%m-%dT%H%M")
        f_hour_i = int(f_hour)
        valid_dt = init_dt + timedelta(hours=f_hour_i)
        return valid_dt, f_hour_i
    return None, None


def get_day_night_label(valid_dt):
    """Map valid hour to day/night period label used in output grouping."""
    if valid_dt is None:
        return None
    return VALID_HOUR_TO_PERIOD.get(valid_dt.hour)


def init_prob_stats():
    """Initialize per-probability-bin accumulators for one grouping key."""
    return {np.round(b, 2): {'sum_obs': 0.0, 'count': 0, 'sum_se': 0.0} for b in PROB_BINS}


def init_threshold_stats():
    """Initialize threshold contingency-table accumulators for one key."""
    return {t: {'hits': 0, 'misses': 0, 'fa': 0, 'cn': 0} for t in THRESHOLDS}


def load_regions(region_file, name_col):
    """Load and validate region polygons used for spatial verification splits."""
    if not region_file.exists():
        raise FileNotFoundError(f"Region file not found: {region_file}")

    gdf = gpd.read_file(region_file)
    if gdf.empty:
        raise ValueError(f"Region file has no polygons: {region_file}")

    if name_col not in gdf.columns:
        raise ValueError(
            f"Region name column '{name_col}' not found in {region_file}. "
            f"Available columns: {list(gdf.columns)}"
        )

    gdf = gdf[[name_col, "geometry"]].dropna(subset=[name_col, "geometry"]).copy()
    if gdf.empty:
        raise ValueError("No valid named geometries found in region file.")

    return gdf


def build_region_masks(gdf, name_col, raster_shape, raster_transform, raster_crs):
    """Rasterize each region polygon onto the forecast grid as boolean masks.

    Returns:
        tuple[dict[str, np.ndarray], np.ndarray]: (region masks, merged CONUS
        mask) where True values indicate cells inside a region.
    """
    if gdf.crs is None:
        raise ValueError("Region file has no CRS. It must match raster CRS.")
    if raster_crs is None:
        raise ValueError("Raster CRS is missing; cannot verify projection consistency.")

    region_crs = CRS.from_user_input(gdf.crs)
    target_crs = CRS.from_user_input(raster_crs)
    if not region_crs.equals(target_crs):
        print(
            "Region CRS differs from raster CRS; reprojecting region polygons to match raster grid. "
            f"Region CRS: {region_crs.to_string()} | Raster CRS: {target_crs.to_string()}"
        )
        gdf = gdf.to_crs(target_crs)

    region_masks = {}
    for _, row in gdf.iterrows():
        r_name = str(row[name_col])
        poly_mask = geometry_mask(
            [row.geometry],
            out_shape=raster_shape,
            transform=raster_transform,
            invert=True,
            all_touched=False,
        )
        if r_name in region_masks:
            region_masks[r_name] = region_masks[r_name] | poly_mask
        else:
            region_masks[r_name] = poly_mask

    conus_mask = np.zeros(raster_shape, dtype=bool)
    for m in region_masks.values():
        conus_mask |= m

    return region_masks, conus_mask


regions_gdf = load_regions(REGION_FILE, REGION_NAME_COL)

# --- Monthly processing loop ---
# For each interval/month, aggregate stats keyed by (region, day/night period,
# lead hour), then write a monthly CSV for that interval.
for interval in PROB_THUNDER_INTERVALS:
    interval_str = f"{interval:02d}"
    # Example mapping: interval 06 -> truth folder nldn_06_20km.
    truth_dir = BASE_TRUTH_ROOT / f"nldn_{interval_str}_20km"
    # Example mapping: interval 06 -> forecast pattern *tstm06*.tif.
    fct_pattern = f"*tstm{interval_str}*.tif"

    if not truth_dir.exists():
        print(f"Truth folder not found for interval {interval_str}: {truth_dir}. Skipping interval.")
        continue

    for year in YEARS:
        for month in MONTHS:
            month_str = f"{month:02d}"
            csv_name = OUT_DIR / f"verif_{interval_str}_{year}_{month_str}.csv"
            print(f"Checking output: {csv_name}")
            if csv_name.exists():
                print("Output already exists, skipping")
                continue

            fct_month_path = BASE_FCST / str(year) / month_str
            print(f"Checking forecast path: {fct_month_path}")
            if not fct_month_path.exists():
                print("Forecast path does not exist, skipping")
                continue

            print(f"Processing interval {interval_str}, {year}-{month_str}...")

            # 1. Setup Aggregators (created lazily after the first valid raster pair)
            region_masks_flat = None
            # Keyed by (region, period, forecast_hour)
            stats_by_key = {}
            t_stats_by_key = {}

            fct_files = sorted(fct_month_path.rglob(fct_pattern))

            for f_path in fct_files:
                valid_dt, forecast_hour = get_file_times(f_path.name)
                period = get_day_night_label(valid_dt)
                if valid_dt is None or forecast_hour is None or period is None:
                    continue

                t_path = truth_dir / f"nldn_{interval_str}h_{valid_dt.strftime('%Y%m%d_%H00Z')}.tif"

                if t_path.exists():
                    try:
                        with rxr.open_rasterio(f_path, mask_and_scale=True) as ds_f, \
                             rxr.open_rasterio(t_path, mask_and_scale=True) as ds_o:

                            if region_masks_flat is None:
                                # Build raster-aligned region masks once, then reuse.
                                raster_shape = ds_f.values[0].shape
                                raster_transform = ds_f.rio.transform()
                                raster_crs = ds_f.rio.crs

                                region_masks, conus_mask = build_region_masks(
                                    regions_gdf,
                                    REGION_NAME_COL,
                                    raster_shape,
                                    raster_transform,
                                    raster_crs,
                                )

                                region_masks[CONUS_REGION_NAME] = conus_mask
                                region_masks_flat = {k: v.ravel() for k, v in region_masks.items()}

                            # --- SCALING FIX ---
                            # Divide by 100 to convert 0-100 to 0.0-1.0
                            f_vals = ds_f.values[0].flatten() / 100.0
                            o_vals = ds_o.values[0].flatten() # Already 0 or 1

                            mask = ~np.isnan(f_vals) & ~np.isnan(o_vals)

                            for r_name, rmask in region_masks_flat.items():
                                rm = mask & rmask
                                if not np.any(rm):
                                    continue

                                key = (r_name, period, int(forecast_hour))
                                if key not in stats_by_key:
                                    stats_by_key[key] = init_prob_stats()
                                    t_stats_by_key[key] = init_threshold_stats()

                                f_v = f_vals[rm]
                                o_v = o_vals[rm]

                                # A. Probabilistic Binning (for Reliability/Brier)
                                indices = np.digitize(f_v, PROB_BINS) - 1
                                for i in range(len(PROB_BINS)):
                                    m = (indices == i)
                                    if np.any(m):
                                        b_val = np.round(PROB_BINS[i], 2)
                                        stats_by_key[key][b_val]['sum_obs'] += np.sum(o_v[m])
                                        stats_by_key[key][b_val]['count'] += np.sum(m)
                                        stats_by_key[key][b_val]['sum_se'] += np.sum((f_v[m] - o_v[m])**2)

                                # B. Threshold Verification (Contingency Table)
                                for t in THRESHOLDS:
                                    f_yes = (f_v >= t)
                                    o_yes = (o_v == 1)
                                    t_stats_by_key[key][t]['hits'] += np.sum(f_yes & o_yes)
                                    t_stats_by_key[key][t]['misses'] += np.sum(~f_yes & o_yes)
                                    t_stats_by_key[key][t]['fa'] += np.sum(f_yes & ~o_yes)
                                    t_stats_by_key[key][t]['cn'] += np.sum(~f_yes & ~o_yes)

                    except Exception as e:
                        print(f"Error on {f_path.name}: {e}")

            # 2. Save Monthly Results
            if region_masks_flat is None:
                print(f"No valid raster pairs found for interval {interval_str}, {year}-{month_str}; skipping output")
                continue

            # Combine probabilistic and threshold stats per key into one CSV.
            # Key dimensions: region + day/night period + forecast hour.
            # interval_hour is added as a column for downstream filtering.
            out_frames = []
            for (r_name, period, forecast_hour), prob_stats in stats_by_key.items():
                df_prob = pd.DataFrame.from_dict(
                    prob_stats,
                    orient='index'
                ).reset_index().rename(columns={'index': 'prob_bin'})

                df_prob['region'] = r_name
                df_prob['period'] = period
                df_prob['forecast_hour'] = int(forecast_hour)
                df_prob['year'] = year
                df_prob['month'] = month
                df_prob['interval_hour'] = interval

                for t in THRESHOLDS:
                    for metric in ['hits', 'misses', 'fa', 'cn']:
                        df_prob[f"t{int(t*100)}_{metric}"] = t_stats_by_key[(r_name, period, forecast_hour)][t][metric]

                out_frames.append(df_prob)

            df_out = pd.concat(out_frames, ignore_index=True)
            df_out.to_csv(csv_name, index=False)