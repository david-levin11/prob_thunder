#!/usr/bin/env python

"""Verify NBM thunder probabilities against Alaska NLDN/GLD lightning truth.

Script purpose:
- Pair each forecast thunder-probability raster with the matching valid-time
  lightning truth raster.
- Evaluate skill over the full Alaska raster domain only.
- Aggregate two verification views in one monthly output table:
    1) Probability-bin statistics for reliability and Brier Score components.
    2) Threshold contingency-table counts for thresholds from 0.1 to 0.9.

High-level flow:
1. For each configured interval and year/month, scan matching forecast files.
2. Parse valid time and lead hour from forecast filename.
3. Build matching truth-raster filename.
4. Read forecast and truth rasters.
5. Use all valid overlapping raster cells as the Alaska verification domain.
6. Accumulate probabilistic-bin and threshold statistics by:
      period, forecast_hour
7. Write one monthly CSV per interval.

Inputs:
- BASE_FCST: Root directory with forecast GeoTIFF files.
- BASE_TRUTH_ROOT: Root directory containing truth subfolders such as
  nldn_06_20km and nldn_12_20km.

Output:
- OUT_DIR/verif_II_YYYY_MM.csv where II is the configured interval,
  for example 06 or 12.
"""

import os
import re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rioxarray as rxr


# --- CONFIG ---
BASE_TRUTH_ROOT = Path(
    r"C:\Users\David.Levin\NBMLightningVer\gld_rasters"
)

BASE_FCST = Path(
    r"C:\Users\David.Levin\NBMLightningVer\nbm_data"
)

OUT_DIR = Path(
    r"C:\Users\David.Levin\NBMLightningVer\monthly_stats"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2023, 2024, 2025]
MONTHS = range(3, 11)  # March through October

PROB_THUNDER_INTERVALS = [6, 12]

PROB_BINS = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
THRESHOLDS = np.round(np.arange(0.1, 1.0, 0.1), 2).tolist()

VALID_HOUR_TO_PERIOD = {
    0: "day",
    6: "day",
    12: "night",
    18: "night",
}

DOMAIN_NAME = "Alaska"

# Forecast rasters appear to be stored as 0-100 percent.
# Set to False if your rasters are already 0.0-1.0.
FORECAST_IS_PERCENT = True

# Truth rasters should be 0/1.
# If they contain counts, this script converts truth to yes/no using > 0.
TRUTH_YES_THRESHOLD = 0


def get_file_times(filename):
    """Extract valid datetime and forecast lead hour from a forecast filename.

    Expected filename fragment:
        YYYY-MM-DDTHHMM_FXXX

    Example:
        something_2024-06-01T0000_F012_tstm06.tif

    Returns:
        tuple[datetime | None, int | None]
    """

    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{4})_F(\d{3})", filename)

    if not match:
        return None, None

    init_str, f_hour = match.groups()
    init_dt = datetime.strptime(init_str, "%Y-%m-%dT%H%M")
    forecast_hour = int(f_hour)
    valid_dt = init_dt + timedelta(hours=forecast_hour)

    return valid_dt, forecast_hour


def get_day_night_label(valid_dt):
    """Map valid hour to day/night period label."""

    if valid_dt is None:
        return None

    return VALID_HOUR_TO_PERIOD.get(valid_dt.hour)


def init_prob_stats():
    """Initialize per-probability-bin accumulators for one grouping key."""

    return {
        np.round(b, 2): {
            "sum_obs": 0.0,
            "count": 0,
            "sum_se": 0.0,
        }
        for b in PROB_BINS
    }


def init_threshold_stats():
    """Initialize threshold contingency-table accumulators for one grouping key."""

    return {
        t: {
            "hits": 0,
            "misses": 0,
            "fa": 0,
            "cn": 0,
        }
        for t in THRESHOLDS
    }


def read_raster_pair(f_path, t_path):
    """Read forecast and truth rasters and return flattened valid arrays."""

    with rxr.open_rasterio(f_path, mask_and_scale=True) as ds_f, \
         rxr.open_rasterio(t_path, mask_and_scale=True) as ds_o:

        f_vals = ds_f.values[0].astype(float).ravel()
        o_vals = ds_o.values[0].astype(float).ravel()

    if FORECAST_IS_PERCENT:
        f_vals = f_vals / 100.0

    # Truth may already be 0/1. If it is counts, convert to yes/no.
    o_vals = np.where(o_vals > TRUTH_YES_THRESHOLD, 1.0, 0.0)

    valid_mask = ~np.isnan(f_vals) & ~np.isnan(o_vals)

    return f_vals[valid_mask], o_vals[valid_mask]


def diagnose_high_probability_pairs(
    min_prob_threshold=0.50,
    top_n=50,
    output_name="high_prob_pair_diagnostics.csv",
):
    """
    Scan all paired NBM/GLD rasters and identify the cases with the highest
    forecast thunder probabilities.

    For each forecast/truth pair, this reports:
    - max NBM thunder probability
    - number of forecast pixels exceeding selected thresholds
    - number of observed GLD pixels
    - overlap between high-probability forecast pixels and observed GLD pixels

    This is intended to answer:
    "When NBM actually produces rare high probabilities, do those areas overlap
    observed lightning?"
    """

    records = []

    thresholds_to_check = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    for interval in PROB_THUNDER_INTERVALS:
        interval_str = f"{interval:02d}"

        truth_dir = BASE_TRUTH_ROOT / f"gld_{interval_str}_20km"
        fct_pattern = f"*tstm{interval_str}*.tif"

        if not truth_dir.exists():
            print(f"Truth folder not found for interval {interval_str}: {truth_dir}")
            continue

        for year in YEARS:
            for month in MONTHS:
                month_str = f"{month:02d}"
                fct_month_path = BASE_FCST / str(year) / month_str

                if not fct_month_path.exists():
                    continue

                fct_files = sorted(fct_month_path.rglob(fct_pattern))

                print(
                    f"Scanning interval {interval_str}, {year}-{month_str}: "
                    f"{len(fct_files)} forecast files"
                )

                for f_path in fct_files:
                    valid_dt, forecast_hour = get_file_times(f_path.name)
                    period = get_day_night_label(valid_dt)

                    if valid_dt is None or forecast_hour is None or period is None:
                        continue

                    t_path = truth_dir / f"gld_{interval_str}h_{valid_dt.strftime('%Y%m%d_%H00Z')}.tif"

                    if not t_path.exists():
                        continue

                    try:
                        with rxr.open_rasterio(f_path, mask_and_scale=True) as ds_f, \
                             rxr.open_rasterio(t_path, mask_and_scale=True) as ds_o:

                            f_raw = ds_f.values[0].astype(float)
                            o_raw = ds_o.values[0].astype(float)

                            if FORECAST_IS_PERCENT:
                                f_prob = f_raw / 100.0
                            else:
                                f_prob = f_raw

                            o_yes = o_raw > TRUTH_YES_THRESHOLD

                            valid = ~np.isnan(f_prob) & ~np.isnan(o_raw)

                            if not np.any(valid):
                                continue

                            f_valid = f_prob[valid]
                            o_valid = o_yes[valid]

                            max_prob = float(np.nanmax(f_valid))
                            mean_prob = float(np.nanmean(f_valid))

                            obs_pixels = int(np.sum(o_valid))
                            total_valid_pixels = int(np.sum(valid))

                            # Location of maximum forecast probability.
                            # This is in raster row/col coordinates.
                            max_idx_flat = np.nanargmax(np.where(valid, f_prob, np.nan))
                            max_row, max_col = np.unravel_index(max_idx_flat, f_prob.shape)

                            # Was there GLD observed at the max-probability pixel?
                            obs_at_max_prob_pixel = bool(o_yes[max_row, max_col])

                            record = {
                                "interval_hour": interval,
                                "year": year,
                                "month": month,
                                "valid_dt": valid_dt,
                                "forecast_hour": int(forecast_hour),
                                "period": period,
                                "max_prob": max_prob,
                                "mean_prob": mean_prob,
                                "obs_pixels": obs_pixels,
                                "total_valid_pixels": total_valid_pixels,
                                "obs_fraction_domain": obs_pixels / total_valid_pixels,
                                "max_prob_row": int(max_row),
                                "max_prob_col": int(max_col),
                                "obs_at_max_prob_pixel": obs_at_max_prob_pixel,
                                "forecast_file": str(f_path),
                                "truth_file": str(t_path),
                            }

                            # Add threshold-based overlap diagnostics.
                            for thresh in thresholds_to_check:
                                f_yes = valid & (f_prob >= thresh)
                                n_f_yes = int(np.sum(f_yes))

                                if n_f_yes > 0:
                                    n_overlap = int(np.sum(f_yes & o_yes))
                                    obs_rate_in_f_yes = n_overlap / n_f_yes
                                else:
                                    n_overlap = 0
                                    obs_rate_in_f_yes = np.nan

                                record[f"n_fcst_ge_{int(thresh * 100):02d}"] = n_f_yes
                                record[f"n_overlap_ge_{int(thresh * 100):02d}"] = n_overlap
                                record[f"obs_rate_ge_{int(thresh * 100):02d}"] = obs_rate_in_f_yes

                            records.append(record)

                    except Exception as e:
                        print(f"Error diagnosing {f_path.name}: {e}")

    if not records:
        print("No paired forecast/truth rasters found.")
        return pd.DataFrame()

    df_diag = pd.DataFrame(records)

    # Sort by highest forecast probability first.
    df_diag = df_diag.sort_values(
        ["max_prob", "obs_pixels"],
        ascending=[False, False],
    ).reset_index(drop=True)

    out_path = OUT_DIR / output_name
    df_diag.to_csv(out_path, index=False)

    print(f"\nWrote diagnostic file: {out_path}")
    print(f"Total paired rasters diagnosed: {len(df_diag):,}")

    print("\nTop high-probability cases:")
    display_cols = [
        "interval_hour",
        "valid_dt",
        "forecast_hour",
        "period",
        "max_prob",
        "obs_pixels",
        "n_fcst_ge_50",
        "n_overlap_ge_50",
        "obs_rate_ge_50",
        "n_fcst_ge_80",
        "n_overlap_ge_80",
        "obs_rate_ge_80",
        "forecast_file",
        "truth_file",
    ]

    available_cols = [c for c in display_cols if c in df_diag.columns]
    print(df_diag[available_cols].head(top_n).to_string(index=False))

    # Optional filtered view: cases where max probability exceeded your threshold.
    high_cases = df_diag[df_diag["max_prob"] >= min_prob_threshold].copy()

    print(
        f"\nCases with max_prob >= {min_prob_threshold:.2f}: "
        f"{len(high_cases):,}"
    )

    if not high_cases.empty:
        print(
            high_cases[available_cols]
            .head(top_n)
            .to_string(index=False)
        )

    return df_diag


def accumulate_stats(f_v, o_v, prob_stats, threshold_stats):
    """Update probability-bin and threshold stats for one forecast/truth pair."""

    # A. Probability binning for reliability/Brier components.
    #
    # np.digitize with bins 0.0, 0.1, ..., 1.0 produces:
    #   0.00 <= p < 0.10 -> bin 0.0
    #   0.10 <= p < 0.20 -> bin 0.1
    #   ...
    # Values equal to 1.0 need to be clipped into the final bin.
    indices = np.digitize(f_v, PROB_BINS, right=False) - 1
    indices = np.clip(indices, 0, len(PROB_BINS) - 1)

    for i, b in enumerate(PROB_BINS):
        m = indices == i

        if not np.any(m):
            continue

        b_val = np.round(b, 2)

        prob_stats[b_val]["sum_obs"] += np.sum(o_v[m])
        prob_stats[b_val]["count"] += np.sum(m)
        prob_stats[b_val]["sum_se"] += np.sum((f_v[m] - o_v[m]) ** 2)

    # B. Threshold contingency-table verification.
    o_yes = o_v == 1

    for t in THRESHOLDS:
        f_yes = f_v >= t

        threshold_stats[t]["hits"] += np.sum(f_yes & o_yes)
        threshold_stats[t]["misses"] += np.sum(~f_yes & o_yes)
        threshold_stats[t]["fa"] += np.sum(f_yes & ~o_yes)
        threshold_stats[t]["cn"] += np.sum(~f_yes & ~o_yes)


def process_month(interval, year, month):
    """Process one interval/year/month and write one monthly CSV."""

    interval_str = f"{interval:02d}"
    month_str = f"{month:02d}"

    truth_dir = BASE_TRUTH_ROOT / f"gld_{interval_str}_20km"
    fct_pattern = f"*tstm{interval_str}*.tif"

    csv_name = OUT_DIR / f"verif_{interval_str}_{year}_{month_str}.csv"

    print(f"\nProcessing interval {interval_str}, {year}-{month_str}")
    print(f"Output: {csv_name}")

    if csv_name.exists():
        print("Output already exists, skipping.")
        return

    if not truth_dir.exists():
        print(f"Truth folder not found: {truth_dir}. Skipping.")
        return

    fct_month_path = BASE_FCST / str(year) / month_str

    if not fct_month_path.exists():
        print(f"Forecast path does not exist: {fct_month_path}. Skipping.")
        return

    fct_files = sorted(fct_month_path.rglob(fct_pattern))

    if not fct_files:
        print(f"No forecast files found using pattern {fct_pattern}")
        return

    # Keyed by (domain, period, forecast_hour)
    stats_by_key = {}
    t_stats_by_key = {}

    pair_count = 0

    for f_path in fct_files:
        valid_dt, forecast_hour = get_file_times(f_path.name)
        period = get_day_night_label(valid_dt)

        if valid_dt is None or forecast_hour is None or period is None:
            continue

        t_path = truth_dir / f"gld_{interval_str}h_{valid_dt.strftime('%Y%m%d_%H00Z')}.tif"

        if not t_path.exists():
            continue

        try:
            f_v, o_v = read_raster_pair(f_path, t_path)

            if f_v.size == 0:
                continue

            key = (DOMAIN_NAME, period, int(forecast_hour))

            if key not in stats_by_key:
                stats_by_key[key] = init_prob_stats()
                t_stats_by_key[key] = init_threshold_stats()

            accumulate_stats(
                f_v=f_v,
                o_v=o_v,
                prob_stats=stats_by_key[key],
                threshold_stats=t_stats_by_key[key],
            )

            pair_count += 1

        except Exception as e:
            print(f"Error on {f_path.name}: {e}")

    if pair_count == 0:
        print(f"No valid raster pairs found for interval {interval_str}, {year}-{month_str}")
        return

    out_frames = []

    for (domain, period, forecast_hour), prob_stats in stats_by_key.items():
        df_prob = (
            pd.DataFrame.from_dict(prob_stats, orient="index")
            .reset_index()
            .rename(columns={"index": "prob_bin"})
        )

        df_prob["region"] = domain
        df_prob["period"] = period
        df_prob["forecast_hour"] = int(forecast_hour)
        df_prob["year"] = year
        df_prob["month"] = month
        df_prob["interval_hour"] = interval

        for t in THRESHOLDS:
            for metric in ["hits", "misses", "fa", "cn"]:
                df_prob[f"t{int(t * 100)}_{metric}"] = (
                    t_stats_by_key[(domain, period, forecast_hour)][t][metric]
                )

        out_frames.append(df_prob)

    df_out = pd.concat(out_frames, ignore_index=True)
    df_out.to_csv(csv_name, index=False)

    print(f"Wrote {len(df_out):,} rows from {pair_count:,} raster pairs.")


def main():
    """Run Alaska-only verification for all configured intervals/months."""

    for interval in PROB_THUNDER_INTERVALS:
        for year in YEARS:
            for month in MONTHS:
                process_month(interval, year, month)


if __name__ == "__main__":
    # Normal monthly verification:
    # main()

    # High-probability NBM/GLD pair diagnostic:
    diagnose_high_probability_pairs(
        min_prob_threshold=0.50,
        top_n=50,
        output_name="high_prob_pair_diagnostics.csv",
    )
    diag = pd.read_csv(OUT_DIR / "high_prob_pair_diagnostics.csv")

    summary = pd.DataFrame({
        "threshold": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        "num_rasters_with_pixels": [
            (diag[f"n_fcst_ge_{t:02d}"] > 0).sum()
            for t in [10, 20, 30, 40, 50, 60, 70, 80, 90]
        ],
        "total_forecast_pixels": [
            diag[f"n_fcst_ge_{t:02d}"].sum()
            for t in [10, 20, 30, 40, 50, 60, 70, 80, 90]
        ],
        "total_overlap_pixels": [
            diag[f"n_overlap_ge_{t:02d}"].sum()
            for t in [10, 20, 30, 40, 50, 60, 70, 80, 90]
        ],
    })

    summary["obs_rate"] = (
        summary["total_overlap_pixels"] / summary["total_forecast_pixels"]
    )

    print(summary)