#!/usr/bin/env python

"""Verify NBM thunder probabilities against saved monthly GLD climatology.

This script uses saved monthly-only GLD climatology rasters produced by:
    build_gld_monthly_climo_no_period.py

The reference climatology is keyed by:
    interval_hour + month

It is NOT split by day/night. The NBM verification results can still be grouped
and plotted by day/night period, but both periods use the same monthly GLD
climatology reference.

Workflow:
1. Pair NBM thunder probability rasters with matching GLD truth rasters.
2. Load the matching monthly GLD climatology raster.
3. Apply Alaska state mask.
4. Compute pair-level Brier Score for NBM and climatology.
5. Compute Brier Skill Score:
       BSS = 1 - BS_NBM / BS_CLIMO
6. Save pair-level and aggregated CSV summaries.
7. Make Brier/BSS plots.
"""

import re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rioxarray as rxr
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_TRUTH_ROOT = Path(
    r"C:\Users\David.Levin\NBMLightningVer\gld_rasters"
)

BASE_FCST = Path(
    r"C:\Users\David.Levin\NBMLightningVer\nbm_data"
)

CLIMO_DIR = Path(
    r"C:\Users\David.Levin\NBMLightningVer\gld_climatology"
)

OUT_DIR = Path(
    r"C:\Users\David.Levin\NBMLightningVer\brier_gld_climo"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ALASKA_BOUNDARY_FILE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\reference\cb_2018_us_state_5m.shp"
)

VERIFY_YEARS = [2023, 2024, 2025]
MONTHS = range(3, 11)  # March through October
INTERVAL_HOURS = [6, 12]

VALID_HOUR_TO_PERIOD = {
    0: "day",
    6: "day",
    12: "night",
    18: "night",
}

FORECAST_IS_PERCENT = True
TRUTH_YES_THRESHOLD = 0
ALL_TOUCHED = False

SAVE_FIGS = True
SHOW_FIGS = True

ALASKA_MASK_CACHE = {}

REBUILD_PAIR_RESULTS = False
REBUILD_AGG_RESULTS = False

PAIR_CSV = OUT_DIR / "brier_against_saved_gld_monthly_climo_by_pair.csv"
AGG_CSV = OUT_DIR / "brier_against_saved_gld_monthly_climo_aggregated_by_valid_hour.csv"


# ---------------------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------------------

def get_file_times(filename: str):
    """Extract valid datetime and forecast lead hour from NBM filename.

    Expected filename fragment:
        YYYY-MM-DDTHHMM_FXXX

    The datetime is treated as initialization time and FXXX as forecast lead.
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
    """Map valid hour to day/night period."""
    if valid_dt is None:
        return None
    return VALID_HOUR_TO_PERIOD.get(valid_dt.hour)


def weighted_mean(values, weights):
    """Return weighted mean with NaN safety."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))

def get_valid_hours_for_period(period):
    """Return the expected valid hours for each verification period."""

    if period == "day":
        return [0, 6]

    if period == "night":
        return [12, 18]

    raise ValueError(f"Unknown period: {period}")


def finite_min_max(*arrays):
    """Return finite min/max across one or more arrays/Series."""

    vals = []

    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size > 0:
            vals.append(a)

    if not vals:
        return np.nan, np.nan

    all_vals = np.concatenate(vals)
    return float(np.nanmin(all_vals)), float(np.nanmax(all_vals))


def padded_limits(vmin, vmax, pad_fraction=0.08, min_pad=0.01):
    """Create padded axis limits."""

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None

    if vmin == vmax:
        pad = max(abs(vmin) * pad_fraction, min_pad)
    else:
        pad = max((vmax - vmin) * pad_fraction, min_pad)

    return vmin - pad, vmax + pad


# ---------------------------------------------------------------------
# ALASKA MASK
# ---------------------------------------------------------------------

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
    """Rasterize Alaska boundary onto the raster grid."""
    raster_shape = ds_template.values[0].shape
    raster_transform = ds_template.rio.transform()
    raster_crs = ds_template.rio.crs

    cache_key = (
        raster_shape,
        tuple(round(x, 6) for x in raster_transform),
        str(raster_crs),
        ALL_TOUCHED,
    )

    if cache_key in ALASKA_MASK_CACHE:
        return ALASKA_MASK_CACHE[cache_key]

    alaska = load_alaska_boundary(raster_crs)

    alaska_mask = geometry_mask(
        alaska.geometry,
        out_shape=raster_shape,
        transform=raster_transform,
        invert=True,
        all_touched=ALL_TOUCHED,
    )

    ALASKA_MASK_CACHE[cache_key] = alaska_mask
    print(f"Built Alaska mask: {alaska_mask.sum():,} grid cells inside Alaska")

    return alaska_mask


# ---------------------------------------------------------------------
# CLIMATOLOGY / FILE PATHS
# ---------------------------------------------------------------------

def get_climo_path(interval: int, month: int) -> Path:
    """Return saved monthly-only GLD climatology raster path."""
    return CLIMO_DIR / f"gld_climo_{interval:02d}h_{month:02d}_prob.tif"


def get_truth_path(interval: int, valid_dt: datetime) -> Path:
    """Return matching GLD truth raster path."""
    interval_str = f"{interval:02d}"
    truth_dir = BASE_TRUTH_ROOT / f"gld_{interval_str}_20km"
    return truth_dir / f"gld_{interval_str}h_{valid_dt.strftime('%Y%m%d_%H00Z')}.tif"


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------

def evaluate_against_saved_climo():
    """Compute pair-level Brier and BSS using saved monthly GLD climatology."""
    records = []

    for interval in INTERVAL_HOURS:
        interval_str = f"{interval:02d}"
        fct_pattern = f"*tstm{interval_str}*.tif"

        for year in VERIFY_YEARS:
            for month in MONTHS:
                month_str = f"{month:02d}"
                fct_month_path = BASE_FCST / str(year) / month_str

                if not fct_month_path.exists():
                    print(f"Missing forecast path: {fct_month_path}")
                    continue

                fct_files = sorted(fct_month_path.rglob(fct_pattern))
                print(
                    f"Evaluating interval={interval_str}, {year}-{month_str}: "
                    f"{len(fct_files)} forecast files"
                )

                for f_path in fct_files:
                    valid_dt, forecast_hour = get_file_times(f_path.name)
                    period = get_day_night_label(valid_dt)

                    if valid_dt is None or forecast_hour is None or period is None:
                        continue

                    truth_path = get_truth_path(interval, valid_dt)
                    if not truth_path.exists():
                        continue

                    climo_path = get_climo_path(interval, valid_dt.month)
                    if not climo_path.exists():
                        print(f"Missing climo raster: {climo_path}")
                        continue

                    try:
                        with rxr.open_rasterio(f_path, mask_and_scale=True) as ds_f, \
                             rxr.open_rasterio(truth_path, mask_and_scale=True) as ds_o, \
                             rxr.open_rasterio(climo_path, mask_and_scale=True) as ds_c:

                            f_raw = ds_f.values[0].astype(float)
                            o_raw = ds_o.values[0].astype(float)
                            c_prob = ds_c.values[0].astype(float)

                            if f_raw.shape != o_raw.shape or f_raw.shape != c_prob.shape:
                                raise ValueError(
                                    f"Shape mismatch for {f_path.name}: "
                                    f"forecast={f_raw.shape}, truth={o_raw.shape}, climo={c_prob.shape}"
                                )

                            f_prob = f_raw / 100.0 if FORECAST_IS_PERCENT else f_raw
                            o_yes = np.where(o_raw > TRUTH_YES_THRESHOLD, 1.0, 0.0)

                            alaska_mask = build_alaska_mask(ds_f)

                            valid = (
                                ~np.isnan(f_prob) &
                                ~np.isnan(o_raw) &
                                ~np.isnan(c_prob) &
                                alaska_mask
                            )

                            if not np.any(valid):
                                continue

                            f_v = f_prob[valid]
                            o_v = o_yes[valid]
                            c_v = c_prob[valid]

                            se_nbm = (f_v - o_v) ** 2
                            se_climo = (c_v - o_v) ** 2

                            bs_nbm = float(np.mean(se_nbm))
                            bs_climo = float(np.mean(se_climo))
                            bss = 1.0 - (bs_nbm / bs_climo) if bs_climo > 0 else np.nan

                            records.append({
                                "interval_hour": interval,
                                "year": year,
                                "month": month,
                                "valid_dt": valid_dt,
                                "forecast_hour": int(forecast_hour),
                                "period": period,
                                "count": int(valid.sum()),
                                "obs_count": int(o_v.sum()),
                                "obs_frequency": float(o_v.mean()),
                                "mean_forecast_prob": float(f_v.mean()),
                                "max_forecast_prob": float(f_v.max()),
                                "mean_climo_prob": float(c_v.mean()),
                                "max_climo_prob": float(c_v.max()),
                                "brier_score_nbm": bs_nbm,
                                "brier_score_climo": bs_climo,
                                "brier_skill_score": bss,
                                "forecast_file": str(f_path),
                                "truth_file": str(truth_path),
                                "climo_file": str(climo_path),
                            })

                    except Exception as e:
                        print(f"Error on {f_path.name}: {e}")

    pair_df = pd.DataFrame(records)
    out_csv = OUT_DIR / "brier_against_monthly_gld_climo_by_pair.csv"
    pair_df.to_csv(out_csv, index=False)
    print(f"\nSaved pair-level results: {out_csv}")

    return pair_df


def aggregate_results(pair_df):
    """Aggregate pair-level results by interval, period, forecast hour, and valid hour."""

    pair_df = pair_df.copy()
    pair_df["valid_dt"] = pd.to_datetime(pair_df["valid_dt"])
    pair_df["valid_hour"] = pair_df["valid_dt"].dt.hour

    rows = []

    group_cols = ["interval_hour", "period", "forecast_hour", "valid_hour"]

    for keys, group in pair_df.groupby(group_cols):
        interval, period, fh, valid_hour = keys

        w = group["count"].astype(float)

        bs_nbm = weighted_mean(group["brier_score_nbm"], w)
        bs_climo = weighted_mean(group["brier_score_climo"], w)
        bss = 1.0 - (bs_nbm / bs_climo) if bs_climo > 0 else np.nan

        total_count = int(group["count"].sum())
        total_obs = int(group["obs_count"].sum())

        rows.append({
            "interval_hour": interval,
            "period": period,
            "forecast_hour": fh,
            "valid_hour": int(valid_hour),
            "total_count": total_count,
            "total_obs": total_obs,
            "obs_frequency": total_obs / total_count if total_count > 0 else np.nan,
            "mean_forecast_prob_weighted": weighted_mean(group["mean_forecast_prob"], w),
            "mean_climo_prob_weighted": weighted_mean(group["mean_climo_prob"], w),
            "brier_score_nbm": bs_nbm,
            "brier_score_climo": bs_climo,
            "brier_skill_score": bss,
            "n_pairs": len(group),
        })

    agg = pd.DataFrame(rows)

    out_csv = OUT_DIR / "brier_against_saved_gld_monthly_climo_aggregated_by_valid_hour.csv"
    agg.to_csv(out_csv, index=False)
    print(f"Saved aggregated results: {out_csv}")

    return agg


def aggregate_monthly_results(pair_df: pd.DataFrame):
    """Aggregate pair-level BSS by year/month/interval/period/forecast hour."""
    rows = []
    group_cols = ["interval_hour", "period", "year", "month", "forecast_hour"]

    for keys, group in pair_df.groupby(group_cols):
        interval, period, year, month, fh = keys
        w = group["count"].astype(float)

        bs_nbm = weighted_mean(group["brier_score_nbm"], w)
        bs_climo = weighted_mean(group["brier_score_climo"], w)
        bss = 1.0 - (bs_nbm / bs_climo) if bs_climo > 0 else np.nan

        rows.append({
            "interval_hour": interval,
            "period": period,
            "year": year,
            "month": month,
            "forecast_hour": fh,
            "total_count": int(group["count"].sum()),
            "total_obs": int(group["obs_count"].sum()),
            "brier_score_nbm": bs_nbm,
            "brier_score_climo": bs_climo,
            "brier_skill_score": bss,
            "n_pairs": int(len(group)),
        })

    monthly = pd.DataFrame(rows).sort_values(
        ["interval_hour", "period", "year", "month", "forecast_hour"]
    )

    out_csv = OUT_DIR / "brier_against_monthly_gld_climo_by_month.csv"
    monthly.to_csv(out_csv, index=False)
    print(f"Saved monthly aggregated results: {out_csv}")

    return monthly


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def _finish_plot(fig, out_path: Path):
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)


def plot_bss_by_valid_hour_panels(agg):
    """
    Plot Brier Skill Score by forecast hour.

    Creates separate 2-panel plots:
      - Day: valid 00Z and 06Z
      - Night: valid 12Z and 18Z

    Uses common y-axis limits within each interval/period plot.
    """

    for interval in sorted(agg["interval_hour"].unique()):
        for period in ["day", "night"]:
            valid_hours = get_valid_hours_for_period(period)

            sub_ip = agg[
                (agg["interval_hour"] == interval) &
                (agg["period"] == period) &
                (agg["valid_hour"].isin(valid_hours))
            ].copy()

            if sub_ip.empty:
                print(f"No BSS data for interval={interval}, period={period}")
                continue

            ymin, ymax = finite_min_max(sub_ip["brier_skill_score"])
            ylim = padded_limits(ymin, ymax, pad_fraction=0.10, min_pad=0.02)

            fig, axes = plt.subplots(
                1, 2,
                figsize=(14, 5.5),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )

            for ax, valid_hour in zip(axes, valid_hours):
                sub = sub_ip[sub_ip["valid_hour"] == valid_hour].sort_values("forecast_hour")

                if sub.empty:
                    ax.set_title(f"Valid {valid_hour:02d}Z\nNo data")
                    ax.axhline(0, color="black", linestyle="--", linewidth=1)
                    ax.grid(alpha=0.25)
                    continue

                ax.plot(
                    sub["forecast_hour"],
                    sub["brier_skill_score"],
                    marker="o",
                    linewidth=2,
                    label=f"Valid {valid_hour:02d}Z",
                )

                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.set_title(f"Valid {valid_hour:02d}Z")
                ax.set_xlabel("Forecast hour")
                ax.grid(alpha=0.25)
                ax.legend()

                if ylim is not None:
                    ax.set_ylim(*ylim)

            axes[0].set_ylabel("Brier Skill Score")

            fig.suptitle(
                f"Brier Skill Score vs Monthly GLD Climatology\n"
                f"{interval:02d}-hr NBM Thunder Probability, {period.title()} Valid Windows",
                fontsize=15,
            )

            out_path = PLOT_DIR / f"bss_valid_hour_panels_{interval:02d}h_{period}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {out_path}")

            plt.show()

def plot_bs_comparison_by_valid_hour_panels(agg):
    """
    Plot NBM and climatology Brier Score by forecast hour.

    Creates separate 2-panel plots:
      - Day: valid 00Z and 06Z
      - Night: valid 12Z and 18Z

    Uses common y-axis limits within each interval/period plot.
    """

    for interval in sorted(agg["interval_hour"].unique()):
        for period in ["day", "night"]:
            valid_hours = get_valid_hours_for_period(period)

            sub_ip = agg[
                (agg["interval_hour"] == interval) &
                (agg["period"] == period) &
                (agg["valid_hour"].isin(valid_hours))
            ].copy()

            if sub_ip.empty:
                print(f"No Brier Score data for interval={interval}, period={period}")
                continue

            ymin, ymax = finite_min_max(
                sub_ip["brier_score_nbm"],
                sub_ip["brier_score_climo"],
            )
            ylim = padded_limits(ymin, ymax, pad_fraction=0.10, min_pad=0.001)

            fig, axes = plt.subplots(
                1, 2,
                figsize=(14, 5.5),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )

            for ax, valid_hour in zip(axes, valid_hours):
                sub = sub_ip[sub_ip["valid_hour"] == valid_hour].sort_values("forecast_hour")

                if sub.empty:
                    ax.set_title(f"Valid {valid_hour:02d}Z\nNo data")
                    ax.grid(alpha=0.25)
                    continue

                ax.plot(
                    sub["forecast_hour"],
                    sub["brier_score_nbm"],
                    marker="o",
                    linewidth=2,
                    label="NBM",
                )

                ax.plot(
                    sub["forecast_hour"],
                    sub["brier_score_climo"],
                    marker="o",
                    linewidth=2,
                    label="Monthly GLD Climo",
                )

                ax.set_title(f"Valid {valid_hour:02d}Z")
                ax.set_xlabel("Forecast hour")
                ax.grid(alpha=0.25)
                ax.legend()

                if ylim is not None:
                    ax.set_ylim(*ylim)

            axes[0].set_ylabel("Brier Score")

            fig.suptitle(
                f"Brier Score Comparison\n"
                f"{interval:02d}-hr, {period.title()} Valid Windows",
                fontsize=15,
            )

            out_path = PLOT_DIR / f"bs_comparison_valid_hour_panels_{interval:02d}h_{period}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {out_path}")

            plt.show()

def plot_monthly_bss_heatmap_by_valid_hour(pair_df):
    """Plot monthly/lead-time BSS heatmaps separately by valid hour."""

    pair_df = pair_df.copy()
    pair_df["valid_dt"] = pd.to_datetime(pair_df["valid_dt"])
    pair_df["valid_hour"] = pair_df["valid_dt"].dt.hour

    for interval in sorted(pair_df["interval_hour"].unique()):
        for period in sorted(pair_df["period"].unique()):
            for valid_hour in [0, 6, 12, 18]:
                sub = pair_df[
                    (pair_df["interval_hour"] == interval) &
                    (pair_df["period"] == period) &
                    (pair_df["valid_hour"] == valid_hour)
                ].copy()

                if sub.empty:
                    continue

                rows = []

                for keys, group in sub.groupby(["year", "month", "forecast_hour"]):
                    year, month, fh = keys
                    w = group["count"].astype(float)

                    bs_nbm = weighted_mean(group["brier_score_nbm"], w)
                    bs_climo = weighted_mean(group["brier_score_climo"], w)

                    bss = 1.0 - (bs_nbm / bs_climo) if bs_climo > 0 else np.nan

                    rows.append({
                        "year": year,
                        "month": month,
                        "forecast_hour": fh,
                        "bss": bss,
                    })

                mdf = pd.DataFrame(rows)

                if mdf.empty:
                    continue

                mdf["year_month"] = (
                    mdf["year"].astype(str) + "-" +
                    mdf["month"].astype(str).str.zfill(2)
                )

                pivot = mdf.pivot_table(
                    index="forecast_hour",
                    columns="year_month",
                    values="bss",
                    aggfunc="mean",
                ).sort_index()

                fig, ax = plt.subplots(figsize=(14, 7))

                im = ax.imshow(
                    pivot.values,
                    aspect="auto",
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=-1,
                    vmax=1,
                )

                ax.set_title(
                    f"Monthly BSS vs Saved GLD Monthly Climatology\n"
                    f"{interval:02d}-hr, {period.title()}, Valid {valid_hour:02d}Z"
                )
                ax.set_xlabel("Month")
                ax.set_ylabel("Forecast hour")

                ax.set_xticks(np.arange(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, rotation=45, ha="right")

                ax.set_yticks(np.arange(len(pivot.index)))
                ax.set_yticklabels(pivot.index)

                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Brier Skill Score")

                plt.tight_layout()

                out_path = (
                    PLOT_DIR /
                    f"monthly_bss_heatmap_saved_climo_{interval:02d}h_{period}_valid{valid_hour:02d}z.png"
                )
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"Saved: {out_path}")

                plt.show()

def plot_mean_probability_comparison(agg: pd.DataFrame):
    """Plot weighted mean NBM probability, climo probability, and observed frequency."""
    for interval in sorted(agg["interval_hour"].unique()):
        for period in sorted(agg["period"].unique()):
            sub = agg[
                (agg["interval_hour"] == interval) &
                (agg["period"] == period)
            ].sort_values("forecast_hour")

            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                sub["forecast_hour"],
                sub["mean_forecast_prob_weighted"] * 100,
                marker="o",
                linewidth=2,
                label="Mean NBM probability",
            )
            ax.plot(
                sub["forecast_hour"],
                sub["mean_climo_prob_weighted"] * 100,
                marker="o",
                linewidth=2,
                label="Mean GLD climo probability",
            )
            ax.plot(
                sub["forecast_hour"],
                sub["obs_frequency"] * 100,
                marker="o",
                linewidth=2,
                label="Observed frequency",
            )

            ax.set_title(
                f"Mean Probability Comparison\n"
                f"No Day/Night Climo Split, {interval:02d}-hr, {period.title()}"
            )
            ax.set_xlabel("Forecast hour")
            ax.set_ylabel("Probability / Frequency (%)")
            ax.grid(alpha=0.25)
            ax.legend()

            out_path = PLOT_DIR / f"mean_probability_comparison_no_period_{interval:02d}h_{period}.png"
            _finish_plot(fig, out_path)

def plot_probability_diagnostics_by_valid_hour_panels(agg):
    """
    Plot observed frequency, mean NBM probability, and mean GLD climatology
    by forecast hour in separate valid-hour panels.

    Creates separate 2-panel plots:
      - Day: valid 00Z and 06Z
      - Night: valid 12Z and 18Z

    Uses common y-axis limits within each interval/period plot.
    """

    for interval in sorted(agg["interval_hour"].unique()):
        for period in ["day", "night"]:
            valid_hours = get_valid_hours_for_period(period)

            sub_ip = agg[
                (agg["interval_hour"] == interval) &
                (agg["period"] == period) &
                (agg["valid_hour"].isin(valid_hours))
            ].copy()

            if sub_ip.empty:
                print(f"No probability diagnostic data for interval={interval}, period={period}")
                continue

            ymin, ymax = finite_min_max(
                sub_ip["obs_frequency"],
                sub_ip["mean_forecast_prob_weighted"],
                sub_ip["mean_climo_prob_weighted"],
            )
            ylim = padded_limits(ymin, ymax, pad_fraction=0.10, min_pad=0.005)

            # Probabilities/frequencies should not go below zero
            if ylim is not None:
                ylim = (max(0.0, ylim[0]), ylim[1])

            fig, axes = plt.subplots(
                1, 2,
                figsize=(14, 5.5),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )

            for ax, valid_hour in zip(axes, valid_hours):
                sub = sub_ip[sub_ip["valid_hour"] == valid_hour].sort_values("forecast_hour")

                if sub.empty:
                    ax.set_title(f"Valid {valid_hour:02d}Z\nNo data")
                    ax.grid(alpha=0.25)
                    continue

                ax.plot(
                    sub["forecast_hour"],
                    sub["obs_frequency"],
                    marker="o",
                    linewidth=2,
                    label="Observed frequency",
                )

                ax.plot(
                    sub["forecast_hour"],
                    sub["mean_forecast_prob_weighted"],
                    marker="s",
                    linewidth=2,
                    label="Mean NBM probability",
                )

                ax.plot(
                    sub["forecast_hour"],
                    sub["mean_climo_prob_weighted"],
                    marker="^",
                    linewidth=2,
                    label="Mean GLD climo probability",
                )

                ax.set_title(f"Valid {valid_hour:02d}Z")
                ax.set_xlabel("Forecast hour")
                ax.grid(alpha=0.25)
                ax.legend()

                if ylim is not None:
                    ax.set_ylim(*ylim)

            axes[0].set_ylabel("Probability / frequency")

            fig.suptitle(
                f"Probability Diagnostics\n"
                f"{interval:02d}-hr, {period.title()} Valid Windows",
                fontsize=15,
            )

            out_path = PLOT_DIR / f"probability_diagnostics_valid_hour_panels_{interval:02d}h_{period}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {out_path}")

            plt.show()
# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    """Run Brier/BSS plotting workflow.

    By default, this reuses existing CSV files if they already exist.
    Set REBUILD_PAIR_RESULTS=True to rerun the slow raster-based verification.
    """

    # ------------------------------------------------------------
    # 1. Load or create pair-level Brier/BSS results
    # ------------------------------------------------------------
    if PAIR_CSV.exists() and not REBUILD_PAIR_RESULTS:
        print(f"Loading existing pair-level results: {PAIR_CSV}")
        pair_df = pd.read_csv(PAIR_CSV, parse_dates=["valid_dt"])
    else:
        print("Running raster-based Brier/BSS evaluation...")
        pair_df = evaluate_against_saved_climo()

        if pair_df.empty:
            print("No pair-level Brier results created.")
            return

        pair_df.to_csv(PAIR_CSV, index=False)
        print(f"Saved pair-level results: {PAIR_CSV}")

    # Make sure valid_dt and valid_hour exist
    pair_df["valid_dt"] = pd.to_datetime(pair_df["valid_dt"])

    if "valid_hour" not in pair_df.columns:
        pair_df["valid_hour"] = pair_df["valid_dt"].dt.hour

    # ------------------------------------------------------------
    # 2. Load or create aggregated results
    # ------------------------------------------------------------
    if AGG_CSV.exists() and not REBUILD_AGG_RESULTS:
        print(f"Loading existing aggregated results: {AGG_CSV}")
        agg = pd.read_csv(AGG_CSV)
    else:
        print("Aggregating pair-level results by valid hour...")
        agg = aggregate_results(pair_df)
        agg.to_csv(AGG_CSV, index=False)
        print(f"Saved aggregated results: {AGG_CSV}")

    print("\nAggregated Brier / BSS results by valid hour:")
    print(agg.to_string(index=False))

    # ------------------------------------------------------------
    # 3. Plot from existing/loaded results
    # ------------------------------------------------------------
    plot_bss_by_valid_hour_panels(agg)
    plot_bs_comparison_by_valid_hour_panels(agg)
    plot_probability_diagnostics_by_valid_hour_panels(agg)

    # Optional, if you added the valid-hour heatmap function:
    # plot_monthly_bss_heatmap_by_valid_hour(pair_df)

if __name__ == "__main__":
    main()