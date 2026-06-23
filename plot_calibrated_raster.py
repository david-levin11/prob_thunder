#!/usr/bin/env python

"""Plot raw NBM thunder probability and isotonic-calibrated probability.

This script:
1. Reads a single NBM thunder probability raster.
2. Determines interval_hour and day/night period from the filename.
3. Loads the matching weighted isotonic calibration model.
4. Applies the calibration to each valid pixel.
5. Optionally writes a calibrated GeoTIFF.
6. Plots raw vs calibrated probability side by side.

Expected model paths:
    C:/Users/David.Levin/NBMLightningVer/calibration_aicc_total/models/
        isotonic_aicc_total_12h_day.joblib

    C:/Users/David.Levin/NBMLightningVer/calibration_union/models/
        isotonic_union_12h_day.joblib

etc.
"""

from pathlib import Path
from datetime import datetime, timedelta
import re

import numpy as np
import rioxarray as rxr
import rasterio
import joblib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATASET_NAME = "aicc_total"
# Options:
#   "gld"
#   "aicc_total"
#   "union"

MODEL_BASE = Path(
    rf"C:\Users\David.Levin\NBMLightningVer\calibration_{DATASET_NAME}\models"
)

OUT_DIR = Path(
    rf"C:\Users\David.Levin\NBMLightningVer\calibration_{DATASET_NAME}\sample_maps"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick one NBM thunder raster to test.
FORECAST_FILE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\blendv5.0_alaska_tstm12_2026-06-17T12_00_2026-06-19T06_00.tif"
)


# # Pick one NBM thunder raster to test.
# FORECAST_FILE = Path(
#     r"C:\Users\David.Levin\NBMLightningVer\nbm_data\2025\06\18\1300\tstm12\blendv4.3_alaska_tstm12_2025-06-18T1300_F017.tif"
# )

FORECAST_IS_PERCENT = True

# If True, write the calibrated probability raster.
WRITE_CALIBRATED_RASTER = True

# If True, save the side-by-side plot.
SAVE_PNG = True

# If True, show the plot interactively.
SHOW_PLOT = True

# ---------------------------------------------------------------------
# PLOTTING OPTIONS
# ---------------------------------------------------------------------

# If True, plot categorical probability classes.
# If False, plot continuous probabilities using PROB_LEVELS.
USE_CATEGORICAL_PLOTTING = False

# Continuous plot masking threshold only.
# Values <= this threshold are transparent in continuous plots.
PLOT_MIN_PROB = 0.009

PROB_LEVELS = np.array([
    0.00,
    0.01,   # 1%
    0.025,  # 2.5%
    0.05,   # 5%
    0.075,  # 7.5%
    0.10,   # 10%
    0.20,   # 20%
    0.30,   # 30%
    0.50,   # 50%
    0.70,   # 70%
    1.00,   # 100%
])

PROB_LEVEL_LABELS = [
    "0%",
    "1%",
    "2.5%",
    "5%",
    "7.5%",
    "10%",
    "20%",
    "30%",
    "50%",
    "70%",
    "100%",
]

PROB_COLORS = [
    "#f7fbff",  # 0–1%
    "#dbeef7",  # 1–2.5%
    "#b6d7e8",  # 2.5–5%
    "#fff2b2",  # 5–7.5%
    "#fed976",  # 7.5–10%
    "#feb24c",  # 10–20%
    "#fd8d3c",  # 20–30%
    "#f03b20",  # 30–50%
    "#bd0026",  # 50–70%
    "#800026",  # 70–100%
]

# Categorical probability classes. Values are 0-1 probabilities.
# lower bound is inclusive; upper bound is exclusive, except the final
# high-end category effectively includes everything up to 1.0.
CATEGORY_DEFS = [
    {"label": "Isolated",   "min": 0.05, "max": 0.30, "color": "yellow"},
    {"label": "Scattered",  "min": 0.30, "max": 0.51, "color": "orange"},
    {"label": "Numerous",   "min": 0.51, "max": 0.701, "color": "red"},
    {"label": "Widespread", "min": 0.701, "max": 1.01, "color": "purple"},
]


VALID_HOUR_TO_PERIOD = {
    0: "day",
    6: "day",
    12: "night",
    18: "night",
}

# ---------------------------------------------------------------------
# CALIBRATION OPTIONS
# ---------------------------------------------------------------------

# With the updated calibration model trained on mean_fcst_prob, it is reasonable
# to apply the model over the full non-missing field. Exact raw zeros can still
# be forced back to zero so a true 0% NBM area does not become nonzero.
CALIBRATE_MIN_RAW_PROB = 0.00

# For a true reliability calibration comparison, leave this False.
# Set True only if you intentionally want an upward-only operational stretch.
UPWARD_ONLY_CALIBRATION = False

ZERO_STAYS_ZERO = True

# Print counts of raw/calibrated grid cells exceeding key probabilities.
PRINT_FOOTPRINT_DIAGNOSTICS = True
FOOTPRINT_THRESHOLDS = [0.01, 0.05, 0.10, 0.30, 0.51, 0.70]

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def get_nbm_alaska_crs():
    """Cartopy CRS matching the Alaska NBM polar stereographic grid."""

    globe = ccrs.Globe(
        semimajor_axis=6371200,
        semiminor_axis=6371200,
        ellipse=None,
    )

    return ccrs.NorthPolarStereo(
        true_scale_latitude=60,
        central_longitude=-150,
        globe=globe,
    )


def parse_forecast_file_info(path: Path):
    """Parse init time, valid time, forecast hour, interval, and period from NBM filename/path.

    Supports both filename styles:

    1) Old archive style:
       blendv4.3_alaska_tstm12_2025-06-18T1300_F017.tif

    2) New exported style:
       blendv5.0_alaska_tstm12_2026-06-14T12_00_2026-06-15T06_00.tif
    """

    name = path.name

    # ------------------------------------------------------------
    # Parse interval from tstm06 or tstm12 in filename or parent path
    # ------------------------------------------------------------
    interval_match = re.search(r"tstm(\d{2})", str(path))

    if not interval_match:
        raise ValueError(f"Could not parse tstm interval from path: {path}")

    interval_hour = int(interval_match.group(1))

    # ------------------------------------------------------------
    # Pattern 1: old style with init and forecast hour
    # Example: 2025-06-18T1300_F017
    # ------------------------------------------------------------
    old_match = re.search(
        r"(\d{4}-\d{2}-\d{2}T\d{4})_F(\d{3})",
        name,
    )

    if old_match:
        init_str, fh_str = old_match.groups()
        init_dt = datetime.strptime(init_str, "%Y-%m-%dT%H%M")
        forecast_hour = int(fh_str)
        valid_dt = init_dt + timedelta(hours=forecast_hour)

    else:
        # ------------------------------------------------------------
        # Pattern 2: new style with init time and valid time
        # Example: 2026-06-14T12_00_2026-06-15T06_00
        # ------------------------------------------------------------
        new_match = re.search(
            r"(\d{4}-\d{2}-\d{2}T\d{2})_(\d{2})_"
            r"(\d{4}-\d{2}-\d{2}T\d{2})_(\d{2})",
            name,
        )

        if not new_match:
            raise ValueError(
                f"Could not parse NBM timing from: {name}\n\n"
                "Expected one of these filename patterns:\n"
                "  Old style: 2025-06-18T1300_F017\n"
                "  New style: 2026-06-14T12_00_2026-06-15T06_00"
            )

        init_datehour, init_minute, valid_datehour, valid_minute = new_match.groups()

        init_dt = datetime.strptime(
            f"{init_datehour}_{init_minute}",
            "%Y-%m-%dT%H_%M",
        )

        valid_dt = datetime.strptime(
            f"{valid_datehour}_{valid_minute}",
            "%Y-%m-%dT%H_%M",
        )

        forecast_hour = int(round((valid_dt - init_dt).total_seconds() / 3600.0))

        if forecast_hour < 0:
            raise ValueError(
                f"Parsed negative forecast hour from {name}: "
                f"init={init_dt}, valid={valid_dt}"
            )

    period = VALID_HOUR_TO_PERIOD.get(valid_dt.hour)

    if period is None:
        raise ValueError(
            f"Valid hour {valid_dt.hour} is not mapped to day/night period."
        )

    return {
        "init_dt": init_dt,
        "valid_dt": valid_dt,
        "forecast_hour": forecast_hour,
        "interval_hour": interval_hour,
        "period": period,
    }


def get_model_path(dataset_name, interval_hour, period):
    """Return the matching isotonic calibration model path."""

    return MODEL_BASE / f"isotonic_{dataset_name}_{interval_hour:02d}h_{period}.joblib"


def apply_calibration(
    raw_prob,
    model,
    calibrate_min_raw_prob=CALIBRATE_MIN_RAW_PROB,
    upward_only=UPWARD_ONLY_CALIBRATION,
    zero_stays_zero=ZERO_STAYS_ZERO,
):
    """Apply isotonic calibration while preserving low-end probability footprint."""

    calibrated = raw_prob.astype("float32").copy()

    valid = np.isfinite(raw_prob)
    to_calibrate = valid & (raw_prob >= calibrate_min_raw_prob)

    if np.any(to_calibrate):
        x = raw_prob[to_calibrate].astype(float)
        x = np.clip(x, 0.0, 1.0)

        y = model.predict(x).astype("float32")

        if upward_only:
            y = np.maximum(y, x).astype("float32")

        calibrated[to_calibrate] = y

    if zero_stays_zero:
        calibrated[valid & (raw_prob == 0.0)] = 0.0

    return calibrated

def write_calibrated_raster(template_ds, calibrated_prob, out_path):
    """Write calibrated probability raster as float32 GeoTIFF."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    nodata = -9999.0
    out_arr = calibrated_prob.astype("float32").copy()
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
                f"Could not overwrite {out_path}. Close ArcPro/QGIS/Python viewers "
                "or use a different output path."
            ) from e

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_arr, 1)

    print(f"Wrote calibrated raster: {out_path}")


def probability_category_label(cat):
    """Create a human-readable legend label for one category definition."""

    cmin = cat["min"] * 100.0
    cmax = cat["max"] * 100.0

    if cat["max"] >= 1.0:
        return f"{cat['label']} (>={cmin:.0f}%)"

    # Display upper edge as the highest whole-percent category value.
    # Example: max=0.30 means 5-29%.
    upper_display = cmax - 1.0

    if upper_display <= cmin:
        return f"{cat['label']} ({cmin:.0f}-{cmax:.0f}%)"

    return f"{cat['label']} ({cmin:.0f}-{upper_display:.0f}%)"


def make_categorical_array(prob_arr, category_defs):
    """Convert continuous probabilities into categorical plotting codes.

    Returns
    -------
    cat_arr : np.ndarray
        Float array with NaN where transparent, otherwise category codes 1..N.
    cmap : ListedColormap
        Categorical colormap matching category_defs.
    legend_handles : list[Patch]
        Legend handles for category labels.
    """

    cat_arr = np.full(prob_arr.shape, np.nan, dtype="float32")
    colors = []
    legend_handles = []

    for i, cat in enumerate(category_defs, start=1):
        cmin = float(cat["min"])
        cmax = float(cat["max"])
        color = cat["color"]

        mask = np.isfinite(prob_arr) & (prob_arr >= cmin) & (prob_arr < cmax)
        cat_arr[mask] = float(i)

        colors.append(color)
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor="black",
                label=probability_category_label(cat),
            )
        )

    cmap = ListedColormap(colors)
    cmap.set_bad(alpha=0.0)

    return cat_arr, cmap, legend_handles


def print_footprint_diagnostics(raw_prob, calibrated_prob):
    """Print raw vs calibrated probability footprint counts."""

    valid = np.isfinite(raw_prob) & np.isfinite(calibrated_prob)

    print("\nFootprint diagnostics:")
    print(f"  Valid pixels: {int(valid.sum()):,}")

    for t in FOOTPRINT_THRESHOLDS:
        raw_count = int(np.count_nonzero(valid & (raw_prob >= t)))
        cal_count = int(np.count_nonzero(valid & (calibrated_prob >= t)))
        diff = cal_count - raw_count

        if raw_count > 0:
            pct_change = 100.0 * diff / raw_count
            pct_text = f"{pct_change:+.1f}%"
        else:
            pct_text = "n/a"

        print(
            f"  >= {t * 100:5.1f}%: "
            f"raw={raw_count:,}, calibrated={cal_count:,}, "
            f"diff={diff:+,} ({pct_text})"
        )


def plot_raw_vs_calibrated(
    ds,
    raw_prob,
    calibrated_prob,
    info,
    model_path,
    out_png=None,
):
    """Plot raw and calibrated NBM probabilities side by side."""

    raster_crs = get_nbm_alaska_crs()

    left, bottom, right, top = ds.rio.bounds()
    extent = [left, right, bottom, top]

    if USE_CATEGORICAL_PLOTTING:
        raw_cat, raw_cmap, raw_legend_handles = make_categorical_array(
            raw_prob, CATEGORY_DEFS
        )
        cal_cat, cal_cmap, cal_legend_handles = make_categorical_array(
            calibrated_prob, CATEGORY_DEFS
        )

        raw_plot = np.ma.masked_invalid(raw_cat)
        cal_plot = np.ma.masked_invalid(cal_cat)
    else:
        raw_plot = np.ma.masked_where(
            (~np.isfinite(raw_prob)) | (raw_prob <= PLOT_MIN_PROB),
            raw_prob,
        )

        cal_plot = np.ma.masked_where(
            (~np.isfinite(calibrated_prob)) | (calibrated_prob <= PLOT_MIN_PROB),
            calibrated_prob,
        )

        cmap = ListedColormap(PROB_COLORS)
        cmap.set_bad(alpha=0.0)

        norm = BoundaryNorm(
            PROB_LEVELS,
            ncolors=cmap.N,
            clip=True,
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8),
        subplot_kw={"projection": raster_crs},
        constrained_layout=True,
    )

    def add_background(ax):
        ax.add_feature(cfeature.LAND, facecolor="0.90", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="black", zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black", zorder=3)

        ax.set_extent(extent, crs=raster_crs)

        ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=False,
            linewidth=0.3,
            color="gray",
            alpha=0.4,
            linestyle="--",
        )

    # Raw panel
    ax = axes[0]
    add_background(ax)

    if USE_CATEGORICAL_PLOTTING:
        im0 = ax.imshow(
            raw_plot,
            origin="upper",
            extent=extent,
            transform=raster_crs,
            cmap=raw_cmap,
            vmin=1,
            vmax=len(CATEGORY_DEFS),
            alpha=0.90,
            zorder=2,
        )
    else:
        im0 = ax.imshow(
            raw_plot,
            origin="upper",
            extent=extent,
            transform=raster_crs,
            cmap=cmap,
            norm=norm,
            alpha=0.90,
            zorder=2,
        )

    raw_title_label = (
        "Raw NBM Thunder Probability Categories"
        if USE_CATEGORICAL_PLOTTING
        else "Raw NBM Thunder Probability"
    )
    ax.set_title(
        f"{raw_title_label}\n"
        f"Max: {np.nanmax(raw_prob):.2f}, Mean: {np.nanmean(raw_prob):.4f}"
    )

    # Calibrated panel
    ax = axes[1]
    add_background(ax)

    if USE_CATEGORICAL_PLOTTING:
        im1 = ax.imshow(
            cal_plot,
            origin="upper",
            extent=extent,
            transform=raster_crs,
            cmap=cal_cmap,
            vmin=1,
            vmax=len(CATEGORY_DEFS),
            alpha=0.90,
            zorder=2,
        )
    else:
        im1 = ax.imshow(
            cal_plot,
            origin="upper",
            extent=extent,
            transform=raster_crs,
            cmap=cmap,
            norm=norm,
            alpha=0.90,
            zorder=2,
        )

    cal_title_label = (
        "Calibrated NBM Thunder Probability Categories"
        if USE_CATEGORICAL_PLOTTING
        else "Calibrated NBM Thunder Probability"
    )
    ax.set_title(
        f"{cal_title_label}\n"
        f"Max: {np.nanmax(calibrated_prob):.2f}, Mean: {np.nanmean(calibrated_prob):.4f}"
    )

    if USE_CATEGORICAL_PLOTTING:
        for ax in axes:
            ax.legend(
                handles=cal_legend_handles,
                loc="lower left",
                frameon=True,
                framealpha=0.92,
                title="Probability category",
            )
    else:
        cbar = fig.colorbar(
            im1,
            ax=axes,
            orientation="horizontal",
            fraction=0.046,
            pad=0.05,
            boundaries=PROB_LEVELS,
            ticks=PROB_LEVELS,
            spacing="uniform",
        )

        cbar.set_label("Thunder probability")
        cbar.ax.set_xticklabels(PROB_LEVEL_LABELS)

    fig.suptitle(
        f"Raw vs {DATASET_NAME} Isotonic-Calibrated NBM Thunder Probability\n"
        f"Init: {info['init_dt']:%Y-%m-%d %HZ}, "
        f"F{info['forecast_hour']:03d}, "
        f"Valid: {info['valid_dt']:%Y-%m-%d %HZ}, "
        f"{info['interval_hour']:02d}h {info['period'].title()}\n"
        f"Model: {model_path.name}",
        fontsize=12,
    )

    if out_png is not None:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {out_png}")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    info = parse_forecast_file_info(FORECAST_FILE)

    print("Forecast info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    model_path = get_model_path(
        dataset_name=DATASET_NAME,
        interval_hour=info["interval_hour"],
        period=info["period"],
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Calibration model not found:\n{model_path}\n\n"
            "Check DATASET_NAME, interval, period, and model directory."
        )

    print(f"\nLoading calibration model: {model_path}")
    model = joblib.load(model_path)

    with rxr.open_rasterio(FORECAST_FILE, mask_and_scale=True) as ds:
        raw = ds.values[0].astype(float)

        if FORECAST_IS_PERCENT:
            raw_prob = raw / 100.0
        else:
            raw_prob = raw

        raw_prob = np.where(np.isfinite(raw_prob), np.clip(raw_prob, 0.0, 1.0), np.nan)

        calibrated_prob = apply_calibration(raw_prob, model)

        if PRINT_FOOTPRINT_DIAGNOSTICS:
            print_footprint_diagnostics(raw_prob, calibrated_prob)

        print("\nRaw probability summary:")
        print(f"  min:  {np.nanmin(raw_prob):.4f}")
        print(f"  mean: {np.nanmean(raw_prob):.4f}")
        print(f"  max:  {np.nanmax(raw_prob):.4f}")

        print("\nCalibrated probability summary:")
        print(f"  min:  {np.nanmin(calibrated_prob):.4f}")
        print(f"  mean: {np.nanmean(calibrated_prob):.4f}")
        print(f"  max:  {np.nanmax(calibrated_prob):.4f}")

        valid_time_str = info["valid_dt"].strftime("%Y%m%d_%H00Z")
        base_name = (
            f"nbm_{DATASET_NAME}_calibrated_"
            f"tstm{info['interval_hour']:02d}_"
            f"{valid_time_str}_F{info['forecast_hour']:03d}"
        )

        if WRITE_CALIBRATED_RASTER:
            out_tif = OUT_DIR / f"{base_name}.tif"
            write_calibrated_raster(ds, calibrated_prob, out_tif)

        if SAVE_PNG:
            out_png = OUT_DIR / f"{base_name}_raw_vs_calibrated.png"
        else:
            out_png = None

        plot_raw_vs_calibrated(
            ds=ds,
            raw_prob=raw_prob,
            calibrated_prob=calibrated_prob,
            info=info,
            model_path=model_path,
            out_png=out_png,
        )


if __name__ == "__main__":
    main()