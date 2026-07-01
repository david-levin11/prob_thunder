#!/usr/bin/env python

"""
Read one NBM thunder GRIB2 file, apply isotonic CSV calibration on the fly,
and plot raw vs calibrated probabilities.

Does not write output.

Requires:
    xarray
    cfgrib
    numpy
    matplotlib
    cartopy

Calibration CSV format:
    raw_probability,calibrated_probability

CSV values should be in 0-1 probability units.
NBM GRIB values are assumed to be 0-100 percent.
"""

from pathlib import Path
import csv

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.colors import BoundaryNorm, ListedColormap


# ---------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------

INPUT_GRIB_FILE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\blend.t12z.core.f036.ak.grib2"
)

CSV_MODEL_FILE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\calibration_aicc_total\csv_models\isotonic_aicc_total_12h_day.csv"
)

# NBM thunder probabilities are usually 0-100 percent.
INPUT_IS_PERCENT = True

# Keep raw zero values as exact zero after calibration.
ZERO_STAYS_ZERO = True

# If True, plot categorical thunder coverage classes.
# If False, plot continuous probability bins.
USE_CATEGORICAL_PLOTTING = False

# Mask values <= this probability in continuous plots.
PLOT_MIN_PROB = 0.0

# ---------------------------------------------------------------------
# CONTINUOUS PROBABILITY COLORS
# ---------------------------------------------------------------------

PROB_LEVELS = np.array([
    0.00,
    0.01,
    0.025,
    0.05,
    0.075,
    0.10,
    0.20,
    0.30,
    0.50,
    0.70,
    1.00,
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
    "#f7fbff",  # 0-1%
    "#dbeef7",  # 1-2.5%
    "#b6d7e8",  # 2.5-5%
    "#fff2b2",  # 5-7.5%
    "#fed976",  # 7.5-10%
    "#feb24c",  # 10-20%
    "#fd8d3c",  # 20-30%
    "#f03b20",  # 30-50%
    "#bd0026",  # 50-70%
    "#800026",  # 70-100%
]

# ---------------------------------------------------------------------
# CATEGORICAL PROBABILITY COLORS
# ---------------------------------------------------------------------

CATEGORY_DEFS = [
    {"label": "Isolated",   "min": 0.05, "max": 0.30, "color": "yellow"},
    {"label": "Scattered",  "min": 0.30, "max": 0.51, "color": "orange"},
    {"label": "Numerous",   "min": 0.51, "max": 0.701, "color": "red"},
    {"label": "Widespread", "min": 0.701, "max": 1.01, "color": "purple"},
]


# ---------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------

def get_nbm_alaska_crs():
    """Cartopy CRS matching the NBM Alaska polar stereographic grid."""

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


# ---------------------------------------------------------------------
# CALIBRATION
# ---------------------------------------------------------------------

def load_calibration_csv(csv_file):
    """Load isotonic calibration CSV into sorted arrays."""

    raw_probs = []
    calibrated_probs = []

    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            raw_probs.append(float(row["raw_probability"]))
            calibrated_probs.append(float(row["calibrated_probability"]))

    raw_probs = np.asarray(raw_probs, dtype="float64")
    calibrated_probs = np.asarray(calibrated_probs, dtype="float64")

    if raw_probs.size < 2:
        raise ValueError(f"Calibration CSV has too few rows: {csv_file}")

    order = np.argsort(raw_probs)

    return raw_probs[order], calibrated_probs[order]


def apply_calibration(raw_prob, raw_lookup, calibrated_lookup):
    """Apply CSV isotonic calibration to a probability array in 0-1 units."""

    calibrated = np.full(raw_prob.shape, np.nan, dtype="float32")

    valid = np.isfinite(raw_prob)

    if not np.any(valid):
        return calibrated

    x = raw_prob[valid].astype("float64")
    x = np.clip(x, 0.0, 1.0)

    y = np.interp(
        x,
        raw_lookup,
        calibrated_lookup,
        left=calibrated_lookup[0],
        right=calibrated_lookup[-1],
    )

    y = np.clip(y, 0.0, 1.0)

    if ZERO_STAYS_ZERO:
        y[x == 0.0] = 0.0

    calibrated[valid] = y.astype("float32")

    return calibrated


# ---------------------------------------------------------------------
# GRIB READING
# ---------------------------------------------------------------------

def open_single_message_grib(path):
    """Open one-message GRIB file with xarray/cfgrib."""

    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",  # do not create .idx sidecar file
        },
    )

    data_vars = list(ds.data_vars)

    if len(data_vars) == 1:
        var_name = data_vars[0]
    else:
        # Fallback: use the first variable with at least 2 dimensions.
        candidates = [v for v in data_vars if ds[v].ndim >= 2]

        if not candidates:
            raise ValueError(f"Could not identify gridded variable: {data_vars}")

        var_name = candidates[0]

    da = ds[var_name]

    return ds, da, var_name


def get_plot_xy(da):
    """Return longitude/latitude coordinate arrays for plotting."""

    if "longitude" not in da.coords or "latitude" not in da.coords:
        raise ValueError(
            "Expected latitude/longitude coordinates in GRIB dataset, "
            f"but found coords: {list(da.coords)}"
        )

    lon = da["longitude"].values
    lat = da["latitude"].values

    # Convert 0-360 longitudes to -180 to 180 if needed.
    lon = np.where(lon > 180.0, lon - 360.0, lon)

    return lon, lat


# ---------------------------------------------------------------------
# CATEGORICAL PLOTTING
# ---------------------------------------------------------------------

def make_categorical_array(prob_arr, category_defs):
    """Convert continuous probabilities to category codes."""

    cat_arr = np.full(prob_arr.shape, np.nan, dtype="float32")
    colors = []
    labels = []

    for i, cat in enumerate(category_defs, start=1):
        cmin = cat["min"]
        cmax = cat["max"]

        mask = np.isfinite(prob_arr) & (prob_arr >= cmin) & (prob_arr < cmax)
        cat_arr[mask] = float(i)

        colors.append(cat["color"])

        if cmax >= 1.0:
            labels.append(f"{cat['label']} ≥{cmin * 100:.0f}%")
        else:
            labels.append(f"{cat['label']} {cmin * 100:.0f}-{cmax * 100:.0f}%")

    cmap = ListedColormap(colors)
    cmap.set_bad(alpha=0.0)

    return cat_arr, cmap, labels


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def plot_raw_vs_calibrated(da, raw_prob, calibrated_prob, var_name):
    """Plot raw and calibrated probability arrays."""

    map_crs = get_nbm_alaska_crs()

    data_crs = ccrs.PlateCarree()

    lon, lat = get_plot_xy(da)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8),
        subplot_kw={"projection": map_crs},
        constrained_layout=True,
    )

    def add_background(ax):
        ax.add_feature(cfeature.LAND, facecolor="0.90", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="black", zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black", zorder=3)

        # Set extent from the lon/lat data.
        lon_min = -178
        lon_max = -130
        lat_min = 55
        lat_max = 72

        pad_x = 2.0
        pad_y = 1.0

        ax.set_extent(
            [lon_min - pad_x, lon_max + pad_x, lat_min - pad_y, lat_max + pad_y],
            crs=data_crs,
        )

        ax.gridlines(
            crs=data_crs,
            draw_labels=False,
            linewidth=0.3,
            color="gray",
            alpha=0.4,
            linestyle="--",
        )

    if USE_CATEGORICAL_PLOTTING:
        raw_plot, raw_cmap, labels = make_categorical_array(raw_prob, CATEGORY_DEFS)
        cal_plot, cal_cmap, _ = make_categorical_array(calibrated_prob, CATEGORY_DEFS)

        raw_plot = np.ma.masked_invalid(raw_plot)
        cal_plot = np.ma.masked_invalid(cal_plot)

        for ax, arr, cmap, title, prob in [
            (axes[0], raw_plot, raw_cmap, "Raw NBM thunder categories", raw_prob),
            (axes[1], cal_plot, cal_cmap, "Calibrated thunder categories", calibrated_prob),
        ]:
            add_background(ax)

            im = ax.pcolormesh(
                lon,
                lat,
                arr,
                transform=data_crs,
                cmap=cmap,
                vmin=1,
                vmax=len(CATEGORY_DEFS),
                shading="auto",
                zorder=2,
            )

            ax.set_title(
                f"{title}\n"
                f"Max: {np.nanmax(prob):.2f}, Mean: {np.nanmean(prob):.4f}"
            )

        # Categorical legend
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=cat["color"], edgecolor="black", label=label)
            for cat, label in zip(CATEGORY_DEFS, labels)
        ]

        axes[1].legend(
            handles=handles,
            loc="lower left",
            frameon=True,
            framealpha=0.92,
            title="Probability category",
        )

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

        last_im = None

        for ax, arr, title, prob in [
            (axes[0], raw_plot, "Raw NBM thunder probability", raw_prob),
            (axes[1], cal_plot, "Calibrated thunder probability", calibrated_prob),
        ]:
            add_background(ax)

            im = ax.pcolormesh(
                lon,
                lat,
                arr,
                transform=data_crs,
                cmap=cmap,
                norm=norm,
                shading="auto",
                zorder=2,
            )

            last_im = im

            ax.set_title(
                f"{title}\n"
                f"Max: {np.nanmax(prob):.2f}, Mean: {np.nanmean(prob):.4f}"
            )

        cbar = fig.colorbar(
            last_im,
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
        cbar.ax.tick_params(labelsize=8)

    valid_time = da.coords.get("valid_time", None)
    step = da.coords.get("step", None)

    title_bits = [f"GRIB: {INPUT_GRIB_FILE.name}", f"Variable: {var_name}"]

    if valid_time is not None:
        title_bits.append(f"Valid: {np.datetime_as_string(valid_time.values, unit='h')}")

    if step is not None:
        title_bits.append(f"Step: {step.values}")

    fig.suptitle("\n".join(title_bits), fontsize=12)

    plt.show()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print(f"Opening GRIB:\n  {INPUT_GRIB_FILE}")
    ds, da, var_name = open_single_message_grib(INPUT_GRIB_FILE)

    print("\nDataset:")
    print(ds)

    print(f"\nUsing variable: {var_name}")

    raw_values = da.values.astype("float32")

    if INPUT_IS_PERCENT:
        raw_prob = raw_values / 100.0
    else:
        raw_prob = raw_values

    raw_prob = np.where(
        np.isfinite(raw_prob),
        np.clip(raw_prob, 0.0, 1.0),
        np.nan,
    ).astype("float32")

    print(f"\nLoading calibration CSV:\n  {CSV_MODEL_FILE}")
    raw_lookup, calibrated_lookup = load_calibration_csv(CSV_MODEL_FILE)

    calibrated_prob = apply_calibration(
        raw_prob=raw_prob,
        raw_lookup=raw_lookup,
        calibrated_lookup=calibrated_lookup,
    )

    print("\nRaw probability summary:")
    print(f"  min:  {np.nanmin(raw_prob):.4f}")
    print(f"  mean: {np.nanmean(raw_prob):.4f}")
    print(f"  max:  {np.nanmax(raw_prob):.4f}")

    print("\nCalibrated probability summary:")
    print(f"  min:  {np.nanmin(calibrated_prob):.4f}")
    print(f"  mean: {np.nanmean(calibrated_prob):.4f}")
    print(f"  max:  {np.nanmax(calibrated_prob):.4f}")

    plot_raw_vs_calibrated(
        da=da,
        raw_prob=raw_prob,
        calibrated_prob=calibrated_prob,
        var_name=var_name,
    )


if __name__ == "__main__":
    main()