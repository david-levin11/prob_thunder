#!/usr/bin/env python

"""
Apply an isotonic calibration CSV to one NBM probability grid.

Requires:
    numpy
    rasterio

Does NOT require:
    joblib
    sklearn

Expected calibration CSV columns:
    raw_probability,calibrated_probability

The CSV probabilities should be in 0-1 units.
"""

from pathlib import Path
import csv

import numpy as np
import rasterio


# ---------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------

CSV_MODEL_FILE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\calibration_aicc_total\csv_models\isotonic_aicc_total_12h_night.csv"
)

INPUT_GRID_FILE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\blendv5.0_alaska_tstm12_2026-06-17T12_00_2026-06-19T06_00.tif"
)

OUTPUT_GRID_FILE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\blendv5.0_alaska_tstm12_2026-06-17T12_00_2026-06-19T06_00_calibrated.tif"
)

# Set True if the input NBM grid is stored as 0-100 percent.
# Set False if already stored as 0-1 probability.
INPUT_IS_PERCENT = True

# Set True to write output as 0-100 percent.
# Set False to write output as 0-1 probability.
OUTPUT_AS_PERCENT = True

# Keep raw 0% pixels as exactly 0%.
ZERO_STAYS_ZERO = True

# Output nodata value.
OUTPUT_NODATA = -9999.0


# ---------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------

def load_calibration_csv(csv_file):
    """Load calibration CSV into sorted NumPy arrays."""

    raw_probs = []
    calibrated_probs = []

    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            raw_probs.append(float(row["raw_probability"]))
            calibrated_probs.append(float(row["calibrated_probability"]))

    if len(raw_probs) < 2:
        raise ValueError(f"Calibration CSV has too few rows: {csv_file}")

    raw_probs = np.asarray(raw_probs, dtype="float64")
    calibrated_probs = np.asarray(calibrated_probs, dtype="float64")

    order = np.argsort(raw_probs)

    return raw_probs[order], calibrated_probs[order]


def apply_calibration(raw_prob, raw_lookup, calibrated_lookup):
    """Apply calibration lookup to a 2D probability grid in 0-1 units."""

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


def main():
    if not CSV_MODEL_FILE.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {CSV_MODEL_FILE}")

    if not INPUT_GRID_FILE.exists():
        raise FileNotFoundError(f"Input grid not found: {INPUT_GRID_FILE}")

    print(f"Loading calibration CSV:\n  {CSV_MODEL_FILE}")
    raw_lookup, calibrated_lookup = load_calibration_csv(CSV_MODEL_FILE)

    print(f"Reading input grid:\n  {INPUT_GRID_FILE}")
    with rasterio.open(INPUT_GRID_FILE) as src:
        grid = src.read(1).astype("float32")
        profile = src.profile.copy()
        input_nodata = src.nodata

    if input_nodata is not None:
        grid = np.where(grid == input_nodata, np.nan, grid)

    raw_prob = grid.astype("float32")

    if INPUT_IS_PERCENT:
        raw_prob = raw_prob / 100.0

    raw_prob = np.where(
        np.isfinite(raw_prob),
        np.clip(raw_prob, 0.0, 1.0),
        np.nan,
    ).astype("float32")

    calibrated_prob = apply_calibration(
        raw_prob=raw_prob,
        raw_lookup=raw_lookup,
        calibrated_lookup=calibrated_lookup,
    )

    print("\nRaw probability summary, 0-1 units:")
    print(f"  min:  {np.nanmin(raw_prob):.4f}")
    print(f"  mean: {np.nanmean(raw_prob):.4f}")
    print(f"  max:  {np.nanmax(raw_prob):.4f}")

    print("\nCalibrated probability summary, 0-1 units:")
    print(f"  min:  {np.nanmin(calibrated_prob):.4f}")
    print(f"  mean: {np.nanmean(calibrated_prob):.4f}")
    print(f"  max:  {np.nanmax(calibrated_prob):.4f}")

    out_grid = calibrated_prob.copy()

    if OUTPUT_AS_PERCENT:
        out_grid = out_grid * 100.0

    out_grid = out_grid.astype("float32")
    out_grid[~np.isfinite(out_grid)] = OUTPUT_NODATA

    profile.update(
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate",
    )

    OUTPUT_GRID_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_GRID_FILE.exists():
        OUTPUT_GRID_FILE.unlink()

    print(f"\nWriting calibrated grid:\n  {OUTPUT_GRID_FILE}")
    with rasterio.open(OUTPUT_GRID_FILE, "w", **profile) as dst:
        dst.write(out_grid, 1)

    print("Done.")


if __name__ == "__main__":
    main()