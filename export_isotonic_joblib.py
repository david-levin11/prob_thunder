#!/usr/bin/env python

"""
Export fitted sklearn IsotonicRegression .joblib models to portable CSV files.

Run this script on the machine where joblib/sklearn are available.

Output CSV format:

    raw_probability,calibrated_probability
    0.0000000000,0.0000000000
    0.0123456789,0.0345678901
    ...

These CSVs can then be used on a locked-down system without joblib or sklearn.
"""

from pathlib import Path
import csv

import joblib


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATASET_NAME = "aicc_total"
# Options:
#   "gld"
#   "aicc_total"
#   "union"

MODEL_DIR = Path(
    rf"C:\Users\David.Levin\NBMLightningVer\calibration_{DATASET_NAME}\models"
)

OUT_DIR = Path(
    rf"C:\Users\David.Levin\NBMLightningVer\calibration_{DATASET_NAME}\csv_models"
)

INTERVAL_HOURS = [6, 12]
PERIODS = ["day", "night"]


# ---------------------------------------------------------------------
# EXPORT FUNCTIONS
# ---------------------------------------------------------------------

def export_isotonic_model_to_csv(model_path: Path, out_csv: Path):
    """Export one fitted sklearn IsotonicRegression model to CSV."""

    if not model_path.exists():
        print(f"Missing model, skipping: {model_path}")
        return False

    model = joblib.load(model_path)

    if not hasattr(model, "X_thresholds_") or not hasattr(model, "y_thresholds_"):
        raise TypeError(
            f"Model does not look like a fitted sklearn IsotonicRegression model:\n"
            f"{model_path}"
        )

    raw_thresholds = model.X_thresholds_
    calibrated_thresholds = model.y_thresholds_

    if len(raw_thresholds) != len(calibrated_thresholds):
        raise ValueError(
            f"Threshold length mismatch in model:\n{model_path}"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["raw_probability", "calibrated_probability"])

        for raw_p, cal_p in zip(raw_thresholds, calibrated_thresholds):
            writer.writerow([
                f"{float(raw_p):.10f}",
                f"{float(cal_p):.10f}",
            ])

    print(f"Exported: {out_csv}")
    return True


def main():
    print(f"Reading models from: {MODEL_DIR}")
    print(f"Writing CSV models to: {OUT_DIR}")

    exported = 0

    for interval in INTERVAL_HOURS:
        for period in PERIODS:
            model_name = f"isotonic_{DATASET_NAME}_{interval:02d}h_{period}.joblib"
            csv_name = f"isotonic_{DATASET_NAME}_{interval:02d}h_{period}.csv"

            model_path = MODEL_DIR / model_name
            out_csv = OUT_DIR / csv_name

            success = export_isotonic_model_to_csv(model_path, out_csv)

            if success:
                exported += 1

    print(f"\nExport complete. CSV models written: {exported}")


if __name__ == "__main__":
    main()