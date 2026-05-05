"""Aggregate one giant GLD/Vaisala CSV into rolling-window CSV point extracts.

Input file:
- GLD_AK_20-24.csv

Expected input columns:
- Lat,Lon,Ip,mult,SMA,SMI,Chi,Cloud/CG,DateTime

Output layout:
- Aggregated CSV files are written under INPUT_DIR into nldn_06_obs and nldn_12_obs.
- Output names follow nldn_XXh_YYYYMMDD_HHMMZ.csv where XX is the window.
- Output columns match the reference NLDN aggregation code:
  valid_datetime, lat, lon, polarity, strength, flashes
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# --- Configuration ---
INPUT_DIR = r"C:\Users\David.Levin\NBMLightningVer"
INPUT_FILE = os.path.join(INPUT_DIR, "GLD_AK_20-24.csv")

OUTPUT_DIR_06 = os.path.join(INPUT_DIR, "gld_06_obs")
OUTPUT_DIR_12 = os.path.join(INPUT_DIR, "gld_12_obs")

for d in [OUTPUT_DIR_06, OUTPUT_DIR_12]:
    os.makedirs(d, exist_ok=True)

YEARS = [2020, 2021, 2022, 2023, 2024]
MONTHS = range(3, 11)  # March through October
VALID_HOURS = [0, 6, 12, 18]
WINDOWS = [6, 12]

# Set True to only include Cloud/CG == G or CG.
CG_ONLY = False

# Optional quality-control filters.
# Set to None to disable.
MAX_CHI = None          # Example: 5.0
MAX_SMA = None          # Example: 10.0

OUT_COLS = ["valid_datetime", "lat", "lon", "polarity", "strength", "flashes"]


def load_gld_file() -> pd.DataFrame:
    """Load and normalize the giant GLD/Vaisala CSV file."""

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Could not find input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = ["Lat", "Lon", "Ip", "mult", "SMA", "SMI", "Chi", "Cloud/CG", "DateTime"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Parse timestamp. Assuming DateTime is UTC.
    df["dt"] = pd.to_datetime(df["DateTime"], errors="coerce")

    # Rename / normalize values.
    df["lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Lon"], errors="coerce")
    df["ip"] = pd.to_numeric(df["Ip"], errors="coerce")
    df["mult"] = pd.to_numeric(df["mult"], errors="coerce")
    df["SMA"] = pd.to_numeric(df["SMA"], errors="coerce")
    df["SMI"] = pd.to_numeric(df["SMI"], errors="coerce")
    df["Chi"] = pd.to_numeric(df["Chi"], errors="coerce")

    df = df.dropna(subset=["dt", "lat", "lon", "ip"])

    if CG_ONLY:
        df["cloud_cg"] = df["Cloud/CG"].astype(str).str.upper().str.strip()
        df = df[df["cloud_cg"].isin(["G", "CG"])]

    if MAX_CHI is not None:
        df = df[df["Chi"] <= MAX_CHI]

    if MAX_SMA is not None:
        df = df[df["SMA"] <= MAX_SMA]

    # Match reference output concept.
    # Reference NLDN polarity is often already + / -.
    df["polarity"] = np.where(df["ip"] < 0, "-", "+")

    # Peak current magnitude.
    df["strength"] = df["ip"].abs()

    # Reference code writes "flashes".
    #GLD file has "mult", but many rows appear to be 0.
    # To avoid zero-lightning detections, treat 0/missing as 1 event.
    df["flashes"] = df["mult"]
    df.loc[df["flashes"].fillna(0) <= 0, "flashes"] = 1
    df["flashes"] = df["flashes"].astype(int)

    # Keep only fields needed downstream.
    df = df[["dt", "lat", "lon", "polarity", "strength", "flashes"]].copy()

    # Sorting speeds up repeated time filtering and makes outputs chronological.
    df = df.sort_values("dt").reset_index(drop=True)

    return df


def get_month_end(year: int, month: int) -> datetime:
    """Return first datetime of the next month."""

    if month == 12:
        return datetime(year + 1, 1, 1)

    return datetime(year, month + 1, 1)


def get_output_folder(window: int) -> str:
    """Return output directory for a given window length."""

    if window == 6:
        return OUTPUT_DIR_06

    if window == 12:
        return OUTPUT_DIR_12

    raise ValueError(f"Unsupported window: {window}")


def write_subset(subset: pd.DataFrame, valid_dt: datetime, window: int) -> None:
    """Write one interval subset to disk using the reference output format."""

    folder = get_output_folder(window)
    filename = f"gld_{window:02d}h_{valid_dt.strftime('%Y%m%d_%H%M')}Z.csv"
    out_path = os.path.join(folder, filename)

    final_df = subset.copy()

    # This matches the reference script:
    final_df["valid_datetime"] = final_df["dt"].dt.strftime("%Y-%m-%d %H:%M")

    final_df = final_df[OUT_COLS]

    final_df.to_csv(out_path, index=False)


def process_aggregations() -> None:
    """Generate 6-hour and 12-hour trailing-window CSVs."""

    print(f"Loading {INPUT_FILE}")
    df = load_gld_file()

    if df.empty:
        print("No data found after loading/filtering.")
        return

    print(f"Loaded {len(df):,} events")
    print(f"Date range: {df['dt'].min()} to {df['dt'].max()}")

    for year in YEARS:
        for month in MONTHS:
            print(f"Processing {year}-{month:02d}...")

            current_day = datetime(year, month, 1)
            end_date = get_month_end(year, month)

            while current_day < end_date:
                for v_hr in VALID_HOURS:
                    valid_dt = current_day.replace(hour=v_hr)

                    for window in WINDOWS:
                        start_dt = valid_dt - timedelta(hours=window)

                        # Filter window [start_dt, valid_dt)
                        mask = (df["dt"] >= start_dt) & (df["dt"] < valid_dt)
                        subset = df.loc[mask].copy()

                        # This mirrors the reference behavior:
                        # only writes a file if there are events in the interval.
                        if not subset.empty:
                            write_subset(subset, valid_dt, window)

                current_day += timedelta(days=1)

    print("Processing Complete.")


if __name__ == "__main__":
    process_aggregations()