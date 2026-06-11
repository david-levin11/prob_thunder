#!/usr/bin/env python

"""Build weighted isotonic calibration curves from binned reliability data.

This script reads monthly verification CSVs produced by the Alaska gridded
verification workflow and builds monotonic probability calibration curves.

Input reliability data columns expected:
    prob_bin
    sum_obs
    count
    period
    forecast_hour
    year
    month
    interval_hour

Calibration target:
    observed_frequency = sum_obs / count

Calibration model:
    weighted isotonic regression

Default grouping:
    interval_hour + period

Example output:
    calibration_union/isotonic_calibration_06h_day.csv
    calibration_union/isotonic_calibration_12h_night.csv
    calibration_union/reliability_raw_vs_calibrated_06h_day.png

Notes:
    - This uses binned reliability data, not individual pixels.
    - The isotonic fit is weighted by the number of samples in each bin.
    - By default, it treats prob_bin as the lower edge of each reliability bin.
      For 10% bins, the representative raw probability is prob_bin + 0.05.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
import joblib


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATASET_NAME = "aicc_total"
# Examples:
#   "gld"
#   "aicc_total"
#   "union"

MONTHLY_STATS_DIRS = {
    "gld": Path(r"C:\Users\David.Levin\NBMLightningVer\gld_monthly_stats"),
    "aicc_total": Path(r"C:\Users\David.Levin\NBMLightningVer\aicc_monthly_stats"),
    "union": Path(r"C:\Users\David.Levin\NBMLightningVer\union_monthly_stats"),
}

OUT_BASE = Path(
    rf"C:\Users\David.Levin\NBMLightningVer\calibration_{DATASET_NAME}"
)

OUT_BASE.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUT_BASE / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = OUT_BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = OUT_BASE / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2023, 2024, 2025]
MONTHS = range(3, 11)

INTERVAL_HOURS = [6, 12]
PERIODS = ["day", "night"]

# Bins in your verification are 0.0, 0.1, ..., 1.0.
# For reliability fitting, use the midpoint of each bin as the representative
# raw probability, e.g. 0.0 bin -> 0.05, 0.1 bin -> 0.15.
USE_BIN_MIDPOINTS = True
BIN_WIDTH = 0.10

# Minimum total samples required for a bin to be included in the fit.
MIN_BIN_COUNT = 1

# Optional: drop the final 1.0 bin if it has tiny or zero counts.
DROP_EMPTY_OR_TINY_ONE_BIN = True
MIN_ONE_BIN_COUNT = 100

# Optional train/test split for first pass.
# If HOLDOUT_YEARS is empty, calibration uses all years.
HOLDOUT_YEARS = [2025]
# Example:
# HOLDOUT_YEARS = [2025]

SAVE_FIGS = True
SHOW_FIGS = True


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def get_monthly_stats_dir():
    """Return monthly stats directory for selected dataset."""

    if DATASET_NAME not in MONTHLY_STATS_DIRS:
        raise ValueError(
            f"Unknown DATASET_NAME={DATASET_NAME!r}. "
            f"Known options: {list(MONTHLY_STATS_DIRS)}"
        )

    return MONTHLY_STATS_DIRS[DATASET_NAME]


def load_monthly_stats():
    """Load selected monthly verification CSVs."""

    stats_dir = get_monthly_stats_dir()

    frames = []
    missing = []

    for interval in INTERVAL_HOURS:
        interval_str = f"{interval:02d}"

        for year in YEARS:
            for month in MONTHS:
                month_str = f"{month:02d}"
                path = stats_dir / f"verif_{interval_str}_{year}_{month_str}.csv"

                if not path.exists():
                    missing.append(path)
                    continue

                df = pd.read_csv(path)

                if "interval_hour" not in df.columns:
                    df["interval_hour"] = interval
                if "year" not in df.columns:
                    df["year"] = year
                if "month" not in df.columns:
                    df["month"] = month

                frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No monthly verification CSVs found in {stats_dir}")

    df_all = pd.concat(frames, ignore_index=True)

    print(f"Loaded {len(frames):,} monthly files from {stats_dir}")
    print(f"Rows: {len(df_all):,}")

    if missing:
        print(f"Missing monthly files: {len(missing):,}")
        for path in missing[:10]:
            print(f"  {path}")

    required = ["prob_bin", "sum_obs", "count", "period", "forecast_hour", "year", "month", "interval_hour"]
    missing_cols = [c for c in required if c not in df_all.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df_all


# ---------------------------------------------------------------------
# CALIBRATION HELPERS
# ---------------------------------------------------------------------

def representative_raw_probability(prob_bin):
    """Convert reliability bin lower edge to representative raw probability."""

    p = np.asarray(prob_bin, dtype=float)

    if USE_BIN_MIDPOINTS:
        # 0.0 -> 0.05, 0.1 -> 0.15, ..., 0.9 -> 0.95
        # Keep 1.0 at 1.0 if it exists.
        out = np.where(p >= 1.0, 1.0, p + BIN_WIDTH / 2.0)
    else:
        out = p

    return np.clip(out, 0.0, 1.0)


def aggregate_reliability_bins(
    df,
    group_cols=("interval_hour", "period", "prob_bin"),
):
    """Aggregate sum_obs/count by reliability bin and compute observed frequency."""

    grouped = (
        df
        .groupby(list(group_cols), as_index=False)
        .agg(
            sum_obs=("sum_obs", "sum"),
            count=("count", "sum"),
        )
    )

    grouped["obs_freq"] = grouped["sum_obs"] / grouped["count"]
    grouped["raw_prob"] = representative_raw_probability(grouped["prob_bin"])

    # Filter bad or empty bins.
    grouped = grouped[
        np.isfinite(grouped["raw_prob"]) &
        np.isfinite(grouped["obs_freq"]) &
        np.isfinite(grouped["count"]) &
        (grouped["count"] >= MIN_BIN_COUNT)
    ].copy()

    if DROP_EMPTY_OR_TINY_ONE_BIN:
        keep = ~(
            np.isclose(grouped["prob_bin"], 1.0) &
            (grouped["count"] < MIN_ONE_BIN_COUNT)
        )
        grouped = grouped[keep].copy()

    return grouped


def fit_isotonic_from_bins(bin_df):
    """Fit weighted isotonic regression from binned reliability points."""

    x = bin_df["raw_prob"].to_numpy(dtype=float)
    y = bin_df["obs_freq"].to_numpy(dtype=float)
    w = bin_df["count"].to_numpy(dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    w = w[order]

    model = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )

    model.fit(x, y, sample_weight=w)

    return model


def calibration_curve_table(model, bin_df, n_grid=101):
    """Create a calibration lookup table for plotting and later use."""

    grid = np.linspace(0.0, 1.0, n_grid)
    calibrated = model.predict(grid)

    table = pd.DataFrame({
        "raw_probability": grid,
        "calibrated_probability": calibrated,
    })

    # Add original binned reliability points for reference separately.
    points = bin_df.copy()
    points = points.sort_values("raw_prob")

    return table, points


def apply_calibration_to_binned_data(model, df):
    """Apply fitted calibration to binned stats and compute calibrated Brier components.

    This is approximate because it uses bin-representative raw probabilities,
    not individual pixel values.
    """

    out = df.copy()

    out["raw_prob"] = representative_raw_probability(out["prob_bin"])
    out["raw_prob_calibrated"] = model.predict(out["raw_prob"].to_numpy(dtype=float))

    # For binned binary observations:
    # Sum squared error for constant forecast p in a bin:
    #   sum[(p - y)^2] = sum_obs*(p-1)^2 + (count-sum_obs)*p^2
    p_raw = out["raw_prob"].to_numpy(dtype=float)
    p_cal = out["raw_prob_calibrated"].to_numpy(dtype=float)
    sum_obs = out["sum_obs"].to_numpy(dtype=float)
    count = out["count"].to_numpy(dtype=float)
    sum_no = count - sum_obs

    out["sum_se_raw_approx"] = sum_obs * (p_raw - 1.0) ** 2 + sum_no * p_raw ** 2
    out["sum_se_calibrated_approx"] = sum_obs * (p_cal - 1.0) ** 2 + sum_no * p_cal ** 2

    return out


def summarize_approx_brier(df):
    """Summarize approximate Brier Score from binned data."""

    total_count = df["count"].sum()

    if total_count <= 0:
        return {
            "count": 0,
            "brier_raw_approx": np.nan,
            "brier_calibrated_approx": np.nan,
            "brier_improvement": np.nan,
        }

    bs_raw = df["sum_se_raw_approx"].sum() / total_count
    bs_cal = df["sum_se_calibrated_approx"].sum() / total_count

    return {
        "count": int(total_count),
        "brier_raw_approx": float(bs_raw),
        "brier_calibrated_approx": float(bs_cal),
        "brier_improvement": float(bs_raw - bs_cal),
    }


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def plot_calibration_curve(interval, period, bin_df, curve_table, out_path):
    """Plot raw reliability points and isotonic calibration curve."""

    fig, ax = plt.subplots(figsize=(8, 7))

    # Perfect reliability line
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="Perfect reliability",
    )

    # Isotonic calibration curve
    ax.plot(
        curve_table["raw_probability"],
        curve_table["calibrated_probability"],
        linewidth=2.5,
        label="Weighted isotonic calibration",
    )

    # Binned observed frequencies
    sizes = np.sqrt(bin_df["count"].to_numpy(dtype=float))
    sizes = 30 + 220 * sizes / np.nanmax(sizes)

    ax.scatter(
        bin_df["raw_prob"],
        bin_df["obs_freq"],
        s=sizes,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.7,
        label="Binned reliability points",
    )

    for _, row in bin_df.iterrows():
        ax.annotate(
            f"{row['prob_bin']:.1f}",
            xy=(row["raw_prob"], row["obs_freq"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Raw NBM thunder probability")
    ax.set_ylabel("Observed lightning frequency / calibrated probability")
    ax.set_title(
        f"{DATASET_NAME} weighted isotonic calibration\n"
        f"{interval:02d}h, {period.title()}"
    )

    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")

    if SAVE_FIGS:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)


def plot_reliability_before_after(interval, period, eval_df, out_path):
    """Plot approximate binned reliability before and after calibration."""

    grouped = (
        eval_df
        .groupby("prob_bin", as_index=False)
        .agg(
            sum_obs=("sum_obs", "sum"),
            count=("count", "sum"),
            raw_prob=("raw_prob", "mean"),
            raw_prob_calibrated=("raw_prob_calibrated", "mean"),
        )
    )

    grouped = grouped[grouped["count"] > 0].copy()
    grouped["obs_freq"] = grouped["sum_obs"] / grouped["count"]

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="Perfect reliability",
    )

    ax.scatter(
        grouped["raw_prob"],
        grouped["obs_freq"],
        s=80,
        edgecolor="black",
        linewidth=0.7,
        label="Raw NBM",
    )

    ax.scatter(
        grouped["raw_prob_calibrated"],
        grouped["obs_freq"],
        s=80,
        marker="s",
        edgecolor="black",
        linewidth=0.7,
        label="Calibrated NBM",
    )

    for _, row in grouped.iterrows():
        ax.plot(
            [row["raw_prob"], row["raw_prob_calibrated"]],
            [row["obs_freq"], row["obs_freq"]],
            linewidth=0.8,
            alpha=0.5,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(
        f"Reliability before/after calibration\n"
        f"{DATASET_NAME}, {interval:02d}h, {period.title()}"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")

    if SAVE_FIGS:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    df_all = load_monthly_stats()

    if HOLDOUT_YEARS:
        train_df = df_all[~df_all["year"].isin(HOLDOUT_YEARS)].copy()
        test_df = df_all[df_all["year"].isin(HOLDOUT_YEARS)].copy()

        print(f"\nUsing holdout years: {HOLDOUT_YEARS}")
        print(f"Training rows: {len(train_df):,}")
        print(f"Testing rows:  {len(test_df):,}")
    else:
        train_df = df_all.copy()
        test_df = df_all.copy()

        print("\nNo holdout years set. Fitting and evaluating on all years.")

    calibration_summaries = []

    for interval in INTERVAL_HOURS:
        for period in PERIODS:
            train_sub = train_df[
                (train_df["interval_hour"] == interval) &
                (train_df["period"] == period)
            ].copy()

            test_sub = test_df[
                (test_df["interval_hour"] == interval) &
                (test_df["period"] == period)
            ].copy()

            if train_sub.empty:
                print(f"\nNo training data for interval={interval}, period={period}")
                continue

            bin_df = aggregate_reliability_bins(train_sub)

            if bin_df.empty:
                print(f"\nNo usable reliability bins for interval={interval}, period={period}")
                continue

            print(f"\nFitting interval={interval:02d}h, period={period}")
            print(
                bin_df[
                    ["prob_bin", "raw_prob", "sum_obs", "count", "obs_freq"]
                ].to_string(index=False)
            )

            model = fit_isotonic_from_bins(bin_df)

            curve_table, points = calibration_curve_table(model, bin_df)

            # Save model and tables.
            model_path = MODEL_DIR / f"isotonic_{DATASET_NAME}_{interval:02d}h_{period}.joblib"
            table_path = TABLE_DIR / f"calibration_curve_{DATASET_NAME}_{interval:02d}h_{period}.csv"
            points_path = TABLE_DIR / f"calibration_points_{DATASET_NAME}_{interval:02d}h_{period}.csv"

            joblib.dump(model, model_path)
            curve_table.to_csv(table_path, index=False)
            points.to_csv(points_path, index=False)

            print(f"Saved model: {model_path}")
            print(f"Saved curve table: {table_path}")
            print(f"Saved binned points: {points_path}")

            plot_calibration_curve(
                interval=interval,
                period=period,
                bin_df=bin_df,
                curve_table=curve_table,
                out_path=PLOT_DIR / f"calibration_curve_{DATASET_NAME}_{interval:02d}h_{period}.png",
            )

            if not test_sub.empty:
                test_bins = aggregate_reliability_bins(test_sub)

                eval_df = apply_calibration_to_binned_data(model, test_bins)
                brier_summary = summarize_approx_brier(eval_df)

                plot_reliability_before_after(
                    interval=interval,
                    period=period,
                    eval_df=eval_df,
                    out_path=PLOT_DIR / f"reliability_before_after_{DATASET_NAME}_{interval:02d}h_{period}.png",
                )

                calibration_summaries.append({
                    "dataset": DATASET_NAME,
                    "interval_hour": interval,
                    "period": period,
                    "train_years": ",".join(str(y) for y in sorted(train_df["year"].unique())),
                    "test_years": ",".join(str(y) for y in sorted(test_df["year"].unique())),
                    **brier_summary,
                })

    summary_df = pd.DataFrame(calibration_summaries)

    summary_path = OUT_BASE / f"calibration_summary_{DATASET_NAME}.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved calibration summary: {summary_path}")

    if not summary_df.empty:
        print("\nCalibration summary:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()