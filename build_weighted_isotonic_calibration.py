#!/usr/bin/env python

"""Build weighted isotonic calibration curves from binned reliability data.

This version is intended for monthly stats created by the updated verification
script that writes:

    sum_fcst
    mean_fcst_prob
    prob_bin_upper
    prob_bin_label

The important change from the older calibration script is that this script uses
actual mean forecast probability inside each reliability bin:

    raw_prob = sum_fcst / count

rather than assuming the raw probability is the bin midpoint. This is especially
important for the low-end bins such as 0-1%, 1-5%, and 5-10%.

Default grouping:
    interval_hour + period

Outputs:
    calibration_<dataset>/models/isotonic_<dataset>_<interval>h_<period>.joblib
    calibration_<dataset>/tables/calibration_curve_<dataset>_<interval>h_<period>.csv
    calibration_<dataset>/tables/calibration_points_<dataset>_<interval>h_<period>.csv
    calibration_<dataset>/plots/*.png
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

PLOT_DIR = OUT_BASE / "plots"
MODEL_DIR = OUT_BASE / "models"
TABLE_DIR = OUT_BASE / "tables"

for p in [OUT_BASE, PLOT_DIR, MODEL_DIR, TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

YEARS = [2023, 2024, 2025]
MONTHS = range(3, 11)

INTERVAL_HOURS = [6, 12]
PERIODS = ["day", "night"]

# Minimum total samples required for a bin to be included in the isotonic fit.
MIN_BIN_COUNT = 1

# Drop bins with very tiny counts. This is mainly useful for rare high-end bins.
DROP_TINY_BINS = True
MIN_TINY_BIN_COUNT = 100

# Apply the tiny-bin filter only above this raw-probability threshold.
# This keeps low-probability bins, even if one split bin is small, but removes
# very sparse high-end points that can make the curve noisy.
TINY_BIN_FILTER_MIN_RAW_PROB = 0.60

# Optional train/test split.
# If HOLDOUT_YEARS is empty, calibration trains and evaluates on all years.
HOLDOUT_YEARS = [2025]

# If True, save a second model/table where calibrated probability is not allowed
# to be lower than the raw probability. This is useful for a probability-stretch
# product, but the ordinary isotonic model remains the statistically calibrated one.
SAVE_UPWARD_ONLY_TABLE = True

# Optional lower and upper bounds on the isotonic output.
# Leaving these as 0 and 1 is the usual probabilistic calibration choice.
ISOTONIC_Y_MIN = 0.0
ISOTONIC_Y_MAX = 1.0

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

    required = [
        "prob_bin",
        "sum_obs",
        "count",
        "sum_fcst",
        "period",
        "forecast_hour",
        "year",
        "month",
        "interval_hour",
    ]
    missing_cols = [c for c in required if c not in df_all.columns]

    if missing_cols:
        raise ValueError(
            "Missing required columns from monthly stats: "
            f"{missing_cols}\n\n"
            "This calibration script expects monthly stats from the updated "
            "verification script that writes sum_fcst and mean_fcst_prob. "
            "Rerun the updated verification script first."
        )

    # Ensure numeric columns are numeric.
    for col in ["prob_bin", "sum_obs", "count", "sum_fcst"]:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    if "prob_bin_upper" in df_all.columns:
        df_all["prob_bin_upper"] = pd.to_numeric(df_all["prob_bin_upper"], errors="coerce")

    return df_all


# ---------------------------------------------------------------------
# CALIBRATION HELPERS
# ---------------------------------------------------------------------

def aggregate_reliability_bins(
    df,
    group_cols=("interval_hour", "period", "prob_bin"),
):
    """Aggregate reliability bins and compute actual mean raw forecast probability.

    The key correction is:

        raw_prob = sum_fcst / count

    This replaces the older midpoint approximation:

        raw_prob = prob_bin + bin_width / 2
    """

    agg_dict = {
        "sum_obs": ("sum_obs", "sum"),
        "count": ("count", "sum"),
        "sum_fcst": ("sum_fcst", "sum"),
    }

    if "prob_bin_upper" in df.columns:
        agg_dict["prob_bin_upper"] = ("prob_bin_upper", "first")

    grouped = (
        df
        .groupby(list(group_cols), as_index=False)
        .agg(**agg_dict)
    )

    grouped["obs_freq"] = grouped["sum_obs"] / grouped["count"]
    grouped["raw_prob"] = grouped["sum_fcst"] / grouped["count"]

    # Optional label for plots/tables.
    if "prob_bin_upper" in grouped.columns:
        grouped["prob_bin_label"] = grouped.apply(
            lambda r: f"{r['prob_bin']:.2f}-{r['prob_bin_upper']:.2f}",
            axis=1,
        )
    else:
        grouped["prob_bin_label"] = grouped["prob_bin"].map(lambda x: f"{x:.2f}")

    # Filter bad or empty bins.
    grouped = grouped[
        np.isfinite(grouped["raw_prob"]) &
        np.isfinite(grouped["obs_freq"]) &
        np.isfinite(grouped["count"]) &
        (grouped["count"] >= MIN_BIN_COUNT)
    ].copy()

    # Guard against tiny high-end bins dominating the tail.
    if DROP_TINY_BINS:
        keep = ~(
            (grouped["raw_prob"] >= TINY_BIN_FILTER_MIN_RAW_PROB) &
            (grouped["count"] < MIN_TINY_BIN_COUNT)
        )
        dropped = grouped[~keep].copy()
        if not dropped.empty:
            print("\nDropping tiny high-end bins from isotonic fit:")
            cols = ["prob_bin", "prob_bin_label", "raw_prob", "obs_freq", "count"]
            cols = [c for c in cols if c in dropped.columns]
            print(dropped[cols].to_string(index=False))
        grouped = grouped[keep].copy()

    grouped = grouped.sort_values("raw_prob").reset_index(drop=True)

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
        y_min=ISOTONIC_Y_MIN,
        y_max=ISOTONIC_Y_MAX,
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

    if SAVE_UPWARD_ONLY_TABLE:
        table["calibrated_probability_upward_only"] = np.maximum(
            table["calibrated_probability"],
            table["raw_probability"],
        )

    # Add original binned reliability points for reference separately.
    points = bin_df.copy().sort_values("raw_prob")
    points["calibrated_at_raw_prob"] = model.predict(
        points["raw_prob"].to_numpy(dtype=float)
    )

    if SAVE_UPWARD_ONLY_TABLE:
        points["calibrated_at_raw_prob_upward_only"] = np.maximum(
            points["calibrated_at_raw_prob"],
            points["raw_prob"],
        )

    return table, points


def apply_calibration_to_binned_data(model, df):
    """Apply fitted calibration to binned stats and compute approximate Brier components.

    This is still approximate because each bin is represented by its actual mean
    raw forecast probability, not by individual pixel probabilities. It is much
    better than the old midpoint approximation, especially in the low bins.
    """

    out = df.copy()

    # Aggregate first if needed, so raw_prob uses total sum_fcst / total count.
    if "raw_prob" not in out.columns:
        out = aggregate_reliability_bins(out)

    out["raw_prob_calibrated"] = model.predict(out["raw_prob"].to_numpy(dtype=float))

    if SAVE_UPWARD_ONLY_TABLE:
        out["raw_prob_calibrated_upward_only"] = np.maximum(
            out["raw_prob_calibrated"],
            out["raw_prob"],
        )

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

    if SAVE_UPWARD_ONLY_TABLE:
        p_cal_up = out["raw_prob_calibrated_upward_only"].to_numpy(dtype=float)
        out["sum_se_calibrated_upward_only_approx"] = (
            sum_obs * (p_cal_up - 1.0) ** 2 + sum_no * p_cal_up ** 2
        )

    return out


def summarize_approx_brier(df):
    """Summarize approximate Brier Score from binned data."""

    total_count = df["count"].sum()

    if total_count <= 0:
        out = {
            "count": 0,
            "brier_raw_approx": np.nan,
            "brier_calibrated_approx": np.nan,
            "brier_improvement": np.nan,
        }
        if SAVE_UPWARD_ONLY_TABLE:
            out.update({
                "brier_calibrated_upward_only_approx": np.nan,
                "brier_improvement_upward_only": np.nan,
            })
        return out

    bs_raw = df["sum_se_raw_approx"].sum() / total_count
    bs_cal = df["sum_se_calibrated_approx"].sum() / total_count

    out = {
        "count": int(total_count),
        "brier_raw_approx": float(bs_raw),
        "brier_calibrated_approx": float(bs_cal),
        "brier_improvement": float(bs_raw - bs_cal),
    }

    if SAVE_UPWARD_ONLY_TABLE and "sum_se_calibrated_upward_only_approx" in df.columns:
        bs_cal_up = df["sum_se_calibrated_upward_only_approx"].sum() / total_count
        out.update({
            "brier_calibrated_upward_only_approx": float(bs_cal_up),
            "brier_improvement_upward_only": float(bs_raw - bs_cal_up),
        })

    return out


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def plot_calibration_curve(interval, period, bin_df, curve_table, out_path):
    """Plot raw reliability points and isotonic calibration curve."""

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="Perfect reliability",
    )

    ax.plot(
        curve_table["raw_probability"],
        curve_table["calibrated_probability"],
        linewidth=2.5,
        label="Weighted isotonic calibration",
    )

    if SAVE_UPWARD_ONLY_TABLE:
        ax.plot(
            curve_table["raw_probability"],
            curve_table["calibrated_probability_upward_only"],
            linewidth=1.8,
            linestyle=":",
            label="Upward-only version",
        )

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
        label = row.get("prob_bin_label", f"{row['prob_bin']:.2f}")
        ax.annotate(
            label,
            xy=(row["raw_prob"], row["obs_freq"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Actual mean raw NBM thunder probability in bin")
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

    grouped = eval_df.copy()

    if grouped.empty:
        return

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

    if SAVE_UPWARD_ONLY_TABLE:
        ax.scatter(
            grouped["raw_prob_calibrated_upward_only"],
            grouped["obs_freq"],
            s=70,
            marker="^",
            edgecolor="black",
            linewidth=0.7,
            label="Upward-only calibrated",
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
                    ["prob_bin", "prob_bin_label", "raw_prob", "sum_obs", "count", "obs_freq"]
                ].to_string(index=False)
            )

            model = fit_isotonic_from_bins(bin_df)
            curve_table, points = calibration_curve_table(model, bin_df)

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

                eval_path = TABLE_DIR / f"evaluation_bins_{DATASET_NAME}_{interval:02d}h_{period}.csv"
                eval_df.to_csv(eval_path, index=False)
                print(f"Saved evaluation bins: {eval_path}")

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
