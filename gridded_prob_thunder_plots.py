#!/usr/bin/env python

"""Plot Alaska NBM/GLD thunder verification diagnostics from monthly gridded stats.

Purpose:
- Reads monthly CSV files created by the Alaska-masked gridded verification script.
- Computes contingency-table metrics from stored threshold counts.
- Computes reliability / observed-frequency metrics by probability bin.
- Creates CSI-vs-threshold and sharpness plots for Alaska.
- Optionally creates reliability plots by forecast hour.

Expected input monthly files:
    verif_06_YYYY_MM.csv
    verif_12_YYYY_MM.csv

Expected input columns include:
    prob_bin, sum_obs, count, sum_se,
    region, period, forecast_hour, year, month, interval_hour,
    t10_hits, t10_misses, t10_fa, t10_cn, ... t90_*

Outputs:
    PNG plots written to PLOT_DIR.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

MONTHLY_STATS_DIR = Path(
    r"C:\Users\David.Levin\NBMLightningVer\monthly_stats"
)

PLOT_DIR = Path(
    r"C:\Users\David.Levin\NBMLightningVer\plots"
)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2023, 2024, 2025]
MONTHS = range(3, 11)  # March through October

INTERVAL_HOURS = [6, 12]
PERIODS = ["day", "night"]

# Minimum number of grid-cell samples required to plot a reliability point.
MIN_BIN_COUNT = 1000

# If True, save plots to disk.
SAVE_FIGS = True

# If True, show plots interactively.
SHOW_FIGS = True


# ---------------------------------------------------------------------
# METRIC HELPERS
# ---------------------------------------------------------------------

def safe_divide(num, den):
    """Return num / den, using NaN where denominator is zero."""

    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)

    out = np.full_like(num, np.nan, dtype=float)
    valid = den != 0
    out[valid] = num[valid] / den[valid]

    return out


def get_thresholds_from_columns(df):
    """Find threshold values from columns like t10_hits, t50_hits, etc."""

    thresholds = sorted(
        {
            int(c.split("_")[0][1:])
            for c in df.columns
            if c.startswith("t") and c.endswith("_hits")
        }
    )

    return thresholds


def add_probability_bin_metrics(df):
    """Add reliability-style metrics from sum_obs/count/sum_se."""

    df = df.copy()

    df["obs_no"] = df["count"] - df["sum_obs"]
    df["observed_frequency"] = safe_divide(df["sum_obs"], df["count"])
    df["brier_score_bin"] = safe_divide(df["sum_se"], df["count"])

    # Your prob_bin is the lower edge: 0.3 means 30-40%.
    # Midpoint is better for plotting.
    df["prob_bin_midpoint"] = df["prob_bin"] + 0.05
    df.loc[df["prob_bin"] >= 1.0, "prob_bin_midpoint"] = 1.0

    return df


def threshold_metric_long_table(
    df,
    group_cols=("interval_hour", "period", "forecast_hour", "year", "month"),
):
    """
    Convert wide threshold contingency columns into a long table.

    Important:
    In your monthly output, threshold counts are repeated on every probability-bin
    row within the same interval/period/forecast_hour/year/month group. Therefore
    this function takes the first row per group instead of summing across prob_bin.
    """

    thresholds = get_thresholds_from_columns(df)

    rows = []

    for keys, group in df.groupby(list(group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_dict = dict(zip(group_cols, keys))
        row0 = group.iloc[0]

        for t in thresholds:
            h = row0[f"t{t}_hits"]
            m = row0[f"t{t}_misses"]
            fa = row0[f"t{t}_fa"]
            cn = row0[f"t{t}_cn"]

            rows.append({
                **key_dict,
                "threshold": t,
                "hits": h,
                "misses": m,
                "fa": fa,
                "cn": cn,
                "CSI": h / (h + m + fa) if (h + m + fa) > 0 else np.nan,
                "POD": h / (h + m) if (h + m) > 0 else np.nan,
                "FAR": fa / (h + fa) if (h + fa) > 0 else np.nan,
                "SR": h / (h + fa) if (h + fa) > 0 else np.nan,
                "BIAS": (h + fa) / (h + m) if (h + m) > 0 else np.nan,
                "ACC": (h + cn) / (h + m + fa + cn)
                    if (h + m + fa + cn) > 0 else np.nan,
                "BASE_RATE": (h + m) / (h + m + fa + cn)
                    if (h + m + fa + cn) > 0 else np.nan,
            })

    return pd.DataFrame(rows)


def aggregate_threshold_metrics(
    threshold_long,
    group_cols=("interval_hour", "period", "forecast_hour", "threshold"),
):
    """
    Aggregate threshold contingency counts across years/months, then recompute metrics.
    """

    grouped = (
        threshold_long
        .groupby(list(group_cols), as_index=False)
        .agg(
            hits=("hits", "sum"),
            misses=("misses", "sum"),
            fa=("fa", "sum"),
            cn=("cn", "sum"),
        )
    )

    h = grouped["hits"]
    m = grouped["misses"]
    fa = grouped["fa"]
    cn = grouped["cn"]

    grouped["CSI"] = safe_divide(h, h + m + fa)
    grouped["POD"] = safe_divide(h, h + m)
    grouped["FAR"] = safe_divide(fa, h + fa)
    grouped["SR"] = safe_divide(h, h + fa)
    grouped["BIAS"] = safe_divide(h + fa, h + m)
    grouped["ACC"] = safe_divide(h + cn, h + m + fa + cn)
    grouped["BASE_RATE"] = safe_divide(h + m, h + m + fa + cn)

    return grouped


def probability_bin_reliability_table(
    df,
    group_cols=("interval_hour", "period", "forecast_hour", "prob_bin"),
):
    """Aggregate probability-bin stats across selected months/years."""

    grouped = (
        df.groupby(list(group_cols), as_index=False)
          .agg(
              sum_obs=("sum_obs", "sum"),
              count=("count", "sum"),
              sum_se=("sum_se", "sum"),
          )
    )

    grouped = add_probability_bin_metrics(grouped)

    return grouped


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def load_monthly_stats(
    monthly_stats_dir=MONTHLY_STATS_DIR,
    years=YEARS,
    months=MONTHS,
    intervals=INTERVAL_HOURS,
):
    """Load selected monthly verification CSV files."""

    frames = []
    missing = []

    for interval in intervals:
        interval_str = f"{interval:02d}"

        for year in years:
            for month in months:
                month_str = f"{month:02d}"
                path = monthly_stats_dir / f"verif_{interval_str}_{year}_{month_str}.csv"

                if not path.exists():
                    missing.append(path)
                    continue

                df = pd.read_csv(path)

                # Defensively set these if missing.
                if "interval_hour" not in df.columns:
                    df["interval_hour"] = interval
                if "year" not in df.columns:
                    df["year"] = year
                if "month" not in df.columns:
                    df["month"] = month

                frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No monthly verification CSVs found in {monthly_stats_dir}"
        )

    df_all = pd.concat(frames, ignore_index=True)

    print(f"Loaded {len(frames)} monthly files")
    print(f"Rows: {len(df_all):,}")

    if missing:
        print(f"Missing files: {len(missing)}")
        print("First few missing:")
        for p in missing[:10]:
            print(f"  {p}")

    return df_all


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def plot_csi_by_forecast_hour(
    threshold_stats,
    interval_hour,
    period,
    title_suffix="Alaska GLD/NBM",
    save_fig=SAVE_FIGS,
    show_fig=SHOW_FIGS,
):
    """Plot CSI vs probability threshold for each forecast hour."""

    sub = threshold_stats[
        (threshold_stats["interval_hour"] == interval_hour) &
        (threshold_stats["period"] == period)
    ].copy()

    if sub.empty:
        print(f"No threshold data for interval={interval_hour}, period={period}")
        return

    min_fh = sub["forecast_hour"].min()
    max_fh = sub["forecast_hour"].max()

    cmap = cm.get_cmap("coolwarm")

    fig, ax = plt.subplots(figsize=(11, 7))

    for fh, group in sub.groupby("forecast_hour"):
        group = group.sort_values("threshold")

        color_val = (fh - min_fh) / (max_fh - min_fh) if max_fh > min_fh else 0.5
        color = cmap(color_val)

        ax.plot(
            group["threshold"],
            group["CSI"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"F{int(fh)}",
        )

        if not group["CSI"].isna().all():
            max_idx = group["CSI"].idxmax()
            max_row = group.loc[max_idx]

            ax.plot(
                max_row["threshold"],
                max_row["CSI"],
                marker="o",
                markersize=7,
                color=color,
            )

            ax.text(
                max_row["threshold"] + 1,
                max_row["CSI"],
                f"F{int(fh)}",
                color=color,
                fontsize=8,
                fontweight="bold",
                va="bottom",
            )

    ax.set_title(
        f"CSI vs Forecast Probability Threshold\n"
        f"{title_suffix}, {interval_hour}-hr, {period.title()}"
    )
    ax.set_xlabel("Forecast probability threshold (%)")
    ax.set_ylabel("CSI")
    ax.set_xticks(sorted(sub["threshold"].unique()))
    ax.grid(alpha=0.25)

    plt.tight_layout()

    if save_fig:
        out_path = PLOT_DIR / f"csi_threshold_{interval_hour:02d}h_{period}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def plot_sharpness(
    bin_stats,
    interval_hour,
    period,
    title_suffix="Alaska GLD/NBM",
    save_fig=SAVE_FIGS,
    show_fig=SHOW_FIGS,
):
    """
    Plot sharpness from aggregated grid-cell counts by probability bin.

    Sharpness here means the relative frequency of forecast probabilities,
    independent of whether lightning occurred.
    """

    sub = bin_stats[
        (bin_stats["interval_hour"] == interval_hour) &
        (bin_stats["period"] == period)
    ].copy()

    if sub.empty:
        print(f"No bin data for interval={interval_hour}, period={period}")
        return

    sharp = (
        sub.groupby("prob_bin", as_index=False)
           .agg(count=("count", "sum"))
           .sort_values("prob_bin")
    )

    sharp["relative_frequency"] = sharp["count"] / sharp["count"].sum()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        sharp["prob_bin"] * 100,
        sharp["relative_frequency"],
        width=8,
        align="edge",
        edgecolor="black",
        alpha=0.75,
    )

    ax.set_title(
        f"Sharpness Diagram\n"
        f"{title_suffix}, {interval_hour}-hr, {period.title()}"
    )
    ax.set_xlabel("Forecast probability bin lower edge (%)")
    ax.set_ylabel("Relative frequency")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-7)
    ax.set_xticks(np.arange(0, 110, 10))
    ax.grid(alpha=0.25, axis="y")

    plt.tight_layout()

    if save_fig:
        out_path = PLOT_DIR / f"sharpness_{interval_hour:02d}h_{period}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def plot_reliability_by_forecast_hour(
    bin_stats,
    interval_hour,
    period,
    min_count=MIN_BIN_COUNT,
    title_suffix="Alaska GLD/NBM",
    save_fig=SAVE_FIGS,
    show_fig=SHOW_FIGS,
):
    """Plot observed frequency vs forecast probability by forecast hour."""

    sub = bin_stats[
        (bin_stats["interval_hour"] == interval_hour) &
        (bin_stats["period"] == period) &
        (bin_stats["count"] >= min_count)
    ].copy()

    if sub.empty:
        print(
            f"No reliability data for interval={interval_hour}, period={period} "
            f"with count >= {min_count}"
        )
        return

    min_fh = sub["forecast_hour"].min()
    max_fh = sub["forecast_hour"].max()

    cmap = cm.get_cmap("coolwarm")

    fig, ax = plt.subplots(figsize=(10, 7))

    for fh, group in sub.groupby("forecast_hour"):
        group = group.sort_values("prob_bin_midpoint")

        color_val = (fh - min_fh) / (max_fh - min_fh) if max_fh > min_fh else 0.5
        color = cmap(color_val)

        ax.plot(
            group["prob_bin_midpoint"] * 100,
            group["observed_frequency"] * 100,
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"F{int(fh)}",
        )

    ax.plot([0, 100], [0, 100], color="black", linestyle="--", linewidth=1)

    ax.set_title(
        f"Reliability by Forecast Hour\n"
        f"{title_suffix}, {interval_hour}-hr, {period.title()}"
    )
    ax.set_xlabel("Forecast probability bin midpoint (%)")
    ax.set_ylabel("Observed lightning frequency (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)

    plt.tight_layout()

    if save_fig:
        out_path = PLOT_DIR / f"reliability_{interval_hour:02d}h_{period}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def plot_csi_and_sharpness_panel(
    threshold_stats,
    bin_stats,
    interval_hour,
    title_suffix="Alaska GLD/NBM",
    save_fig=SAVE_FIGS,
    show_fig=SHOW_FIGS,
):
    """
    Create a 2x2 panel similar to the old CONUS notebook:
    - CSI vs threshold for day
    - CSI vs threshold for night
    - Sharpness for day
    - Sharpness for night
    """

    stats_day = threshold_stats[
        (threshold_stats["interval_hour"] == interval_hour) &
        (threshold_stats["period"] == "day")
    ].copy()

    stats_night = threshold_stats[
        (threshold_stats["interval_hour"] == interval_hour) &
        (threshold_stats["period"] == "night")
    ].copy()

    bin_day = bin_stats[
        (bin_stats["interval_hour"] == interval_hour) &
        (bin_stats["period"] == "day")
    ].copy()

    bin_night = bin_stats[
        (bin_stats["interval_hour"] == interval_hour) &
        (bin_stats["period"] == "night")
    ].copy()

    all_fhs = set()
    if not stats_day.empty:
        all_fhs.update(stats_day["forecast_hour"].unique())
    if not stats_night.empty:
        all_fhs.update(stats_night["forecast_hour"].unique())

    if not all_fhs:
        print(f"No data for interval={interval_hour}")
        return

    min_fh = min(all_fhs)
    max_fh = max(all_fhs)

    cmap = cm.get_cmap("coolwarm")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1])

    ax_top_d = fig.add_subplot(gs[0, 0])
    ax_top_n = fig.add_subplot(gs[0, 1], sharey=ax_top_d, sharex=ax_top_d)
    ax_bot_d = fig.add_subplot(gs[1, 0])
    ax_bot_n = fig.add_subplot(gs[1, 1], sharey=ax_bot_d, sharex=ax_bot_d)

    fig.suptitle(
        f"CSI vs Threshold & Sharpness Diagram\n"
        f"{title_suffix}, {interval_hour}-hr, "
        f"{min(YEARS)}-{max(YEARS)}, Mar-Oct",
        fontsize=16,
        y=0.96,
    )

    def plot_csi_curves(ax, stats, title):
        if stats.empty:
            ax.set_title(f"{title}\nNo data")
            return

        for fh, group in stats.groupby("forecast_hour"):
            group = group.sort_values("threshold")

            color_val = (fh - min_fh) / (max_fh - min_fh) if max_fh > min_fh else 0.5
            color = cmap(color_val)

            ax.plot(
                group["threshold"],
                group["CSI"],
                color=color,
                linewidth=1.7,
                label=f"F{int(fh)}",
            )

            if not group["CSI"].isna().all():
                max_idx = group["CSI"].idxmax()
                max_row = group.loc[max_idx]

                ax.plot(
                    max_row["threshold"],
                    max_row["CSI"],
                    marker="o",
                    markersize=5,
                    color=color,
                )

                ax.text(
                    max_row["threshold"] + 1,
                    max_row["CSI"],
                    f"F{int(fh)}",
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                    va="bottom",
                )

        ax.set_title(title)
        ax.set_xlabel("Threshold (%)")
        ax.set_ylabel("CSI")
        ax.set_xticks(sorted(stats["threshold"].unique()))
        ax.grid(alpha=0.2)

    def plot_sharpness_from_bins(ax, bins, title):
        if bins.empty:
            ax.set_title(f"{title}\nNo data")
            return

        sharp = (
            bins.groupby("prob_bin", as_index=False)
                .agg(count=("count", "sum"))
                .sort_values("prob_bin")
        )

        sharp["relative_frequency"] = sharp["count"] / sharp["count"].sum()

        ax.bar(
            sharp["prob_bin"] * 100,
            sharp["relative_frequency"],
            width=8,
            align="edge",
            edgecolor="black",
            alpha=0.75,
        )

        ax.set_title(title)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-7)
        ax.set_xlabel("Forecast Probability Bin Lower Edge (%)")
        ax.set_ylabel("Relative Frequency")
        ax.set_xticks(np.arange(0, 110, 10))
        ax.grid(alpha=0.2, axis="y")

    plot_csi_curves(
        ax_top_d,
        stats_day,
        "CSI vs Threshold for Each Forecast Hour (Day)",
    )

    plot_csi_curves(
        ax_top_n,
        stats_night,
        "CSI vs Threshold for Each Forecast Hour (Night)",
    )

    plot_sharpness_from_bins(
        ax_bot_d,
        bin_day,
        "Sharpness Diagram (Day)",
    )

    plot_sharpness_from_bins(
        ax_bot_n,
        bin_night,
        "Sharpness Diagram (Night)",
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    if save_fig:
        out_path = PLOT_DIR / f"csi_sharpness_{interval_hour:02d}h_alaska.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    """Run the full plotting workflow."""

    df = load_monthly_stats()

    # Add bin-level reliability metrics to raw monthly rows.
    df = add_probability_bin_metrics(df)

    # Build threshold metrics from existing tXX contingency columns.
    threshold_long = threshold_metric_long_table(df)

    # Aggregate across years/months.
    threshold_stats = aggregate_threshold_metrics(threshold_long)

    # Aggregate probability-bin stats across years/months.
    bin_stats = probability_bin_reliability_table(df)

    # Save intermediate tables for inspection.
    threshold_csv = PLOT_DIR / "threshold_metrics_aggregated.csv"
    bin_csv = PLOT_DIR / "probability_bin_metrics_aggregated.csv"

    threshold_stats.to_csv(threshold_csv, index=False)
    bin_stats.to_csv(bin_csv, index=False)

    print(f"Saved: {threshold_csv}")
    print(f"Saved: {bin_csv}")

    # Create plots.
    for interval_hour in INTERVAL_HOURS:
        plot_csi_and_sharpness_panel(
            threshold_stats,
            bin_stats,
            interval_hour=interval_hour,
            title_suffix="Alaska GLD/NBM",
        )

        for period in PERIODS:
            plot_csi_by_forecast_hour(
                threshold_stats,
                interval_hour=interval_hour,
                period=period,
                title_suffix="Alaska GLD/NBM",
            )

            plot_sharpness(
                bin_stats,
                interval_hour=interval_hour,
                period=period,
                title_suffix="Alaska GLD/NBM",
            )

            plot_reliability_by_forecast_hour(
                bin_stats,
                interval_hour=interval_hour,
                period=period,
                min_count=MIN_BIN_COUNT,
                title_suffix="Alaska GLD/NBM",
            )


if __name__ == "__main__":
    main()