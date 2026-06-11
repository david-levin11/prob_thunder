#!/usr/bin/env python

"""Plot CSI, POD, FAR, and Bias from Alaska gridded monthly verification stats.

This script expects monthly CSVs from the Alaska-masked verification workflow.

Required threshold columns:
    t1_hits, t1_misses, t1_fa, t1_cn
    t5_hits, t5_misses, t5_fa, t5_cn
    t10_hits, ...

If t1/t5 are missing, rerun the raster verification script after setting:
    THRESHOLDS = [0.01, 0.05] + np.round(np.arange(0.10, 1.0, 0.10), 2).tolist()
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

MONTHLY_STATS_DIR = Path(
    r"C:\Users\David.Levin\NBMLightningVer\union_monthly_stats"
)

PLOT_DIR = Path(
    r"C:\Users\David.Levin\NBMLightningVer\plots\contingency_low_thresholds_union"
)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2023, 2024, 2025]
MONTHS = range(3, 11)

INTERVAL_HOURS = [6, 12]
PERIODS = ["day", "night"]

SAVE_FIGS = True
SHOW_FIGS = True


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def get_metric_ylim(stats, metric, pad_fraction=0.08, min_pad=0.02):
    """
    Compute a common y-axis range for a metric across the full stats dataframe.

    This ensures all plots for that metric use the same scale and do not clip data.
    """
    metric = metric.upper()

    vals = pd.to_numeric(stats[metric], errors="coerce")
    vals = vals[np.isfinite(vals)]

    if vals.empty:
        return None

    ymin = vals.min()
    ymax = vals.max()

    if ymin == ymax:
        pad = max(abs(ymax) * pad_fraction, min_pad)
    else:
        pad = max((ymax - ymin) * pad_fraction, min_pad)

    ymin = ymin - pad
    ymax = ymax + pad

    # Metrics bounded between 0 and 1
    if metric in ["CSI", "POD", "FAR", "SR", "ACC"]:
        ymin = max(0.0, ymin)
        ymax = min(1.0, ymax)

    return ymin, ymax


def get_metric_label(metric):
    """Pretty label for plot axes/titles."""
    metric = metric.upper()

    labels = {
        "CSI": "Critical Success Index",
        "POD": "Probability of Detection",
        "FAR": "False Alarm Ratio",
        "SR": "Success Ratio",
        "BIAS": "Frequency Bias",
        "ACC": "Accuracy",
    }

    return labels.get(metric, metric)


def get_valid_hours_for_period(period):
    """Return expected valid UTC hours for each period."""
    if period == "day":
        return [0, 6]
    if period == "night":
        return [12, 18]
    raise ValueError(f"Unknown period: {period}")


def safe_divide(num, den):
    """Return num / den, using NaN where denominator is zero."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)

    out = np.full_like(num, np.nan, dtype=float)
    valid = den != 0
    out[valid] = num[valid] / den[valid]

    return out


def get_valid_hours_for_period(period):
    """Return valid UTC hours associated with each period label."""

    if period == "day":
        return [0, 6]

    if period == "night":
        return [12, 18]

    raise ValueError(f"Unknown period: {period}")


def get_thresholds_from_columns(df):
    """Detect thresholds from columns like t1_hits, t5_hits, t10_hits."""

    thresholds = sorted(
        {
            int(c.split("_")[0][1:])
            for c in df.columns
            if c.startswith("t") and c.endswith("_hits")
        }
    )

    return thresholds


def finite_min_max(*arrays):
    """Return finite min/max across arrays or Series."""

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
    """Return padded plotting limits."""

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None

    if vmin == vmax:
        pad = max(abs(vmin) * pad_fraction, min_pad)
    else:
        pad = max((vmax - vmin) * pad_fraction, min_pad)

    return vmin - pad, vmax + pad


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def load_monthly_stats():
    """Load selected monthly verification CSVs."""

    frames = []
    missing = []

    for interval in INTERVAL_HOURS:
        interval_str = f"{interval:02d}"

        for year in YEARS:
            for month in MONTHS:
                month_str = f"{month:02d}"
                path = MONTHLY_STATS_DIR / f"verif_{interval_str}_{year}_{month_str}.csv"

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
        raise FileNotFoundError(f"No monthly verification CSVs found in {MONTHLY_STATS_DIR}")

    df_all = pd.concat(frames, ignore_index=True)

    print(f"Loaded {len(frames)} monthly files")
    print(f"Rows: {len(df_all):,}")

    if missing:
        print(f"Missing files: {len(missing)}")
        for p in missing[:10]:
            print(f"  {p}")

    thresholds = get_thresholds_from_columns(df_all)
    print(f"Detected thresholds: {thresholds}")

    if 1 not in thresholds or 5 not in thresholds:
        print(
            "\nWARNING: t1 or t5 threshold columns were not found.\n"
            "Rerun the raster verification script with:\n"
            "THRESHOLDS = [0.01, 0.05] + np.round(np.arange(0.10, 1.0, 0.10), 2).tolist()\n"
        )

    return df_all


# ---------------------------------------------------------------------
# CONTINGENCY METRICS
# ---------------------------------------------------------------------

def threshold_metric_long_table(
    df,
    group_cols=("interval_hour", "period", "forecast_hour", "year", "month"),
):
    """
    Convert wide threshold columns into a long table.

    Important:
    In your monthly output, threshold counts are repeated on every prob_bin row
    for the same interval/period/forecast_hour/year/month. Therefore, this
    function takes the first row per group instead of summing across prob_bin.
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
            })

    return pd.DataFrame(rows)


def add_contingency_metrics(df):
    """Add CSI, POD, FAR, SR, Bias, and Accuracy."""

    df = df.copy()

    h = df["hits"]
    m = df["misses"]
    fa = df["fa"]
    cn = df["cn"]

    df["CSI"] = safe_divide(h, h + m + fa)
    df["POD"] = safe_divide(h, h + m)
    df["FAR"] = safe_divide(fa, h + fa)
    df["SR"] = safe_divide(h, h + fa)
    df["BIAS"] = safe_divide(h + fa, h + m)
    df["ACC"] = safe_divide(h + cn, h + m + fa + cn)

    return df


def aggregate_threshold_metrics(
    threshold_long,
    group_cols=("interval_hour", "period", "forecast_hour", "threshold"),
):
    """Aggregate counts across years/months and recompute metrics."""

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

    grouped = add_contingency_metrics(grouped)

    return grouped


# ---------------------------------------------------------------------
# OPTIONAL: ADD VALID HOUR BASED ON FORECAST HOUR PATTERN
# ---------------------------------------------------------------------

def infer_valid_hour_from_period_and_forecast_hour(df):
    """
    Infer valid hour from the known period/forecast-hour pattern in your monthly stats.

    Your monthly stats do not include valid_dt, but you observed:
      Day   uses valid 00Z and 06Z
      Night uses valid 12Z and 18Z

    For the NBM cadence in your output, forecast hours alternate valid hours.

    This function assigns:
      day:
        odd sequence alternates 00/06 based on forecast hour
      night:
        matching pattern shifted to 12/18

    If you later add valid_hour directly to the verification CSV, use that instead.
    """

    df = df.copy()

    def _valid_hour(row):
        period = row["period"]
        fh = int(row["forecast_hour"])

        # For your 6-hour example:
        # F011 -> 00Z
        # F017 -> 06Z
        # F023 -> 00Z
        # F029 -> 06Z
        #
        # These differ by 6 hr. Integer division by 6 gives alternating groups.
        alt = (fh // 6) % 2

        if period == "day":
            return 0 if alt == 1 else 6

        if period == "night":
            return 12 if alt == 1 else 18

        return np.nan

    if "valid_hour" not in df.columns:
        df["valid_hour"] = df.apply(_valid_hour, axis=1).astype("Int64")

    return df

def plot_metric_by_threshold_valid_hour_panels(
    stats,
    metric,
    ylim=None,
    legend_ncol=2,
    label_best_points=True,
):
    """
    Plot one contingency metric by threshold and forecast hour.

    Produces separate 2-panel plots:
      Day:   00Z and 06Z
      Night: 12Z and 18Z

    Adds:
      - common y-axis range for the metric
      - shared lead-time legend
      - optional labels at best points for each forecast hour
    """

    metric = metric.upper()
    metric_label = get_metric_label(metric)

    if ylim is None:
        metric_ylim = get_metric_ylim(stats, metric)
    else:
        metric_ylim = ylim

    # Consistent colors for forecast hours across all figures
    all_fhs = sorted(stats["forecast_hour"].dropna().unique())
    cmap = plt.get_cmap("turbo", len(all_fhs))
    fh_to_color = {fh: cmap(i) for i, fh in enumerate(all_fhs)}

    for interval in sorted(stats["interval_hour"].unique()):
        for period in PERIODS:
            valid_hours = get_valid_hours_for_period(period)

            sub_ip = stats[
                (stats["interval_hour"] == interval) &
                (stats["period"] == period) &
                (stats["valid_hour"].isin(valid_hours))
            ].copy()

            if sub_ip.empty:
                print(f"No {metric} data for interval={interval}, period={period}")
                continue

            fig, axes = plt.subplots(
                1, 2,
                figsize=(17, 5.8),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )

            for ax, valid_hour in zip(axes, valid_hours):
                sub_vh = sub_ip[sub_ip["valid_hour"] == valid_hour].copy()

                if sub_vh.empty:
                    ax.set_title(f"Valid {valid_hour:02d}Z\nNo data")
                    ax.grid(alpha=0.25)
                    continue

                for fh, group in sub_vh.groupby("forecast_hour"):
                    group = group.sort_values("threshold")
                    color = fh_to_color.get(fh, "black")

                    ax.plot(
                        group["threshold"],
                        group[metric],
                        marker="o",
                        linewidth=1.8,
                        alpha=0.85,
                        color=color,
                        label=f"F{int(fh)}",
                    )

                    # Highlight and optionally label best point
                    valid_group = group.dropna(subset=[metric])

                    if not valid_group.empty:
                        if metric == "FAR":
                            idx = valid_group[metric].idxmin()
                        else:
                            idx = valid_group[metric].idxmax()

                        best = valid_group.loc[idx]

                        ax.scatter(
                            best["threshold"],
                            best[metric],
                            s=45,
                            color=color,
                            edgecolor="black",
                            linewidth=0.5,
                            zorder=5,
                        )

                        if label_best_points:
                            ax.annotate(
                                f"F{int(fh)}",
                                xy=(best["threshold"], best[metric]),
                                xytext=(4, 4),
                                textcoords="offset points",
                                fontsize=8,
                                color=color,
                                fontweight="bold",
                                ha="left",
                                va="bottom",
                            )

                ax.set_title(f"Valid {valid_hour:02d}Z")
                ax.set_xlabel("Forecast probability threshold (%)")
                ax.grid(alpha=0.25)
                ax.set_xticks(sorted(sub_ip["threshold"].unique()))

                if metric_ylim is not None:
                    ax.set_ylim(*metric_ylim)

            axes[0].set_ylabel(metric_label)

            fig.suptitle(
                f"{metric_label} vs Forecast Probability Threshold\n"
                f"{interval:02d}-hr Thunder Probability, {period.title()} Valid Windows",
                fontsize=15,
            )

            # Shared legend outside the plot
            handles = []
            labels = []

            for fh in all_fhs:
                if fh in sub_ip["forecast_hour"].unique():
                    handles.append(
                        plt.Line2D(
                            [0], [0],
                            color=fh_to_color[fh],
                            marker="o",
                            linewidth=1.8,
                        )
                    )
                    labels.append(f"F{int(fh)}")

            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=8,
                title="Lead Time",
                ncol=legend_ncol,
                frameon=True,
            )

            out_path = PLOT_DIR / f"{metric.lower()}_threshold_{interval:02d}h_{period}.png"

            if SAVE_FIGS:
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"Saved: {out_path}")

            if SHOW_FIGS:
                plt.show()
            else:
                plt.close(fig)

def summarize_best_csi(stats):
    """
    For each interval/period/valid_hour/forecast_hour, find the threshold
    with the maximum CSI and report CSI, POD, FAR, and Bias at that threshold.
    """

    rows = []

    group_cols = ["interval_hour", "period", "valid_hour", "forecast_hour"]

    for keys, group in stats.groupby(group_cols):
        interval, period, valid_hour, fh = keys

        group = group.dropna(subset=["CSI"])

        if group.empty:
            continue

        best = group.loc[group["CSI"].idxmax()]

        rows.append({
            "interval_hour": interval,
            "period": period,
            "valid_hour": valid_hour,
            "forecast_hour": fh,
            "best_threshold": best["threshold"],
            "best_CSI": best["CSI"],
            "POD_at_best_CSI": best["POD"],
            "FAR_at_best_CSI": best["FAR"],
            "BIAS_at_best_CSI": best["BIAS"],
            "hits": best["hits"],
            "misses": best["misses"],
            "fa": best["fa"],
            "cn": best["cn"],
        })

    return pd.DataFrame(rows)


def plot_best_csi_summary(best_df):
    """
    Plot best CSI threshold plus POD/FAR at that threshold by forecast hour.
    """

    for interval in sorted(best_df["interval_hour"].unique()):
        for period in PERIODS:
            valid_hours = get_valid_hours_for_period(period)

            sub_ip = best_df[
                (best_df["interval_hour"] == interval) &
                (best_df["period"] == period) &
                (best_df["valid_hour"].isin(valid_hours))
            ].copy()

            if sub_ip.empty:
                continue

            fig, axes = plt.subplots(
                3, 2,
                figsize=(15, 12),
                sharex=True,
                constrained_layout=True,
            )

            for col, valid_hour in enumerate(valid_hours):
                sub = sub_ip[sub_ip["valid_hour"] == valid_hour].sort_values("forecast_hour")

                if sub.empty:
                    continue

                axes[0, col].plot(
                    sub["forecast_hour"],
                    sub["best_threshold"],
                    marker="o",
                    linewidth=2,
                )
                axes[0, col].set_title(f"Valid {valid_hour:02d}Z")
                axes[0, col].set_ylabel("Best CSI Threshold (%)")
                axes[0, col].grid(alpha=0.25)

                axes[1, col].plot(
                    sub["forecast_hour"],
                    sub["best_CSI"],
                    marker="o",
                    linewidth=2,
                    label="CSI",
                )
                axes[1, col].plot(
                    sub["forecast_hour"],
                    sub["POD_at_best_CSI"],
                    marker="o",
                    linewidth=2,
                    label="POD",
                )
                axes[1, col].plot(
                    sub["forecast_hour"],
                    sub["FAR_at_best_CSI"],
                    marker="o",
                    linewidth=2,
                    label="FAR",
                )
                axes[1, col].set_ylabel("Metric value")
                axes[1, col].set_ylim(0, 1)
                axes[1, col].grid(alpha=0.25)
                axes[1, col].legend()

                axes[2, col].plot(
                    sub["forecast_hour"],
                    sub["BIAS_at_best_CSI"],
                    marker="o",
                    linewidth=2,
                    color="purple",
                )
                axes[2, col].axhline(1, color="black", linestyle="--", linewidth=1)
                axes[2, col].set_ylabel("Frequency Bias")
                axes[2, col].set_xlabel("Forecast hour")
                axes[2, col].grid(alpha=0.25)

            fig.suptitle(
                f"Best CSI Threshold and Associated POD/FAR\n"
                f"{interval:02d}-hr Thunder Probability, {period.title()} Valid Windows",
                fontsize=15,
            )

            out_path = PLOT_DIR / f"best_csi_summary_{interval:02d}h_{period}.png"

            if SAVE_FIGS:
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"Saved: {out_path}")

            if SHOW_FIGS:
                plt.show()
            else:
                plt.close(fig)

def main():
    df = load_monthly_stats()

    threshold_long = threshold_metric_long_table(df)
    threshold_stats = aggregate_threshold_metrics(threshold_long)

    # Add/infer valid_hour for 2-panel plotting
    threshold_stats = infer_valid_hour_from_period_and_forecast_hour(threshold_stats)

    out_csv = PLOT_DIR / "threshold_metrics_with_1_5_10.csv"
    threshold_stats.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Core metric plots
    plot_metric_by_threshold_valid_hour_panels(
        threshold_stats,
        metric="CSI",
        ylim=None,
    )

    plot_metric_by_threshold_valid_hour_panels(
        threshold_stats,
        metric="POD",
        ylim=(0, 1),
    )

    plot_metric_by_threshold_valid_hour_panels(
        threshold_stats,
        metric="FAR",
        ylim=(0, 1),
    )

    # Best CSI summary
    best = summarize_best_csi(threshold_stats)

    best_csv = PLOT_DIR / "best_csi_threshold_summary.csv"
    best.to_csv(best_csv, index=False)
    print(f"Saved: {best_csv}")

    plot_best_csi_summary(best)


if __name__ == "__main__":
    main()