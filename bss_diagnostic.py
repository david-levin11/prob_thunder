import pandas as pd

pair_df = pd.read_csv(
    r"C:\Users\David.Levin\NBMLightningVer\brier_gld_climo\brier_against_monthly_gld_climo_by_pair.csv",
    parse_dates=["valid_dt"]
)

pair_df["valid_hour"] = pair_df["valid_dt"].dt.hour

valid_hour_counts = (
    pair_df.groupby(["interval_hour", "period", "forecast_hour", "valid_hour"])
           .size()
           .reset_index(name="n_pairs")
           .sort_values(["interval_hour", "period", "forecast_hour", "valid_hour"])
)

print(valid_hour_counts.to_string(index=False))