import pandas as pd
from pathlib import Path

# --- paths ---
data_dir = Path(r"C:\Users\David.Levin\NBMLightningVer")

old_csv = data_dir / "GLD_AK_20-24.csv"
new_txt = data_dir / "2025_AK.txt"
out_csv = data_dir / "GLD_AK_20-25.csv"


def standardize_datetime_string(x):
    """
    Standardize DateTime strings to:
    YYYY-MM-DD HH:MM:SS.fffffffff

    Handles:
    - 2024-06-06 23:31:00
    - 2025-01-06 11:24:27.391869729

    This does NOT parse/reformat with pandas, so nanoseconds are preserved.
    """
    if pd.isna(x):
        return x

    x = str(x).strip()

    if "." in x:
        base, frac = x.split(".", 1)
        frac = frac.ljust(9, "0")[:9]
        return f"{base}.{frac}"

    return f"{x}.000000000"


# ------------------------------------------------------------
# 1. Read existing CSV while preserving DateTime exactly as text
# ------------------------------------------------------------
old = pd.read_csv(
    old_csv,
    dtype={
        "DateTime": "string",
        "Cloud/CG": "string",
    }
)

old["DateTime"] = old["DateTime"].apply(standardize_datetime_string)


# ------------------------------------------------------------
# 2. Read new whitespace-delimited TXT file
# ------------------------------------------------------------
new = pd.read_csv(
    new_txt,
    sep=r"\s+",
    header=None,
    names=[
        "Date", "Time", "Lat", "Lon", "Ip", "mult",
        "SMA", "SMI", "Chi", "Cloud/CG"
    ],
    dtype={
        "Date": "string",
        "Time": "string",
        "Lat": "float64",
        "Lon": "float64",
        "Ip": "float64",
        "mult": "Int64",
        "SMA": "float64",
        "SMI": "float64",
        "Chi": "float64",
        "Cloud/CG": "string",
    }
)

# Combine Date + Time into your existing CSV's DateTime format
new["DateTime"] = new["Date"].str.strip() + " " + new["Time"].str.strip()
new["DateTime"] = new["DateTime"].apply(standardize_datetime_string)

# Reorder to match the existing CSV
new = new[
    ["Lat", "Lon", "Ip", "mult", "SMA", "SMI", "Chi", "Cloud/CG", "DateTime"]
]


# ------------------------------------------------------------
# 3. Append
# ------------------------------------------------------------
combined = pd.concat([old, new], ignore_index=True)


# ------------------------------------------------------------
# 4. Validate datetimes without overwriting them
# ------------------------------------------------------------
combined["_dt_sort"] = pd.to_datetime(
    combined["DateTime"],
    errors="coerce",
    format="mixed"
)

bad = combined[combined["_dt_sort"].isna()]

if not bad.empty:
    print(f"WARNING: {len(bad):,} rows failed DateTime parsing.")
    print(bad[["Lat", "Lon", "Ip", "Cloud/CG", "DateTime"]].head(30))
    raise ValueError("Stopping because some DateTime values could not be parsed.")


# ------------------------------------------------------------
# 5. Sort chronologically, but keep original DateTime strings
# ------------------------------------------------------------
combined = (
    combined
    .sort_values("_dt_sort")
    .drop(columns="_dt_sort")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 6. Write final combined file
# ------------------------------------------------------------
combined.to_csv(out_csv, index=False)

print(f"Wrote {len(combined):,} rows to {out_csv}")