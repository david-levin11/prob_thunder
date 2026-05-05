"""Aggregate daily NLDN text files into rolling-window CSV point extracts.

Overview:
- Reads VAISALA daily lightning text files from INPUT_DIR.
- Builds event timestamps from year/month/day/hour/minute columns.
- Iterates configured years, months, valid hours, and trailing window lengths.
- Filters events to each interval [valid time - window, valid time).
- Writes interval CSV files to window-specific output folders.

Output layout:
- Aggregated CSV files are written under INPUT_DIR into nldn_06_obs and nldn_12_obs.
- Output names follow nldn_XXh_YYYYMMDD_HHMMZ.csv where XX is the window.
"""

import pandas as pd
import os
from datetime import datetime, timedelta

# --- Configuration ---
INPUT_DIR = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\nldn"
OUTPUT_DIR_06 = os.path.join(INPUT_DIR, "nldn_06_obs")
OUTPUT_DIR_12 = os.path.join(INPUT_DIR, "nldn_12_obs")

# Create directories if they don't exist
for d in [OUTPUT_DIR_06, OUTPUT_DIR_12]:
    if not os.path.exists(d):
        os.makedirs(d)

# Years and Months to process
YEARS = [2023, 2024, 2025]
MONTHS = range(3, 11)  # March to October
VALID_HOURS = [0, 6, 12, 18]
WINDOWS = [6, 12]

def load_nldn_file(file_date):
    """Load one VAISALA daily lightning file into a normalized DataFrame.

    Args:
        file_date: Date used to construct VAISALA_YYYYMMDD.txt input name.

    Returns:
        pandas.DataFrame: Empty DataFrame when file is missing/unreadable, else
        columns ['dt', 'lat', 'lon', 'polarity', 'strength', 'flashes'] where
        'dt' is a pandas datetime timestamp built from source date-time fields.
    """
    file_name = f"VAISALA_{file_date.strftime('%Y%m%d')}.txt"
    file_path = os.path.join(INPUT_DIR, file_name)
    
    if not os.path.exists(file_path):
        return pd.DataFrame()

    try:
        # Columns: 0:Yr, 1:Mo, 2:Dy, 3:Hr, 4:Mn, 5:Sec, 6:Lat, 7:Lon, 8:Sens, 9:Pol, 10:Str, 11:Fls
        df = pd.read_csv(file_path, sep=r'\s+', header=None, 
                         names=['year', 'month', 'day', 'hour', 'minute', 'second', 
                                'lat', 'lon', 'sensors', 'polarity', 'strength', 'flashes'])
        
        # Create datetime objects efficiently
        df['dt'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
        return df[['dt', 'lat', 'lon', 'polarity', 'strength', 'flashes']]
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return pd.DataFrame()

def process_aggregations():
    """Generate aggregated NLDN CSV files for all configured intervals.

    For each (year, month, day, valid hour) combination, this function creates
    both 6-hour and 12-hour trailing-window extracts. It loads the minimum set
    of source daily files needed for each interval (current day and possibly
    previous day when crossing midnight), filters records by timestamp, and
    writes matching events to the corresponding output folder.
    """
    # Iterate through each year and month
    for year in YEARS:
        for month in MONTHS:
            print(f"Processing {year}-{month:02d}...")
            
            # Get list of days in this month
            start_date = datetime(year, month, 1)
            # Find last day of month
            if month == 12:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, month + 1, 1)
            
            current_day = start_date
            while current_day < end_date:
                for v_hr in VALID_HOURS:
                    valid_dt = current_day.replace(hour=v_hr)
                    
                    for window in WINDOWS:
                        start_dt = valid_dt - timedelta(hours=window)
                        
                        # Load required files (current day and previous day if window spans midnight)
                        days_needed = {start_dt.date(), valid_dt.date()}
                        combined_df = pd.concat([load_nldn_file(d) for d in days_needed], ignore_index=True)
                        
                        if combined_df.empty:
                            continue
                            
                        # Filter for the window [start_dt, valid_dt)
                        mask = (combined_df['dt'] >= start_dt) & (combined_df['dt'] < valid_dt)
                        subset = combined_df[mask].copy()
                        
                        if not subset.empty:
                            # Format columns as requested
                            subset['valid_datetime'] = subset['dt'].dt.strftime('%Y-%m-%d %H:%M')
                            out_cols = ['valid_datetime', 'lat', 'lon', 'polarity', 'strength', 'flashes']
                            final_df = subset[out_cols]
                            
                            # Save file
                            folder = OUTPUT_DIR_06 if window == 6 else OUTPUT_DIR_12
                            filename = f"nldn_{window:02}h_{valid_dt.strftime('%Y%m%d_%H%M')}Z.csv"
                            final_df.to_csv(os.path.join(folder, filename), index=False)
                            
                current_day += timedelta(days=1)

if __name__ == "__main__":
    process_aggregations()
    print("Processing Complete.")