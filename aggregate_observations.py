"""Aggregate hourly METAR thunder/lightning observations into interval products.

Overview:
- Reads monthly CONUS observation files named conus_obs_YYYY_MM.csv.
- Aggregates observed_thunder and observed_lightning by station at configurable
    intervals (1, 3, 6, or special 12-hour rolling windows).
- Preserves station metadata in each aggregated output row.
- Writes one CSV per valid timestamp to a year/month/day folder hierarchy.

Special handling:
- For intervals greater than 1 hour, the script optionally loads a short buffer
    from the previous month so early-month windows can aggregate correctly.
- For 12-hour output, rolling 12-hour maxima are sampled at 00, 06, 12, and 18Z.
"""

import pandas as pd
import os
import gc

# Configuration
input_dir = r'N:\data\nbm\prob_thunder\metar'
output_base_dir = r'C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\metar'

def process_monthly_file(year, month, interval_hrs=1):
    """Aggregate one month of station observations to a target interval.

    Args:
        year: Four-digit year (for example, 2024).
        month: Month number 1-12.
        interval_hrs: Aggregation interval in hours. Supports standard resample
            intervals (1, 3, 6, etc.) and special 12-hour rolling logic.

    Notes:
        - Input files are expected at input_dir/conus_obs_YYYY_MM.csv.
        - Output files are written per timestamp as:
          conus_obs.{interval}hr.YYYYMMDDHH.csv
        - Output folder structure is: output_base_dir/YYYY/MM/DD
    """
    file_name = f'conus_obs_{year}_{month:02d}.csv'
    file_path = os.path.join(input_dir, file_name)
    if not os.path.exists(file_path):
        print(f"--- Skipping {file_name}: File not found. ---")
        return

    print(f"\n>>> Starting {interval_hrs}hr aggregation for {file_name}...")
    
    cols_to_load = ['station', 'valid', 'observed_thunder', 'observed_lightning', 
                    'station_name', 'lat', 'lon', 'elevation_m', 'elevation_ft']
    
    df = pd.read_csv(file_path, parse_dates=['valid'], usecols=cols_to_load)
    df['valid'] = pd.to_datetime(df['valid'], errors='coerce')
    
    # Load previous month if interval > 1 to provide buffer
    if interval_hrs > 1:
        prev_month = 12 if month == 1 else month - 1
        prev_year = year - 1 if month == 1 else year
        prev_file = f'conus_obs_{prev_year}_{prev_month:02d}.csv'
        prev_path = os.path.join(input_dir, prev_file)
        
        if os.path.exists(prev_path):
            print(f"    Loading buffer from previous month ({prev_file})...")
            df_prev = pd.read_csv(prev_path, parse_dates=['valid'], usecols=cols_to_load)
            df_prev['valid'] = pd.to_datetime(df_prev['valid'], errors='coerce')
            
            # Keep only the last 3 days of the previous month for the buffer to save memory
            max_date = df_prev['valid'].max()
            if not pd.isna(max_date):
                cutoff = max_date - pd.Timedelta(days=3)
                df_prev = df_prev[df_prev['valid'] >= cutoff]
            
            df = pd.concat([df_prev, df], ignore_index=True)
            del df_prev

    df = df.sort_values(['station', 'valid'])
    
    # ---------------------------------------------------------
    # SPECIAL LOGIC FOR 12-HOUR (Rolling 12h every 6h)
    # ---------------------------------------------------------
    if interval_hrs == 12:
        # 1. De-duplicate station/valid pairs first
        df = df.groupby(['station', 'valid'], as_index=False).agg({
            'observed_thunder': 'max',
            'observed_lightning': 'max',
            'station_name': 'first',
            'lat': 'first',
            'lon': 'first',
            'elevation_m': 'first',
            'elevation_ft': 'first'
        })

        # 2. Set index to 'valid' for resampling
        df.set_index('valid', inplace=True)
        
        # 3. Resample to 1h to group sub-hourly reports (like METARs at xx:53) into the nearest hour
        df_1h = df.groupby('station').resample('1h', closed='right', label='right', origin='start_day').agg({
            'observed_thunder': 'max',
            'observed_lightning': 'max',
            'station_name': 'first',
            'lat': 'first',
            'lon': 'first',
            'elevation_m': 'first',
            'elevation_ft': 'first'
        })
        
        # IMPORTANT: Remove the 'station' column if it was preserved, then reset the MultiIndex
        df_1h = df_1h.drop(columns=['station'], errors='ignore').reset_index()
        
        # 4. Fill metadata forward (now that 'station' is just a normal column)
        metadata_cols = ['station_name', 'lat', 'lon', 'elevation_m', 'elevation_ft']
        for col in metadata_cols:
            df_1h[col] = df_1h.groupby('station', group_keys=False)[col].ffill()

        # 5. Calculate 12-hour ROLLING max
        # We group by station and apply rolling to the thunder/lightning columns
        df_1h['observed_thunder'] = df_1h.groupby('station')['observed_thunder'].transform(lambda x: x.rolling(window=12, min_periods=1).max())
        df_1h['observed_lightning'] = df_1h.groupby('station')['observed_lightning'].transform(lambda x: x.rolling(window=12, min_periods=1).max())
        
        agg_df = df_1h.reset_index()

        # 6. Filter for only 0, 6, 12, 18Z valid times
        agg_df = agg_df[agg_df['valid'].dt.hour.isin([0, 6, 12, 18])]
        
    else:
        # STANDARD LOGIC for 1, 3, 6 hour intervals
        df.set_index('valid', inplace=True)
        freq_str = f'{interval_hrs}h'
        groups = df.groupby('station').resample(freq_str, closed='right', label='right', origin='start_day')
        
        agg_df = groups.agg({
            'observed_thunder': 'max',
            'observed_lightning': 'max',
            'station_name': 'first',
            'lat': 'first',
            'lon': 'first',
            'elevation_m': 'first',
            'elevation_ft': 'first'
        }).reset_index()
    
    # ---------------------------------------------------------
    # CLEANUP AND EXPORT
    # ---------------------------------------------------------
    # Filter agg_df to only include the current year and month being processed
    agg_df = agg_df[(agg_df['valid'].dt.year == year) & (agg_df['valid'].dt.month == month)]
    
    agg_df.dropna(subset=['observed_thunder'], inplace=True)
    
    unique_hours = sorted(agg_df['valid'].unique())
    
    for hr in unique_hours:
        hr_ts = pd.to_datetime(hr)
        timestamp_str = hr_ts.strftime('%Y%m%d%H')
        out_name = f"conus_obs.{interval_hrs}hr.{timestamp_str}.csv"
        
        out_dir = os.path.join(output_base_dir, f'{hr_ts.year}', f"{hr_ts.month:02d}", f"{hr_ts.day:02d}")
        os.makedirs(out_dir, exist_ok=True)
        
        out_path = os.path.join(out_dir, out_name)
        hour_data = agg_df[agg_df['valid'] == hr]
        hour_data.to_csv(out_path, index=False)
        print(f"   Saved: {out_name}")

    print(f">>> Completed {interval_hrs}hr for {file_name}.\n")
    del df, agg_df
    gc.collect()

# --- Execution ---
# Configure interval set and monthly range for batch processing.
intervals_to_run = [1, 3]

for year in range(2023, 2026):
    for month in range(3, 11):
        for interval in intervals_to_run:
            process_monthly_file(year, month, interval_hrs=interval)