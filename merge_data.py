"""Merge NBM station statistics with matching aggregated METAR observations.

Overview:
- Scans NBM-derived CSV outputs beneath nbm_root.
- Parses initialization time and forecast hour from each NBM filename to derive
    the corresponding valid datetime.
- Maps each NBM thunder threshold folder (for example, tstm12) to the matching
    observation interval file (for example, 12hr).
- Normalizes station identifiers so NBM and METAR station codes align.
- Writes merged station-level verification tables beneath output_merged_root,
    preserving the NBM directory structure.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import multiprocessing

# --- Configuration ---
nbm_root = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\prob_thunder_csv"
obs_root = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\metar"
output_merged_root = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\prob_thunder_merged"

# Filters
# YEARS = ["2023", "2024", "2025"]
# MONTHS = [f"{m:02d}" for m in range(3, 11)]
# THRESHOLDS = ["tstm01", "tstm03", "tstm06", "tstm12"]
YEARS = ["2023", "2024", "2025"]
MONTHS = [f"{m:02d}" for m in range(3, 11)]
THRESHOLDS = ["tstm12"]

def get_valid_datetime(filename):
    """Parse an NBM CSV filename and return its valid datetime.

    Expected filename pattern:
        blendv4.1_conus_tstm12_2023-08-08T1300_F017.csv

    The valid time is computed as initialization time plus forecast hour.
    """
    try:
        # Strip .csv and split by underscore
        clean_name = filename.replace('.csv', '')
        parts = clean_name.split('_')
        
        # The timestamp is the 4th element (index 3)
        init_str = parts[3] 
        
        # The Lead hour is the 5th element (index 4)
        fhr_str = parts[4].replace('F', '')
        
        # Convert to datetime
        init_dt = datetime.strptime(init_str, "%Y-%m-%dT%H%M")
        valid_dt = init_dt + timedelta(hours=int(fhr_str))
        
        return valid_dt
    except Exception as e:
        print(f"Parsing error on {filename}: {e}")
        return None

def normalize_stations(df, col_name):
    """Standardize station IDs so NBM and METAR files join reliably.

    This trims whitespace, uppercases values, and removes a leading ``K`` from
    5-character identifiers such as ``KDEN`` so they match 4-character METAR
    station codes like ``DEN``.
    """
    df[col_name] = df[col_name].astype(str).str.strip().str.upper()
    # If NBM uses 5 chars (KDEN) and Obs uses 4 (DEN), this trims the K
    df[col_name] = df[col_name].apply(lambda x: x[1:] if (len(x) == 5 and x.startswith('K')) else x)
    return df

def process_file(task_data):
    """Merge one NBM CSV with its matching observation CSV.

    Args:
        task_data: Tuple of (root, file, current_threshold).

    Notes:
        The observation file path is inferred from the NBM filename valid time
        and the threshold folder name (for example, ``tstm12`` -> ``12hr``).
    """
    root, file, current_threshold = task_data
    valid_dt = get_valid_datetime(file)
    if not valid_dt:
        return

    # Map threshold folder name to observation aggregation interval.
    interval_val = current_threshold.replace('tstm', '').lstrip('0')
    obs_interval_str = f"{interval_val}hr"

    # Build full paths for the NBM source and expected observation match.
    nbm_path = os.path.join(root, file)
    obs_filename = f"conus_obs.{obs_interval_str}.{valid_dt.strftime('%Y%m%d%H')}.csv"
    obs_path = os.path.join(
        obs_root, 
        valid_dt.strftime("%Y"), 
        valid_dt.strftime("%m"), 
        valid_dt.strftime("%d"), 
        obs_filename
    )

    if not os.path.exists(obs_path):
        return

    try:
        # Load both tables and normalize station keys before joining.
        nbm_df = pd.read_csv(nbm_path)
        obs_df = pd.read_csv(obs_path, usecols=['station', 'observed_thunder', 'observed_lightning'])
        
        count_nbm_orig = len(nbm_df)
        nbm_df = normalize_stations(nbm_df, 'station')
        obs_df = normalize_stations(obs_df, 'station')

        # Keep only stations found in both the forecast and observation files.
        merged = pd.merge(nbm_df, obs_df, on='station', how='inner')

        if not merged.empty:
            # Preserve the NBM directory layout in the merged output tree.
            rel_path = os.path.relpath(root, nbm_root)
            out_dir = os.path.join(output_merged_root, rel_path)
            os.makedirs(out_dir, exist_ok=True)
            
            out_file = os.path.join(out_dir, file)
            merged.to_csv(out_file, index=False)
            print(f"Success: {file} | Matched {len(merged)}/{count_nbm_orig} stations")
            
    except Exception as e:
        print(f"Error on {file}: {e}")

def merge_nbm_obs():
    """Discover matching NBM CSV files and merge them in parallel.

    The search is limited by the configured year/month/threshold filters. Each
    discovered file becomes one multiprocessing task handled by process_file().
    """
    print(f"--- Starting Merge: {datetime.now().strftime('%H:%M:%S')} ---")
    
    tasks = []
    for year in YEARS:
        for month in MONTHS:
            month_path = os.path.join(nbm_root, year, month)
            if not os.path.exists(month_path):
                continue
            
            for root, dirs, files in os.walk(month_path):
                path_parts = os.path.normpath(root).split(os.sep)
                current_threshold = next((t for t in THRESHOLDS if t in path_parts), None)
                
                if not current_threshold:
                    continue

                for file in files:
                    if file.endswith(".csv"):
                        tasks.append((root, file, current_threshold))

    print(f"Total files to process: {len(tasks)}")
    
    # Leave one core free for system responsiveness during batch processing.
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_cores} cores...")
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(process_file, tasks)

    print(f"--- Process Complete: {datetime.now().strftime('%H:%M:%S')} ---")

if __name__ == "__main__":
    merge_nbm_obs()