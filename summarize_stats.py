"""Build verification datasets and summarize NBM probabilistic forecast skill.

Overview:
- Step 1 scans merged NBM/observation CSV files and consolidates them into
    parquet datasets grouped by thunder threshold, cycle, and optional WFO filter.
- Step 2 loads each parquet dataset and computes forecast verification metrics
    for one or more probability fields such as point_prob, mean, max, and pct90.
- Output products include parquet datasets for efficient reuse and summary CSVs
    with Brier Score, Brier Skill Score, ROC AUC, POD, FAR, and POFD.

Key assumptions:
- Input CSV filenames follow the standardized NBM naming convention used by the
    merge step.
- Probability columns are stored on a 0-100 scale and converted internally to
    0-1 probabilities during metric calculations.
"""

import os
import glob
import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# ==========================================
# CONFIGURATION
# ==========================================
base_csv_dir = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\prob_thunder_merged"
output_dir = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\prob_thunder_results"
stations_csv_path = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\reference\stations.csv"

YEARS = ["2023", "2024", "2025"]
MONTHS = [f"{m:02d}" for m in range(3, 11)]  # March (03) to October (10)
CYCLES = ["0100", "1300"]
THRESHOLDS = ["tstm01","tstm03","tstm06"]
PROB_THRESHOLDS = [20,50,80]

# List of WFOs to process individually. If empty, all stations are processed together.
PROCESS_WFO = []  # e.g., ["FGZ", "PSR"]

# The probability columns from your CSV to evaluate
EVAL_COLUMNS = ['point_prob', 'min', 'max', 'mean', 'median', 'pct90', 'majority']
# Truth column to evaluate against. Can be observed_combined, observed_thunder, or observed_lightning
TRUTH_COLUMN = 'observed_combined'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Regex to extract info from filename, e.g., blendv4.1_conus_tstm12_2023-03-02T1300_F023.csv
filename_pattern = re.compile(r"blendv([\d\.]+)_conus_(tstm\d{2})_(\d{4}-\d{2}-\d{2})T(\d{4})_F(\d{3})\.csv")


# ==========================================
# STEP 1: Harvest CSVs into Parquet Dataset
# ==========================================
def build_parquet_dataset():
    """Harvest merged CSV files into analysis-ready parquet datasets.

    The function iterates by threshold, cycle, and optional WFO filter to limit
    memory use. For each matched CSV, it appends forecast metadata extracted
    from the filename, derives a combined observed event flag, and writes a
    parquet file that can be reused for repeated verification runs.
    """
    print("Starting Step 1: Harvesting data into Parquet...")
    
    # Load station metadata used for optional WFO subsetting and region fields.
    stations_df = pd.read_csv(stations_csv_path)
    
    # Process one threshold and cycle at a time to save memory
    for threshold in THRESHOLDS:
        for cycle in CYCLES:
            # Process all stations together unless a WFO-specific queue is configured.
            wfo_queue = PROCESS_WFO if PROCESS_WFO else [None]
            
            for focus_wfo in wfo_queue:
                q_label = focus_wfo if focus_wfo else "ALL"
                print(f"Processing threshold {threshold} | cycle {cycle} | WFO: {q_label}...")
                group_dataframes = []
                
                # Restrict to stations belonging to the requested WFO when needed.
                allowed_stations = None
                if focus_wfo:
                    w_mask = stations_df['wfo'] == focus_wfo
                    allowed_stations = set(stations_df.loc[w_mask, 'station'])

                for year in YEARS:
                    for month in MONTHS:
                        month_path = os.path.join(base_csv_dir, year, month)
                        if not os.path.exists(month_path):
                            continue
                            
                        days = [d for d in os.listdir(month_path) if os.path.isdir(os.path.join(month_path, d))]
                        for day in days:
                            target_dir = os.path.join(base_csv_dir, year, month, day, cycle, threshold)
                            if not os.path.exists(target_dir):
                                continue
                            
                            csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
                            for csv_file in csv_files:
                                filename = os.path.basename(csv_file)
                                match = filename_pattern.search(filename)
                                
                                if match:
                                    version = match.group(1)
                                    fhr = int(match.group(5))
                                    
                                    # Read CSV
                                    try:
                                        df = pd.read_csv(csv_file)
                                    except Exception as e:
                                        print(f"Error reading {filename}: {e}")
                                        continue
                                        
                                    # Drop station_1 if it exists as it is a duplicate of station
                                    if 'station_1' in df.columns:
                                        df.drop(columns=['station_1'], inplace=True)
                                        
                                    if allowed_stations is not None:
                                        if 'station' in df.columns:
                                            df = df[df['station'].isin(allowed_stations)]
                                    
                                    if df.empty:
                                        continue
                                    
                                    # Add forecast metadata derived from the filename and loop context.
                                    df['nbm_version'] = version
                                    df['forecast_hour'] = fhr
                                    df['threshold_type'] = threshold
                                    df['cycle'] = cycle
                                    df['valid_date'] = match.group(3)
                                    if focus_wfo:
                                       df['wfo_filter'] = focus_wfo
                                    
                                    # Treat either observed thunder or lightning as an event occurrence.
                                    df['observed_combined'] = ((df.get('observed_thunder', 0) > 0) | (df.get('observed_lightning', 0) > 0)).astype(float)
                                    
                                    # Preserve missing truth when both source observation columns are absent.
                                    if 'observed_thunder' in df.columns and 'observed_lightning' in df.columns:
                                        both_missing = df['observed_thunder'].isna() & df['observed_lightning'].isna()
                                        df.loc[both_missing, 'observed_combined'] = np.nan
                                    
                                    # Retain only forecast, truth, and grouping fields needed downstream.
                                    cols_to_keep = [
                                        'station', 'lat', 'lon',
                                        'point_prob', 'min', 'max', 'mean', 'median', 'pct90', 'majority',
                                        'observed_thunder', 'observed_lightning', 'observed_combined', 
                                        'nbm_version', 'forecast_hour', 'threshold_type', 'cycle', 'valid_date', 'wfo_filter'
                                    ]
                                    
                                    # Guard against malformed files that may be missing expected columns.
                                    existing_cols = [c for c in cols_to_keep if c in df.columns]
                                    df = df[existing_cols]
                                    
                                    group_dataframes.append(df)

                if group_dataframes:
                    # Combine one threshold/cycle/WFO slice into a reusable parquet dataset.
                    group_df = pd.concat(group_dataframes, ignore_index=True)
                    
                    # Reattach station metadata cleanly without merge suffix clutter.
                    cols_to_add = ['wfo', 'state', 'region']
                    for col in cols_to_add:
                        if col in group_df.columns:
                            group_df.drop(columns=[col], inplace=True)
                            
                    # Merge on the 'station' column
                    group_df = group_df.merge(stations_df[['station'] + cols_to_add], on='station', how='left')
                    
                    if focus_wfo:
                        out_filename = f"prob_thunder_{threshold}_{cycle}_{focus_wfo}.parquet"
                    else:
                        out_filename = f"prob_thunder_{threshold}_{cycle}.parquet"
                        
                    out_path = os.path.join(output_dir, out_filename)
                    print(f"  Saving {out_filename} ({len(group_df)} rows)...")
                    group_df.to_parquet(out_path, engine='pyarrow', index=False)
                    
                    # Free memory before moving back to the top of the loop
                    del group_df
                    del group_dataframes
                else:
                    print(f"  No data found for {threshold} | {cycle} | {q_label}.")

    print("Step 1 Complete!\n")


# ==========================================
# STEP 2: Calculate Summary Statistics
# ==========================================
def calculate_statistics():
    """Compute verification metrics from harvested parquet datasets.

    For each parquet file, the function groups rows by forecast metadata and
    evaluates each configured probability field against the selected truth
    column. Metrics include Brier Score, Brier Skill Score, ROC AUC, and
    contingency-table statistics at multiple probability thresholds.
    """
    print("Starting Step 2: Calculating verification statistics...")
    
    # Each parquet file corresponds to one threshold/cycle/WFO data slice.
    parquet_files = glob.glob(os.path.join(output_dir, "prob_thunder_*.parquet"))
    if not parquet_files:
        print("Error: No Parquet datasets found in output_dir. Please run Step 1 first.")
        return

    # Load only fields required for grouping and score calculations.
    columns_to_load = ['nbm_version', 'threshold_type', 'forecast_hour', 'cycle', TRUTH_COLUMN] + EVAL_COLUMNS
    # WFO filter is optional because not every parquet file contains it.
    columns_to_load.append('wfo_filter')
    
    # Process one Parquet dataset at a time
    for pf in parquet_files:
        results = []
        print(f"Loading and processing {os.path.basename(pf)}...")
        try:
            # Intersect requested columns with the parquet schema for robustness.
            from pyarrow.parquet import read_schema
            file_cols = read_schema(pf).names
            actual_cols_to_load = [col for col in columns_to_load if col in file_cols]
            
            df = pd.read_parquet(pf, engine='pyarrow', columns=actual_cols_to_load)
        except Exception as e:
            print(f"Error loading {os.path.basename(pf)}: {e}")
            continue
            
        # Downcast large float columns after loading to reduce memory pressure.
        for col in EVAL_COLUMNS:
            if col in df.columns and df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')

        # Include WFO when present so regional subsets get separate score rows.
        groupby_cols = ['nbm_version', 'threshold_type', 'forecast_hour', 'cycle']
        if 'wfo_filter' in df.columns:
            groupby_cols.append('wfo_filter')
            
        grouped = df.groupby(groupby_cols)
        
        for name, group in grouped:
            if 'wfo_filter' in df.columns:
                version, threshold, fhr, cycle_hr, wfo_f = name
            else:
                version, threshold, fhr, cycle_hr = name
                wfo_f = ""
            
            # Skip malformed groups that do not contain the requested truth field.
            if TRUTH_COLUMN not in group.columns:
                continue
                
            truth = group[TRUTH_COLUMN].values
        
            # Evaluate each forecast probability representation independently.
            for eval_col in EVAL_COLUMNS:
                if eval_col not in group.columns:
                    continue
                    
                # Convert percent probabilities to fractions for metric calculations.
                forecasts = group[eval_col].values / 100.0
                
                # Mask missing forecast/truth pairs before computing skill metrics.
                valid_mask = ~np.isnan(forecasts) & ~np.isnan(truth)
                valid_forecasts = forecasts[valid_mask]
                valid_truth = truth[valid_mask]
                
                if len(valid_truth) == 0:
                    continue
                    
                # Brier Score measures mean squared probability error.
                brier_score = np.mean((valid_forecasts - valid_truth) ** 2)
                
                # Use sample climatology as the Brier Skill Score reference forecast.
                clim_prob = np.mean(valid_truth)
                bs_clim = np.mean((clim_prob - valid_truth) ** 2)
                
                # Skill is undefined if the sample has no event variance.
                if bs_clim > 0:
                    bss = 1.0 - (brier_score / bs_clim)
                else:
                    bss = np.nan
                
                # ROC AUC requires both event and non-event cases in the sample.
                if len(np.unique(valid_truth)) > 1:
                    roc_auc = roc_auc_score(valid_truth, valid_forecasts)
                else:
                    roc_auc = np.nan
                    
                # Start one output row per grouping and evaluated probability field.
                stat_row = {
                    'nbm_version': version,
                    'threshold_type': threshold,
                    'forecast_hour': fhr,
                    'cycle': cycle_hr,
                    'wfo': wfo_f,            # <--- ADD THIS LINE
                    'eval_metric': eval_col,
                    'total_samples': len(valid_truth),
                    'brier_score': brier_score,
                    'brier_skill_score': bss,
                    'roc_auc': roc_auc
                }
                
                # Build contingency-table metrics at configured decision thresholds.
                for p_thresh in PROB_THRESHOLDS:
                    frac_thresh = p_thresh / 100.0
                    forecast_yes = (valid_forecasts >= frac_thresh)
                    
                    hits = np.sum(forecast_yes & (valid_truth == 1))
                    false_alarms = np.sum(forecast_yes & (valid_truth == 0))
                    misses = np.sum((~forecast_yes) & (valid_truth == 1))
                    correct_negatives = np.sum((~forecast_yes) & (valid_truth == 0))
                    
                    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
                    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
                    pofd = false_alarms / (false_alarms + correct_negatives) if (false_alarms + correct_negatives) > 0 else np.nan
                    
                    stat_row[f'POD_{p_thresh}'] = pod
                    stat_row[f'FAR_{p_thresh}'] = far
                    stat_row[f'POFD_{p_thresh}'] = pofd
                    stat_row[f'Hits_{p_thresh}'] = hits
                    stat_row[f'Misses_{p_thresh}'] = misses
                    stat_row[f'FalseAlarms_{p_thresh}'] = false_alarms
                    stat_row[f'CorrectNegatives_{p_thresh}'] = correct_negatives
                    
                results.append(stat_row)
            
        # Release per-file objects before moving to the next parquet dataset.
        del df
        del grouped
        
        if not results:
            print(f"  No statistics could be calculated for {os.path.basename(pf)}.")
            continue
            
        # Convert accumulated rows to a final summary table.
        results_df = pd.DataFrame(results)
        
        # Keep output stable and easy to scan across cycles and lead times.
        sort_cols = ['nbm_version', 'threshold_type', 'cycle', 'forecast_hour', 'wfo', 'eval_metric']
        # Drop the empty WFO column when no station subset was requested.
        if 'wfo' in results_df.columns and (results_df['wfo'] == "").all():
            sort_cols.remove('wfo')
            results_df.drop('wfo', axis=1, inplace=True)
            
        results_df.sort_values(by=sort_cols, inplace=True)
        
        # Output to a file matching the parquet file input name
        base_name = os.path.basename(pf).replace('.parquet', '')
        stats_csv_path = os.path.join(output_dir, f"{base_name}_stats.csv")
        results_df.to_csv(stats_csv_path, index=False)
        
        print(f"  Statistics saved to: {os.path.basename(stats_csv_path)}")

    print("Step 2 Complete!\n")

if __name__ == '__main__':
    # Build parquet inputs first; then Step 2 can be rerun independently if desired.
    build_parquet_dataset()
    
    # Uncomment to generate verification summary CSVs from the parquet datasets.
    # calculate_statistics()