"""Compute zonal and point-based NBM verification statistics for station buffers.

Overview:
- Scans downloaded NBM GeoTIFF files beneath INPUT_ROOT.
- Filters rasters by configured year, month, cycle, and thunder threshold.
- Uses multiprocessing so each worker runs ArcPy zonal statistics on one raster.
- Joins station-buffer metadata, exports results to CSV, and extracts the raster
    value at each station point as ``point_prob``.
- Normalizes the final CSV schema and removes ArcPy sidecar files.

Output layout:
- CSV outputs mirror the INPUT_ROOT folder structure under OUTPUT_ROOT.
- Each raster produces one CSV with summary statistics for each station zone.
"""

import arcpy
import os
import pandas as pd
import sys
import time
import gc
from multiprocessing import Pool, cpu_count

# --- Configuration ---
INPUT_ROOT = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\prob_thunder"
OUTPUT_ROOT = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\prob_thunder_csv"
ZONE_DATA = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\NBM_Verif.gdb\stations_buffer_20km"
ZONE_FIELD = "station"

# arcpy.env.workspace = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\NBM_Verif.gdb"
arcpy.env.overwriteOutput = True
# arcpy.env.parallelProcessingFactor = "100%"

YEARS = ["2023"]
MONTHS = [f"{m:02d}" for m in range(3, 4)]  # March to October
CYCLES = ["0100", "1300"]
THRESHOLDS = ["tstm01"]  # You can add others like "tstm01", etc.
PROCESS_AGAIN = True

# Pandas Formatting Configuration
ROUND_MAP = {
    "lat": 4, "lon": 4, "elevation_ft": 0, "mean": 2, 
    "std": 2, "sum": 2, "majority_percent": 1, "minority_percent": 1
}
DROP_COLUMNS = {
    "zone_code", "area", "buff_dist", "orig_fid", 
    "shape_length", "shape_area"
}
INT_COLUMNS = {
    "count", "min", "max", "range", "variety", "majority", 
    "minority", "median", "pct90", "majority_count", "minority_count"
}

def worker_init():
    """Initialize each worker process with the Spatial Analyst license."""
    arcpy.CheckOutExtension("Spatial")

def clean_csv(csv_path):
    """Normalize exported CSV field names and numeric formatting.

    Steps:
    - Lowercase and trim column names.
    - Drop ArcPy fields not needed for downstream verification work.
    - Round selected floating-point columns.
    - Convert count/statistic fields to nullable integer dtype.
    """
    try:
        df = pd.read_csv(csv_path)
        # Standardize column names
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Drop unnecessary columns
        cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Round floats
        for col, dec in ROUND_MAP.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(dec)

        # Convert integers (handle NaNs with Int64)
        for col in INT_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Pandas Error on {os.path.basename(csv_path)}: {e}")

def process_single_raster(raster_info):
    """Process one raster into a station-level CSV of zonal statistics.

    Args:
        raster_info: Tuple of (input_raster, output_csv).

    Returns:
        str: Short status message indicating created output or processing error.

    Notes:
        The workflow computes polygon-based zonal statistics over station buffer
        zones, then separately extracts the raster value at each station point
        into a ``point_prob`` field.
    """
    input_raster, output_csv = raster_info
    
    # Ensure directory exists (multiprocessing safe)
    dest_folder = os.path.dirname(output_csv)
    if not os.path.exists(dest_folder):
        try:
            os.makedirs(dest_folder)
        except: pass

    try:
        pid = os.getpid()
        temp_table = f"memory\\stats_{pid}"
        # temp_table = f"stats_{pid}"
        temp_view = f"view_{pid}"

        # 1. Compute zonal statistics over each station buffer polygon.
        arcpy.sa.ZonalStatisticsAsTable(
            in_zone_data=ZONE_DATA,
            zone_field=ZONE_FIELD,
            in_value_raster=input_raster,
            out_table=temp_table,
            statistics_type="ALL",
            percentile_values=[90]
        )

        # 2. Join station-zone metadata back onto the zonal statistics table.
        arcpy.management.MakeTableView(temp_table, temp_view)
        arcpy.management.AddJoin(temp_view, ZONE_FIELD, ZONE_DATA, ZONE_FIELD)
        
        # 3. Export joined results to CSV while suppressing ArcPy shape warnings.
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            arcpy.conversion.ExportTable(temp_view, output_csv)
        
        # Cleanup arcpy objects immediately
        arcpy.management.Delete(temp_table)
        arcpy.management.Delete(temp_view)
        
        # 3.5 Sample the raster at station-point locations for direct point value.
        spatial_ref = arcpy.SpatialReference(4326)
        memory_fc = f"memory\\temp_pts_{pid}"
        # memory_fc = f"temp_pts_{pid}"
        
        arcpy.management.XYTableToPoint(
            in_table=output_csv,
            out_feature_class=memory_fc,
            x_field="lon",
            y_field="lat",
            z_field="",
            coordinate_system=spatial_ref
        )
        
        arcpy.sa.ExtractMultiValuesToPoints(
            in_point_features=memory_fc,
            in_rasters=[[input_raster, "point_prob"]],
            bilinear_interpolate_values="NONE" 
        )
        
        original_df = pd.read_csv(output_csv)
        arr = arcpy.da.FeatureClassToNumPyArray(memory_fc, ["point_prob"])
        extracted_df = pd.DataFrame(arr)
        original_df["point_prob"] = extracted_df["point_prob"]
        original_df.to_csv(output_csv, index=False)
        arcpy.management.Delete(memory_fc)

        # 4. Standardize final CSV schema and numeric formatting.
        clean_csv(output_csv)

        # Remove the XML sidecar file arcpy generates
        xml_file = output_csv + ".xml"
        if os.path.exists(xml_file):
            os.remove(xml_file)

        return f"[CREATED] {os.path.basename(output_csv)}"

    except Exception as e:
        return f"[ERROR] {os.path.basename(input_raster)}: {str(e)}"

def main():
    """Discover matching rasters, dispatch multiprocessing work, and clean output.

    This is the batch entrypoint for the script. It builds a task list from the
    configured search filters, runs worker processes with process recycling to
    reduce ArcPy memory growth, and removes generated schema.ini files after all
    CSV exports finish.
    """
    start_time = time.time()
    tasks = []

    print("--- Searching for GeoTIFFs ---")
    for year in YEARS:
        for month in MONTHS:
            month_dir = os.path.join(INPUT_ROOT, year, month)
            if not os.path.exists(month_dir):
                continue
            
            for root, dirs, files in os.walk(month_dir):
                path_parts = os.path.normpath(root).split(os.sep)
                
                # Only keep folders matching the configured cycle and threshold.
                is_valid_cycle = any(c in path_parts for c in CYCLES)
                is_valid_threshold = any(t in path_parts for t in THRESHOLDS)

                if is_valid_cycle and is_valid_threshold:
                    for file in files:
                        if file.lower().endswith(".tif"):
                            in_path = os.path.join(root, file)
                            
                            # Mirror the raster folder layout beneath OUTPUT_ROOT.
                            rel_path = os.path.relpath(root, INPUT_ROOT)
                            out_csv = os.path.join(OUTPUT_ROOT, rel_path, f"{os.path.splitext(file)[0]}.csv")
                            
                            if PROCESS_AGAIN or not os.path.exists(out_csv):
                                tasks.append((in_path, out_csv))

    total_files = len(tasks)
    if total_files == 0:
        print("No new files found to process. Exiting.")
        return

    print(f"Total tasks to process: {total_files}")
    
    # Refresh workers periodically to limit long-run ArcPy memory growth.
    num_workers = max(1, cpu_count() - 1)
    pool = Pool(processes=num_workers, initializer=worker_init, maxtasksperchild=20)
    
    try:
        # Stream completions as workers finish instead of waiting for order.
        for i, result in enumerate(pool.imap_unordered(process_single_raster, tasks), 1):
            sys.stdout.write(f"\rProgress: {i}/{total_files} | {result}")
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        print("\n\n!!! Termination signal (Ctrl+C) detected. Killing workers... !!!")
        pool.terminate()
        pool.join()
        sys.exit(1)
        
    else:
        pool.close()
        pool.join()
        
    print("\n--- Cleaning up schema.ini files ---")
    deleted_schema_count = 0
    for root_dir, _, files in os.walk(OUTPUT_ROOT):
        if "schema.ini" in files:
            schema_path = os.path.join(root_dir, "schema.ini")
            try:
                os.remove(schema_path)
                deleted_schema_count += 1
            except Exception:
                pass
    if deleted_schema_count > 0:
        print(f"Removed {deleted_schema_count} schema.ini files.")
        
    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f"\n\n--- Processing Complete ---")
    print(f"Total Time: {duration:.2f} minutes")
    print(f"Average: {((end_time - start_time) / total_files):.2f} seconds per file")

if __name__ == "__main__":
    main()