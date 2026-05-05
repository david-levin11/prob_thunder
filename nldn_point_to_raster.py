"""Convert NLDN point CSV files into binary rasters on the NBM verification grid.

Overview:
- Reads lightning-point CSV files from one or more configured input folders.
- Detects longitude and latitude field names from common header variants.
- Converts each CSV to an in-memory point feature class.
- Rasterizes point presence onto the NBM template grid using ArcPy environment
    settings for snap raster, extent, cell size, and output projection.
- Saves a binary raster where 1 indicates at least one lightning point in a
    grid cell and 0 indicates no lightning point in that cell.

Output layout:
- Rasters are written beneath RASTER_OUTPUT_BASE with subfolders matching the
    input folder names, such as nldn_06_obs and nldn_12_obs.
"""

import arcpy
import os
import glob

# --- Configuration ---
# Path to project database
arcpy.env.workspace = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\NBM_Verif.gdb"

# Path to an existing NBM file (GRIB or TIFF) to act as the grid template
NBM_TEMPLATE = arcpy.Raster("NBM_JFWPRB")

# Directories we created in the previous step
CSV_FOLDERS = [
    r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\nldn\nldn_06_obs",
    r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\nldn\nldn_12_obs"
]

# Where to save the final rasters
RASTER_OUTPUT_BASE = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\nldn\nldn_rasters"

# Set to an integer (e.g., 1) to process only that many files total, then exit.
# Keep as None for normal full processing.
MAX_FILES_TO_PROCESS = None

# --- Environment Setup ---
arcpy.env.overwriteOutput = True
arcpy.env.snapRaster = NBM_TEMPLATE
arcpy.env.cellSize = NBM_TEMPLATE
arcpy.env.extent = NBM_TEMPLATE
arcpy.env.outputCoordinateSystem = arcpy.Describe(NBM_TEMPLATE).spatialReference

def _normalize_field_name(name):
    """Normalize CSV header text for flexible longitude/latitude matching."""
    # Handle case differences, whitespace, and UTF-8 BOM in CSV headers.
    return name.lstrip("\ufeff").strip().lower()

def _get_xy_fields(table_path):
    """Return detected longitude and latitude field names for a table.

    Args:
        table_path: Path to the CSV or table to inspect.

    Returns:
        tuple[str, str]: The source x-field and y-field names.

    Raises:
        RuntimeError: If no recognizable longitude/latitude headers are found.
    """
    fields = [f.name for f in arcpy.ListFields(table_path)]
    normalized = {_normalize_field_name(name): name for name in fields}

    x_candidates = ["lon", "longitude", "long", "x", "xcoord", "x_coord"]
    y_candidates = ["lat", "latitude", "y", "ycoord", "y_coord"]

    x_field = next((normalized[c] for c in x_candidates if c in normalized), None)
    y_field = next((normalized[c] for c in y_candidates if c in normalized), None)

    if x_field is None or y_field is None:
        raise RuntimeError(
            f"Could not find longitude/latitude fields in {os.path.basename(table_path)}. "
            f"Available fields: {fields}"
        )

    return x_field, y_field

def csv_to_binary_raster():
    """Convert all configured NLDN CSV files into binary rasters.

    The function preserves a simple folder mapping from input CSV directory to
    output raster directory, processes each CSV independently, and cleans up
    temporary in-memory ArcPy datasets after each file.
    """
    if not os.path.exists(RASTER_OUTPUT_BASE):
        os.makedirs(RASTER_OUTPUT_BASE)

    attempted_count = 0

    for folder in CSV_FOLDERS:
        # Use the source folder name to organize output rasters by interval bucket.
        suffix = os.path.basename(folder)
        if suffix.endswith("_obs"):
            suffix = suffix[:-4]
        out_subfolder = os.path.join(RASTER_OUTPUT_BASE, suffix)
        if not os.path.exists(out_subfolder):
            os.makedirs(out_subfolder)

        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        print(f"Processing {len(csv_files)} files in {suffix}...")

        for csv_path in csv_files:
            attempted_count += 1
            file_name = os.path.basename(csv_path).replace(".csv", "")
            temp_points = "memory\\temp_points"
            temp_ras = "memory\\temp_ras"
            out_raster = os.path.join(out_subfolder, f"{file_name}.tif")

            try:
                lon_field, lat_field = _get_xy_fields(csv_path)

                # 1. Load CSV as Point Feature Class (Temporary)
                arcpy.management.XYTableToPoint(
                    csv_path,
                    temp_points,
                    lon_field,
                    lat_field,
                    None,
                    arcpy.SpatialReference(4326)
                )

                # Use any unique point identifier because only point presence matters.
                oid_field = arcpy.Describe(temp_points).OIDFieldName
                arcpy.conversion.PointToRaster(
                    temp_points,
                    oid_field,
                    temp_ras,
                    "MOST_FREQUENT",
                    "NONE",
                    arcpy.env.cellSize
                )

                # Convert populated cells to 1 and empty cells to 0.
                binary_ras = arcpy.sa.Con(arcpy.sa.IsNull(arcpy.Raster(temp_ras)), 0, 1)

                # Save the final binary raster to disk.
                binary_ras.save(out_raster)
                print(f"   [✓] Created: {file_name}.tif")

                if MAX_FILES_TO_PROCESS is not None and attempted_count >= MAX_FILES_TO_PROCESS:
                    print(f"Reached test limit ({MAX_FILES_TO_PROCESS} file). Exiting early.")
                    return

            except Exception as e:
                print(f"   [!] Failed to process {file_name}: {e}")
            finally:
                # Clean up in-memory objects before moving to the next file.
                if arcpy.Exists(temp_points):
                    arcpy.management.Delete(temp_points)
                if arcpy.Exists(temp_ras):
                    arcpy.management.Delete(temp_ras)

            if MAX_FILES_TO_PROCESS is not None and attempted_count >= MAX_FILES_TO_PROCESS:
                print(f"Reached test limit ({MAX_FILES_TO_PROCESS} file). Exiting early.")
                return

if __name__ == "__main__":
    csv_to_binary_raster()
    print("Processing Complete.")
