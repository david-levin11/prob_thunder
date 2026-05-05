#!/home/chad.kahler/anaconda3/envs/verif/bin/python

"""Dilate binary NLDN raster grids by a configurable radial distance.

Overview:
- Reads binary lightning rasters from configured subfolders under base_in.
- Computes a circular footprint using grid spacing and radius_km.
- Applies a maximum filter so any lightning-hit cell expands to nearby cells
  within the requested radius.
- Preserves raster metadata and CRS when writing output files.

Output layout:
- For each input subfolder (for example, nldn_06), results are written to a
  sibling output folder named <subfolder>_20km beneath base_out.
- Output filenames match the source filenames.
"""

import os
from pathlib import Path
import rioxarray as rxr
import numpy as np
from scipy import ndimage
import xarray as xr

def process_lightning_batch(base_in, base_out, radius_km=20, max_files=None):
    """Process NLDN raster folders and write radius-dilated versions.

    Args:
        base_in: Root directory containing source folders such as nldn_06 and
            nldn_12 with GeoTIFF raster files.
        base_out: Root directory where output folders are created.
        radius_km: Dilation radius in kilometers. Converted to pixels from
            raster x-resolution.
        max_files: Optional cap on total files processed across all folders.
            Use None to process every file found.

    Notes:
        The routine assumes projected raster units are meters (for example,
        LCC grids) when converting kilometers to pixel distance.
    """
    # Define the subdirectories we want to process
    subdirs = ['nldn_06', 'nldn_12']
    # subdirs = ['nldn_06']
    processed_files = 0
    
    for sub in subdirs:
        input_path = Path(base_in) / sub
        output_path = Path(base_out) / f"{sub}_20km"
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all .tif files
        tif_files = sorted(input_path.glob("*.tif"))
        print(f"Processing {len(tif_files)} files in {sub}...")
        
        for file in tif_files:
            if max_files is not None and processed_files >= max_files:
                print(f"Reached test limit ({max_files} file(s)). Stopping.")
                return

            try:
                # 1. Load the truth grid
                ds = rxr.open_rasterio(file, mask_and_scale=True)
                
                # 2. Calculate pixel radius for 20km
                # Assumes LCC units are in meters
                res_x = abs(ds.rio.resolution()[0])
                radius_px = int(round((radius_km * 1000.0) / res_x))
                
                # 3. Create circular footprint
                y, x = np.ogrid[-radius_px : radius_px+1, -radius_px : radius_px+1]
                footprint = x**2 + y**2 <= radius_px**2
                
                # 4. Apply Maximum Filter
                # We handle the data as a numpy array, then put it back in Xarray
                data = np.nan_to_num(ds.values[0], nan=0)
                dilated_data = ndimage.maximum_filter(data, footprint=footprint)
                
                # 5. Rebuild as DataArray to keep metadata/CRS
                output_da = xr.DataArray(
                    dilated_data[np.newaxis, ...], # Maintain (band, y, x) shape
                    coords=ds.coords,
                    dims=ds.dims,
                    attrs=ds.attrs
                )
                output_da.rio.write_crs(ds.rio.crs, inplace=True)
                
                # 6. Save to new location
                out_file = output_path / file.name
                output_da.rio.to_raster(out_file, compress='lzw')
                
                print(f"  Done: {file.name}")
                processed_files += 1
                
            except Exception as e:
                print(f"  Error processing {file.name}: {e}")

# --- Configuration ---
input_base_dir = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\nldn\nldn_rasters"
output_base_dir = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\nldn\nldn_rasters" 
MAX_FILES = None  # Set to None to process all files

process_lightning_batch(input_base_dir, output_base_dir, max_files=MAX_FILES)