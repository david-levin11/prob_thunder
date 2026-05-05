"""Build a master station metadata list from monthly CONUS METAR CSV files.

Overview:
- Scans a target directory for files matching ``conus_obs_*.csv``.
- Reads station-level metadata columns from each file.
- De-duplicates stations within each file and again across all files.
- Writes a single station catalog to ``stations.csv`` in the same directory.

Input requirements:
- Source files must include: ``station``, ``station_name``, ``lat``, ``lon``,
  and ``elevation_ft`` columns.
"""

import pandas as pd
import glob
import os

# Source directory and file matching pattern.
directory = r'C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\prob_thunder\metar'
pattern = 'conus_obs_*.csv'
output_file = os.path.join(directory, 'stations.csv')

# Metadata columns retained in the final station list.
metadata_cols = ['station', 'station_name', 'lat', 'lon', 'elevation_ft']

def build_station_list():
    """Create a deduplicated station metadata table from matched METAR files.

    The function reads only metadata columns for efficiency, then combines all
    results and keeps one row per station identifier.
    """
    all_stations = []
    
    # Search for files matching the pattern
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    
    print(f"Found {len(files)} files. Starting processing...")

    for file in files:
        try:
            # Read only the metadata columns to save memory
            # usecols speeds up the process significantly
            df = pd.read_csv(file, usecols=metadata_cols)
            
            # Drop duplicates within the current file immediately
            df_unique = df.drop_duplicates(subset=['station'])
            all_stations.append(df_unique)
            print(f"Processed: {os.path.basename(file)}")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Combine per-file station tables into one frame.
    master_df = pd.concat(all_stations, ignore_index=True)

    # A station can appear in multiple monthly files; keep first instance found.
    master_df = master_df.drop_duplicates(subset=['station'])

    # Write consolidated station metadata.
    master_df.to_csv(output_file, index=False)
    print(f"Success! Master station list saved to: {output_file}")
    print(f"Total unique stations found: {len(master_df)}")

if __name__ == "__main__":
    build_station_list()