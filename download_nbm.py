"""Download NBM CONUS thunder probability TIFFs from NOAA public S3 storage.

Overview:
- Iterates daily initialization cycles over a user-defined date range.
- Detects the operational NBM version by run date (4.1, 4.2, or 4.3).
- Scans S3 prefixes for selected thunder probability elements (for example,
    tstm01/tstm03/tstm06/tstm12).
- Computes forecast lead time from file valid-end timestamp and keeps files up
    to forecast hour 174.
- Saves files in a date/init/element folder layout and renames each output to a
    standardized FXXX suffix format.

Expected inputs:
- start_date_str/end_date_str: YYYY-MM-DD
- init_hours: list like ["0100", "1300"]
- target_elements: list like ["tstm12"]
"""

import boto3
from botocore.config import Config
from botocore import UNSIGNED
import os
import datetime
from datetime import timedelta
import concurrent.futures

def get_nbm_version(target_date):
    """Return the operational NBM version string for a model run date.

    Args:
            target_date: Date of the initialization cycle.

    Returns:
            str: One of "4.1", "4.2", or "4.3".
    """
    v4_2_date = datetime.date(2024, 5, 15)
    v4_3_date = datetime.date(2025, 5, 27)
    
    if target_date >= v4_3_date:
        return "4.3"
    elif target_date >= v4_2_date:
        return "4.2"
    else:
        return "4.1"

def download_s3_file(s3_client, bucket, key, local_path):
    """Download one S3 object to disk if it is not already present.

    Returns a status string used by the caller for progress/error reporting.
    """
    if os.path.exists(local_path):
        return f"Already exists: {local_path}"
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        s3_client.download_file(bucket, key, local_path)
        return f"Downloaded: {local_path}"
    except Exception as e:
        return f"Error downloading {key}: {e}"

def fetch_nbm_f174_renamed(start_date_str, end_date_str, save_dir, target_elements, init_hours):
    """Fetch and rename NBM thunder-probability TIFFs through forecast hour 174.

    For each date, initialization time, and target element, this function lists
    matching S3 objects, parses valid-end timestamps from filenames, computes
    lead time relative to initialization time, and downloads files whose lead
    time is <= 174 hours.

    Downloaded files are renamed to:
        blendv{version}_conus_{element}_{init}_F{forecast_hour:03d}.tif

    Args:
        start_date_str: Inclusive start date in YYYY-MM-DD format.
        end_date_str: Inclusive end date in YYYY-MM-DD format.
        save_dir: Root local output directory.
        target_elements: Element folders to request (for example, tstm12).
        init_hours: Initialization times in HHMM format.
    """
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_name = 'noaa-nbm-pds'
    
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    
    print(f"Starting NBM download from {start_date} to {end_date}...")
    print(f"Targeting: {target_elements} out to Forecast Hour 174")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        
        current_date = start_date
        while current_date <= end_date:
            version = get_nbm_version(current_date)
            date_str = current_date.strftime("%Y/%m/%d") 
            
            for init_hhmm in init_hours:
                # Build initialization datetime used for forecast-hour calculation.
                init_dt = datetime.datetime.strptime(f"{current_date.strftime('%Y%m%d')}{init_hhmm}", "%Y%m%d%H%M")
                
                for element in target_elements:
                    prefix = f"blendv{version}/conus/{date_str}/{init_hhmm}/{element}/"
                    
                    try:
                        paginator = s3.get_paginator('list_objects_v2')
                        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
                        
                        for page in pages:
                            if 'Contents' not in page:
                                continue
                                
                            for obj in page['Contents']:
                                key = obj['Key']
                                if not key.endswith('.tif'):
                                    continue
                                
                                filename = os.path.basename(key)
                                
                                try:
                                    # Filename contains init and valid-end timestamps.
                                    # Example: blendv4.1_conus_tstm12_2023-03-01T13:00_2023-03-07T12:00.tif
                                    end_time_str = filename.replace('.tif', '').split('_')[-1]
                                    valid_end_dt = datetime.datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M")
                                    
                                    # Convert valid-end minus init time to integer forecast hour.
                                    time_diff = valid_end_dt - init_dt
                                    forecast_hour = int(time_diff.total_seconds() / 3600)
                                    
                                    # Keep files through F174 only.
                                    if forecast_hour <= 174:
                                        
                                        # Standardized local filename with explicit lead time tag.
                                        clean_init_str = init_dt.strftime("%Y-%m-%dT%H%M")
                                        new_filename = f"blendv{version}_conus_{element}_{clean_init_str}_F{forecast_hour:03d}.tif"
                                        
                                        local_file_path = os.path.join(
                                            save_dir, 
                                            str(current_date.year), 
                                            f"{current_date.month:02d}", 
                                            f"{current_date.day:02d}", 
                                            init_hhmm,
                                            element, 
                                            new_filename
                                        )
                                        
                                        futures.append(executor.submit(download_s3_file, s3, bucket_name, key, local_file_path))
                                        
                                except Exception as parse_e:
                                    print(f"Skipping unparseable file {filename}: {parse_e}")
                                    
                    except Exception as e:
                        print(f"Error listing {prefix}: {e}")
                        
            current_date += timedelta(days=1)
            
        total_files = len(futures)
        print(f"\nFound {total_files} files within F174. Beginning download...")
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            
            result_msg = future.result()
            if "Error" in result_msg:
                print(f"❌ {result_msg}")
                
            if completed % 100 == 0: 
                print(f"Processed {completed} of {total_files} files...")

    print("\n✅ Download complete! Files are capped at F174 and renamed to standardized FXXX format.")

# Example usage. Edit date range, elements, init hours, and destination as needed.
if __name__ == "__main__":
    START_DATE = "2025-03-01"
    END_DATE = "2025-10-31" 
    
    TARGET_ELEMENTS = ['tstm01', 'tstm03', 'tstm06', 'tstm12'] # You can add 'tstm01', 'tstm03', etc.
    INIT_HOURS = ['0100', '1300']
    
    SAVE_DIRECTORY = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\prob_thunder"
    
    fetch_nbm_f174_renamed(START_DATE, END_DATE, SAVE_DIRECTORY, TARGET_ELEMENTS, INIT_HOURS)