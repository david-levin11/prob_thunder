"""Download archived monthly NLDN tar files from NOAA NCEI.

Overview:
- Builds monthly archive filenames for the configured year and month range.
- Requests each tar file from the restricted NCEI NLDN archive.
- Skips files already present locally unless re-download is enabled.
- Saves each successful download into a single local output directory.

Access note:
- The archive is under a restricted NCEI path. A 403 response usually means the
    required network access or credentials are not available in the current session.
"""

import os
import requests

# --- Configuration ---
BASE_URL = "https://www.ncei.noaa.gov/data/restricted/national-lightning-detection-network/archive/"
OUTPUT_DIR = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\nldn"

YEARS = [2023, 2024, 2025]
MONTHS = range(2, 11)  # March (3) thru October (10)

download_again = False  # Set True to retry files even if they already exist locally.

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

def download_nldn_data():
    """Download configured monthly NLDN archive tar files to OUTPUT_DIR.

    The function reuses a single requests session for efficiency, checks whether
    each target file is already present, and prints a simple status line for
    success, common HTTP failures, or unexpected request exceptions.
    """
    # Use a session for better performance if downloading many files
    with requests.Session() as session:
        # Note: If credentials are required, uncomment the line below:
        # session.auth = ('your_username', 'your_password')

        for year in YEARS:
            for month in MONTHS:
                # Monthly archive naming convention used by the NCEI NLDN endpoint.
                file_name = f"9603_{year}{month:02d}.tar"
                url = f"{BASE_URL}{file_name}"
                save_path = os.path.join(OUTPUT_DIR, file_name)

                if not download_again and os.path.exists(save_path):
                    print(f"File already exists: {file_name}. Skipping download.")
                    continue

                print(f"Attempting to download: {file_name}...", end=" ", flush=True)

                try:
                    response = session.get(url, stream=True, timeout=30)
                    
                    if response.status_code == 200:
                        # Stream to disk in chunks so large tar files are not held in memory.
                        with open(save_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print("Done.")
                    elif response.status_code == 404:
                        print("File not found (404).")
                    elif response.status_code == 403:
                        print("Access Forbidden (403). Ensure VPN is connected.")
                    else:
                        print(f"Failed (Status: {response.status_code}).")

                except Exception as e:
                    print(f"Error: {e}")

if __name__ == "__main__":
    download_nldn_data()