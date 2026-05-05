"""Collect ASOS METAR observations over CONUS and flag thunder/lightning mentions.

Overview:
- Queries station metadata for each CONUS state ASOS network from the IEM Mesonet API.
- Downloads METAR observations for a user-provided date range.
- Adds two binary flags derived from METAR text:
    - ``observed_thunder``: thunder-related weather codes (TS, VCTS, etc.)
    - ``observed_lightning``: explicit lightning tokens (LTG...)
- Appends standardized rows to a single CSV output file.

Expected date format:
- ``YYYY-MM-DD`` for both start and end dates.
"""

import requests
import pandas as pd
import io
import time
import os

def get_network_stations(network):
    """Return active station IDs and metadata for one IEM network.

    Args:
        network: Network identifier such as ``IA_ASOS``.

    Returns:
        tuple[list[str], pandas.DataFrame | None]:
            - Station IDs used in the METAR request.
            - Station metadata with coordinates and elevation, or ``None`` on failure.
    """
    url = f"https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        station_ids = []
        metadata = []
        for feat in data['features']:
            st_id = feat['id']
            station_ids.append(st_id)
            
            geom = feat.get('geometry', {}).get('coordinates', [None, None])
            lon, lat = geom[0], geom[1]
            elev_m = feat['properties'].get('elevation', 0.0)
            elev_ft = elev_m * 3.28084 if elev_m is not None else 0.0
            
            metadata.append({
                'station': st_id,
                'station_name': feat['properties'].get('sname', ''),
                'lat': round(lat, 4) if lat else None,
                'lon': round(lon, 4) if lon else None,
                'elevation_m': round(elev_m, 1) if elev_m else None,
                'elevation_ft': round(elev_ft, 1) if elev_ft else None
            })
            
        return station_ids, pd.DataFrame(metadata)
    except Exception as e:
        print(f"Failed to fetch stations for {network}: {e}")
        return [], None

def download_conus_thunder_data(start_date, end_date, output_csv):
    """Download METAR data for CONUS ASOS networks and append flagged observations.

    The function loops through a fixed list of CONUS states, requests each state's
    ASOS METAR feed for the provided date range, computes thunder/lightning flags,
    merges station metadata, then appends data to ``output_csv``.

    Args:
        start_date: Inclusive start date in ``YYYY-MM-DD`` format.
        end_date: Inclusive end date in ``YYYY-MM-DD`` format.
        output_csv: Target CSV path. Created if missing, otherwise appended.
    """
    s_year, s_month, s_day = start_date.split('-')
    e_year, e_month, e_day = end_date.split('-')
    
    # conus_states = ['AZ', 'CA'] # Testing list
    conus_states = [
        'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 
        'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 
        'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 
        'WV', 'WI', 'WY'
    ]
    
    print(f"Starting CONUS extraction from {start_date} to {end_date}...")
    base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    
    total_obs = 0
    total_ts_hits = 0
    total_ltg_hits = 0
    
    # --- REGEX PATTERNS ---
    # 1. Strict Thunder (TS, VCTS, TSB, TSE)
    thunder_regex = r'(?:\s|^)[+-]?(?:VC)?TS[A-Z]*\b|\bTS[BE]\d{4}(?:[BE]\d{4})?\b'
    # 2. Lightning Codes (LTG, LTGIC, LTG DSNT, etc.)
    lightning_regex = r'\bLTG[A-Z]*\b'

    for state in conus_states:
        network = f"{state}_ASOS"
        print(f"\n--- Processing {network} ---")
        
        stations, metadata_df = get_network_stations(network)
        if not stations:
            continue
            
        print(f"Found {len(stations)} stations. Downloading METARs...")
        
        params = {
            'network': network,
            'station': ",".join(stations),
            'tz': 'Etc/UTC',
            'data': 'metar',
            'year1': s_year, 'month1': s_month, 'day1': s_day,
            'year2': e_year, 'month2': e_month, 'day2': e_day,
            'format': 'onlycomma'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=120)
            response.raise_for_status()
            
            df = pd.read_csv(io.StringIO(response.text), comment='#')
            
            if df.empty or 'metar' not in df.columns:
                print("No data found.")
                continue
                
            # --- THE NEW VECTORIZED LOGIC ---
            # 1. Check for Thunder (1 or 0)
            df['observed_thunder'] = df['metar'].str.contains(thunder_regex, regex=True, na=False).astype(int)
            
            # 2. Check for Lightning (1 or 0)
            df['observed_lightning'] = df['metar'].str.contains(lightning_regex, regex=True, na=False).astype(int)
            
            # 3. Merge metadata
            df = df.merge(metadata_df, on='station', how='left')
            
            # 4. Enforce exact column order
            final_cols = ['station', 'valid', 'metar', 'observed_thunder', 'observed_lightning',
                          'station_name', 'lat', 'lon', 'elevation_m', 'elevation_ft']
            
            for col in final_cols:
                if col not in df.columns:
                    df[col] = ''
            df = df[final_cols]
            
            # Save to CSV
            write_header = not os.path.exists(output_csv)
            df.to_csv(output_csv, mode='a', index=False, header=write_header)
            
            # Tally up the stats for the printout
            obs_count = len(df)
            ts_hits = df['observed_thunder'].sum()
            ltg_hits = df['observed_lightning'].sum()
            
            total_obs += obs_count
            total_ts_hits += ts_hits
            total_ltg_hits += ltg_hits
            
            print(f"Total Obs: {obs_count:,} | Thunder: {ts_hits:,} | Lightning: {ltg_hits:,}")
            
        except Exception as e:
            print(f"Error: {e}")
            
        time.sleep(2)
        
    print(f"\n✅ Complete! Extracted {total_obs:,} total observations.")
    print(f"Total Thunder Hits: {total_ts_hits:,}")
    print(f"Total Lightning Hits: {total_ltg_hits:,}")
    print(f"Data saved to: {output_csv}")

if __name__ == "__main__":
    
    # Example invocation. Edit dates/path as needed for batch collection windows.
    START = "2023-02-01"
    END = "2023-02-28"
    SAVE_PATH = r"C:\Users\chad.kahler\Documents\ArcGIS\Projects\NBM_Verif\Data\prob_thunder\metar\conus_obs_2023_02.csv"
    
    download_conus_thunder_data(START, END, SAVE_PATH)