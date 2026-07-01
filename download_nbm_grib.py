import os
import re
import requests


search_string = ":TSTM:surface:24-36 hour acc fcst"

remote_url = "https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.20260617/12/core/blend.t12z.core.f036.ak.grib2"

local_filename = os.path.basename(remote_url)

def download_subset(remote_url, local_filename, search_string):
  print("   > Downloading a subset of NBM gribs")
  local_file = local_filename
  
  idx = remote_url+".idx"
  print("IDX file = " + idx)
  r = requests.get(idx)
  if not r.ok:
    print('     ❌ SORRY! Status Code:', r.status_code, r.reason)
    print(f'      ❌ It does not look like the index file exists: {idx}')

  lines = r.text.split('\n')
  expr = re.compile(search_string)
  expr
  byte_ranges = {}
  for n, line in enumerate(lines, start=1):
    # n is the line number (starting from 1) so that when we call for
    # `lines[n]` it will give us the next line. (Clear as mud??)
    # Use the compiled regular expression to search the line
    #print(">> Searching throgh this line: " + line)
    if expr.search(line):
      # aka, if the line contains the string we are looking for...
      # Get the beginning byte in the line we found
      parts = line.split(':')
      rangestart = int(parts[1])
      # Get the beginning byte in the next line...
      if n+1 < len(lines):
        # ...if there is a next line
        parts = lines[n].split(':')
        rangeend = int(parts[1])
      else:
        # ...if there isn't a next line, then go to the end of the file.
        rangeend = ''

        # Store the byte-range string in our dictionary,
        # and keep the line information too so we can refer back to it.
      byte_ranges[f'{rangestart}-{rangeend}'] = line
      #print(line)
    #else:
      #print(">>>  Could not find search string!")
  #print(">>  Number of items in byteRange:" + str(len(byte_ranges)))
  for i, (byteRange, line) in enumerate(byte_ranges.items()):

    if i == 0:
      # If we are working on the first item, overwrite the existing file.
      curl = f'curl -s --range {byteRange} {remote_url} > {local_file}'
      #print(">>  Adding curl command: " + curl)
    else:
      # If we are working on not the first item, append the existing file.
      curl = f'curl -s --range {byteRange} {remote_url} >> {local_file}'
      #print("Adding curl command: " + curl)
    #print('>>  Parsing line: ' + line)
    try:
      num, byte, date, var, level, forecast, _ = line.split(':')
    except:
      pass
      #print(">>>  Can't get num/byte/etc from this line, so skipping...")

    #print(f'  Downloading GRIB line [{num:>3}]: variable={var}, level={level}, forecast={forecast}')
    #print(f'  Downloading GRIB line: variable={var}, level={level}, forecast={forecast}')
    #print("Running the curl command...")
    os.system(curl)

  if os.path.exists(local_file):
    print(f'      ✅ Success! Searched for [{search_string}] and got [{len(byte_ranges)}] GRIB fields and saved as {local_file}')
    return local_file
  else:
    print(print(f'      ❌ Unsuccessful! Searched for [{search_string}] and did not find anything!'))

if __name__ == "__main__":
  download_subset(remote_url, local_filename, search_string)