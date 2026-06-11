import numpy as np
import rioxarray as rxr

path = r"C:\Users\David.Levin\NBMLightningVer\aicc_lightning\rasters_total\aicc_total_12_20km\aicc_total_12h_20240619_0000Z.tif"

ds = rxr.open_rasterio(path, mask_and_scale=True)
arr = ds.values[0]

print("Shape:", arr.shape)
print("Min:", np.nanmin(arr))
print("Max:", np.nanmax(arr))
print("NaN pixels:", np.count_nonzero(np.isnan(arr)))
print("Zero pixels:", np.count_nonzero(arr == 0))
print("Lightning pixels:", np.count_nonzero(arr == 1))
print("Unique finite values:", np.unique(arr[np.isfinite(arr)]))