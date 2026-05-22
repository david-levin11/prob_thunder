#!/usr/bin/env python

"""Dilate binary AICC lightning truth rasters by a configurable radius.

Input raster values:
    0   = no lightning
    1   = lightning
    255 = NoData / outside Alaska

Output raster values:
    0   = no lightning
    1   = lightning after dilation
    255 = NoData / outside Alaska

The dilation expands lightning-hit cells outward by radius_km using a circular
footprint based on the raster grid spacing.
"""

from pathlib import Path

import numpy as np
import rioxarray as rxr
import rasterio
from scipy import ndimage


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_IN = Path(
    r"C:\Users\David.Levin\NBMLightningVer\aicc_lightning\rasters_total"
)

BASE_OUT = Path(
    r"C:\Users\David.Levin\NBMLightningVer\aicc_lightning\rasters_total_dilated"
)

# Update these to match your actual raw AICC raster folders.
INPUT_SUBDIRS = [
    "aicc_total_06_20km",
    "aicc_total_12_20km",
]

RADIUS_KM = 20

INPUT_NODATA = 255
OUTPUT_NODATA = 255

SKIP_EXISTING = True

# Set to an integer for testing, e.g. 10. Use None for all files.
MAX_FILES = None


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def circular_footprint(radius_px):
    """Create a circular boolean footprint with the requested pixel radius."""

    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    return (x**2 + y**2) <= radius_px**2


def get_radius_pixels(ds, radius_km):
    """Convert dilation radius from km to pixels using raster resolution."""

    res_x, res_y = ds.rio.resolution()

    # Pixel size should be in projected meters.
    # Use the average of x/y resolution in case they differ slightly.
    pixel_size_m = (abs(res_x) + abs(res_y)) / 2.0

    radius_px = int(round((radius_km * 1000.0) / pixel_size_m))

    if radius_px < 1:
        radius_px = 1

    return radius_px, pixel_size_m


def write_uint8_raster_like(ds, arr, out_path):
    """Write uint8 GeoTIFF using metadata from input raster."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not SKIP_EXISTING:
        try:
            out_path.unlink()
        except PermissionError as e:
            raise PermissionError(
                f"Cannot overwrite {out_path}. Close ArcPro/QGIS/Python viewers "
                "or write to a new output directory."
            ) from e

    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": ds.rio.crs,
        "transform": ds.rio.transform(),
        "nodata": OUTPUT_NODATA,
        "compress": "deflate",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype("uint8"), 1)


def dilate_one_file(in_path, out_path, radius_km):
    """Dilate one binary lightning raster."""

    if SKIP_EXISTING and out_path.exists():
        print(f"  Skipping existing: {out_path.name}")
        return False

    with rxr.open_rasterio(in_path, mask_and_scale=False) as ds:
        arr = ds.values[0].astype("uint8")

        radius_px, pixel_size_m = get_radius_pixels(ds, radius_km)
        footprint = circular_footprint(radius_px)

        nodata_mask = arr == INPUT_NODATA
        lightning_mask = arr == 1

        # Binary dilation expands only lightning pixels.
        dilated = ndimage.binary_dilation(
            lightning_mask,
            structure=footprint,
        )

        out = np.zeros(arr.shape, dtype="uint8")
        out[dilated] = 1

        # Preserve NoData outside Alaska.
        out[nodata_mask] = OUTPUT_NODATA

        write_uint8_raster_like(ds, out, out_path)

    lightning_before = int(np.count_nonzero(lightning_mask))
    lightning_after = int(np.count_nonzero(out == 1))
    nodata_pixels = int(np.count_nonzero(out == OUTPUT_NODATA))

    print(
        f"  Done: {in_path.name} | "
        f"radius={radius_km} km ≈ {radius_px} px "
        f"(pixel={pixel_size_m:.1f} m) | "
        f"lightning {lightning_before:,} -> {lightning_after:,} pixels | "
        f"nodata={nodata_pixels:,}"
    )

    return True


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def process_batch():
    """Dilate all configured AICC truth-raster folders."""

    BASE_OUT.mkdir(parents=True, exist_ok=True)

    processed = 0

    for subdir in INPUT_SUBDIRS:
        input_dir = BASE_IN / subdir

        # Example:
        # input:  aicc_total_06_20km
        # output: aicc_total_06_20km_dilated
        output_dir = BASE_OUT / f"{subdir}_dilated"

        if not input_dir.exists():
            print(f"Input directory not found: {input_dir}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(input_dir.glob("*.tif"))

        print(f"\nProcessing {len(tif_files):,} files in {input_dir}")

        for in_path in tif_files:
            if MAX_FILES is not None and processed >= MAX_FILES:
                print(f"Reached MAX_FILES={MAX_FILES}. Stopping.")
                return

            out_path = output_dir / in_path.name

            try:
                wrote = dilate_one_file(
                    in_path=in_path,
                    out_path=out_path,
                    radius_km=RADIUS_KM,
                )

                if wrote:
                    processed += 1

            except Exception as e:
                print(f"  Error processing {in_path.name}: {e}")

    print(f"\nFinished. Files processed: {processed:,}")


if __name__ == "__main__":
    process_batch()