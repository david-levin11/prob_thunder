#!/usr/bin/env python

"""Create union lightning truth rasters from GLD and AICC dilated rasters.

Union rule:
    output = 1 if GLD == 1 OR AICC == 1
    output = 0 if GLD == 0 AND AICC == 0
    output = 255 where outside valid domain / NoData

Expected input values:
    0   = no lightning
    1   = lightning
    255 = NoData / outside Alaska

Output values:
    0   = no lightning in either dataset
    1   = lightning in either GLD or AICC
    255 = NoData

The script pairs rasters by valid time parsed from filenames like:
    gld_06h_20240619_0000Z.tif
    aicc_total_06h_20240619_0000Z.tif
"""

from pathlib import Path
import re

import numpy as np
import rasterio


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

GLD_BASE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\gld_rasters"
)

AICC_BASE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\aicc_lightning\rasters_total_dilated"
)

OUT_BASE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\union_lightning_rasters"
)

GLD_SUBDIRS = [
    "gld_06_20km",
    "gld_12_20km",
]

AICC_SUBDIRS = [
    "aicc_total_06_20km_dilated",
    "aicc_total_12_20km_dilated",
]

WINDOWS = [6, 12]

OUTPUT_PREFIX = "union"

INPUT_NODATA = 255
OUTPUT_NODATA = 255

SKIP_EXISTING = True
MAX_FILES = None

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def parse_window_and_valid_time(path):
    """
    Parse window and valid time from a raster filename.

    Handles examples like:
        gld_06h_20240619_0000Z.tif
        aicc_total_12h_20240619_0000Z.tif
        union_06h_20240619_0000Z.tif

    Returns:
        tuple[int, str] or None
        Example: (6, "20240619_0000Z")
    """

    match = re.search(r"_(\d{2})h_(\d{8}_\d{4}Z)\.tif$", path.name)

    if not match:
        return None

    window = int(match.group(1))
    valid_str = match.group(2)

    return window, valid_str


def index_rasters(base_dir, subdirs, windows=None):
    """
    Index rasters by (window, valid_time_string).

    Args:
        base_dir: root folder
        subdirs: list of subfolders to search
        windows: optional list of allowed window hours

    Returns:
        dict[(int, str), Path]
    """

    index = {}

    for subdir in subdirs:
        search_dir = base_dir / subdir

        if not search_dir.exists():
            print(f"WARNING: directory not found: {search_dir}")
            continue

        rasters = sorted(search_dir.glob("*.tif"))

        print(f"  {subdir}: found {len(rasters):,} rasters")

        for path in rasters:
            parsed = parse_window_and_valid_time(path)

            if parsed is None:
                continue

            window, valid_str = parsed

            if windows is not None and window not in windows:
                continue

            key = (window, valid_str)

            if key in index:
                print(f"WARNING: duplicate raster for {key}:")
                print(f"  Existing: {index[key]}")
                print(f"  New:      {path}")
                print("  Keeping existing.")
                continue

            index[key] = path

    return index


def read_binary_raster(path):
    """
    Read raster as raw uint8/array without masking.

    Returns:
        arr, profile
    """

    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()

    return arr, profile


def create_union_array(gld_arr, aicc_arr):
    """
    Create binary union array from GLD and AICC rasters.

    Uses NoData if both are NoData.
    If one dataset is valid and the other is NoData, use the valid dataset.
    """

    gld_nodata = gld_arr == INPUT_NODATA
    aicc_nodata = aicc_arr == INPUT_NODATA

    gld_yes = gld_arr == 1
    aicc_yes = aicc_arr == 1

    both_nodata = gld_nodata & aicc_nodata

    union = np.zeros(gld_arr.shape, dtype="uint8")

    # Lightning if either dataset observed lightning.
    union[gld_yes | aicc_yes] = 1

    # NoData only where both datasets are NoData.
    union[both_nodata] = OUTPUT_NODATA

    return union


def write_union_raster(out_path, arr, profile):
    """Write union raster using source raster metadata."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        if SKIP_EXISTING:
            return False

        try:
            out_path.unlink()
        except PermissionError as e:
            raise PermissionError(
                f"Cannot overwrite {out_path}. Close ArcPro/QGIS/Python viewers "
                "or write to a new output directory."
            ) from e

    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate",
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype("uint8"), 1)

    return True


def output_path(window, valid_str):
    """Return output path for union raster."""

    window_str = f"{window:02d}"

    out_dir = OUT_BASE / f"{OUTPUT_PREFIX}_{window_str}_20km"
    out_name = f"{OUTPUT_PREFIX}_{window_str}h_{valid_str}.tif"

    return out_dir / out_name


def print_counts(label, arr):
    """Print value counts for quick diagnostics."""

    n_yes = int(np.count_nonzero(arr == 1))
    n_zero = int(np.count_nonzero(arr == 0))
    n_nodata = int(np.count_nonzero(arr == OUTPUT_NODATA))

    print(
        f"{label}: "
        f"lightning={n_yes:,}, "
        f"zero={n_zero:,}, "
        f"nodata={n_nodata:,}"
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print("Indexing GLD rasters...")
    gld_index = index_rasters(GLD_BASE, GLD_SUBDIRS, windows=WINDOWS)
    print(f"Found {len(gld_index):,} GLD rasters")

    print("\nIndexing AICC rasters...")
    aicc_index = index_rasters(AICC_BASE, AICC_SUBDIRS, windows=WINDOWS)
    print(f"Found {len(aicc_index):,} AICC rasters")

    common_keys = sorted(set(gld_index) & set(aicc_index))
    gld_only = sorted(set(gld_index) - set(aicc_index))
    aicc_only = sorted(set(aicc_index) - set(gld_index))

    print(f"\nMatched raster pairs: {len(common_keys):,}")
    print(f"GLD-only rasters:     {len(gld_only):,}")
    print(f"AICC-only rasters:    {len(aicc_only):,}")

    if gld_only:
        print("\nFirst few GLD-only keys:")
        for key in gld_only[:10]:
            print(f"  {key}")

    if aicc_only:
        print("\nFirst few AICC-only keys:")
        for key in aicc_only[:10]:
            print(f"  {key}")

    written = 0
    skipped = 0
    errors = 0

    for key in common_keys:
        window, valid_str = key

        gld_path = gld_index[key]
        aicc_path = aicc_index[key]
        out_path = output_path(window, valid_str)

        if SKIP_EXISTING and out_path.exists():
            skipped += 1
            continue

        try:
            gld_arr, gld_profile = read_binary_raster(gld_path)
            aicc_arr, _ = read_binary_raster(aicc_path)

            if gld_arr.shape != aicc_arr.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"GLD {gld_arr.shape}, AICC {aicc_arr.shape}"
                )

            union_arr = create_union_array(gld_arr, aicc_arr)

            did_write = write_union_raster(out_path, union_arr, gld_profile)

            if did_write:
                written += 1

                print(f"\nWrote {out_path.name}")
                print_counts("  GLD  ", gld_arr)
                print_counts("  AICC ", aicc_arr)
                print_counts("  UNION", union_arr)

            else:
                skipped += 1

            if MAX_FILES is not None and written >= MAX_FILES:
                print(f"\nReached MAX_FILES={MAX_FILES}. Stopping.")
                break

        except Exception as e:
            errors += 1
            print(f"\nERROR processing {key}")
            print(f"  GLD:  {gld_path}")
            print(f"  AICC: {aicc_path}")
            print(f"  {e}")

    print("\nDone.")
    print(f"Written: {written:,}")
    print(f"Skipped: {skipped:,}")
    print(f"Errors:  {errors:,}")


if __name__ == "__main__":
    main()