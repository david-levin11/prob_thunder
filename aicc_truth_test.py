from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
from matplotlib.colors import ListedColormap, BoundaryNorm


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

RASTER_PATH = Path(
    r"C:\Users\David.Levin\NBMLightningVer\aicc_lightning\rasters_total_test"
    r"\aicc_total_12_20km\aicc_total_12h_20240619_0000Z.tif"
)

# Optional: save PNG next to raster
SAVE_PNG = True


# ---------------------------------------------------------------------
# READ RASTER
# ---------------------------------------------------------------------

ds = rxr.open_rasterio(RASTER_PATH, mask_and_scale=True)
arr = ds.values[0]

print("\nRaster:")
print(RASTER_PATH)

print("\nMetadata:")
print("Shape:", arr.shape)
print("CRS:", ds.rio.crs)
print("Transform:", ds.rio.transform())
print("Bounds:", ds.rio.bounds())
print("Nodata:", ds.rio.nodata)

print("\nValue diagnostics:")
print("Min:", np.nanmin(arr))
print("Max:", np.nanmax(arr))
print("NaN pixels:", np.count_nonzero(np.isnan(arr)))
print("Zero pixels:", np.count_nonzero(arr == 0))
print("Lightning pixels:", np.count_nonzero(arr == 1))
print("Unique finite values:", np.unique(arr[np.isfinite(arr)]))

# Raster extent for plotting
left, bottom, right, top = ds.rio.bounds()
extent = [left, right, bottom, top]


# ---------------------------------------------------------------------
# PLOT 1: FULL DOMAIN
# ---------------------------------------------------------------------

# Binary color map:
# 0 = white/no lightning
# 1 = red/lightning
cmap = ListedColormap(["white", "red"])
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(
    arr,
    origin="upper",
    extent=extent,
    cmap=cmap,
    norm=norm,
)

ax.set_title(
    "AICC Lightning Truth Raster\n"
    f"{RASTER_PATH.name}"
)
ax.set_xlabel("Raster X")
ax.set_ylabel("Raster Y")

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["No lightning", "Lightning"])

plt.tight_layout()

if SAVE_PNG:
    out_png = RASTER_PATH.with_suffix(".full_domain.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_png}")

plt.show()


# ---------------------------------------------------------------------
# PLOT 2: ZOOM TO LIGHTNING PIXELS
# ---------------------------------------------------------------------

rows, cols = np.where(arr == 1)

if len(rows) == 0:
    print("\nNo lightning pixels found to zoom to.")
else:
    pad = 50  # grid cells around lightning area

    r0 = max(rows.min() - pad, 0)
    r1 = min(rows.max() + pad, arr.shape[0])
    c0 = max(cols.min() - pad, 0)
    c1 = min(cols.max() + pad, arr.shape[1])

    zoom = arr[r0:r1, c0:c1]

    # Convert pixel row/col bounds to map coordinates.
    transform = ds.rio.transform()

    x_left, y_top = transform * (c0, r0)
    x_right, y_bottom = transform * (c1, r1)
    zoom_extent = [x_left, x_right, y_bottom, y_top]

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        zoom,
        origin="upper",
        extent=zoom_extent,
        cmap=cmap,
        norm=norm,
    )

    ax.set_title(
        "AICC Lightning Truth Raster — Zoom to Lightning Area\n"
        f"{RASTER_PATH.name}"
    )
    ax.set_xlabel("Raster X")
    ax.set_ylabel("Raster Y")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["No lightning", "Lightning"])

    plt.tight_layout()

    if SAVE_PNG:
        out_png = RASTER_PATH.with_suffix(".zoom_lightning.png")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_png}")

    plt.show()

ds.close()