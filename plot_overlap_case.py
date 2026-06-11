import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


INPUT_NODATA = 255


def get_nbm_alaska_crs():
    """
    Cartopy CRS matching the Alaska NBM polar stereographic grid.

    Raster metadata showed:
    - Polar Stereographic
    - latitude_of_origin = 60
    - central_meridian = -150, equivalent to 210
    - spherical earth radius = 6371200 m
    """
    globe = ccrs.Globe(
        semimajor_axis=6371200,
        semiminor_axis=6371200,
        ellipse=None,
    )

    return ccrs.NorthPolarStereo(
        true_scale_latitude=60,
        central_longitude=-150,
        globe=globe,
    )


def read_binary_truth(path):
    """
    Read binary truth raster and return:
      arr_raw: raw raster values
      arr_plot: masked array where non-lightning values are transparent
      yes_mask: True where lightning == 1
    """

    ds = rxr.open_rasterio(path, mask_and_scale=True)
    arr = ds.values[0]

    # With mask_and_scale=True, NoData should become NaN.
    yes_mask = arr == 1

    # Plot only lightning pixels; mask zeros and NaNs.
    arr_plot = np.ma.masked_where(~yes_mask, arr)

    return ds, arr, arr_plot, yes_mask


def plot_gld_aicc_truth_overlap_cartopy(
    gld_file,
    aicc_file,
    zoom_to_raster=True,
    save_png=False,
    out_png=None,
):
    """
    Plot GLD and AICC dilated truth rasters plus overlap and union panels.

    Panels:
      1. GLD only
      2. AICC only
      3. Overlap categories: GLD only / AICC only / both
      4. Union: GLD OR AICC
    """

    raster_crs = get_nbm_alaska_crs()

    ds_gld, gld_arr, gld_plot, gld_yes = read_binary_truth(gld_file)
    ds_aicc, aicc_arr, aicc_plot, aicc_yes = read_binary_truth(aicc_file)

    try:
        print("GLD shape: ", gld_arr.shape)
        print("AICC shape:", aicc_arr.shape)
        print("GLD CRS:", ds_gld.rio.crs)
        print("AICC CRS:", ds_aicc.rio.crs)
        print("GLD bounds:", ds_gld.rio.bounds())
        print("AICC bounds:", ds_aicc.rio.bounds())

        if gld_arr.shape != aicc_arr.shape:
            raise ValueError(f"Shape mismatch: GLD {gld_arr.shape}, AICC {aicc_arr.shape}")

        left, bottom, right, top = ds_gld.rio.bounds()
        extent = [left, right, bottom, top]

        valid = np.isfinite(gld_arr) | np.isfinite(aicc_arr)

        gld_only = gld_yes & ~aicc_yes
        aicc_only = aicc_yes & ~gld_yes
        both = gld_yes & aicc_yes
        union = gld_yes | aicc_yes

        print("\nTruth-raster comparison:")
        print("  GLD lightning pixels:       ", int(gld_yes.sum()))
        print("  AICC lightning pixels:      ", int(aicc_yes.sum()))
        print("  GLD only pixels:            ", int(gld_only.sum()))
        print("  AICC only pixels:           ", int(aicc_only.sum()))
        print("  Both datasets pixels:       ", int(both.sum()))
        print("  Union lightning pixels:     ", int(union.sum()))
        print("  Valid grid cells:           ", int(valid.sum()))

        # Overlap categorical panel:
        # 0 = no lightning
        # 1 = GLD only
        # 2 = AICC only
        # 3 = both
        overlap = np.zeros(gld_arr.shape, dtype="float32")
        overlap[gld_only] = 1
        overlap[aicc_only] = 2
        overlap[both] = 3
        overlap[~valid] = np.nan
        overlap_plot = np.ma.masked_where((overlap == 0) | np.isnan(overlap), overlap)

        # Union plot: show only union lightning pixels
        union_arr = np.zeros(gld_arr.shape, dtype="float32")
        union_arr[union] = 1
        union_arr[~valid] = np.nan
        union_plot = np.ma.masked_where(~union, union_arr)

        gld_cmap = ListedColormap(["deepskyblue"])
        aicc_cmap = ListedColormap(["orange"])
        union_cmap = ListedColormap(["red"])

        overlap_cmap = ListedColormap([
            "deepskyblue",  # 1 GLD only
            "orange",       # 2 AICC only
            "purple",       # 3 both
        ])
        overlap_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], overlap_cmap.N)

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(16, 14),
            subplot_kw={"projection": raster_crs},
            constrained_layout=True,
        )

        axes = axes.ravel()
        def add_cartopy_background(ax):
            ax.add_feature(cfeature.LAND, facecolor="0.90", edgecolor="none", zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="black", zorder=3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black", zorder=3)

            if zoom_to_raster:
                ax.set_extent(extent, crs=raster_crs)

            ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=False,
                linewidth=0.3,
                color="gray",
                alpha=0.4,
                linestyle="--",
            )

        # 1. GLD
        ax = axes[0]
        add_cartopy_background(ax)
        ax.imshow(
            gld_plot,
            origin="upper",
            extent=extent,
            transform=raster_crs,
            cmap=gld_cmap,
            vmin=1,
            vmax=1,
            alpha=0.9,
            zorder=4,
        )
        ax.set_title(f"GLD Dilated Truth\n{int(gld_yes.sum()):,} lightning pixels")

        # 2. AICC
        ax = axes[1]
        add_cartopy_background(ax)
        ax.imshow(
            aicc_plot,
            origin="upper",
            extent=extent,
            transform=raster_crs,
            cmap=aicc_cmap,
            vmin=1,
            vmax=1,
            alpha=0.9,
            zorder=4,
        )
        ax.set_title(f"AICC Dilated Truth\n{int(aicc_yes.sum()):,} lightning pixels")

        # 3. Overlap
        ax = axes[2]
        add_cartopy_background(ax)
        ax.imshow(
            overlap_plot,
            origin="upper",
            extent=extent,
            transform=raster_crs,
            cmap=overlap_cmap,
            norm=overlap_norm,
            alpha=0.9,
            zorder=4,
        )
        ax.set_title(
            "Overlap Categories\n"
            f"Both: {int(both.sum()):,} pixels"
        )

        overlap_legend = [
            Patch(facecolor="deepskyblue", edgecolor="black", label=f"GLD only: {int(gld_only.sum()):,}"),
            Patch(facecolor="orange", edgecolor="black", label=f"AICC only: {int(aicc_only.sum()):,}"),
            Patch(facecolor="purple", edgecolor="black", label=f"Both: {int(both.sum()):,}"),
        ]

        ax.legend(
            handles=overlap_legend,
            loc="lower right",
            frameon=True,
        )

        # 4. Union
        ax = axes[3]
        add_cartopy_background(ax)
        ax.imshow(
            union_plot,
            origin="upper",
            extent=extent,
            transform=raster_crs,
            cmap=union_cmap,
            vmin=1,
            vmax=1,
            alpha=0.9,
            zorder=4,
        )
        ax.set_title(
            "Union Truth\n"
            f"GLD OR AICC: {int(union.sum()):,} pixels"
        )

        union_legend = [
            Patch(facecolor="red", edgecolor="black", label=f"Union: {int(union.sum()):,}"),
        ]
        ax.legend(handles=union_legend, loc="lower right", frameon=True)

        fig.suptitle(
            f"GLD vs AICC Dilated Lightning Truth and Union\n"
            f"GLD: {gld_file}\n"
            f"AICC: {aicc_file}",
            fontsize=11,
        )

        if save_png:
            if out_png is None:
                out_png = "gld_aicc_truth_overlap_union.png"

            fig.savefig(out_png, dpi=250, bbox_inches="tight")
            print(f"Saved: {out_png}")

        plt.show()

    finally:
        ds_gld.close()
        ds_aicc.close()

gld_file = r"C:\Users\David.Levin\NBMLightningVer\gld_rasters\gld_12_20km\gld_12h_20250619_0600Z.tif"

aicc_file = r"C:\Users\David.Levin\NBMLightningVer\aicc_lightning\rasters_total_dilated\aicc_total_12_20km_dilated\aicc_total_12h_20250619_0600Z.tif"

plot_gld_aicc_truth_overlap_cartopy(
    gld_file=gld_file,
    aicc_file=aicc_file,
    save_png=True,
    out_png=r"C:\Users\David.Levin\NBMLightningVer\gld_aicc_overlap_20250619_0600Z.png",
)