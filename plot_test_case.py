import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D


FORECAST_IS_PERCENT = True
TRUTH_YES_THRESHOLD = 0


def get_nbm_alaska_crs():
    """
    Cartopy CRS matching the Alaska NBM polar stereographic grid.

    Your raster metadata showed:
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


def plot_forecast_truth_pair_cartopy(
    forecast_file,
    truth_file,
    highlight_threshold=0.50,
    zoom_to_raster=True,
):
    """
    Plot NBM thunder probability, GLD truth raster, and overlay panel
    using Cartopy land/coastline features as a background.
    """

    raster_crs = get_nbm_alaska_crs()

    with rxr.open_rasterio(forecast_file, mask_and_scale=True) as ds_f, \
         rxr.open_rasterio(truth_file, mask_and_scale=True) as ds_o:

        f_arr = ds_f.values[0].astype(float)
        o_arr = ds_o.values[0].astype(float)

        if FORECAST_IS_PERCENT:
            f_arr = f_arr / 100.0

        # Plotting-only version of forecast array:
        # hide values below 1%
        f_plot = np.ma.masked_where(f_arr < 0.01, f_arr)

        o_bin = np.where(o_arr > TRUTH_YES_THRESHOLD, 1.0, 0.0)

        valid = ~np.isnan(f_arr) & ~np.isnan(o_arr)
        high_mask = valid & (f_arr >= highlight_threshold)
        obs_mask = valid & (o_bin == 1)

        # Forecast probability bins
        f_levels = np.linspace(0, 1, 11)
        f_cmap = plt.get_cmap("YlOrRd", len(f_levels) - 1).copy()
        f_cmap.set_bad(alpha=0.0)  # transparent below 1% because those values are masked
        f_norm = BoundaryNorm(f_levels, f_cmap.N)
        print("Forecast shape:", f_arr.shape)
        print("Truth shape:   ", o_bin.shape)
        print("Forecast CRS:", ds_f.rio.crs)
        print("Truth CRS:   ", ds_o.rio.crs)
        print("Forecast bounds:", ds_f.rio.bounds())
        print("Truth bounds:   ", ds_o.rio.bounds())

        left, bottom, right, top = ds_f.rio.bounds()
        extent = [left, right, bottom, top]

        valid = ~np.isnan(f_arr) & ~np.isnan(o_arr)
        high_mask = valid & (f_arr >= highlight_threshold)
        obs_mask = valid & (o_bin == 1)

        print(f"\nSummary for forecast >= {highlight_threshold:.2f}")
        print("  Valid pixels:        ", int(valid.sum()))
        print("  Forecast high pixels:", int(high_mask.sum()))
        print("  Observed yes pixels: ", int(obs_mask.sum()))
        print("  Overlap pixels:      ", int((high_mask & obs_mask).sum()))
        print("  Max forecast prob:   ", float(np.nanmax(f_arr)))

        # Forecast probability bins
        f_levels = np.linspace(0, 1, 11)
        f_cmap = plt.get_cmap("YlOrRd", len(f_levels) - 1)
        f_norm = BoundaryNorm(f_levels, f_cmap.N)

        # Binary observed raster
        o_cmap = ListedColormap(["white", "deepskyblue"])
        o_norm = BoundaryNorm([-0.5, 0.5, 1.5], o_cmap.N)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(22, 8),
            subplot_kw={"projection": raster_crs},
            constrained_layout=True,
        )

        def add_cartopy_background(ax):
            # Land/coastline context
            ax.add_feature(cfeature.LAND, facecolor="0.90", edgecolor="none", zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="black", zorder=3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black", zorder=3)

            # Optional: lakes/rivers can clutter Alaska, but useful sometimes.
            ax.add_feature(cfeature.LAKES, facecolor="white", edgecolor="0.5", linewidth=0.3, zorder=2)

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

        # ------------------------------------------------------------
        # 1. Forecast probability
        # ------------------------------------------------------------
        ax = axes[0]
        add_cartopy_background(ax)

        # Forecast panel
        im0 = ax.imshow(
            f_plot,
            origin="upper",
            extent=extent,
            transform=raster_crs,
            cmap=f_cmap,
            norm=f_norm,
            alpha=0.85,
            zorder=2,
        )

        ax.set_title("NBM Thunder Probability")

        cbar0 = plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
        cbar0.set_label("Forecast probability")
        cbar0.set_ticks(f_levels)

        # ------------------------------------------------------------
        # 2. Observed GLD raster
        # ------------------------------------------------------------
        ax = axes[1]
        add_cartopy_background(ax)

        im1 = ax.imshow(
            o_bin,
            origin="upper",
            extent=extent,
            transform=raster_crs,
            cmap=o_cmap,
            norm=o_norm,
            alpha=0.75,
            zorder=2,
        )

        ax.set_title("Observed GLD Raster")

        cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label("Observed lightning")
        cbar1.set_ticks([0, 1])
        cbar1.set_ticklabels(["No", "Yes"])

        # ------------------------------------------------------------
        # 3. Overlay
        # ------------------------------------------------------------
        ax = axes[2]
        add_cartopy_background(ax)

        # Overlay panel
        im2 = ax.imshow(
            f_plot,
            origin="upper",
            extent=extent,
            transform=raster_crs,
            cmap=f_cmap,
            norm=f_norm,
            alpha=0.85,
            zorder=2,
        )

        if np.any(obs_mask):
            ax.contour(
                obs_mask.astype(int),
                levels=[0.5],
                colors="cyan",
                linewidths=0.8,
                origin="upper",
                extent=extent,
                transform=raster_crs,
                zorder=4,
            )

        if np.any(high_mask):
            ax.contour(
                high_mask.astype(int),
                levels=[0.5],
                colors="black",
                linewidths=1.7,
                origin="upper",
                extent=extent,
                transform=raster_crs,
                zorder=5,
            )

        ax.set_title(
            f"Overlay\n"
            f"Black = forecast >= {highlight_threshold:.2f}; Cyan = observed GLD"
        )

        cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
        cbar2.set_label("Forecast probability")
        cbar2.set_ticks(f_levels)

        legend_lines = [
            Line2D([0], [0], color="black", lw=2, label=f"Forecast >= {highlight_threshold:.2f}"),
            Line2D([0], [0], color="cyan", lw=2, label="Observed lightning"),
            Line2D([0], [0], color="black", lw=1, label="Coastline"),
        ]

        ax.legend(handles=legend_lines, loc="lower right")

        fig.suptitle(
            f"Forecast vs Observed Lightning\n"
            f"{forecast_file}\n"
            f"{truth_file}",
            fontsize=11,
        )

        plt.show()

forecast_file = r"C:\Users\David.Levin\NBMLightningVer\nbm_data\2025\06\18\1300\tstm12\blendv4.3_alaska_tstm12_2025-06-18T1300_F017.tif"
truth_file = r"C:\Users\David.Levin\NBMLightningVer\gld_rasters\gld_12_20km\gld_12h_20250619_0600Z.tif"

plot_forecast_truth_pair_cartopy(
    forecast_file=forecast_file,
    truth_file=truth_file,
    highlight_threshold=0.50,
)