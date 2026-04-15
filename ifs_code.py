import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ecmwf.opendata import Client
import warnings
import time
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

client = Client(source="azure")

steps = [3, 6, 12, 48, 120]

lat_max, lat_min = 46.0, 30.0
lon_min, lon_max = 128.0, 146.0

# ── Output size ──────────────────────────────────────────────────────────────
PLOT_DPI    = 100          # was 150  → smaller PNG / faster GIF frame load
FIG_SIZE    = (7, 4.5)     # was (8,5) → tighter frame
GIF_RESIZE  = (700, 450)   # final pixel size forced for every GIF frame
GIF_DURATION= 1200         # ms per frame (slightly slower = easier to read)

# ── Contour tuning ───────────────────────────────────────────────────────────
CONTOUR_LEVELS   = 8       # was 15  → far less clutter
CONTOUR_INTERVAL = None    # set e.g. 4 to force fixed hPa spacing (None = auto)

print("Starting combined Temperature + Pressure processing...")

frame_paths = []

for step in steps:
    t_file   = f"temp_2t_{step}.grib"
    p_file   = f"temp_sp_{step}.grib"
    plot_img = f"plot_combined_{step}.png"

    print(f"Step {step}h: ", end="", flush=True)

    ds_t = ds_p = None

    try:
        # ── Download ─────────────────────────────────────────────────────────
        client.retrieve(model="ifs", type="fc", param="2t",
                        step=step, date=-1, time=0, target=t_file)
        client.retrieve(model="ifs", type="fc", param="sp",
                        step=step, date=-1, time=0, target=p_file)

        ds_t = xr.open_dataset(t_file, engine="cfgrib")
        ds_p = xr.open_dataset(p_file, engine="cfgrib")

        t = ds_t["t2m"].sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        ) - 273.15

        p = ds_p["sp"].sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        ) / 100.0

        # ── Contour levels: fixed interval keeps labels readable ──────────────
        if CONTOUR_INTERVAL:
            p_min = float(p.min())
            p_max = float(p.max())
            import numpy as np
            levels = np.arange(
                round(p_min / CONTOUR_INTERVAL) * CONTOUR_INTERVAL,
                round(p_max / CONTOUR_INTERVAL) * CONTOUR_INTERVAL + CONTOUR_INTERVAL,
                CONTOUR_INTERVAL
            )
        else:
            levels = CONTOUR_LEVELS

        # ── Time labels ───────────────────────────────────────────────────────
        base_time      = pd.to_datetime(ds_t.time.values)
        valid_time     = base_time + pd.Timedelta(hours=int(step))
        valid_time_str = valid_time.strftime("%Y-%m-%d %H:%M UTC")

        # ── Plot ──────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=FIG_SIZE)
        ax  = plt.axes(projection=ccrs.PlateCarree())

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS,   linestyle=":", linewidth=0.5)
        ax.add_feature(cfeature.LAND,      facecolor="lightgray", alpha=0.3)

        # Temperature shading
        t.plot(
            ax=ax,
            cmap="RdYlBu_r",
            transform=ccrs.PlateCarree(),
            cbar_kwargs={"label": "°C", "shrink": 0.85, "pad": 0.03}
        )

        # Pressure contours — fewer, thicker, well-labelled
        cs = p.plot.contour(
            ax=ax,
            colors="black",
            linewidths=1.2,
            levels=levels,
            alpha=0.7
        )
        ax.clabel(cs, inline=True, fontsize=7, fmt="%d hPa",
                  inline_spacing=4)

        # Gridlines (subtle)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                          color="gray", alpha=0.5, linestyle="--")
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlocator = mticker.MultipleLocator(4)
        gl.ylocator = mticker.MultipleLocator(4)
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}

        ax.set_title(
            f"IFS  |  2m Temperature & Surface Pressure  |  +{step}h\n"
            f"Valid: {valid_time_str}",
            fontsize=9, pad=6
        )

        plt.tight_layout(pad=0.5)
        plt.savefig(plot_img, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

        frame_paths.append(plot_img)
        print("Done")

    except Exception as e:
        print(f"FAILED: {e}")

    finally:
        if ds_t: ds_t.close()
        if ds_p: ds_p.close()
        for f in [t_file, p_file]:
            if os.path.exists(f):
                os.remove(f)

    time.sleep(2)

# ── Build optimised GIF ───────────────────────────────────────────────────────
if frame_paths:
    gif_name = "temp_pressure_evolution.gif"
    frames   = []

    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize(GIF_RESIZE, Image.LANCZOS)
        # Quantise to 128 colours → much smaller file size
        img = img.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        frames.append(img)

    frames[0].save(
        gif_name,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_DURATION,
        loop=0,
        optimize=True       # Pillow palette optimisation pass
    )
    print(f"\nGIF saved: {gif_name}  ({os.path.getsize(gif_name)/1024:.0f} KB)")
else:
    print("No frames produced — GIF skipped.")
