import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ecmwf.opendata import Client
import numpy as np
import imageio
import warnings
import time

warnings.filterwarnings("ignore")
client = Client(source="azure")

# 1. Create a unique timestamp for this specific run
timestamp = time.strftime("%Y%m%d_%H%M")

steps = [6, 12, 120, 240]
lat_max, lat_min = 46.0, 30.0
lon_min, lon_max = 128.0, 146.0

variables = {
    "2t": ["Temperature", "coolwarm", "°C", "t2m"],
    "sp": ["Surface Pressure", "viridis", "hPa", "sp"],
    "tp": ["Total Precipitation", "YlGnBu", "mm", "tp"]
}

temp_frames = []

print(f"Starting Batch Processing for run: {timestamp}")

for var_code, info in variables.items():
    var_name, var_cmap, var_unit, data_key = info
    print(f"\n>>> Variable: {var_name}")
    
    for step in steps:
        # We add the timestamp to the filename so Git sees a change every time
        target_grib = f"temp_{var_code}_{step}_{timestamp}.grib"
        plot_img = f"plot_{var_code}_{step}_{timestamp}.png"
        
        print(f"    Step {step}h: ", end="", flush=True)
        
        try:
            client.retrieve(model="ifs", type="fc", param=var_code, step=step, target=target_grib)
            ds = xr.open_dataset(target_grib, engine="cfgrib")
            
            if data_key not in ds.data_vars:
                raise KeyError(f"Missing {data_key}")

            data = ds[data_key].sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

            if len(data.dims) > 2:
                level_dim = [d for d in data.dims if d not in ['latitude', 'longitude', 'time']][0]
                data = data.isel({level_dim: 0})

            if var_code == "2t": data = data - 273.15
            elif var_code == "sp": data = data / 100.0
            elif var_code == "tp": data = data * 1000.0

            fig = plt.figure(figsize=(10, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=var_cmap, cbar_kwargs={'label': var_unit})
            
            time_str = np.datetime_as_string(ds.time.values[0] + np.timedelta64(step, 'h'), unit='h').replace('T', ' ')
            plt.title(f"IFS {var_name}\n{time_str}", fontsize=12, fontweight='bold')
            
            plt.savefig(plot_img, dpi=150, bbox_inches='tight')
            plt.close()
            
            if var_code == "2t":
                temp_frames.append(plt.imread(plot_img))
                
            print("Done.")
            ds.close()
            os.remove(target_grib)
            time.sleep(2)

        except Exception as e:
            print(f"FAILED: {e}")
        finally:
            if os.path.exists(target_grib):
                try: os.remove(target_grib)
                except: pass

if len(temp_frames) > 0:
    print("\n>>> Stitching Animation...")
    imageio.mimsave('weather_animation.gif', temp_frames, fps=0.7)
    print("Success! GIF created.")

print("\nAll tasks complete.")
