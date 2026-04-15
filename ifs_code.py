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

# --- CONFIGURATION ---
steps = [6, 12, 120, 240] 
lat_max, lat_min = 46.0, 30.0
lon_min, lon_max = 128.0, 146.0

variables = {
    "2t": ["Temperature", "coolwarm", "°C", "t2m"],
    "sp": ["Surface Pressure", "viridis", "hPa", "sp"],
    "tp": ["Total Precipitation", "YlGnBu", "mm", "tp"]
}

run_id = time.strftime("%Y%m%d_%H%M")
temp_frames = []

print(f"🚀 Starting IFS Pipeline (Run ID: {run_id})")

for var_code, info in variables.items():
    var_name, var_cmap, var_unit, data_key = info
    print(f"\nProcessing {var_name}...")
    
    for step in steps:
        target_grib = f"temp_{var_code}_{step}_{run_id}.grib"
        plot_img = f"plot_{var_code}_{step}.png"
        
        print(f"  - Step {step}h: ", end="", flush=True)
        
        try:
            client.retrieve(model="ifs", type="fc", param=var_code, step=step, target=target_grib)
            ds = xr.open_dataset(target_grib, engine="cfgrib")
            
            # Fix for scalar time
            raw_time = np.atleast_1d(ds.time.values)
            start_time = raw_time[0]
            forecast_time = start_time + np.timedelta64(step, 'h')
            time_str = np.datetime_as_string(forecast_time, unit='h').replace('T', ' ')
            
            data = ds[data_key].sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

            if len(data.dims) > 2:
                other_dims = [d for d in data.dims if d not in ['latitude', 'longitude']]
                if other_dims:
                    data = data.isel({other_dims[0]: 0})

            if var_code == "2t": data = data - 273.15
            elif var_code == "sp": data = data / 100.0
            elif var_code == "tp": data = data * 1000.0

            # --- FIXED PLOTTING SECTION ---
            # 1. Use a fixed figure size
            fig = plt.figure(figsize=(10, 6)) 
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            
            data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=var_cmap, cbar_kwargs={'label': var_unit})
            
            plt.title(f"IFS {var_name}\n{time_str}", fontsize=12, fontweight='bold')
            
            # 2. Use tight_layout instead of bbox_inches='tight'
            # This adjusts elements INSIDE the fixed box, rather than changing the box size.
            plt.tight_layout() 
            
            # 3. Save without bbox_inches='tight' to keep dimensions constant
            plt.savefig(plot_img, dpi=150) 
            plt.close()
            
            if var_code == "2t":
                # Read the image back in to ensure it is a numpy array for imageio
                temp_frames.append(plt.imread(plot_img))
                
            print("✅ Done")
            ds.close()

        except Exception as e:
            print(f"❌ FAILED: {e}")
        finally:
            if os.path.exists(target_grib): os.remove(target_grib)
            if os.path.exists(target_grib + ".idx"): os.remove(target_grib + ".idx")
            for f in os.listdir('.'):
                if f.endswith('.idx'):
                    try: os.remove(f)
                    except: pass
        
        time.sleep(2)


print("\n🏁 All processes completed.")
