import pandas as pd
import xarray as xr
import numpy as np

# --------------------------------------------------
# FILES
# --------------------------------------------------

GFS_FILE = "data/forecasts/gfs_20240801_f006_india.parquet"
IMD_FILE = "data/processed/india_rainfall_2024.nc"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading GFS...")
gfs = pd.read_parquet(GFS_FILE)

print("Loading IMD...")
imd = xr.open_dataset(IMD_FILE)

rain = imd["rainfall"]

# --------------------------------------------------
# SELECT IMD DATE
# --------------------------------------------------

imd_date = pd.Timestamp("2024-08-01")

obs = rain.sel(TIME=imd_date)

# --------------------------------------------------
# MATCH EACH GFS GRID POINT TO IMD GRID
# --------------------------------------------------

print("Matching GFS grid to IMD grid...")

matched = []

for _, row in gfs.iterrows():

    lat = row["latitude"]
    lon = row["longitude"]

    try:
        imd_value = obs.sel(
            LATITUDE=lat,
            LONGITUDE=lon,
            method="nearest"
        ).values.item()
    except Exception:
        imd_value = np.nan

    matched.append(imd_value)

gfs["imd_rainfall_mm"] = matched

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n========== MATCH RESULT ==========")

print("GFS rows:", len(gfs))

print(
    "Valid IMD matches:",
    gfs["imd_rainfall_mm"].notna().sum()
)

print(
    "Missing IMD matches:",
    gfs["imd_rainfall_mm"].isna().sum()
)

print("\nSample:")
print(
    gfs[
        [
            "latitude",
            "longitude",
            "gfs_precip_mm",
            "imd_rainfall_mm"
        ]
    ].head(20).to_string(index=False)
)

# --------------------------------------------------
# SAVE TEST FILE
# --------------------------------------------------

output = "data/processed/gfs_imd_match_test.parquet"

gfs.to_parquet(output, index=False)

print("\nSaved:")
print(output)
