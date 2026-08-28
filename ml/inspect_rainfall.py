from pathlib import Path
import xarray as xr


file_path = Path("data/raw/RF25_ind2020_rfp25.nc")

dataset = xr.open_dataset(file_path, engine="scipy")

rainfall = dataset["RAINFALL"]


print("\n===== ASTRA BLEND: RAINFALL CHECK =====\n")

print("Rainfall variable:")
print(rainfall)

print("\nRainfall attributes:")
print(rainfall.attrs)

print("\nMinimum rainfall:")
print(float(rainfall.min(skipna=True)))

print("\nMaximum rainfall:")
print(float(rainfall.max(skipna=True)))

print("\nAverage rainfall:")
print(float(rainfall.mean(skipna=True)))

print("\nNumber of missing values:")
print(int(rainfall.isnull().sum()))

print("\n===== MISSING VALUE INFORMATION =====")

print("Encoded fill value:")
print(rainfall.encoding.get("_FillValue"))

print("\nMissing-value attribute:")
print(rainfall.attrs.get("_FillValue"))