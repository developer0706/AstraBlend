from pathlib import Path

import xarray as xr


RAW_FILE = Path("data/raw/RF25_ind2020_rfp25.nc")
OUTPUT_FILE = Path("data/processed/rainfall_2020_clean.nc")


def main() -> None:
    print("Loading IMD rainfall data...")

    dataset = xr.open_dataset(RAW_FILE)

    rainfall = dataset["RAINFALL"]

    # xarray has already decoded the IMD -999 fill value as NaN.
    # Keep valid rainfall values and preserve missing locations.
    rainfall_clean = rainfall.where(rainfall.notnull())

    clean_dataset = rainfall_clean.to_dataset(name="rainfall")

    clean_dataset["rainfall"].attrs["long_name"] = "IMD gridded daily rainfall"
    clean_dataset["rainfall"].attrs["units"] = "mm"
    clean_dataset["rainfall"].attrs["source"] = "India Meteorological Department"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    clean_dataset.to_netcdf(OUTPUT_FILE)

    valid_values = rainfall_clean.notnull().sum().item()
    missing_values = rainfall_clean.isnull().sum().item()

    print("\n===== ASTRA BLEND CLEANING COMPLETE =====")
    print(f"Input file:  {RAW_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Valid values: {valid_values:,}")
    print(f"Missing values: {missing_values:,}")
    print(f"Minimum rainfall: {rainfall_clean.min().item():.2f} mm")
    print(f"Maximum rainfall: {rainfall_clean.max().item():.2f} mm")
    print(f"Average rainfall: {rainfall_clean.mean().item():.2f} mm")


if __name__ == "__main__":
    main()