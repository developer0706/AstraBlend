from pathlib import Path

import xarray as xr


RAINFALL_FILE = Path("data/processed/rainfall_2020_clean.nc")
MASK_FILE = Path("data/processed/india_mask_2020.nc")
OUTPUT_FILE = Path("data/processed/india_rainfall_2020.nc")


def main() -> None:
    print("Loading rainfall data...")
    rainfall_ds = xr.open_dataset(RAINFALL_FILE)

    print("Loading India mask...")
    mask_ds = xr.open_dataset(MASK_FILE)

    rainfall = rainfall_ds["rainfall"]
    india_mask = mask_ds["india_land_mask"]

    # Apply the 2D India mask to every day in the rainfall dataset.
    india_rainfall = rainfall.where(india_mask)

    output = india_rainfall.to_dataset(name="rainfall")

    output["rainfall"].attrs["long_name"] = (
        "IMD daily gridded rainfall over India"
    )
    output["rainfall"].attrs["units"] = "mm"
    output["rainfall"].attrs["source"] = "India Meteorological Department"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(OUTPUT_FILE)

    valid_values = int(india_rainfall.notnull().sum().item())
    missing_values = int(india_rainfall.isnull().sum().item())

    print()
    print("===== ASTRA BLEND: INDIA RAINFALL =====")
    print(f"Output file:       {OUTPUT_FILE}")
    print(f"Valid values:      {valid_values:,}")
    print(f"Missing values:    {missing_values:,}")
    print(
        f"Minimum rainfall:  "
        f"{float(india_rainfall.min(skipna=True)):.2f} mm"
    )
    print(
        f"Maximum rainfall:  "
        f"{float(india_rainfall.max(skipna=True)):.2f} mm"
    )
    print(
        f"Average rainfall:  "
        f"{float(india_rainfall.mean(skipna=True)):.2f} mm"
    )


if __name__ == "__main__":
    main()