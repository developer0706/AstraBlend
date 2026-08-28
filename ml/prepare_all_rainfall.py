from pathlib import Path

import xarray as xr


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MASK_FILE = PROCESSED_DIR / "india_mask_2020.nc"


def process_year(input_file: Path) -> None:
    year = input_file.stem.replace("RF25_ind", "").replace("_rfp25", "")

    output_file = PROCESSED_DIR / f"india_rainfall_{year}.nc"

    print()
    print(f"===== Processing {year} =====")

    rainfall_ds = xr.open_dataset(input_file)
    rainfall = rainfall_ds["RAINFALL"]

    mask_ds = xr.open_dataset(MASK_FILE)
    india_mask = mask_ds["india_land_mask"]

    india_rainfall = rainfall.where(india_mask)

    output = india_rainfall.to_dataset(name="rainfall")

    output["rainfall"].attrs["long_name"] = (
        f"IMD daily gridded rainfall over India - {year}"
    )
    output["rainfall"].attrs["units"] = "mm"
    output["rainfall"].attrs["source"] = (
        "India Meteorological Department"
    )

    output.to_netcdf(output_file)

    valid_count = int(india_rainfall.notnull().sum().item())
    missing_count = int(india_rainfall.isnull().sum().item())

    print(f"Saved: {output_file}")
    print(f"Valid values: {valid_count:,}")
    print(f"Missing values: {missing_count:,}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        RAW_DIR.glob("RF25_ind*_rfp25.nc")
    )

    if not files:
        raise FileNotFoundError(
            "No IMD rainfall files found in data/raw/"
        )

    print("===== ASTRA BLEND: BATCH RAINFALL PROCESSING =====")
    print(f"Found {len(files)} IMD files.")

    for file in files:
        process_year(file)

    print()
    print("===== ALL YEARS COMPLETE =====")


if __name__ == "__main__":
    main()