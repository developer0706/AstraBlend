from pathlib import Path

import xarray as xr


IMD_FILE = Path("data/processed/india_rainfall_2024.nc")


def inspect_imd() -> xr.Dataset:
    """Load and inspect the IMD rainfall dataset."""
    dataset = xr.open_dataset(IMD_FILE)

    print("===== IMD DATASET =====")
    print(dataset)

    print()
    print("Latitude:")
    print(dataset["LATITUDE"].values[:5], "...")

    print()
    print("Longitude:")
    print(dataset["LONGITUDE"].values[:5], "...")

    print()
    print("Time:")
    print(dataset["TIME"].values[:5], "...")

    return dataset


def main() -> None:
    dataset = inspect_imd()

    rainfall = dataset["rainfall"]

    print()
    print("===== IMD RAINFALL =====")
    print("Dimensions:", rainfall.dims)
    print("Shape:", rainfall.shape)
    print("Units:", rainfall.attrs.get("units"))

    print()
    print("Next stage:")
    print("1. Load GFS forecast")
    print("2. Convert GFS precipitation to mm")
    print("3. Align GFS to IMD grid")
    print("4. Align valid time")
    print("5. Compare forecast with observation")


if __name__ == "__main__":
    main()