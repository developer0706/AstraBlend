from pathlib import Path

import xarray as xr


FILE = Path("data/processed/rainfall_2020_clean.nc")


def main() -> None:
    dataset = xr.open_dataset(FILE)

    rainfall = dataset["rainfall"]

    print("===== ASTRA BLEND: GEOGRAPHY CHECK =====")

    print(f"Latitude points:  {rainfall.sizes['LATITUDE']}")
    print(f"Longitude points: {rainfall.sizes['LONGITUDE']}")

    print(
        f"Latitude range:  "
        f"{float(rainfall.LATITUDE.min()):.2f}° "
        f"to "
        f"{float(rainfall.LATITUDE.max()):.2f}°"
    )

    print(
        f"Longitude range: "
        f"{float(rainfall.LONGITUDE.min()):.2f}° "
        f"to "
        f"{float(rainfall.LONGITUDE.max()):.2f}°"
    )

    print(f"Time points:     {rainfall.sizes['TIME']}")

    print(
        f"First date:      "
        f"{rainfall.TIME.min().values}"
    )

    print(
        f"Last date:       "
        f"{rainfall.TIME.max().values}"
    )


if __name__ == "__main__":
    main()