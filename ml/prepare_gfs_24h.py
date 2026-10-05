from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "gfs_india_f024.nc"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "gfs_india_f024.parquet"
)


def main():
    print("===== ASTRA BLEND: PREPARE GFS +24H =====")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    with h5py.File(INPUT_FILE, "r") as file:
        latitude = file["latitude"][:]
        longitude = file["longitude"][:]
        precipitation = file["tp"][:]

        forecast_time_raw = file["time"][()]
        step_raw = file["step"][()]

        units = file["tp"].attrs.get("units", b"unknown")
        step_type = file["tp"].attrs.get(
            "GRIB_stepType",
            b"unknown",
        )

    if isinstance(units, bytes):
        units = units.decode()

    if isinstance(step_type, bytes):
        step_type = step_type.decode()

    forecast_time = pd.to_datetime(
        forecast_time_raw,
        unit="s",
        origin="unix",
    )

    lead_hours = float(step_raw)

    print()
    print("===== GFS METADATA =====")
    print("Forecast run:", forecast_time)
    print("Lead hours:", lead_hours)
    print("Variable: tp")
    print("Units:", units)
    print("Step type:", step_type)

    if lead_hours != 24:
        print(
            "WARNING: expected 24-hour lead, "
            f"got {lead_hours}"
        )

    # India region.
    lat_mask = (
        (latitude >= 6.0)
        & (latitude <= 37.0)
    )

    lon_mask = (
        (longitude >= 68.0)
        & (longitude <= 98.0)
    )

    india_lat = latitude[lat_mask]
    india_lon = longitude[lon_mask]

    india_precip = precipitation[
        np.ix_(lat_mask, lon_mask)
    ]

    lat_grid, lon_grid = np.meshgrid(
        india_lat,
        india_lon,
        indexing="ij",
    )

    df = pd.DataFrame(
        {
            "forecast_time": forecast_time,
            "lead_hours": lead_hours,
            "valid_time": (
                forecast_time
                + pd.Timedelta(hours=lead_hours)
            ),
            "latitude": lat_grid.ravel(),
            "longitude": lon_grid.ravel(),
            "gfs_precip": india_precip.ravel(),
        }
    )

    df = df[
        np.isfinite(df["gfs_precip"])
    ].copy()

    df["source"] = "NOAA GFS 0.25"
    df["variable"] = "total_precipitation"
    df["units"] = units

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("===== GFS +24H INDIA DATA =====")
    print("Rows:", f"{len(df):,}")
    print(
        "Valid time:",
        df["valid_time"].iloc[0],
    )
    print(
        "Rainfall min:",
        float(df["gfs_precip"].min()),
    )
    print(
        "Rainfall max:",
        float(df["gfs_precip"].max()),
    )
    print(
        "Rainfall mean:",
        float(df["gfs_precip"].mean()),
    )

    print()
    print("Output:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()