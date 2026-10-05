from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "gfs_india_f006.nc"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "gfs_india_f006.parquet"
)


# India region
MIN_LAT = 6.0
MAX_LAT = 37.0
MIN_LON = 68.0
MAX_LON = 98.0


def main():
    print("===== ASTRA BLEND: PREPARE GFS =====")

    with h5py.File(INPUT_FILE, "r") as file:

        lat = file["latitude"][:]
        lon = file["longitude"][:]
        precipitation = file["tp"][:]

        forecast_time = file["time"][()]
        forecast_step = file["step"][()]

        tp_attrs = dict(file["tp"].attrs)

    # ------------------------------------------------------
    # Convert time metadata
    # ------------------------------------------------------

    forecast_time = pd.to_datetime(
        forecast_time,
        unit="s",
        origin="unix",
    )

    # step is stored as hours according to the file metadata.
    lead_hours = float(forecast_step)

    # ------------------------------------------------------
    # Select India region
    # ------------------------------------------------------

    lat_mask = (
        (lat >= MIN_LAT)
        & (lat <= MAX_LAT)
    )

    lon_mask = (
        (lon >= MIN_LON)
        & (lon <= MAX_LON)
    )

    india_lat = lat[lat_mask]
    india_lon = lon[lon_mask]

    india_precip = precipitation[
        np.ix_(lat_mask, lon_mask)
    ]

    # ------------------------------------------------------
    # Check units
    # ------------------------------------------------------

    units = tp_attrs.get("units", b"unknown")

    if isinstance(units, bytes):
        units = units.decode()

    print()
    print("===== GFS METADATA =====")
    print("Forecast run:", forecast_time)
    print("Lead hours:", lead_hours)
    print("Variable: tp")
    print("Units:", units)
    print("Step type:", tp_attrs.get("GRIB_stepType"))

    # ------------------------------------------------------
    # Build table
    # ------------------------------------------------------

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

    # Remove missing values.
    df = df[np.isfinite(df["gfs_precip"])].copy()

    # GFS total precipitation is typically represented in
    # kg/m² or equivalent precipitation units. For liquid
    # precipitation, 1 kg/m² corresponds approximately to 1 mm.
    #
    # We leave the raw value unchanged here and record the
    # original units rather than silently converting it.

    df["source"] = "NOAA GFS 0.25"
    df["variable"] = "total_precipitation"

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("===== INDIA GFS DATASET =====")
    print("Rows:", f"{len(df):,}")
    print("Latitude range:",
          df["latitude"].min(),
          "to",
          df["latitude"].max())
    print("Longitude range:",
          df["longitude"].min(),
          "to",
          df["longitude"].max())

    print(
        "Precipitation min:",
        float(df["gfs_precip"].min())
    )

    print(
        "Precipitation max:",
        float(df["gfs_precip"].max())
    )

    print(
        "Precipitation mean:",
        float(df["gfs_precip"].mean())
    )

    print()
    print("Valid time:", df["valid_time"].iloc[0])
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()