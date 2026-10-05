from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "rainfall_ml_features.parquet"


YEARS = range(2020, 2026)


def load_year(year: int) -> xr.DataArray:
    """Load one year's processed India rainfall dataset."""

    file_path = PROCESSED_DIR / f"india_rainfall_{year}.nc"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    dataset = xr.open_dataset(file_path)

    if "rainfall" not in dataset:
        raise ValueError(f"'rainfall' variable missing from {file_path}")

    return dataset["rainfall"]


def prepare_year(year: int) -> pd.DataFrame:
    """Convert one year's rainfall grid into ML-ready rows."""

    print(f"Processing {year}...")

    rainfall = load_year(year)

    # Convert the xarray grid into a table.
    dataframe = rainfall.to_dataframe(
        name="rainfall_mm"
    ).reset_index()

    dataframe = dataframe.rename(
        columns={
            "TIME": "date",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
        }
    )

    # Sort before calculating time-based features.
    dataframe = dataframe.sort_values(
        ["latitude", "longitude", "date"]
    )

    # Rainfall from previous days.
    grouped = dataframe.groupby(
        ["latitude", "longitude"],
        sort=False,
    )["rainfall_mm"]

    dataframe["rainfall_lag_1d"] = grouped.shift(1)
    dataframe["rainfall_lag_3d"] = grouped.shift(3)

    # Rolling rainfall sums.
    dataframe["rainfall_rolling_3d"] = (
        dataframe.groupby(
            ["latitude", "longitude"],
            sort=False,
        )["rainfall_mm"]
        .transform(
            lambda series: series.rolling(3).sum()
        )
    )

    dataframe["rainfall_rolling_7d"] = (
        dataframe.groupby(
            ["latitude", "longitude"],
            sort=False,
        )["rainfall_mm"]
        .transform(
            lambda series: series.rolling(7).sum()
        )
    )

    # Calendar features.
    dataframe["month"] = dataframe["date"].dt.month
    dataframe["day_of_year"] = dataframe["date"].dt.dayofyear

    # Monsoon indicator: June–September.
    dataframe["monsoon"] = dataframe["month"].isin(
        [6, 7, 8, 9]
    ).astype("int8")

    # Initial extreme-rain threshold.
    # This is a prototype classification target.
    dataframe["heavy_rain_flag"] = (
        dataframe["rainfall_mm"] >= 64.5
    ).astype("int8")

    return dataframe


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames = []

    for year in YEARS:
        frame = prepare_year(year)
        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    # Remove rows where lag/rolling features cannot yet exist.
    combined = combined.dropna(
        subset=[
            "rainfall_lag_1d",
            "rainfall_lag_3d",
            "rainfall_rolling_3d",
            "rainfall_rolling_7d",
        ]
    )

    # Make sure dates are stored consistently.
    combined["date"] = pd.to_datetime(
        combined["date"]
    )

    # Put columns in a predictable order.
    columns = [
        "date",
        "latitude",
        "longitude",
        "rainfall_mm",
        "rainfall_lag_1d",
        "rainfall_lag_3d",
        "rainfall_rolling_3d",
        "rainfall_rolling_7d",
        "month",
        "day_of_year",
        "monsoon",
        "heavy_rain_flag",
    ]

    combined = combined[columns]

    combined.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("===== ASTRA BLEND: ML FEATURES =====")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows: {len(combined):,}")
    print(f"Columns: {len(combined.columns)}")
    print()
    print(combined.head())


if __name__ == "__main__":
    main()