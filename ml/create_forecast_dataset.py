from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rainfall_ml_features.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rainfall_forecast_dataset.parquet"
)


def main() -> None:
    print("===== ASTRA BLEND: FORECAST DATASET =====")

    print("Loading ML features...")
    df = pd.read_parquet(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    # Make sure each location is ordered chronologically.
    df = df.sort_values(
        ["latitude", "longitude", "date"]
    ).reset_index(drop=True)

    # Tomorrow's rainfall becomes today's prediction target.
    location_group = df.groupby(
        ["latitude", "longitude"],
        sort=False,
    )

    df["target_rainfall_mm"] = (
        location_group["rainfall_mm"].shift(-1)
    )

    df["target_heavy_rain"] = (
        location_group["heavy_rain_flag"].shift(-1)
    )

    # Rows at the very end of each location have no next-day target.
    df = df.dropna(
        subset=[
            "target_rainfall_mm",
            "target_heavy_rain",
        ]
    )

    df["target_heavy_rain"] = (
        df["target_heavy_rain"].astype("int8")
    )

    # Keep only information that would be available
    # when making the forecast.
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
        "target_rainfall_mm",
        "target_heavy_rain",
    ]

    df = df[columns]

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("===== FORECAST DATASET CREATED =====")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print()
    print("Target rainfall:")
    print(df["target_rainfall_mm"].describe())

    print()
    print("Target heavy rain:")
    print(df["target_heavy_rain"].value_counts())

    print()
    print(
        "Heavy-rain percentage:",
        f"{df['target_heavy_rain'].mean() * 100:.3f}%"
    )


if __name__ == "__main__":
    main()