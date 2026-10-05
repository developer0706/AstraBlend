from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rainfall_forecast_dataset.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


def main() -> None:
    print("===== ASTRA BLEND: TIME-BASED DATA SPLIT =====")

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

    df = pd.read_parquet(
        INPUT_FILE,
        columns=columns,
    )

    df["date"] = pd.to_datetime(df["date"])

    train = df[df["date"].dt.year <= 2023]
    validation = df[df["date"].dt.year == 2024]
    test = df[df["date"].dt.year == 2025]

    train_file = OUTPUT_DIR / "train.parquet"
    validation_file = OUTPUT_DIR / "validation.parquet"
    test_file = OUTPUT_DIR / "test.parquet"

    train.to_parquet(train_file, index=False)
    validation.to_parquet(validation_file, index=False)
    test.to_parquet(test_file, index=False)

    print()
    print("===== SPLIT SUMMARY =====")
    print(
        f"Train:      {len(train):,} rows "
        f"({train['date'].min().date()} → "
        f"{train['date'].max().date()})"
    )
    print(
        f"Validation: {len(validation):,} rows "
        f"({validation['date'].min().date()} → "
        f"{validation['date'].max().date()})"
    )
    print(
        f"Test:       {len(test):,} rows "
        f"({test['date'].min().date()} → "
        f"{test['date'].max().date()})"
    )

    print()
    print("Created:")
    print(train_file)
    print(validation_file)
    print(test_file)


if __name__ == "__main__":
    main()