from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMD_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rainfall_forecast_dataset.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_verification.parquet"
)


def main():
    print("===== ASTRA BLEND VERIFICATION TEMPLATE =====")

    imd = pd.read_parquet(
        IMD_FILE,
        columns=[
            "date",
            "latitude",
            "longitude",
            "rainfall_mm",
            "target_rainfall_mm",
            "target_heavy_rain",
        ],
    )

    imd["date"] = pd.to_datetime(imd["date"])

    verification = imd[
        [
            "date",
            "latitude",
            "longitude",
            "target_rainfall_mm",
            "target_heavy_rain",
        ]
    ].copy()

    verification = verification.rename(
        columns={
            "date": "forecast_valid_date",
            "target_rainfall_mm": "imd_observed_rainfall_mm",
            "target_heavy_rain": "imd_heavy_rain",
        }
    )

    verification["gfs_rainfall_mm"] = pd.NA
    verification["gfs_available"] = False
    verification["gfs_lead_hours"] = 24

    verification.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Rows:", f"{len(verification):,}")
    print(
        "Date range:",
        verification["forecast_valid_date"].min(),
        "to",
        verification["forecast_valid_date"].max(),
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()