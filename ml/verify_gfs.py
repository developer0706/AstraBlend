from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GFS_FILE = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "gfs_india_f024.parquet"
)

IMD_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rainfall_forecast_dataset.parquet"
)


def calculate_metrics(
    forecast: np.ndarray,
    observed: np.ndarray,
):
    error = forecast - observed

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    bias = np.mean(error)

    return mae, rmse, bias


def main():
    print("===== ASTRA BLEND: GFS VERIFICATION =====")

    # ------------------------------------------------------
    # Load GFS forecast
    # ------------------------------------------------------

    gfs = pd.read_parquet(GFS_FILE)

    valid_time = pd.to_datetime(
        gfs["valid_time"].iloc[0]
    )

    valid_date = valid_time.normalize()

    print("GFS valid time:", valid_time)

    # ------------------------------------------------------
    # Find matching IMD observation date
    # ------------------------------------------------------

    print()
    print("Searching IMD observations...")

    imd_columns = [
        "date",
        "latitude",
        "longitude",
        "target_rainfall_mm",
    ]

    imd = pd.read_parquet(
        IMD_FILE,
        columns=imd_columns,
    )

    imd["date"] = pd.to_datetime(imd["date"])

    matching = imd[
        imd["date"] == valid_date
    ].copy()

    if matching.empty:
        print()
        print("===== NO MATCHING OBSERVATION =====")
        print(
            f"No IMD observation exists in the current "
            f"dataset for {valid_date.date()}."
        )
        print()
        print(
            "Current IMD dataset ends at:",
            imd["date"].max().date(),
        )
        print(
            "GFS forecast valid date:",
            valid_date.date(),
        )
        print()
        print(
            "Verification cannot be performed yet."
        )
        return

    # ------------------------------------------------------
    # Align GFS with IMD grid
    # ------------------------------------------------------

    print(
        f"IMD observations found: "
        f"{len(matching):,}"
    )

    print()
    print("Aligning GFS with IMD...")

    # GFS is also 0.25°, so use the nearest grid cell.
    merged = gfs[
        [
            "latitude",
            "longitude",
            "gfs_precip",
        ]
    ].merge(
        matching,
        on=[
            "latitude",
            "longitude",
        ],
        how="inner",
    )

    if merged.empty:
        print()
        print(
            "No matching grid coordinates were found."
        )
        return

    print(
        f"Matched grid points: "
        f"{len(merged):,}"
    )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    forecast = merged[
        "gfs_precip"
    ].to_numpy()

    observed = merged[
        "target_rainfall_mm"
    ].to_numpy()

    mae, rmse, bias = calculate_metrics(
        forecast,
        observed,
    )

    print()
    print("===== GFS SKILL =====")
    print(f"MAE:  {mae:.4f} mm")
    print(f"RMSE: {rmse:.4f} mm")
    print(f"Bias: {bias:.4f} mm")

    # ------------------------------------------------------
    # Heavy rainfall detection
    # ------------------------------------------------------

    thresholds = {
        "heavy": 64.5,
        "very_heavy": 115.6,
        "extreme": 204.5,
    }

    print()
    print("===== EVENT DETECTION =====")

    for name, threshold in thresholds.items():

        observed_event = (
            observed >= threshold
        )

        forecast_event = (
            forecast >= threshold
        )

        tp = np.sum(
            forecast_event & observed_event
        )

        fn = np.sum(
            (~forecast_event) & observed_event
        )

        fp = np.sum(
            forecast_event & (~observed_event)
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        print()
        print(
            f"{name.upper()} ≥ {threshold} mm"
        )
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"TP:        {tp}")
        print(f"FN:        {fn}")
        print(f"FP:        {fp}")


if __name__ == "__main__":
    main()