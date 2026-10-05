from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rainfall_forecast_dataset.parquet"
)

HEAVY_THRESHOLD_MM = 64.5


def evaluate_year(df: pd.DataFrame, year: int) -> None:
    """Evaluate the persistence baseline for one year."""

    test = df[df["date"].dt.year == year].copy()

    # Persistence rainfall forecast:
    # tomorrow = today's rainfall.
    test["predicted_rainfall_mm"] = test["rainfall_mm"]

    mae = mean_absolute_error(
        test["target_rainfall_mm"],
        test["predicted_rainfall_mm"],
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["target_rainfall_mm"],
            test["predicted_rainfall_mm"],
        )
    )

    # Persistence heavy-rain forecast.
    actual_heavy = (
        test["target_rainfall_mm"] >= HEAVY_THRESHOLD_MM
    )

    predicted_heavy = (
        test["rainfall_mm"] >= HEAVY_THRESHOLD_MM
    )

    precision = precision_score(
        actual_heavy,
        predicted_heavy,
        zero_division=0,
    )

    recall = recall_score(
        actual_heavy,
        predicted_heavy,
        zero_division=0,
    )

    f1 = f1_score(
        actual_heavy,
        predicted_heavy,
        zero_division=0,
    )

    print()
    print(f"===== PERSISTENCE BASELINE: {year} =====")
    print(f"Samples:   {len(test):,}")
    print(f"MAE:       {mae:.4f} mm")
    print(f"RMSE:      {rmse:.4f} mm")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")


def main() -> None:
    print("===== ASTRA BLEND: BASELINE EVALUATION =====")

    columns = [
        "date",
        "rainfall_mm",
        "target_rainfall_mm",
    ]

    df = pd.read_parquet(
        INPUT_FILE,
        columns=columns,
    )

    df["date"] = pd.to_datetime(df["date"])

    evaluate_year(df, 2024)
    evaluate_year(df, 2025)

    print()
    print("===== BASELINE COMPLETE =====")


if __name__ == "__main__":
    main()