from pathlib import Path
import sqlite3

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_FILE = PROJECT_ROOT / "database" / "astrablend.db"


def calculate_metrics(
    forecast: np.ndarray,
    observed: np.ndarray,
) -> dict[str, float]:
    """Calculate standard forecast verification metrics."""

    forecast = np.asarray(forecast, dtype=float)
    observed = np.asarray(observed, dtype=float)

    valid = np.isfinite(forecast) & np.isfinite(observed)

    forecast = forecast[valid]
    observed = observed[valid]

    if len(forecast) == 0:
        raise ValueError("No valid forecast/observation pairs.")

    error = forecast - observed

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    bias = np.mean(error)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "sample_count": int(len(error)),
    }


def save_skill(
    model_name: str,
    latitude: float,
    longitude: float,
    season: str,
    lead_hours: int,
    metrics: dict[str, float],
) -> None:
    """Save or update a model-skill record."""

    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO forecast_skill (
            model_name,
            latitude,
            longitude,
            season,
            lead_hours,
            mae,
            rmse,
            bias,
            sample_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(
            model_name,
            latitude,
            longitude,
            season,
            lead_hours
        )
        DO UPDATE SET
            mae = excluded.mae,
            rmse = excluded.rmse,
            bias = excluded.bias,
            sample_count = excluded.sample_count
        """,
        (
            model_name,
            latitude,
            longitude,
            season,
            lead_hours,
            metrics["mae"],
            metrics["rmse"],
            metrics["bias"],
            metrics["sample_count"],
        ),
    )

    connection.commit()
    connection.close()


def main() -> None:
    print("===== ASTRA BLEND: SKILL ENGINE TEST =====")

    # Toy values only to test the database pipeline.
    #
    # We are NOT claiming these are real weather forecasts.
    forecast = np.array([10.0, 20.0, 5.0, 30.0])
    observed = np.array([12.0, 18.0, 7.0, 25.0])

    metrics = calculate_metrics(
        forecast,
        observed,
    )

    print("Metrics:")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Bias: {metrics['bias']:.4f}")

    save_skill(
        model_name="TEST_MODEL",
        latitude=23.0,
        longitude=77.0,
        season="monsoon",
        lead_hours=24,
        metrics=metrics,
    )

    print()
    print("Skill record saved to SQLite.")


if __name__ == "__main__":
    main()