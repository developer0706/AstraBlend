from pathlib import Path

import numpy as np
import xarray as xr


IMD_FILE = Path("data/processed/india_rainfall_2024.nc")


def calculate_metrics(
    forecast: xr.DataArray,
    observed: xr.DataArray,
) -> dict[str, float]:
    """
    Calculate standard forecast verification metrics.
    Both arrays must already be on the same grid and time window.
    """

    forecast_values = forecast.values.astype(float)
    observed_values = observed.values.astype(float)

    valid = np.isfinite(forecast_values) & np.isfinite(observed_values)

    forecast_values = forecast_values[valid]
    observed_values = observed_values[valid]

    if len(forecast_values) == 0:
        raise ValueError("No overlapping valid forecast/observation values.")

    error = forecast_values - observed_values

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    bias = np.mean(error)

    correlation = np.corrcoef(
        forecast_values,
        observed_values,
    )[0, 1]

    return {
        "sample_count": int(len(error)),
        "mae_mm": float(mae),
        "rmse_mm": float(rmse),
        "bias_mm": float(bias),
        "correlation": float(correlation),
    }


def main() -> None:
    print("===== ASTRA BLEND: FORECAST VERIFICATION ENGINE =====")

    print("Loading IMD observations...")
    imd = xr.open_dataset(IMD_FILE)

    print("IMD variables:", list(imd.data_vars))
    print("IMD coordinates:", list(imd.coords))

    print()
    print("Verification engine is ready.")
    print()
    print("The forecast dataset must provide:")
    print("- same geographic region")
    print("- compatible grid")
    print("- same rainfall units (mm)")
    print("- same valid time window")

    print()
    print("No forecast comparison has been performed yet.")


if __name__ == "__main__":
    main()