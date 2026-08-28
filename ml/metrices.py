def absolute_error(forecast: float, actual: float) -> float:
    """
    Calculate how far a forecast is from the actual observed value.
    """
    return abs(forecast - actual)


if __name__ == "__main__":
    forecast = 80.0
    actual = 60.0

    error = absolute_error(forecast, actual)

    print(f"Forecast: {forecast} mm")
    print(f"Actual:   {actual} mm")
    print(f"Absolute error: {error} mm")