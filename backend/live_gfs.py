from datetime import datetime
from typing import Dict, List

import requests

from backend.risk_levels import classify_rainfall


# ==========================================================
# Open-Meteo GFS API
# ==========================================================

GFS_API_URL = "https://api.open-meteo.com/v1/gfs"


# ==========================================================
# Fetch live GFS weather data
# ==========================================================

def fetch_live_gfs(
    latitude: float,
    longitude: float,
    forecast_days: int = 2,
) -> Dict:

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "precipitation,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "wind_direction_10m"
        ),
        "models": "gfs_global",
        "forecast_days": forecast_days,
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(
        GFS_API_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise RuntimeError(
            "GFS API response does not contain hourly data."
        )

    hourly = data["hourly"]

    times: List[str] = hourly["time"]
    precipitation: List[float] = hourly["precipitation"]
    humidity: List[float] = hourly["relative_humidity_2m"]
    wind_speed: List[float] = hourly["wind_speed_10m"]
    wind_direction: List[float] = hourly["wind_direction_10m"]

    hourly_forecast = []

    for (
        time_value,
        rain_value,
        humidity_value,
        wind_speed_value,
        wind_direction_value,
    ) in zip(
        times,
        precipitation,
        humidity,
        wind_speed,
        wind_direction,
    ):

        hourly_forecast.append(
            {
                "time": time_value,
                "precipitation_mm": float(rain_value or 0.0),
                "relative_humidity_percent": float(
                    humidity_value or 0.0
                ),
                "wind_speed_kmh": float(
                    wind_speed_value or 0.0
                ),
                "wind_direction_degrees": float(
                    wind_direction_value or 0.0
                ),
            }
        )

    return {
        "source": "NOAA GFS via Open-Meteo GFS API",
        "requested_latitude": latitude,
        "requested_longitude": longitude,
        "grid_latitude": float(data["latitude"]),
        "grid_longitude": float(data["longitude"]),
        "timezone": data["timezone"],
        "retrieved_at": datetime.now().astimezone().isoformat(),
        "hourly": hourly_forecast,
    }


# ==========================================================
# Calculate daily precipitation
# ==========================================================

def daily_precipitation_total(
    forecast: Dict,
    target_date: str,
) -> float:

    total = 0.0

    for item in forecast["hourly"]:
        if item["time"].startswith(target_date):
            total += item["precipitation_mm"]

    return round(total, 2)


# ==========================================================
# Calculate average daily humidity
# ==========================================================

def daily_average_humidity(
    forecast: Dict,
    target_date: str,
) -> float:

    values = []

    for item in forecast["hourly"]:
        if item["time"].startswith(target_date):
            values.append(
                item["relative_humidity_percent"]
            )

    if not values:
        return 0.0

    return round(
        sum(values) / len(values),
        1,
    )


# ==========================================================
# Calculate average daily wind speed
# ==========================================================

def daily_average_wind_speed(
    forecast: Dict,
    target_date: str,
) -> float:

    values = []

    for item in forecast["hourly"]:
        if item["time"].startswith(target_date):
            values.append(
                item["wind_speed_kmh"]
            )

    if not values:
        return 0.0

    return round(
        sum(values) / len(values),
        1,
    )


# ==========================================================
# Calculate maximum daily wind speed
# ==========================================================

def daily_max_wind_speed(
    forecast: Dict,
    target_date: str,
) -> float:

    values = []

    for item in forecast["hourly"]:
        if item["time"].startswith(target_date):
            values.append(
                item["wind_speed_kmh"]
            )

    if not values:
        return 0.0

    return round(
        max(values),
        1,
    )


# ==========================================================
# Calculate average daily wind direction
# ==========================================================

def daily_average_wind_direction(
    forecast: Dict,
    target_date: str,
) -> float:

    values = []

    for item in forecast["hourly"]:
        if item["time"].startswith(target_date):
            values.append(
                item["wind_direction_degrees"]
            )

    if not values:
        return 0.0

    return round(
        sum(values) / len(values),
        1,
    )


# ==========================================================
# Convert wind direction degrees into compass direction
# ==========================================================

def wind_direction_name(
    degrees: float,
) -> str:

    directions = [
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
    ]

    index = round(
        degrees / 45
    ) % 8

    return directions[index]


# ==========================================================
# Get GFS risk + weather for one location
# ==========================================================

def get_location_risk(
    name: str,
    latitude: float,
    longitude: float,
) -> Dict:

    forecast = fetch_live_gfs(
        latitude,
        longitude,
    )

    dates = sorted(
        {
            item["time"][:10]
            for item in forecast["hourly"]
        }
    )

    daily_results = []

    for target_date in dates:

        rainfall_mm = daily_precipitation_total(
            forecast,
            target_date,
        )

        humidity_percent = daily_average_humidity(
            forecast,
            target_date,
        )

        average_wind_speed_kmh = (
            daily_average_wind_speed(
                forecast,
                target_date,
            )
        )

        max_wind_speed_kmh = (
            daily_max_wind_speed(
                forecast,
                target_date,
            )
        )

        wind_direction_degrees = (
            daily_average_wind_direction(
                forecast,
                target_date,
            )
        )

        wind_direction = wind_direction_name(
            wind_direction_degrees
        )

        risk = classify_rainfall(
            rainfall_mm
        )

        daily_results.append(
            {
                "date": target_date,
                "gfs_rainfall_mm": rainfall_mm,
                "humidity_percent": humidity_percent,
                "average_wind_speed_kmh":
                    average_wind_speed_kmh,
                "max_wind_speed_kmh":
                    max_wind_speed_kmh,
                "wind_direction_degrees":
                    wind_direction_degrees,
                "wind_direction":
                    wind_direction,
                "rainfall_category":
                    risk["category"],
                "risk_level":
                    risk["risk_level"],
                "alert":
                    risk["alert"],
                "alert_message":
                    risk["message"],
            }
        )

    return {
        "location": name,
        "latitude": latitude,
        "longitude": longitude,
        "grid_latitude":
            forecast["grid_latitude"],
        "grid_longitude":
            forecast["grid_longitude"],
        "source":
            forecast["source"],
        "retrieved_at":
            forecast["retrieved_at"],
        "forecast":
            daily_results,
    }


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    result = get_location_risk(
        name="Bhubaneswar, Odisha",
        latitude=20.30,
        longitude=85.82,
    )

    print()
    print(
        "===== ASTRABLEND LIVE GFS TEST ====="
    )

    print(
        "Location:",
        result["location"],
    )

    print(
        "Source:",
        result["source"],
    )

    print(
        "Updated:",
        result["retrieved_at"],
    )

    for day in result["forecast"]:

        print()

        print(
            "Date:",
            day["date"],
        )

        print(
            "GFS rainfall:",
            day["gfs_rainfall_mm"],
            "mm",
        )

        print(
            "Humidity:",
            day["humidity_percent"],
            "%",
        )

        print(
            "Average wind speed:",
            day["average_wind_speed_kmh"],
            "km/h",
        )

        print(
            "Maximum wind speed:",
            day["max_wind_speed_kmh"],
            "km/h",
        )

        print(
            "Wind direction:",
            day["wind_direction"],
            "(",
            day["wind_direction_degrees"],
            "degrees )",
        )

        print(
            "Category:",
            day["rainfall_category"],
        )

        print(
            "Risk:",
            day["risk_level"],
        )

        print(
            "Alert:",
            day["alert"],
        )

        print(
            "Message:",
            day["alert_message"],
        )