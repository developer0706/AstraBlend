def classify_rainfall(rainfall_mm: float) -> dict:
    rainfall_mm = max(float(rainfall_mm), 0.0)

    if rainfall_mm < 0.1:
        return {
            "category": "No Rain",
            "risk_level": "LOW",
            "alert": False,
            "message": "No significant rainfall expected.",
        }

    if rainfall_mm <= 15.5:
        return {
            "category": "Light Rain",
            "risk_level": "LOW",
            "alert": False,
            "message": "Light rainfall expected. No major rainfall hazard indicated.",
        }

    if rainfall_mm <= 64.4:
        return {
            "category": "Moderate Rain",
            "risk_level": "MODERATE",
            "alert": True,
            "message": "Moderate rainfall expected. Stay aware of local weather conditions.",
        }

    if rainfall_mm <= 115.5:
        return {
            "category": "Heavy Rain",
            "risk_level": "HIGH",
            "alert": True,
            "message": (
                "Heavy rainfall expected. Waterlogging and localized flooding "
                "may occur in vulnerable areas."
            ),
        }

    if rainfall_mm <= 204.4:
        return {
            "category": "Very Heavy Rain",
            "risk_level": "VERY HIGH",
            "alert": True,
            "message": (
                "Very heavy rainfall expected. High risk of waterlogging, "
                "flooding and disruption in vulnerable areas."
            ),
        }

    return {
        "category": "Extremely Heavy Rain",
        "risk_level": "CRITICAL",
        "alert": True,
        "message": (
            "Extremely heavy rainfall expected. Severe flooding and major "
            "disruption may occur. Follow official emergency guidance."
        ),
    }


if __name__ == "__main__":
    test_values = [
        0.0,
        8.0,
        30.0,
        80.0,
        150.0,
        250.0,
    ]

    print("===== ASTRA BLEND RISK ENGINE TEST =====")

    for rainfall in test_values:
        result = classify_rainfall(rainfall)

        print()
        print("Rainfall:", rainfall, "mm")
        print("Category:", result["category"])
        print("Risk:", result["risk_level"])
        print("Alert:", result["alert"])
        print("Message:", result["message"])