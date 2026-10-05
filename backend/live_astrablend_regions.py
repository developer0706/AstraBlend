from backend.astrablend_live import generate_live_astrablend


# ==========================================================
# Locations to monitor
# ==========================================================

REGIONS = {

    "Odisha": [
        ("Bhubaneswar", 20.30, 85.82),
        ("Balasore", 21.49, 86.93),
        ("Keonjhar", 21.63, 85.58),
        ("Sambalpur", 21.47, 83.97),
        ("Jharsuguda", 21.86, 84.01),
        ("Bargarh", 21.33, 83.62),
    ],

    "West Bengal": [
        ("Kolkata", 22.57, 88.36),
        ("Digha", 21.63, 87.51),
        ("Bankura", 23.23, 87.07),
        ("Purulia", 23.33, 86.36),
    ],

    "Jharkhand": [
        ("Ranchi", 23.34, 85.31),
        ("Jamshedpur", 22.80, 86.20),
        ("Dhanbad", 23.80, 86.43),
    ],

    "Chhattisgarh": [
        ("Raipur", 21.25, 81.63),
        ("Bilaspur", 22.08, 82.15),
        ("Raigarh", 21.90, 83.39),
    ],

    "Assam": [
        ("Guwahati", 26.14, 91.74),
        ("Dibrugarh", 27.47, 94.91),
        ("Silchar", 24.83, 92.78),
    ],

    "Meghalaya": [
        ("Shillong", 25.58, 91.89),
        ("Cherrapunji", 25.27, 91.73),
    ],
}


# ==========================================================
# Risk ranking
# ==========================================================

RISK_ORDER = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
    "VERY HIGH": 3,
    "CRITICAL": 4,
}


# ==========================================================
# Scan all monitored regions
# ==========================================================

def scan_live_astrablend_regions(
    source_date="2026-08-31",
):

    results = {}

    for state, locations in REGIONS.items():

        state_results = []

        for city, latitude, longitude in locations:

            try:

                forecast = generate_live_astrablend(
                    name=f"{city}, {state}",
                    latitude=latitude,
                    longitude=longitude,
                    source_date=source_date,
                )

                forecast["city"] = city
                forecast["state"] = state

                state_results.append(
                    forecast
                )

            except Exception as error:

                state_results.append(
                    {
                        "city": city,
                        "state": state,
                        "error": str(error),
                    }
                )

        valid_results = [
            item
            for item in state_results
            if item.get("blend_ready") is True
        ]

        if valid_results:

            highest = max(
                valid_results,
                key=lambda item: (
                    RISK_ORDER.get(
                        item["risk_level"],
                        -1,
                    ),
                    item[
                        "astrablend_mm"
                    ],
                ),
            )

            mean_astrablend = (
                sum(
                    item[
                        "astrablend_mm"
                    ]
                    for item
                    in valid_results
                )
                /
                len(valid_results)
            )

            summary = {
                "state": state,
                "sampled_locations":
                    len(valid_results),

                "average_astrablend_mm":
                    round(
                        mean_astrablend,
                        2,
                    ),

                "highest_risk_location":
                    highest["city"],

                "highest_risk_level":
                    highest["risk_level"],

                "highest_rainfall_mm":
                    highest[
                        "astrablend_mm"
                    ],

                "highest_category":
                    highest[
                        "rainfall_category"
                    ],
            }

        else:

            summary = {
                "state": state,
                "sampled_locations": 0,
                "status":
                    "No valid forecasts",
            }

        results[state] = {
            "summary": summary,
            "locations": state_results,
        }

    return results


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    results = (
        scan_live_astrablend_regions()
    )

    print()
    print(
        "===== LIVE REGIONAL ASTRABLEND ====="
    )

    for state, data in results.items():

        print()
        print("=" * 65)
        print(state.upper())
        print("=" * 65)

        for item in data["locations"]:

            if "error" in item:

                print(
                    f"{item['city']:15} "
                    f"ERROR: {item['error']}"
                )

                continue

            print(
                f"{item['city']:15} "
                f"ML {item['ml_forecast_mm']:7.2f} | "
                f"GFS {item['gfs_forecast_mm']:7.2f} | "
                f"AB {item['astrablend_mm']:7.2f} | "
                f"{item['rainfall_category']:20} | "
                f"{item['risk_level']}"
            )

        print()
        print("STATE SUMMARY")

        summary = data["summary"]

        for key, value in summary.items():
            print(
                f"{key}: {value}"
            )