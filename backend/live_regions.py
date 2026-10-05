from live_gfs import get_location_risk


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


RISK_ORDER = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
    "VERY HIGH": 3,
    "CRITICAL": 4,
}


def scan_regions():
    print("===== ASTRA BLEND REGIONAL GFS RISK =====")

    for state, locations in REGIONS.items():
        print(f"\n{'=' * 60}")
        print(state.upper())
        print("=" * 60)

        state_results = []

        for city, latitude, longitude in locations:
            try:
                result = get_location_risk(
                    city,
                    latitude,
                    longitude,
                )

                # First returned forecast day
                day = result["forecast"][0]

                state_results.append(
                    {
                        "city": city,
                        **day,
                    }
                )

                print(
                    f"{city:15} "
                    f"{day['gfs_rainfall_mm']:7.2f} mm | "
                    f"{day['rainfall_category']:22} | "
                    f"{day['risk_level']}"
                )

            except Exception as exc:
                print(f"{city:15} ERROR: {exc}")

        if not state_results:
            continue

        highest = max(
            state_results,
            key=lambda item: (
                RISK_ORDER.get(item["risk_level"], -1),
                item["gfs_rainfall_mm"],
            ),
        )

        average_rainfall = sum(
            item["gfs_rainfall_mm"]
            for item in state_results
        ) / len(state_results)

        print()
        print("STATE SUMMARY")
        print("Average sampled rainfall:", round(average_rainfall, 2), "mm")
        print("Highest-risk location:", highest["city"])
        print("Highest rainfall:", highest["gfs_rainfall_mm"], "mm")
        print("Highest category:", highest["rainfall_category"])
        print("GFS risk:", highest["risk_level"])


if __name__ == "__main__":
    scan_regions()