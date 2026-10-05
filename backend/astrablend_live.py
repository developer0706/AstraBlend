import pandas as pd

from backend.live_gfs import (
    get_location_risk,
)

from backend.risk_levels import (
    classify_rainfall,
)

from backend.main import (
    get_live_rainfall_history,
    rainfall_model,
    heavy_rain_model,
    calculate_astrablend,
    HEAVY_RAIN_THRESHOLD,
)

from backend.live_imd import (
    build_live_ml_features,
    feature_dict_to_array,
)


# ==========================================================
# Generate live AstraBlend forecast
# ==========================================================

def generate_live_astrablend(
    name: str,
    latitude: float,
    longitude: float,
    source_date: str = "2026-08-31",
):

    source_date = pd.Timestamp(
        source_date
    ).normalize()

    target_date = (
        source_date
        + pd.Timedelta(days=1)
    )

    target_date_string = str(
        target_date.date()
    )

    # ======================================================
    # 1. IMD rainfall history
    # ======================================================

    live = get_live_rainfall_history(
        latitude,
        longitude,
        source_date,
    )

    if live["missing_files"]:

        return {
            "location": name,
            "blend_ready": False,
            "missing_imd_files":
                live["missing_files"],
        }

    # ======================================================
    # 2. ML features
    # ======================================================

    features = build_live_ml_features(
        live["rainfall_history"],
        source_date,
        live["grid_latitude"],
        live["grid_longitude"],
    )

    X = feature_dict_to_array(
        features
    )

    # ======================================================
    # 3. ML rainfall prediction
    # ======================================================

    ml_rainfall = float(
        rainfall_model.predict(
            X
        )[0]
    )

    ml_rainfall = max(
        0.0,
        ml_rainfall,
    )

    # ======================================================
    # 4. Heavy-rain ML classifier
    # ======================================================

    heavy_probability = float(
        heavy_rain_model.predict_proba(
            X
        )[0, 1]
    )

    heavy_classifier_warning = bool(
        heavy_probability
        >= HEAVY_RAIN_THRESHOLD
    )

    # ======================================================
    # 5. Live GFS
    # ======================================================

    gfs_result = get_location_risk(
        name,
        latitude,
        longitude,
    )

    target_gfs_day = None

    for day in gfs_result["forecast"]:

        if (
            day["date"]
            == target_date_string
        ):
            target_gfs_day = day
            break

    if target_gfs_day is None:

        raise RuntimeError(
            "Live GFS forecast is not available "
            f"for {target_date_string}."
        )

    gfs_rainfall = float(
        target_gfs_day[
            "gfs_rainfall_mm"
        ]
    )

    # ======================================================
    # 6. AstraBlend
    # ======================================================

    astrablend_rainfall = (
        calculate_astrablend(
            ml_rainfall,
            gfs_rainfall,
        )
    )

    # ======================================================
    # 7. Rainfall hazard classification
    # ======================================================

    rainfall_risk = classify_rainfall(
        astrablend_rainfall
    )

    # ======================================================
    # 8. Final response
    # ======================================================

    return {

        "location":
            name,

        "latitude":
            latitude,

        "longitude":
            longitude,

        "imd_grid_latitude":
            live["grid_latitude"],

        "imd_grid_longitude":
            live["grid_longitude"],

        "source_date":
            str(
                source_date.date()
            ),

        "target_date":
            target_date_string,

        # --------------------------
        # Model forecasts
        # --------------------------

        "ml_forecast_mm":
            round(
                ml_rainfall,
                2,
            ),

        "gfs_forecast_mm":
            round(
                gfs_rainfall,
                2,
            ),

        "astrablend_mm":
            round(
                astrablend_rainfall,
                2,
            ),

        # --------------------------
        # Blend
        # --------------------------

        "blend_weights": {
            "ml": 0.60,
            "gfs": 0.40,
        },

        # --------------------------
        # Final rainfall risk
        # --------------------------

        "rainfall_category":
            rainfall_risk[
                "category"
            ],

        "risk_level":
            rainfall_risk[
                "risk_level"
            ],

        "rainfall_alert":
            rainfall_risk[
                "alert"
            ],

        "alert_message":
            rainfall_risk[
                "message"
            ],

        # --------------------------
        # Separate ML hazard model
        # --------------------------

        "heavy_probability_percent":
            round(
                heavy_probability
                * 100,
                2,
            ),

        "heavy_classifier_warning":
            heavy_classifier_warning,

        # --------------------------
        # Metadata
        # --------------------------

        "gfs_source":
            gfs_result[
                "source"
            ],

        "gfs_retrieved_at":
            gfs_result[
                "retrieved_at"
            ],

        "blend_ready":
            True,

        "model":
            "AstraBlend",

        "forecast_method":
            "60% ML + 40% GFS",
    }


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    result = generate_live_astrablend(
        name="Sambalpur, Odisha",
        latitude=21.47,
        longitude=83.97,
        source_date="2026-08-31",
    )

    print()
    print(
        "===== LIVE ASTRABLEND ====="
    )
    print()

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )