from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ==========================================================
# IMD 0.25 degree grid definition
# ==========================================================

IMD_NLON = 135
IMD_NLAT = 129

IMD_LON_START = 66.5
IMD_LAT_START = 6.5

IMD_STEP = 0.25

IMD_MISSING_VALUE = -999.0


# ==========================================================
# Grid coordinates
# ==========================================================

LONGITUDES = (
    IMD_LON_START
    + np.arange(IMD_NLON) * IMD_STEP
)

LATITUDES = (
    IMD_LAT_START
    + np.arange(IMD_NLAT) * IMD_STEP
)


# ==========================================================
# Read one IMD daily binary rainfall file
# ==========================================================

def read_imd_daily_binary(
    file_path,
):
    """
    Read one IMD 0.25 x 0.25 degree daily rainfall
    binary file.

    IMD grid:
        longitude: 66.5 -> 100.0
        latitude : 6.5 -> 38.5
        resolution: 0.25 degree

    Expected grid:
        135 longitude points
        129 latitude points

    Values are rainfall in millimetres.
    """

    file_path = Path(
        file_path
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"IMD rainfall file not found: {file_path}"
        )

    values = np.fromfile(
        file_path,
        dtype=np.float32,
    )

    expected_values = (
        IMD_NLAT
        * IMD_NLON
    )

    if len(values) != expected_values:
        raise RuntimeError(
            "Unexpected IMD rainfall file size. "
            f"Expected {expected_values} float values, "
            f"found {len(values)}."
        )

    rainfall = values.reshape(
        IMD_NLAT,
        IMD_NLON,
    )

    rainfall[
        rainfall <= IMD_MISSING_VALUE
    ] = np.nan

    return rainfall


# ==========================================================
# Get rainfall at nearest IMD grid point
# ==========================================================

def nearest_grid_rainfall(
    rainfall_grid,
    latitude,
    longitude,
):
    """
    Return rainfall at nearest IMD 0.25 degree grid point.
    """

    lat_index = int(
        np.abs(
            LATITUDES - latitude
        ).argmin()
    )

    lon_index = int(
        np.abs(
            LONGITUDES - longitude
        ).argmin()
    )

    rainfall = rainfall_grid[
        lat_index,
        lon_index,
    ]

    return {
        "latitude": float(
            LATITUDES[
                lat_index
            ]
        ),

        "longitude": float(
            LONGITUDES[
                lon_index
            ]
        ),

        "rainfall_mm": (
            None
            if not np.isfinite(
                rainfall
            )
            else float(
                rainfall
            )
        ),
    }


# ==========================================================
# Build ML features from recent rainfall
# ==========================================================

def build_live_ml_features(
    daily_rainfall,
    target_source_date,
    latitude,
    longitude,
):
    """
    Build the same rainfall features used by the trained
    AstraBlend ML rainfall model.

    daily_rainfall must contain at least the last 8 days.

    Expected dictionary form:

        {
            "2026-08-21": rainfall_mm,
            ...
            "2026-08-28": rainfall_mm
        }

    target_source_date is the day used to predict the
    following day's rainfall.
    """

    target_source_date = pd.Timestamp(
        target_source_date
    ).normalize()

    records = []

    for date_string, rainfall in (
        daily_rainfall.items()
    ):

        date = pd.Timestamp(
            date_string
        ).normalize()

        records.append(
            {
                "date": date,
                "rainfall_mm": float(
                    rainfall
                ),
            }
        )

    data = pd.DataFrame(
        records
    ).sort_values(
        "date"
    )

    data = data.set_index(
        "date"
    )

    if target_source_date not in data.index:
        raise RuntimeError(
            "Target source date missing from "
            "daily rainfall observations."
        )

    # Need enough history for rolling features.
    minimum_date = (
        target_source_date
        - pd.Timedelta(days=7)
    )

    required = data.loc[
        minimum_date:
        target_source_date
    ].copy()

    if len(required) < 8:
        raise RuntimeError(
            "At least 8 consecutive rainfall days "
            "are required to build live ML features."
        )

    # ------------------------------------------------------
    # Current rainfall
    # ------------------------------------------------------

    rainfall_mm = float(
        data.loc[
            target_source_date,
            "rainfall_mm"
        ]
    )

    # ------------------------------------------------------
    # Lag 1 day
    # ------------------------------------------------------

    rainfall_lag_1d = float(
        data.loc[
            target_source_date
            - pd.Timedelta(days=1),
            "rainfall_mm"
        ]
    )

    # ------------------------------------------------------
    # Lag 3 days
    # ------------------------------------------------------

    rainfall_lag_3d = float(
        data.loc[
            target_source_date
            - pd.Timedelta(days=3),
            "rainfall_mm"
        ]
    )

    # ------------------------------------------------------
    # Rolling 3-day rainfall
    # ------------------------------------------------------

    rolling_3_start = (
        target_source_date
        - pd.Timedelta(days=2)
    )

    rainfall_rolling_3d = float(
        data.loc[
            rolling_3_start:
            target_source_date,
            "rainfall_mm"
        ].mean()
    )

    # ------------------------------------------------------
    # Rolling 7-day rainfall
    # ------------------------------------------------------

    rolling_7_start = (
        target_source_date
        - pd.Timedelta(days=6)
    )

    rainfall_rolling_7d = float(
        data.loc[
            rolling_7_start:
            target_source_date,
            "rainfall_mm"
        ].mean()
    )

    month = int(
        target_source_date.month
    )

    day_of_year = int(
        target_source_date.dayofyear
    )

    monsoon = int(
        month in [
            6,
            7,
            8,
            9,
        ]
    )

    return {
        "date":
            target_source_date,

        "latitude":
            float(latitude),

        "longitude":
            float(longitude),

        "rainfall_mm":
            rainfall_mm,

        "rainfall_lag_1d":
            rainfall_lag_1d,

        "rainfall_lag_3d":
            rainfall_lag_3d,

        "rainfall_rolling_3d":
            rainfall_rolling_3d,

        "rainfall_rolling_7d":
            rainfall_rolling_7d,

        "month":
            month,

        "day_of_year":
            day_of_year,

        "monsoon":
            monsoon,
    }


# ==========================================================
# Convert feature dictionary to model array
# ==========================================================

def feature_dict_to_array(
    features,
):
    """
    Convert live features to the exact feature order
    expected by rainfall_model.joblib.
    """

    return np.array(
        [[
            features[
                "latitude"
            ],

            features[
                "longitude"
            ],

            features[
                "rainfall_mm"
            ],

            features[
                "rainfall_lag_1d"
            ],

            features[
                "rainfall_lag_3d"
            ],

            features[
                "rainfall_rolling_3d"
            ],

            features[
                "rainfall_rolling_7d"
            ],

            features[
                "month"
            ],

            features[
                "day_of_year"
            ],

            features[
                "monsoon"
            ],
        ]],
        dtype=float,
    )