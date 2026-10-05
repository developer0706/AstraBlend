from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from backend.live_gfs import fetch_live_gfs, daily_precipitation_total

from backend.live_imd import (
    read_imd_daily_binary,
    nearest_grid_rainfall,
    build_live_ml_features,
    feature_dict_to_array,
    LATITUDES,
    LONGITUDES,
)


# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FORECAST_DIR = PROJECT_ROOT / "data" / "forecasts"
LIVE_IMD_DIR = PROJECT_ROOT / "data" / "live_imd"

RAINFALL_MODEL_FILE = MODEL_DIR / "rainfall_model.joblib"
HEAVY_MODEL_FILE = MODEL_DIR / "heavy_rain_model.joblib"
FEATURES_FILE = MODEL_DIR / "features.joblib"

FORECAST_DATA_FILE = (
    DATA_DIR / "rainfall_forecast_dataset.parquet"
)

IMD_2024_FILE = (
    DATA_DIR / "india_rainfall_2024.nc"
)

GFS_FORECAST_FILE = (
    FORECAST_DIR / "gfs_india_f024.parquet"
)


# ==========================================================
# AstraBlend configuration
# ==========================================================

ASTRABLEND_ML_WEIGHT = 0.60
ASTRABLEND_GFS_WEIGHT = 0.40

HEAVY_RAIN_THRESHOLD = 0.15


# ==========================================================
# FastAPI
# ==========================================================

app = FastAPI(
    title="AstraBlend API",
    description=(
        "Hybrid AI-NWP rainfall forecasting API for India"
    ),
    version="1.3.0",
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Load trained models
# ==========================================================

try:

    rainfall_model = load(
        RAINFALL_MODEL_FILE
    )

    heavy_rain_model = load(
        HEAVY_MODEL_FILE
    )

    FEATURES = load(
        FEATURES_FILE
    )

    MODELS_LOADED = True
    MODEL_LOAD_ERROR = None

except Exception as error:

    rainfall_model = None
    heavy_rain_model = None
    FEATURES = None

    MODELS_LOADED = False
    MODEL_LOAD_ERROR = str(error)


# ==========================================================
# Helper: GFS rainfall column
# ==========================================================

def get_gfs_rainfall_column(
    gfs: pd.DataFrame,
):
    if "gfs_precip_mm" in gfs.columns:
        return "gfs_precip_mm"

    if "gfs_precip" in gfs.columns:
        return "gfs_precip"

    raise RuntimeError(
        "Could not find GFS precipitation column. "
        f"Available columns: {list(gfs.columns)}"
    )


# ==========================================================
# Helper: load gridded GFS parquet (used by forecast map)
# ==========================================================

def load_gfs():
    if not GFS_FORECAST_FILE.exists():
        raise FileNotFoundError(
            "GFS forecast file not found: "
            f"{GFS_FORECAST_FILE}"
        )

    gfs = pd.read_parquet(
        GFS_FORECAST_FILE
    )

    if gfs.empty:
        raise RuntimeError(
            "GFS forecast file contains no records."
        )

    get_gfs_rainfall_column(
        gfs
    )

    return gfs


# ==========================================================
# Helper: live GFS forecast for one requested location
# ==========================================================

def get_gfs_forecast(
    latitude: float,
    longitude: float,
):
    forecast = fetch_live_gfs(
        latitude,
        longitude,
        forecast_days=2,
    )

    hourly = forecast.get("hourly", [])

    dates = sorted(
        {
            item["time"][:10]
            for item in hourly
            if item.get("time")
        }
    )

    if len(dates) < 2:
        raise RuntimeError(
            "Live GFS did not return a next-day forecast."
        )

    target_date = dates[0]

    rainfall_mm = daily_precipitation_total(
        forecast,
        target_date,
    )

    return {
        "latitude": forecast["grid_latitude"],
        "longitude": forecast["grid_longitude"],
        "rainfall_mm": max(0.0, float(rainfall_mm)),
        "valid_time": f"{target_date} 00:00:00",
        "forecast_time": forecast["retrieved_at"],
        "lead_hours": 24,
        "source": forecast["source"],
    }


# ==========================================================
# Helper: IMD live filename
# ==========================================================

def get_live_imd_file(
    date,
):

    date = pd.Timestamp(
        date
    )

    filename = (
        "imd_rainfall_"
        f"{date.strftime('%Y%m%d')}"
        ".bin"
    )

    return (
        LIVE_IMD_DIR
        / filename
    )


# ==========================================================
# Helper: read 8-day rainfall history at one location
# ==========================================================

def get_live_rainfall_history(
    latitude: float,
    longitude: float,
    source_date,
):

    source_date = pd.Timestamp(
        source_date
    ).normalize()

    rainfall_history = {}

    missing_files = []

    grid_latitude = None
    grid_longitude = None

    for days_back in range(
        7,
        -1,
        -1,
    ):

        date = (
            source_date
            - pd.Timedelta(
                days=days_back
            )
        )

        file_path = (
            get_live_imd_file(
                date
            )
        )

        if not file_path.exists():

            missing_files.append(
                file_path.name
            )

            continue

        rainfall_grid = (
            read_imd_daily_binary(
                file_path
            )
        )

        grid_result = (
            nearest_grid_rainfall(
                rainfall_grid,
                latitude,
                longitude,
            )
        )

        if (
            grid_result[
                "rainfall_mm"
            ]
            is None
        ):

            raise RuntimeError(
                "IMD rainfall value is missing "
                f"for {date.date()} at the "
                "requested grid point."
            )

        rainfall_history[
            str(date.date())
        ] = (
            grid_result[
                "rainfall_mm"
            ]
        )

        grid_latitude = (
            grid_result[
                "latitude"
            ]
        )

        grid_longitude = (
            grid_result[
                "longitude"
            ]
        )

    return {

        "rainfall_history":
            rainfall_history,

        "missing_files":
            missing_files,

        "grid_latitude":
            grid_latitude,

        "grid_longitude":
            grid_longitude,
    }


# ==========================================================
# Helper: check 8 required IMD files
# ==========================================================

def get_required_imd_files(
    source_date,
):

    source_date = pd.Timestamp(
        source_date
    ).normalize()

    files = []

    missing = []

    for days_back in range(
        7,
        -1,
        -1,
    ):

        date = (
            source_date
            - pd.Timedelta(
                days=days_back
            )
        )

        path = get_live_imd_file(
            date
        )

        files.append(
            {
                "date":
                    date,

                "path":
                    path,
            }
        )

        if not path.exists():

            missing.append(
                path.name
            )

    return (
        files,
        missing,
    )


# ==========================================================
# Helper: build live ML grid for entire India
# ==========================================================

def build_live_ml_grid(
    source_date,
):

    """
    Build all ML features for the IMD 0.25 degree grid.

    The 8 daily IMD files are loaded exactly once.

    Feature order matches rainfall_model.joblib:

        latitude
        longitude
        rainfall_mm
        rainfall_lag_1d
        rainfall_lag_3d
        rainfall_rolling_3d
        rainfall_rolling_7d
        month
        day_of_year
        monsoon
    """

    source_date = pd.Timestamp(
        source_date
    ).normalize()

    (
        required_files,
        missing_files,
    ) = get_required_imd_files(
        source_date
    )

    if missing_files:

        return None, missing_files

    # ------------------------------------------------------
    # Load all 8 grids once
    # ------------------------------------------------------

    grids = {}

    for item in required_files:

        date = item[
            "date"
        ]

        path = item[
            "path"
        ]

        grids[
            date
        ] = read_imd_daily_binary(
            path
        )

    # ------------------------------------------------------
    # Dates used by features
    # ------------------------------------------------------

    d0 = source_date

    d1 = (
        source_date
        - pd.Timedelta(days=1)
    )

    d2 = (
        source_date
        - pd.Timedelta(days=2)
    )

    d3 = (
        source_date
        - pd.Timedelta(days=3)
    )

    d4 = (
        source_date
        - pd.Timedelta(days=4)
    )

    d5 = (
        source_date
        - pd.Timedelta(days=5)
    )

    d6 = (
        source_date
        - pd.Timedelta(days=6)
    )

    # ------------------------------------------------------
    # Current and lagged rainfall
    # ------------------------------------------------------

    rainfall_mm = grids[
        d0
    ]

    rainfall_lag_1d = grids[
        d1
    ]

    rainfall_lag_3d = grids[
        d3
    ]

    # ------------------------------------------------------
    # Rolling 3-day mean
    # ------------------------------------------------------

    rolling_3_stack = np.stack(
        [
            grids[d0],
            grids[d1],
            grids[d2],
        ],
        axis=0,
    )

    rainfall_rolling_3d = (
        np.mean(
            rolling_3_stack,
            axis=0,
        )
    )

    # ------------------------------------------------------
    # Rolling 7-day mean
    # ------------------------------------------------------

    rolling_7_stack = np.stack(
        [
            grids[d0],
            grids[d1],
            grids[d2],
            grids[d3],
            grids[d4],
            grids[d5],
            grids[d6],
        ],
        axis=0,
    )

    rainfall_rolling_7d = (
        np.mean(
            rolling_7_stack,
            axis=0,
        )
    )

    # ------------------------------------------------------
    # Coordinate grid
    # ------------------------------------------------------

    longitude_grid, latitude_grid = (
        np.meshgrid(
            LONGITUDES,
            LATITUDES,
        )
    )

    # ------------------------------------------------------
    # Valid cells
    # ------------------------------------------------------

    valid = (
        np.isfinite(
            rainfall_mm
        )
        &
        np.isfinite(
            rainfall_lag_1d
        )
        &
        np.isfinite(
            rainfall_lag_3d
        )
        &
        np.isfinite(
            rainfall_rolling_3d
        )
        &
        np.isfinite(
            rainfall_rolling_7d
        )
    )

    # ------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------

    month = int(
        source_date.month
    )

    day_of_year = int(
        source_date.dayofyear
    )

    monsoon = int(
        month in [
            6,
            7,
            8,
            9,
        ]
    )

    # ------------------------------------------------------
    # Build dataframe
    # ------------------------------------------------------

    ml_grid = pd.DataFrame(
        {

            "latitude":
                latitude_grid[
                    valid
                ].astype(float),

            "longitude":
                longitude_grid[
                    valid
                ].astype(float),

            "rainfall_mm":
                rainfall_mm[
                    valid
                ].astype(float),

            "rainfall_lag_1d":
                rainfall_lag_1d[
                    valid
                ].astype(float),

            "rainfall_lag_3d":
                rainfall_lag_3d[
                    valid
                ].astype(float),

            "rainfall_rolling_3d":
                rainfall_rolling_3d[
                    valid
                ].astype(float),

            "rainfall_rolling_7d":
                rainfall_rolling_7d[
                    valid
                ].astype(float),
        }
    )

    ml_grid[
        "month"
    ] = month

    ml_grid[
        "day_of_year"
    ] = day_of_year

    ml_grid[
        "monsoon"
    ] = monsoon

    # ------------------------------------------------------
    # ML feature matrix
    # ------------------------------------------------------

    feature_columns = [
        "latitude",
        "longitude",
        "rainfall_mm",
        "rainfall_lag_1d",
        "rainfall_lag_3d",
        "rainfall_rolling_3d",
        "rainfall_rolling_7d",
        "month",
        "day_of_year",
        "monsoon",
    ]

    X = ml_grid[
        feature_columns
    ].to_numpy(
        dtype=float
    )

    # ------------------------------------------------------
    # Batch rainfall predictions
    # ------------------------------------------------------

    ml_predictions = (
        rainfall_model.predict(
            X
        )
    )

    ml_predictions = np.maximum(
        ml_predictions,
        0.0,
    )

    ml_grid[
        "ml_forecast_rainfall_mm"
    ] = ml_predictions

    # ------------------------------------------------------
    # Batch heavy rain probabilities
    # ------------------------------------------------------

    heavy_probabilities = (
        heavy_rain_model.predict_proba(
            X
        )[:, 1]
    )

    ml_grid[
        "heavy_rain_probability"
    ] = heavy_probabilities

    ml_grid[
        "heavy_rain_warning"
    ] = (
        heavy_probabilities
        >= HEAVY_RAIN_THRESHOLD
    )

    return (
        ml_grid,
        [],
    )


# ==========================================================
# Helper: AstraBlend calculation
# ==========================================================

def calculate_astrablend(
    ml_rainfall_mm,
    gfs_rainfall_mm,
):

    result = (
        ASTRABLEND_ML_WEIGHT
        * ml_rainfall_mm
        +
        ASTRABLEND_GFS_WEIGHT
        * gfs_rainfall_mm
    )

    return max(
        0.0,
        float(result),
    )


# ==========================================================
# Root
# ==========================================================

@app.get("/")
def root():

    return {

        "project":
            "AstraBlend",

        "status":
            "running",

        "version":
            "1.3.0",

        "models_loaded":
            MODELS_LOADED,

        "gfs_available":
            GFS_FORECAST_FILE.exists(),

        "live_imd_directory":
            str(
                LIVE_IMD_DIR
            ),

        "blend_weights": {

            "ml":
                ASTRABLEND_ML_WEIGHT,

            "gfs":
                ASTRABLEND_GFS_WEIGHT,
        },
    }


# ==========================================================
# Health
# ==========================================================

@app.get("/health")
def health():

    return {

        "project":
            "AstraBlend",

        "status":
            "healthy",

        "version":
            "1.3.0",

        "models_loaded":
            MODELS_LOADED,

        "model_error":
            MODEL_LOAD_ERROR,

        "gfs_available":
            GFS_FORECAST_FILE.exists(),

        "live_imd_directory_exists":
            LIVE_IMD_DIR.exists(),
    }


# ==========================================================
# Historical rainfall map
# ==========================================================

@app.get("/rainfall-map")
def rainfall_map():

    try:

        if not IMD_2024_FILE.exists():

            raise FileNotFoundError(
                f"IMD file not found: "
                f"{IMD_2024_FILE}"
            )

        dataset = xr.open_dataset(
            IMD_2024_FILE
        )

        rainfall = (
            dataset[
                "rainfall"
            ]
            .sel(
                TIME="2024-08-01"
            )
        )

        values = rainfall.values

        latitudes = (
            dataset[
                "LATITUDE"
            ].values
        )

        longitudes = (
            dataset[
                "LONGITUDE"
            ].values
        )

        points = []

        for i, latitude in enumerate(
            latitudes
        ):

            for j, longitude in enumerate(
                longitudes
            ):

                value = values[
                    i,
                    j,
                ]

                if np.isfinite(
                    value
                ):

                    points.append(
                        {

                            "latitude":
                                float(
                                    latitude
                                ),

                            "longitude":
                                float(
                                    longitude
                                ),

                            "rainfall_mm":
                                float(
                                    value
                                ),
                        }
                    )

        dataset.close()

        return {

            "date":
                "2024-08-01",

            "units":
                "mm",

            "grid":
                "IMD 0.25 degree",

            "points":
                points,
        }

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=(
                "Rainfall map failed: "
                f"{error}"
            ),
        )


# ==========================================================
# Single-location live forecast
# ==========================================================

@app.post("/forecast")
def forecast(
    latitude: float,
    longitude: float,
):

    if not MODELS_LOADED:

        raise HTTPException(

            status_code=503,

            detail=(
                "Models unavailable: "
                f"{MODEL_LOAD_ERROR}"
            ),
        )

    if not (
        -90 <= latitude <= 90
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid latitude.",
        )

    if not (
        -180 <= longitude <= 180
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid longitude.",
        )

    try:

        # ==================================================
        # GFS forecast
        # ==================================================

        gfs_result = (
            get_gfs_forecast(
                latitude,
                longitude,
            )
        )

        if (
            "valid_time"
            not in gfs_result
        ):

            raise RuntimeError(
                "GFS valid_time is missing."
            )

        gfs_valid_time = (
            pd.Timestamp(
                gfs_result[
                    "valid_time"
                ]
            )
        )

        target_date = (
            gfs_valid_time.normalize()
        )

        source_date = (
            target_date
            - pd.Timedelta(
                days=1
            )
        )

        # ==================================================
        # Live IMD history
        # ==================================================

        live_data = (
            get_live_rainfall_history(
                latitude,
                longitude,
                source_date,
            )
        )

        if live_data[
            "missing_files"
        ]:

            return {

                "model":
                    "AstraBlend v1.3",

                "target_date":
                    str(
                        target_date.date()
                    ),

                "required_ml_source_date":
                    str(
                        source_date.date()
                    ),

                "gfs_forecast_rainfall_mm":
                    round(
                        gfs_result[
                            "rainfall_mm"
                        ],
                        2,
                    ),

                "ml_forecast_rainfall_mm":
                    None,

                "astrablend_forecast_rainfall_mm":
                    None,

                "blend_ready":
                    False,

                "date_aligned":
                    False,

                "missing_imd_files":
                    live_data[
                        "missing_files"
                    ],

                "status":
                    (
                        "Live IMD rainfall files "
                        "are required before the "
                        "ML forecast can be generated."
                    ),
            }

        # ==================================================
        # Build ML features
        # ==================================================

        features = (
            build_live_ml_features(

                live_data[
                    "rainfall_history"
                ],

                source_date,

                live_data[
                    "grid_latitude"
                ],

                live_data[
                    "grid_longitude"
                ],
            )
        )

        feature_array = (
            feature_dict_to_array(
                features
            )
        )

        # ==================================================
        # ML rainfall forecast
        # ==================================================

        ml_rainfall = float(

            rainfall_model.predict(
                feature_array
            )[0]

        )

        ml_rainfall = max(
            0.0,
            ml_rainfall,
        )

        # ==================================================
        # Heavy rain classifier
        # ==================================================

        heavy_probability = float(

            heavy_rain_model.predict_proba(
                feature_array
            )[0, 1]

        )

        # ==================================================
        # AstraBlend
        # ==================================================

        gfs_rainfall = (
            gfs_result[
                "rainfall_mm"
            ]
        )

        astrablend_rainfall = (
            calculate_astrablend(
                ml_rainfall,
                gfs_rainfall,
            )
        )

        # ==================================================
        # Response
        # ==================================================

        response = {

            "requested_latitude":
                latitude,

            "requested_longitude":
                longitude,

            "imd_grid_latitude":
                live_data[
                    "grid_latitude"
                ],

            "imd_grid_longitude":
                live_data[
                    "grid_longitude"
                ],

            "gfs_grid_latitude":
                gfs_result[
                    "latitude"
                ],

            "gfs_grid_longitude":
                gfs_result[
                    "longitude"
                ],

            "ml_source_date":
                str(
                    source_date.date()
                ),

            "forecast_target_date":
                str(
                    target_date.date()
                ),

            "gfs_valid_time":
                str(
                    gfs_valid_time
                ),

            "ml_forecast_rainfall_mm":
                round(
                    ml_rainfall,
                    2,
                ),

            "gfs_forecast_rainfall_mm":
                round(
                    gfs_rainfall,
                    2,
                ),

            "astrablend_forecast_rainfall_mm":
                round(
                    astrablend_rainfall,
                    2,
                ),

            "blend_weights": {

                "ml":
                    ASTRABLEND_ML_WEIGHT,

                "gfs":
                    ASTRABLEND_GFS_WEIGHT,
            },

            "heavy_rain_probability":
                round(
                    heavy_probability,
                    4,
                ),

            "heavy_rain_probability_percent":
                round(
                    heavy_probability
                    * 100,
                    2,
                ),

            "heavy_rain_warning":
                bool(
                    heavy_probability
                    >=
                    HEAVY_RAIN_THRESHOLD
                ),

            "date_aligned":
                True,

            "blend_ready":
                True,

            "forecast_method":
                "60% ML + 40% GFS",

            "model":
                "AstraBlend v1.3",
        }

        if (
            "forecast_time"
            in gfs_result
        ):

            response[
                "gfs_forecast_time"
            ] = str(
                gfs_result[
                    "forecast_time"
                ]
            )

        if (
            "lead_hours"
            in gfs_result
        ):

            response[
                "gfs_lead_hours"
            ] = int(
                gfs_result[
                    "lead_hours"
                ]
            )

        return response

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=(
                "Forecast generation failed: "
                f"{error}"
            ),
        )


# ==========================================================
# Live AstraBlend forecast map
# ==========================================================

@app.get("/forecast-map")
def forecast_map():

    """
    Generate a date-aligned AstraBlend rainfall forecast
    for all common IMD/GFS 0.25 degree grid cells.

    IMD live rainfall:
        -> batch ML features
        -> batch ML prediction

    GFS:
        -> +24 hour NWP prediction

    Final rainfall:
        60% ML + 40% GFS

    Heavy-rain probability remains a separate classifier.
    """

    if not MODELS_LOADED:

        raise HTTPException(

            status_code=503,

            detail=(
                "Models unavailable: "
                f"{MODEL_LOAD_ERROR}"
            ),
        )

    try:

        # ==================================================
        # 1. Load GFS
        # ==================================================

        gfs = load_gfs()

        rainfall_column = (
            get_gfs_rainfall_column(
                gfs
            )
        )

        # ==================================================
        # 2. Determine GFS valid date
        # ==================================================

        if "valid_time" not in gfs.columns:

            raise RuntimeError(
                "GFS valid_time column is missing."
            )

        gfs_valid_time = pd.Timestamp(
            gfs[
                "valid_time"
            ].iloc[0]
        )

        target_date = (
            gfs_valid_time.normalize()
        )

        source_date = (
            target_date
            - pd.Timedelta(
                days=1
            )
        )

        # ==================================================
        # 3. Build batch ML grid
        # ==================================================

        (
            ml_grid,
            missing_files,
        ) = build_live_ml_grid(
            source_date
        )

        if missing_files:

            return {

                "model":
                    "AstraBlend v1.3",

                "forecast_target_date":
                    str(
                        target_date.date()
                    ),

                "required_ml_source_date":
                    str(
                        source_date.date()
                    ),

                "blend_ready":
                    False,

                "date_aligned":
                    False,

                "missing_imd_files":
                    missing_files,

                "points":
                    [],

                "status":
                    (
                        "Required live IMD rainfall "
                        "files are missing."
                    ),
            }

        # ==================================================
        # 4. Prepare GFS grid
        # ==================================================

        gfs_map = gfs[
            [
                "latitude",
                "longitude",
                rainfall_column,
            ]
        ].copy()

        gfs_map = gfs_map.rename(
            columns={
                rainfall_column:
                    "gfs_forecast_rainfall_mm"
            }
        )

        gfs_map[
            "gfs_forecast_rainfall_mm"
        ] = (
            pd.to_numeric(
                gfs_map[
                    "gfs_forecast_rainfall_mm"
                ],
                errors="coerce",
            )
            .clip(
                lower=0
            )
        )

        # --------------------------------------------------
        # Quarter-degree key
        # --------------------------------------------------

        ml_grid[
            "lat_key"
        ] = (
            ml_grid[
                "latitude"
            ].round(2)
        )

        ml_grid[
            "lon_key"
        ] = (
            ml_grid[
                "longitude"
            ].round(2)
        )

        gfs_map[
            "lat_key"
        ] = (
            gfs_map[
                "latitude"
            ].round(2)
        )

        gfs_map[
            "lon_key"
        ] = (
            gfs_map[
                "longitude"
            ].round(2)
        )

        # ==================================================
        # 5. Join ML and GFS on common grid points
        # ==================================================

        combined = pd.merge(

            ml_grid,

            gfs_map[
                [
                    "lat_key",
                    "lon_key",
                    "gfs_forecast_rainfall_mm",
                ]
            ],

            on=[
                "lat_key",
                "lon_key",
            ],

            how="inner",
        )

        if combined.empty:

            raise RuntimeError(
                "No common IMD/GFS grid points were found."
            )

        # ==================================================
        # 6. AstraBlend batch calculation
        # ==================================================

        combined[
            "astrablend_forecast_rainfall_mm"
        ] = (

            ASTRABLEND_ML_WEIGHT
            *
            combined[
                "ml_forecast_rainfall_mm"
            ]

            +

            ASTRABLEND_GFS_WEIGHT
            *
            combined[
                "gfs_forecast_rainfall_mm"
            ]
        )

        combined[
            "astrablend_forecast_rainfall_mm"
        ] = combined[
            "astrablend_forecast_rainfall_mm"
        ].clip(
            lower=0
        )

        # ==================================================
        # 7. Build JSON map points
        # ==================================================

        points = []

        for row in combined.itertuples(
            index=False
        ):

            heavy_probability = float(
                row.heavy_rain_probability
            )

            astrablend_value = float(
                row.astrablend_forecast_rainfall_mm
            )

            points.append(
                {

                    "latitude":
                        round(
                            float(
                                row.latitude
                            ),
                            2,
                        ),

                    "longitude":
                        round(
                            float(
                                row.longitude
                            ),
                            2,
                        ),

                    # Compatibility field for existing map
                    "rainfall_mm":
                        round(
                            astrablend_value,
                            2,
                        ),

                    "ml_forecast_rainfall_mm":
                        round(
                            float(
                                row.ml_forecast_rainfall_mm
                            ),
                            2,
                        ),

                    "gfs_forecast_rainfall_mm":
                        round(
                            float(
                                row.gfs_forecast_rainfall_mm
                            ),
                            2,
                        ),

                    "astrablend_forecast_rainfall_mm":
                        round(
                            astrablend_value,
                            2,
                        ),

                    "heavy_rain_probability":
                        round(
                            heavy_probability,
                            4,
                        ),

                    "heavy_rain_probability_percent":
                        round(
                            heavy_probability
                            * 100,
                            2,
                        ),

                    "heavy_rain_warning":
                        bool(
                            row.heavy_rain_warning
                        ),
                }
            )

        # ==================================================
        # 8. Summary statistics
        # ==================================================

        warning_count = int(
            combined[
                "heavy_rain_warning"
            ].sum()
        )

        mean_ml = float(
            combined[
                "ml_forecast_rainfall_mm"
            ].mean()
        )

        mean_gfs = float(
            combined[
                "gfs_forecast_rainfall_mm"
            ].mean()
        )

        mean_blend = float(
            combined[
                "astrablend_forecast_rainfall_mm"
            ].mean()
        )

        max_blend = float(
            combined[
                "astrablend_forecast_rainfall_mm"
            ].max()
        )

        # ==================================================
        # 9. Response
        # ==================================================

        response = {

            "model":
                "AstraBlend v1.3",

            "forecast_method":
                "60% ML + 40% GFS",

            "blend_weights": {

                "ml":
                    ASTRABLEND_ML_WEIGHT,

                "gfs":
                    ASTRABLEND_GFS_WEIGHT,
            },

            "ml_source_date":
                str(
                    source_date.date()
                ),

            "forecast_target_date":
                str(
                    target_date.date()
                ),

            "gfs_valid_time":
                str(
                    gfs_valid_time
                ),

            "date_aligned":
                True,

            "blend_ready":
                True,

            "grid":
                "0.25 degree",

            "point_count":
                len(
                    points
                ),

            "heavy_rain_warning_count":
                warning_count,

            "summary": {

                "mean_ml_rainfall_mm":
                    round(
                        mean_ml,
                        2,
                    ),

                "mean_gfs_rainfall_mm":
                    round(
                        mean_gfs,
                        2,
                    ),

                "mean_astrablend_rainfall_mm":
                    round(
                        mean_blend,
                        2,
                    ),

                "max_astrablend_rainfall_mm":
                    round(
                        max_blend,
                        2,
                    ),
            },

            "points":
                points,
        }

        # ==================================================
        # Optional GFS metadata
        # ==================================================

        if (
            "forecast_time"
            in gfs.columns
        ):

            response[
                "gfs_forecast_time"
            ] = str(
                pd.Timestamp(
                    gfs[
                        "forecast_time"
                    ].iloc[0]
                )
            )

        if (
            "lead_hours"
            in gfs.columns
        ):

            response[
                "gfs_lead_hours"
            ] = int(
                gfs[
                    "lead_hours"
                ].iloc[0]
            )

        return response

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=(
                "Forecast map generation failed: "
                f"{error}"
            ),
        )
# ==========================================================
# Live regional AstraBlend risk
# ==========================================================

@app.get("/live-risk-regions")
def live_risk_regions():

    try:

        from backend.live_astrablend_regions import (
            scan_live_astrablend_regions,
        )

        results = (
            scan_live_astrablend_regions(
                source_date="2026-08-31",
            )
        )

        return {
            "model": "AstraBlend",
            "forecast_method": "60% ML + 40% GFS",
            "source_date": "2026-08-31",
            "forecast_target_date": "2026-09-01",
            "regions": results,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Live regional AstraBlend "
                f"forecast failed: {error}"
            ),
        )