from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = DATA_DIR / "train.parquet"
VALIDATION_FILE = DATA_DIR / "validation.parquet"

RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 1_000_000

FEATURES = [
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

RAINFALL_TARGET = "target_rainfall_mm"
HEAVY_TARGET = "target_heavy_rain"


def load_final_training_data() -> pd.DataFrame:
    print("Loading 2020–2024 training data...")

    train = pd.read_parquet(
        TRAIN_FILE,
        columns=FEATURES + [
            RAINFALL_TARGET,
            HEAVY_TARGET,
        ],
    )

    validation = pd.read_parquet(
        VALIDATION_FILE,
        columns=FEATURES + [
            RAINFALL_TARGET,
            HEAVY_TARGET,
        ],
    )

    combined = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    if len(combined) > TRAIN_SAMPLE_SIZE:
        combined = combined.sample(
            n=TRAIN_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    return combined


def main() -> None:
    print("===== ASTRA BLEND: SAVING PRODUCTION MODELS =====")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_final_training_data()

    X = df[FEATURES]

    # --------------------------------------------------
    # Rainfall regression model
    # --------------------------------------------------
    print()
    print("Training rainfall model...")

    rainfall_model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    rainfall_model.fit(
        X,
        df[RAINFALL_TARGET],
    )

    dump(
        rainfall_model,
        MODEL_DIR / "rainfall_model.joblib",
    )

    print("Saved rainfall_model.joblib")

    # --------------------------------------------------
    # Heavy-rain classifier
    # --------------------------------------------------
    print()
    print("Training heavy-rain model...")

    heavy_model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    heavy_model.fit(
        X,
        df[HEAVY_TARGET].astype("int8"),
    )

    dump(
        heavy_model,
        MODEL_DIR / "heavy_rain_model.joblib",
    )

    print("Saved heavy_rain_model.joblib")

    # Save the feature definition so the API uses
    # exactly the same feature order as training.
    dump(
        FEATURES,
        MODEL_DIR / "features.joblib",
    )

    print("Saved features.joblib")

    print()
    print("===== MODELS SAVED =====")
    print(f"Directory: {MODEL_DIR}")
    print("Files:")
    print(" - rainfall_model.joblib")
    print(" - heavy_rain_model.joblib")
    print(" - features.joblib")


if __name__ == "__main__":
    main()