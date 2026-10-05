from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = DATA_DIR / "train.parquet"
VALIDATION_FILE = DATA_DIR / "validation.parquet"

RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 500_000

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

TARGET = "target_rainfall_mm"


def main() -> None:
    print("===== ASTRA BLEND: FIRST ML MODEL =====")

    print("Loading training data...")
    train = pd.read_parquet(TRAIN_FILE)

    print(f"Training rows available: {len(train):,}")

    if len(train) > TRAIN_SAMPLE_SIZE:
        train = train.sample(
            n=TRAIN_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    print(f"Training sample: {len(train):,}")
    print(f"Number of features: {len(FEATURES)}")

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    print()
    print("Training model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    print()
    print("Loading validation data...")
    validation = pd.read_parquet(VALIDATION_FILE)

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    print("Generating predictions...")
    predictions = model.predict(X_validation)

    mae = mean_absolute_error(
        y_validation,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            predictions,
        )
    )

    print()
    print("===== VALIDATION RESULTS =====")
    print(f"MAE:  {mae:.4f} mm")
    print(f"RMSE: {rmse:.4f} mm")

    print()
    print("===== PERSISTENCE BASELINE =====")
    print("MAE:  3.9590 mm")
    print("RMSE: 12.2229 mm")

    print()
    if mae < 3.9590:
        print("✅ ML model beats persistence on MAE.")
    else:
        print("❌ ML model does not beat persistence on MAE yet.")

    if rmse < 12.2229:
        print("✅ ML model beats persistence on RMSE.")
    else:
        print("❌ ML model does not beat persistence on RMSE yet.")


if __name__ == "__main__":
    main()