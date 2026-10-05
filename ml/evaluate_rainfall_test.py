from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = DATA_DIR / "train.parquet"
VALIDATION_FILE = DATA_DIR / "validation.parquet"
TEST_FILE = DATA_DIR / "test.parquet"

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

TARGET = "target_rainfall_mm"


def main() -> None:
    print("===== ASTRA BLEND: FINAL RAINFALL TEST =====")

    # --------------------------------------------------
    # Load 2020–2024 data for final training
    # --------------------------------------------------
    print("Loading training data...")

    train = pd.read_parquet(
        TRAIN_FILE,
        columns=FEATURES + [TARGET],
    )

    validation = pd.read_parquet(
        VALIDATION_FILE,
        columns=FEATURES + [TARGET],
    )

    train_final = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    print(
        f"Training rows available (2020–2024): "
        f"{len(train_final):,}"
    )

    if len(train_final) > TRAIN_SAMPLE_SIZE:
        train_final = train_final.sample(
            n=TRAIN_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    X_train = train_final[FEATURES]
    y_train = train_final[TARGET]

    print(
        f"Final training sample: "
        f"{len(train_final):,}"
    )

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------
    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    print()
    print("Training final rainfall model...")

    model.fit(X_train, y_train)

    print("Training complete.")

    # --------------------------------------------------
    # 2025 FINAL TEST
    # --------------------------------------------------
    print()
    print("Loading 2025 test data...")

    test = pd.read_parquet(
        TEST_FILE,
        columns=FEATURES + [TARGET],
    )

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print("Generating predictions...")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    # --------------------------------------------------
    # Persistence baseline
    # --------------------------------------------------
    persistence_mae = mean_absolute_error(
        y_test,
        test["rainfall_mm"],
    )

    persistence_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test["rainfall_mm"],
        )
    )

    mae_improvement = (
        (persistence_mae - mae)
        / persistence_mae
        * 100
    )

    rmse_improvement = (
        (persistence_rmse - rmse)
        / persistence_rmse
        * 100
    )

    print()
    print("===== 2025 FINAL TEST RESULTS =====")
    print(f"Samples: {len(test):,}")
    print()
    print("ML model:")
    print(f"MAE:  {mae:.4f} mm")
    print(f"RMSE: {rmse:.4f} mm")

    print()
    print("Persistence:")
    print(f"MAE:  {persistence_mae:.4f} mm")
    print(f"RMSE: {persistence_rmse:.4f} mm")

    print()
    print("Improvement over persistence:")
    print(f"MAE:  {mae_improvement:.2f}%")
    print(f"RMSE: {rmse_improvement:.2f}%")

    print()
    print("===== FINAL TEST COMPLETE =====")
    print("2025 was not used for model selection.")


if __name__ == "__main__":
    main()