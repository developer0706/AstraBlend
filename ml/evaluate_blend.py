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


def train_model(data: pd.DataFrame):
    if len(data) > TRAIN_SAMPLE_SIZE:
        data = data.sample(
            n=TRAIN_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    model.fit(
        data[FEATURES],
        data[TARGET],
    )

    return model


def rmse(actual, predicted):
    return np.sqrt(
        mean_squared_error(actual, predicted)
    )


def main() -> None:
    print("===== ASTRA BLEND: FIRST BLENDING EXPERIMENT =====")

    columns = FEATURES + [TARGET]

    # --------------------------------------------------
    # STEP 1: Train using 2020–2023 only
    # --------------------------------------------------
    print()
    print("Training base ML model on 2020–2023...")

    train = pd.read_parquet(
        TRAIN_FILE,
        columns=columns,
    )

    model = train_model(train)

    # --------------------------------------------------
    # STEP 2: Use 2024 to learn blend weight
    # --------------------------------------------------
    print("Generating 2024 validation predictions...")

    validation = pd.read_parquet(
        VALIDATION_FILE,
        columns=columns,
    )

    ml_validation = model.predict(
        validation[FEATURES]
    )

    persistence_validation = (
        validation["rainfall_mm"].to_numpy()
    )

    actual_validation = (
        validation[TARGET].to_numpy()
    )

    print()
    print("===== SEARCHING FOR BEST 2024 BLEND WEIGHT =====")

    best_weight = None
    best_rmse = float("inf")

    for weight in np.arange(0.0, 1.01, 0.05):

        blended = (
            weight * ml_validation
            + (1.0 - weight) * persistence_validation
        )

        score = rmse(
            actual_validation,
            blended,
        )

        print(
            f"ML weight={weight:.2f} "
            f"Persistence weight={1-weight:.2f} "
            f"RMSE={score:.4f}"
        )

        if score < best_rmse:
            best_rmse = score
            best_weight = weight

    print()
    print("===== SELECTED BLEND =====")
    print(f"ML weight:          {best_weight:.2f}")
    print(
        f"Persistence weight: {1-best_weight:.2f}"
    )
    print(f"Validation RMSE:    {best_rmse:.4f} mm")

    # --------------------------------------------------
    # STEP 3: Refit on 2020–2024
    # --------------------------------------------------
    print()
    print("Training final ML model on 2020–2024...")

    validation_training = pd.read_parquet(
        VALIDATION_FILE,
        columns=columns,
    )

    combined_training = pd.concat(
        [train, validation_training],
        ignore_index=True,
    )

    final_model = train_model(
        combined_training
    )

    # --------------------------------------------------
    # STEP 4: FINAL 2025 TEST
    # --------------------------------------------------
    print("Loading 2025 test data...")

    test = pd.read_parquet(
        TEST_FILE,
        columns=columns,
    )

    ml_test = final_model.predict(
        test[FEATURES]
    )

    persistence_test = (
        test["rainfall_mm"].to_numpy()
    )

    actual_test = (
        test[TARGET].to_numpy()
    )

    blended_test = (
        best_weight * ml_test
        + (1.0 - best_weight) * persistence_test
    )

    ml_mae = mean_absolute_error(
        actual_test,
        ml_test,
    )

    ml_rmse = rmse(
        actual_test,
        ml_test,
    )

    persistence_mae = mean_absolute_error(
        actual_test,
        persistence_test,
    )

    persistence_rmse = rmse(
        actual_test,
        persistence_test,
    )

    blend_mae = mean_absolute_error(
        actual_test,
        blended_test,
    )

    blend_rmse = rmse(
        actual_test,
        blended_test,
    )

    print()
    print("===== 2025 BLEND TEST =====")

    print()
    print("Persistence:")
    print(f"MAE:  {persistence_mae:.4f} mm")
    print(f"RMSE: {persistence_rmse:.4f} mm")

    print()
    print("ML:")
    print(f"MAE:  {ml_mae:.4f} mm")
    print(f"RMSE: {ml_rmse:.4f} mm")

    print()
    print("AstraBlend v0:")
    print(f"MAE:  {blend_mae:.4f} mm")
    print(f"RMSE: {blend_rmse:.4f} mm")

    print()
    print("Improvement vs ML:")

    mae_improvement = (
        (ml_mae - blend_mae)
        / ml_mae
        * 100
    )

    rmse_improvement = (
        (ml_rmse - blend_rmse)
        / ml_rmse
        * 100
    )

    print(f"MAE:  {mae_improvement:.2f}%")
    print(f"RMSE: {rmse_improvement:.2f}%")

    print()
    print("===== BLENDING EXPERIMENT COMPLETE =====")
    print(
        "The blend weight was selected using 2024 only."
    )


if __name__ == "__main__":
    main()