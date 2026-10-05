from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


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


def rmse(actual, predicted):
    return np.sqrt(
        mean_squared_error(actual, predicted)
    )


def train_model(data):
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


def find_best_weight(actual, ml_prediction, persistence):
    best_weight = 0.0
    best_score = float("inf")

    for weight in np.arange(0.0, 1.01, 0.05):
        blended = (
            weight * ml_prediction
            + (1.0 - weight) * persistence
        )

        score = rmse(actual, blended)

        if score < best_score:
            best_score = score
            best_weight = weight

    return best_weight, best_score


def main():
    print("===== ASTRA BLEND: ADAPTIVE WEIGHT EXPERIMENT =====")

    columns = FEATURES + [TARGET]

    # --------------------------------------------------
    # Train on 2020–2023
    # --------------------------------------------------
    train = pd.read_parquet(
        TRAIN_FILE,
        columns=columns,
    )

    model = train_model(train)

    # --------------------------------------------------
    # Learn regime-specific weights on 2024
    # --------------------------------------------------
    validation = pd.read_parquet(
        VALIDATION_FILE,
        columns=columns,
    )

    validation["ml_prediction"] = model.predict(
        validation[FEATURES]
    )

    validation["persistence_prediction"] = (
        validation["rainfall_mm"]
    )

    weights = {}

    print()
    print("===== LEARNING 2024 REGIME WEIGHTS =====")

    for regime_name, subset in [
        ("non_monsoon", validation[validation["monsoon"] == 0]),
        ("monsoon", validation[validation["monsoon"] == 1]),
    ]:

        weight, score = find_best_weight(
            subset[TARGET].to_numpy(),
            subset["ml_prediction"].to_numpy(),
            subset["persistence_prediction"].to_numpy(),
        )

        weights[regime_name] = weight

        print(
            f"{regime_name}: "
            f"ML={weight:.2f}, "
            f"Persistence={1-weight:.2f}, "
            f"RMSE={score:.4f}"
        )

    # --------------------------------------------------
    # Final training: 2020–2024
    # --------------------------------------------------
    validation_training = pd.read_parquet(
        VALIDATION_FILE,
        columns=columns,
    )

    final_training = pd.concat(
        [train, validation_training],
        ignore_index=True,
    )

    final_model = train_model(final_training)

    # --------------------------------------------------
    # Final 2025 test
    # --------------------------------------------------
    test = pd.read_parquet(
        TEST_FILE,
        columns=columns,
    )

    ml_prediction = final_model.predict(
        test[FEATURES]
    )

    persistence = test["rainfall_mm"].to_numpy()
    actual = test[TARGET].to_numpy()

    adaptive_prediction = np.empty(len(test))

    monsoon_mask = test["monsoon"].to_numpy() == 1
    non_monsoon_mask = ~monsoon_mask

    adaptive_prediction[monsoon_mask] = (
        weights["monsoon"] * ml_prediction[monsoon_mask]
        + (1.0 - weights["monsoon"])
        * persistence[monsoon_mask]
    )

    adaptive_prediction[non_monsoon_mask] = (
        weights["non_monsoon"] * ml_prediction[non_monsoon_mask]
        + (1.0 - weights["non_monsoon"])
        * persistence[non_monsoon_mask]
    )

    # --------------------------------------------------
    # Scores
    # --------------------------------------------------
    ml_score = rmse(actual, ml_prediction)
    persistence_score = rmse(actual, persistence)
    adaptive_score = rmse(actual, adaptive_prediction)

    print()
    print("===== 2025 ADAPTIVE TEST =====")
    print(f"Persistence RMSE: {persistence_score:.4f} mm")
    print(f"ML RMSE:          {ml_score:.4f} mm")
    print(f"Adaptive RMSE:    {adaptive_score:.4f} mm")

    improvement = (
        (ml_score - adaptive_score)
        / ml_score
        * 100
    )

    print()
    print(
        f"Adaptive improvement vs ML: "
        f"{improvement:.2f}%"
    )


if __name__ == "__main__":
    main()