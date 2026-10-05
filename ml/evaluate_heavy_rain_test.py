from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = DATA_DIR / "train.parquet"
VALIDATION_FILE = DATA_DIR / "validation.parquet"
TEST_FILE = DATA_DIR / "test.parquet"

RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 1_000_000

DECISION_THRESHOLD = 0.15

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

TARGET = "target_heavy_rain"


def main() -> None:
    print("===== ASTRA BLEND: FINAL HEAVY-RAIN TEST =====")

    # --------------------------------------------------
    # Load 2020–2023 training data
    # --------------------------------------------------
    print("Loading training data...")

    train = pd.read_parquet(
        TRAIN_FILE,
        columns=FEATURES + [TARGET],
    )

    # --------------------------------------------------
    # Add 2024 validation data after model selection.
    # We are now doing the final fit before testing 2025.
    # --------------------------------------------------
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
    y_train = train_final[TARGET].astype("int8")

    print(
        f"Final training sample: "
        f"{len(train_final):,}"
    )

    print()
    print("Training final classifier...")

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    print("Training complete.")

    # --------------------------------------------------
    # FINAL TEST — 2025 ONLY
    # --------------------------------------------------
    print()
    print("Loading 2025 test data...")

    test = pd.read_parquet(
        TEST_FILE,
        columns=FEATURES + [TARGET],
    )

    X_test = test[FEATURES]
    y_test = test[TARGET].astype("int8")

    print("Generating probabilities...")

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype("int8")

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print()
    print("===== 2025 FINAL TEST RESULTS =====")
    print(f"Samples:   {len(test):,}")
    print(f"Threshold: {DECISION_THRESHOLD:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print()
    print("===== TEST COMPLETE =====")
    print("2025 was not used for threshold selection.")


if __name__ == "__main__":
    main()