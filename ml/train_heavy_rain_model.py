from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = DATA_DIR / "train.parquet"
VALIDATION_FILE = DATA_DIR / "validation.parquet"

RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 500_000

HEAVY_THRESHOLD_MM = 64.5

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
    print("===== ASTRA BLEND: HEAVY RAIN MODEL =====")

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------
    print("Loading training data...")

    train = pd.read_parquet(
        TRAIN_FILE,
        columns=FEATURES + [TARGET],
    )

    print(f"Training rows available: {len(train):,}")

    if len(train) > TRAIN_SAMPLE_SIZE:
        train = train.sample(
            n=TRAIN_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    X_train = train[FEATURES]
    y_train = train[TARGET].astype("int8")

    print(f"Training sample: {len(train):,}")
    print(f"Positive examples: {int(y_train.sum()):,}")
    print(
        f"Positive rate: {y_train.mean() * 100:.3f}%"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    print()
    print("Training classifier...")

    model.fit(X_train, y_train)

    print("Training complete.")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    print()
    print("Loading validation data...")

    validation = pd.read_parquet(
        VALIDATION_FILE,
        columns=FEATURES + [TARGET],
    )

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET].astype("int8")

    print("Generating probabilities...")

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # Start with 0.50 probability threshold.
    predictions = (
        probabilities >= 0.50
    ).astype("int8")

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    print()
    print("===== VALIDATION RESULTS =====")
    print(f"Threshold: {HEAVY_THRESHOLD_MM} mm")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print()
    print("===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_validation,
            predictions,
            target_names=[
                "No heavy rain",
                "Heavy rain",
            ],
            zero_division=0,
        )
    )

    print()
    print("===== MODEL COMPLETE =====")


if __name__ == "__main__":
    main()