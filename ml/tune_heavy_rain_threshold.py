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

TARGET = "target_heavy_rain"


def main() -> None:
    print("===== ASTRA BLEND: THRESHOLD TUNING =====")

    train = pd.read_parquet(
        TRAIN_FILE,
        columns=FEATURES + [TARGET],
    )

    if len(train) > TRAIN_SAMPLE_SIZE:
        train = train.sample(
            n=TRAIN_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    X_train = train[FEATURES]
    y_train = train[TARGET].astype("int8")

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    print("Training classifier...")
    model.fit(X_train, y_train)

    validation = pd.read_parquet(
        VALIDATION_FILE,
        columns=FEATURES + [TARGET],
    )

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET].astype("int8")

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    print(f"PR-AUC: {pr_auc:.4f}")
    print()
    print("===== THRESHOLD RESULTS =====")

    best_threshold = None
    best_f1 = -1.0

    for threshold in [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]:
        predictions = (
            probabilities >= threshold
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

        print(
            f"Threshold={threshold:.2f}  "
            f"Precision={precision:.4f}  "
            f"Recall={recall:.4f}  "
            f"F1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print()
    print("===== BEST THRESHOLD =====")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"F1:        {best_f1:.4f}")


if __name__ == "__main__":
    main()