from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

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


def rainfall_category(rainfall):
    if rainfall >= 204.5:
        return 3

    if rainfall >= 115.6:
        return 2

    if rainfall >= 64.5:
        return 1

    return 0


def main():
    print("===== ASTRA BLEND: RAINFALL SEVERITY MODEL =====")

    train = pd.read_parquet(
        TRAIN_FILE,
        columns=FEATURES + ["target_rainfall_mm"],
    )

    validation = pd.read_parquet(
        VALIDATION_FILE,
        columns=FEATURES + ["target_rainfall_mm"],
    )

    # Convert next-day rainfall into IMD severity classes.
    train["target_severity"] = train[
        "target_rainfall_mm"
    ].apply(rainfall_category)

    validation["target_severity"] = validation[
        "target_rainfall_mm"
    ].apply(rainfall_category)

    print()
    print("Training class distribution:")

    print(
        train["target_severity"]
        .value_counts()
        .sort_index()
    )

    # Randomly sample while keeping class representation visible.
    if len(train) > TRAIN_SAMPLE_SIZE:
        train = train.sample(
            n=TRAIN_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    X_train = train[FEATURES]
    y_train = train["target_severity"]

    print()
    print(
        f"Training samples: {len(train):,}"
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    print()
    print("Training severity model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    X_validation = validation[FEATURES]
    y_validation = validation["target_severity"]

    predictions = model.predict(
        X_validation
    )

    probabilities = model.predict_proba(
        X_validation
    )

    print()
    print("===== 2024 VALIDATION =====")

    print(
        classification_report(
            y_validation,
            predictions,
            target_names=[
                "Normal",
                "Heavy",
                "Very Heavy",
                "Extremely Heavy",
            ],
            zero_division=0,
        )
    )

    print()
    print("Example probability output:")
    print(
        probabilities[:5]
    )


if __name__ == "__main__":
    main()