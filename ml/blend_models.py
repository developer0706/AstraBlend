from dataclasses import dataclass

import numpy as np


@dataclass
class ModelPrediction:
    model_name: str
    precipitation_mm: np.ndarray


def weighted_blend(
    predictions: list[ModelPrediction],
    weights: dict[str, float],
) -> np.ndarray:
    """
    Combine multiple model predictions using normalized weights.
    """

    if not predictions:
        raise ValueError("No model predictions were provided.")

    weighted_sum = None
    total_weight = 0.0

    for prediction in predictions:
        if prediction.model_name not in weights:
            raise ValueError(
                f"Missing weight for model: "
                f"{prediction.model_name}"
            )

        weight = float(weights[prediction.model_name])

        if weight < 0:
            raise ValueError(
                f"Negative weight for {prediction.model_name}"
            )

        values = np.asarray(
            prediction.precipitation_mm,
            dtype=float,
        )

        if weighted_sum is None:
            weighted_sum = np.zeros_like(values)

        if values.shape != weighted_sum.shape:
            raise ValueError(
                "All model predictions must have "
                "the same shape."
            )

        weighted_sum += weight * values
        total_weight += weight

    if total_weight <= 0:
        raise ValueError("Total model weight must be positive.")

    return weighted_sum / total_weight