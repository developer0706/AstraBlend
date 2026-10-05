import numpy as np

from blend_models import ModelPrediction, weighted_blend


def main():
    gfs = ModelPrediction(
        model_name="GFS",
        precipitation_mm=np.array([[10.0, 20.0]])
    )

    ml = ModelPrediction(
        model_name="IMD_ML",
        precipitation_mm=np.array([[14.0, 16.0]])
    )

    result = weighted_blend(
        predictions=[gfs, ml],
        weights={
            "GFS": 0.6,
            "IMD_ML": 0.4,
        },
    )

    print("===== BLEND TEST =====")
    print("GFS:", gfs.precipitation_mm)
    print("ML :", ml.precipitation_mm)
    print("Blend:", result)


if __name__ == "__main__":
    main()