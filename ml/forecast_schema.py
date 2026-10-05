from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class ForecastField:
    """
    Standard representation of a weather-model forecast.
    """

    model_name: str
    run_time: datetime
    valid_time: datetime

    latitude: np.ndarray
    longitude: np.ndarray

    precipitation_mm: np.ndarray

    lead_hours: int