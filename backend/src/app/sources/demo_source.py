from datetime import datetime
from app.schemas import Prediction
from app.sources.base_source import BasePredictionSource


class DemoPredictionSource(BasePredictionSource):
    """
    Демонстрационный источник вакансий с тестовыми данными.
    """

    def __init__(self):
        super().__init__(name="Demo Vacancy Source", source_type="demo")

    def get_predictions(self):
        return [
            Prediction(route_id=1, office_from_id=1, timestamp=datetime(2026, 3, 28, 10, 0), predicted_target_2h=10),
            Prediction(route_id=2, office_from_id=1, timestamp=datetime(2026, 3, 28, 10, 0), predicted_target_2h=15),

            Prediction(route_id=3, office_from_id=1, timestamp=datetime(2026, 3, 28, 10, 30), predicted_target_2h=20),

            Prediction(route_id=4, office_from_id=2, timestamp=datetime(2026, 3, 28, 10, 0), predicted_target_2h=8),
        ]
        