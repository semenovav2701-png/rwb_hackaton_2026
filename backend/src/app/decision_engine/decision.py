import math

from src.app.schemas import Prediction
from src.app.decision_engine.base_decision import BaseDecision
from src.app.base_strategy import BaseDecisionStrategy



class VolumeBasedTruckCounter(BaseDecisionStrategy):
    def __init__(self):
        self.volume_car = 10
        self.max_car = 10
        self.min_car = 1
        self.buffer = 0.1

    def decision(self, all_data: list[dict]) -> list[dict]:
        for data in all_data:
            volume = math.ceil(data["total_volume"] * (1 + self.buffer) / self.volume_car)
            data["trucks_needed"] = min(self.max_car, volume) if volume > self.min_car else self.min_car

        return all_data


class Decision(BaseDecision):
    def __init__(self, strategy: BaseDecisionStrategy):
        self.strategy = strategy

    def decision(self, data):
        return self.strategy.decision(data)