from abc import ABC, abstractmethod


class BasePredictionSource(ABC):
    def __init__(self, name: str, source_type: str):
        self.name = name
        self.source_type = source_type

    @abstractmethod
    def get_predictions(self):
        pass