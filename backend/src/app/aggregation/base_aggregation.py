from abc import ABC, abstractmethod


class BaseAggregation(ABC):

    @abstractmethod
    def aggregate(self):
        pass