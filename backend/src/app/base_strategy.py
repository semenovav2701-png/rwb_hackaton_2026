from abc import ABC, abstractmethod


class BaseAggregationStrategy(ABC):

    @abstractmethod
    def aggregate(self):
        pass


class BaseDecisionStrategy(ABC):

    @abstractmethod
    def decision(self):
        pass