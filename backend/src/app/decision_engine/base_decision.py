from abc import ABC, abstractmethod


class BaseDecision(ABC):

    @abstractmethod
    def decision(self):
        pass