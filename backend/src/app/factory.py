from src.app.base_strategy import BaseAggregationStrategy, BaseDecisionStrategy
from src.app.sources.base_source import BasePredictionSource

class AggregationStrategyFactory:
    _creators = {}

    @classmethod
    def register(cls, name: str, creator):
        cls._creators[name] = creator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseAggregationStrategy:
        creator = cls._creators.get(name)
        if creator is None:
            raise ValueError(f"Неизвестный источник данных: {name}")
        return creator(**kwargs)


class DecisionStrategyFactory:
    _creators = {}

    @classmethod
    def register(cls, name: str, creator):
        cls._creators[name] = creator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDecisionStrategy:
        creator = cls._creators.get(name)
        if creator is None:
            raise ValueError(f"Неизвестный источник данных: {name}")
        return creator(**kwargs)


class DataSourceFactory:
    _creators = {}

    @classmethod
    def register(cls, name: str, creator):
        cls._creators[name] = creator

    @classmethod
    def create(cls, name: str, **kwargs) -> BasePredictionSource:
        creator = cls._creators.get(name)
        if creator is None:
            raise ValueError(f"Неизвестный источник данных: {name}")
        return creator(**kwargs)