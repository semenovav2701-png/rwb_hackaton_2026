from fastapi import FastAPI

from src.app.application import Application

from src.app.app_config import AppConfig

from src.app.aggregation.aggregation import StrategyCounter
from src.app.decision_engine.decision import VolumeBasedTruckCounter

from src.app.factory import DecisionStrategyFactory, AggregationStrategyFactory
from src.app.aggregation.aggregation import Aggregation
from src.app.decision_engine.decision import Decision

AggregationStrategyFactory.register("count", StrategyCounter)
DecisionStrategyFactory.register("truck count", VolumeBasedTruckCounter)


def get_application() -> Application:
    config = AppConfig()

    aggregation_strategy = AggregationStrategyFactory.create(config.aggregation_strategy)
    decision_strategy = DecisionStrategyFactory.create(config.decision_strategy)

    aggregation = Aggregation(aggregation_strategy)
    decision = Decision(decision_strategy)

    return Application(aggregation, decision)