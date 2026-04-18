from src.app.aggregation.base_aggregation import BaseAggregation
from src.app.decision_engine.base_decision import BaseDecision
from src.app.schemas import Prediction

class Application:
    def __init__(self, aggregation: BaseAggregation, decision: BaseDecision):
        self.aggregation = aggregation
        self.decision = decision
    

    def run(self, predictions: list[Prediction]):
        result_aggregate = self.aggregation.aggregate(predictions)
        result_decision = self.decision.decision(result_aggregate)

        return result_decision
