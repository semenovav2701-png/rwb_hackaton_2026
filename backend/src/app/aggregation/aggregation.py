from src.app.schemas import Prediction
from src.app.aggregation.base_aggregation import BaseAggregation
from src.app.base_strategy import BaseAggregationStrategy



class StrategyCounter(BaseAggregationStrategy):
    def aggregate(self, predictions: list[Prediction]) -> list[dict]:
        aggr = {}

        for pred in predictions:
            key = (pred.office_from_id, pred.timestamp)
            if aggr.get(key) is not None:
                aggr[key] += pred.predicted_target_2h
            else:
                aggr[key] = pred.predicted_target_2h

        result = []

        for key, val in aggr.items():        
            result.append({"office_from_id": key[0],
                            "timestamp": key[1],
                            "total_volume": val
                        })

        return result


class Aggregation(BaseAggregation):
    def __init__(self, strategy: BaseAggregationStrategy):
        self.strategy = strategy

    def aggregate(self, predictions):
        return self.strategy.aggregate(predictions)