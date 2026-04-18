from tables import Table, EmptyTable, FilledTable
from abc import ABC, abstractmethod


class FeatureStep(ABC):
    @abstractmethod
    def apply(self, table: Table, context: dict) -> None:
        pass


class RouteHourMeanStep(FeatureStep):
    def apply(self, table: Table, context: dict) -> None:
        table.add_route_hour_mean()


class ParsTimestampsStep(FeatureStep):
    def apply(self, table: Table, context: dict) -> None:
        table.pars_timestamps()


class OfficeFromIdStep(FeatureStep):
    def apply(self, table: EmptyTable, context: dict) -> None:
        table.add_office_from_id()


class CyclicHourStep(FeatureStep):
    def apply(self, table: Table, context: dict) -> None:
        table.add_cyclic_hour()


class CyclicDowStep(FeatureStep):
    def apply(self, table: Table, context: dict) -> None:
        table.add_cyclic_dow()


class FlowSpeedStep(FeatureStep):
    def apply(self, table: Table, context: dict) -> None:
        table.add_flow_speed()


class RouteMeanHStep(FeatureStep):
    def apply(self, table: Table, context: dict) -> None:
        table.add_route_mean_h(horizons=[i for i in range(1, 11)])


class AnomalyFlagStep(FeatureStep):
    def apply(self, table: Table, context: dict) -> None:
        table.add_anomaly_flag()


class TargetLagStep(FeatureStep):
    def apply(self, table: Table, context: dict) -> None:
        table.add_target_lags([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


class TargetFeaturesStep(FeatureStep):
    def apply(self, table: Table, context: dict) -> None:
        features = context or []

        for col in features:
            mean_val = table.df[col].mean()
            table.df[f"target_{col}"] = mean_val


class FeaturePipeline:
    def __init__(self, registry: dict | None = None) -> None:
        default_registry = {
            "route_hour_mean": RouteHourMeanStep(),
            "pars_timestamps": ParsTimestampsStep(),
            "office_from_id": OfficeFromIdStep(),
            "cyclic_hour": CyclicHourStep(),
            "cyclic_dow": CyclicDowStep(),
            "flow_speed": FlowSpeedStep(),
            "route_mean_h": RouteMeanHStep(),
            "target_2h_lag": TargetLagStep(),
            "target_features": TargetFeaturesStep(),
            "anomaly_flag": AnomalyFlagStep()
        }
        self.registry = registry if registry is not None else default_registry

    def apply(self, table: Table, steps: list[str], context: dict | None = None) -> None:
        context = context or {}
        for step in steps:
            if step in self.registry:
                cur_context = context.get(step)
                self.registry[step].apply(table=table, context=cur_context)
