import pandas as pd
import numpy as np
import pathlib
from abc import abstractmethod, ABC
import tabloo


class Table(ABC):
    def __init__(self, df: pd.DataFrame, features: list, target_column: str) -> None:
        self.features = features
        self.target_column = target_column
        self.df = df

    def add_column(self, column: pd.Series, column_name: str) -> None:
        self.df[column_name] = column

    def pars_timestamps(self) -> None:
        day_of_month = self.df["timestamp"].dt.day
        day_of_week = self.df["timestamp"].dt.dayofweek
        hour = self.df["timestamp"].dt.hour
        self.add_column(day_of_month, "day_of_month")
        self.add_column(day_of_week, "day_of_week")
        self.add_column(hour, "hour")

    def add_cyclic_hour(self) -> None:
        self.df["cos_hour"] = np.cos(np.pi * 2 * self.df["hour"] / 24)
        self.df["sin_hour"] = np.sin(np.pi * 2 * self.df["hour"] / 24)

    def add_cyclic_dow(self) -> None:
        self.df["cos_dow"] = np.cos(np.pi * 2 * self.df["day_of_week"] / 7)
        self.df["sin_dow"] = np.sin(np.pi * 2 * self.df["day_of_week"] / 7)

    def regular_column_to_log1p(self, column_name: str) -> None:
        self.df[column_name] = np.log1p(self.df[column_name])

    def log1p_to_regular_column(self, column_name: str) -> None:
        self.df[column_name] = np.expm1(self.df[column_name])

    def add_anomaly_flag(self) -> None:
        self.df["ratio"] = self.df[self.target_column] / self.df["route_hour_mean"]
        self.df["anomaly_flag"] = (self.df["ratio"] < 0.6) | (self.df["ratio"] > 1.5)
        self.df = self.df.drop(columns=["ratio"])

    @abstractmethod
    def add_route_hour_mean(self):
        pass

    @abstractmethod
    def add_route_mean_h(self, horizons: list[int]):
        pass

    @abstractmethod
    def add_flow_speed(self):
        pass

    @abstractmethod
    def add_target_lags(self, lags: list[int]) -> None:
        pass


class EmptyTable(Table):
    def __init__(
            self,
            df: pd.DataFrame,
            parental_table: Table,
            features: list,
            target_column: str,
            predict_column: str,
    ) -> None:
        self.parental_table = parental_table
        self.predict_column = predict_column
        super().__init__(df, features, target_column)

    def add_route_hour_mean(self) -> None:
        group = self.parental_table.df.groupby(["route_id", "hour"])
        rename_dict = {self.target_column: "route_hour_mean"}
        mean_targets = group[self.target_column].mean().reset_index().rename(columns=rename_dict)
        self.df = self.df.merge(mean_targets, how="left", on=["route_id", "hour"])

    def get_metrics(self):
        awaited_res = self.df[self.target_column]
        got_res = self.df[self.predict_column]
        residual = (awaited_res - got_res).abs()
        wape = residual.sum() / awaited_res.sum()
        bias = abs(got_res.sum() / awaited_res.sum() - 1)
        # print(f"Mean absolute residual: {round(mean_absolute_error(awaited_res, got_res), 2)}")
        print(f"WAPE: {wape}")
        print(f"Bias: {bias}")
        print(f"Score: {wape + bias}")
        tabloo.show(self.df)

    def add_office_from_id(self):
        group = self.parental_table.df.groupby("route_id")["office_from_id"].first().reset_index()
        self.df = self.df.merge(group, how="left", on=["route_id"])

    def add_flow_speed(self) -> None:
        flow_speeds = self.parental_table.df[["route_id", "flow_speed"]].copy().drop_duplicates()
        self.df = self.df.merge(flow_speeds, how="left", on=["route_id"])

    def add_route_mean_h(self, horizons: list[int]):
        for horizon_range in horizons:
            self.df = self.df.merge(
                self.parental_table.df[["route_id", f"target_2h_in_{horizon_range}_mean"]].drop_duplicates(),
                how="left",
                on=["route_id"]
            )

    def add_target_lags(self, lags: list[int]) -> None:
        for lag in lags:
            cur_lags = (
                self
                .df
                .groupby("route_id")[self.target_column]
                .shift(lag)
            )
            self.df[f"target_2h_lag_{lag}"] = cur_lags
        self.df = self.df.dropna(subset=[f"target_2h_lag_{lag}" for lag in lags])


class FilledTable(Table):
    def add_route_hour_mean(self) -> None:
        group = self.df.groupby(["route_id", "hour"])
        group_sum = group[self.target_column].transform("sum")
        group_length = group[self.target_column].transform("count")
        self.df["route_hour_mean"] = (group_sum - self.df[self.target_column]) / (group_length - 1)

    def add_flow_speed(self) -> None:
        observed_statuses = (
            self.df
            .sort_values(["route_id", "timestamp"])
            [["route_id", "status_1", "status_2", "status_3", "status_4", "status_5", "status_6", "status_7", "status_8"]]
            .copy()
        )
        observed_statuses["shifted_status_8"] = observed_statuses.groupby("route_id")["status_8"].shift(-1)
        observed_statuses["previous_statuses_sum"] = (
            (
                observed_statuses["status_1"] +
                observed_statuses["status_2"] +
                observed_statuses["status_3"] +
                observed_statuses["status_4"] +
                observed_statuses["status_5"] +
                observed_statuses["status_6"] +
                observed_statuses["status_7"]
            ).replace(0, pd.NA)
        )
        observed_statuses = observed_statuses.dropna(subset=["shifted_status_8", "previous_statuses_sum"])
        observed_statuses["ratio"] = \
            observed_statuses["shifted_status_8"] / observed_statuses["previous_statuses_sum"]
        observed_statuses["flow_speed"] = observed_statuses.groupby("route_id")["ratio"].transform("mean")
        flow_speeds = observed_statuses[["route_id", "flow_speed"]].drop_duplicates()
        self.df = self.df.merge(flow_speeds, how="left", on=["route_id"])

    def add_route_mean_h(self, horizons: list[int]) -> None:
        self.df = self.df.sort_values(["route_id", "timestamp"])
        for horizon_range in horizons:
            shifted_targets = self.df[["route_id", self.target_column]].copy()
            shifted_targets[self.target_column] = (
                shifted_targets
                .groupby("route_id")
                [self.target_column]
                .shift(-horizon_range)
            )
            shifted_targets = (
                shifted_targets
                .dropna(subset=[self.target_column])
                .groupby("route_id")
                .mean()
                .reset_index()
                .rename(columns={self.target_column: f"target_2h_in_{horizon_range}_mean"})
            )
            self.df = self.df.merge(shifted_targets, how="left", on=["route_id"])

    def add_target_lags(self, lags: list[int]) -> None:
        for lag in lags:
            cur_lags = (
                self
                .df
                .groupby("route_id")[self.target_column]
                .shift(lag)
            )
            self.df[f"target_2h_lag_{lag}"] = cur_lags
        self.df = self.df.dropna(subset=[f"target_2h_lag_{lag}" for lag in lags])

import os

class DataHandler:
    @staticmethod
    def read_parquet_file(dest: pathlib.Path) -> pd.DataFrame:
        df = pd.read_parquet(dest)
        return df

    def get_dataframes_from_file(
            self,
            train_dest: pathlib.Path,
            train_window: int,
            predict_dest: pathlib.Path | None = None,
            split_date: pd.Timestamp | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_to_train, df_to_predict = None, None
        train_window_days = pd.Timedelta(days=train_window)
        if split_date:
            start_date = split_date - train_window_days
            df = self.read_parquet_file(train_dest)
            df_to_train = df.loc[(df["timestamp"] >= start_date) & (df["timestamp"] < split_date)].copy()
            df_to_predict = df.loc[(df["timestamp"] >= split_date) & (df["timestamp"] < split_date + pd.Timedelta(hours=5))].copy()
        elif predict_dest:
            start_date = pd.Timestamp("2025-05-30 10:30:00") - train_window_days
            df_to_train = self.read_parquet_file(train_dest)
            df_to_train = df_to_train.loc[df_to_train["timestamp"] >= start_date].copy()
            df_to_predict = self.read_parquet_file(predict_dest)
        return df_to_train, df_to_predict

    @staticmethod
    def write_submission(table_to_predict: EmptyTable, dest: pathlib.Path):
        if table_to_predict.df.empty:
            raise ValueError("⚠️ DataFrame для сабмита пустой!")

        if "id" not in table_to_predict.df.columns:
            raise KeyError("⚠️ В DataFrame нет колонки 'id'!")

        if table_to_predict.predict_column not in table_to_predict.df.columns:
            raise KeyError(f"⚠️ В DataFrame нет колонки '{table_to_predict.predict_column}'!")

        n_nan = table_to_predict.df[table_to_predict.predict_column].isna().sum()
        if n_nan > 0:
            print(f"⚠️ Предсказаний NaN: {n_nan}. Они будут записаны как пустые значения в CSV.")

        res = table_to_predict.df[["id"]].copy()
        res["y_pred"] = table_to_predict.df[table_to_predict.predict_column]

        if os.path.exists(dest):
            print(f"⚠️ Файл {dest} уже существует и будет перезаписан!")

        res.to_csv(dest, index=False)
        print(f"✅ Сабмит успешно записан: {dest}")
