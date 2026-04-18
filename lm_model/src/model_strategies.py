import pandas as pd
import numpy as np
import lightgbm as lgb
from abc import ABC, abstractmethod
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


class Strategy(ABC):
    @abstractmethod
    def train_all_horizons(self, horizons, table_to_train, table_to_predict=None):
        pass

    @abstractmethod
    def all_horizons_predict(self, horizons, table_to_predict):
        pass


class TestStrategy(Strategy):
    def train_all_horizons(self, horizons, table_to_train, table_to_predict=None):
        df_full = table_to_train.df.sort_values(["route_id", "timestamp"]).copy()
        target_col = table_to_train.target_column
        features = table_to_train.features
        cat_features = getattr(table_to_train, "categorical_features", [])

        for horizon in horizons:
            print(f"\n🚀 Training horizon: {horizon}")

            df = df_full.copy()
            df[f"{target_col}_shifted"] = df.groupby("route_id")[target_col].shift(-horizon)
            df = df.dropna(subset=[f"{target_col}_shifted"])

            if len(df) == 0:
                print(f"❌ Empty dataset for horizon {horizon}")
                continue

            split_idx = int(len(df) * 0.8)
            df_train = df.iloc[:split_idx].copy()
            df_val = df.iloc[split_idx:].copy()

            if len(df_train) == 0 or len(df_val) == 0:
                print(f"⚠️ horizon {horizon}: fallback training without eval_set")
                X_train = df[features]
                y_train = df[f"{target_col}_shifted"]
                model = horizons[horizon]

                if isinstance(model, LGBMRegressor):
                    model.fit(X_train, y_train, categorical_feature=cat_features)
                elif isinstance(model, CatBoostRegressor):
                    model.fit(X_train, y_train, verbose=100)
                else:
                    model.fit(X_train, y_train)
                continue

            X_train = df_train[features]
            y_train = df_train[f"{target_col}_shifted"]
            X_val = df_val[features]
            y_val = df_val[f"{target_col}_shifted"]

            print(f"train: {X_train.shape}, val: {X_val.shape}")

            model = horizons[horizon]

            if isinstance(model, CatBoostRegressor):
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    verbose=100,
                )
            elif isinstance(model, LGBMRegressor):
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    categorical_feature=cat_features,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
                )
            else:
                model.fit(X_train, y_train)

    def all_horizons_predict(self, horizons, table_to_predict):
        df = table_to_predict.df.sort_values(["route_id", "timestamp"]).copy()
        parent_df = table_to_predict.parental_table.df.sort_values(["route_id", "timestamp"]).copy()
        features = table_to_predict.features

        last_known = parent_df.groupby("route_id").tail(1).copy()
        pred_frames = []

        for horizon in horizons:
            preds = horizons[horizon].predict(last_known[features])

            temp = last_known[["route_id", "timestamp"]].copy()
            temp["timestamp"] = temp["timestamp"] + pd.Timedelta(minutes=horizon * 30)
            temp[table_to_predict.predict_column] = preds
            pred_frames.append(temp)

        predict_df = pd.concat(pred_frames, ignore_index=True)
        table_to_predict.df = df.merge(predict_df, on=["route_id", "timestamp"], how="left")


class SubmissionStrategy(Strategy):
    def train_all_horizons(self, horizons, table_to_train, table_to_predict=None):
        df_full = table_to_train.df.sort_values(["route_id", "timestamp"]).copy()
        target_col = table_to_train.target_column
        features = table_to_train.features
        cat_features = getattr(table_to_train, "categorical_features", [])

        for horizon in horizons:
            print(f"\n🚀 Training horizon: {horizon}")

            df = df_full.copy()
            df[f"{target_col}_shifted"] = df.groupby("route_id")[target_col].shift(-horizon)
            df = df.dropna(subset=[f"{target_col}_shifted"])

            if len(df) == 0:
                print(f"❌ Empty dataset for horizon {horizon}")
                continue

            X = df[features]
            y = df[f"{target_col}_shifted"]

            model = horizons[horizon]

            if isinstance(model, CatBoostRegressor):
                model.fit(X, y, verbose=100)
            elif isinstance(model, LGBMRegressor):
                model.fit(X, y, categorical_feature=cat_features)
            else:
                model.fit(X, y)

    def all_horizons_predict(self, horizons, table_to_predict):
        df = table_to_predict.df.sort_values(["route_id", "timestamp"]).copy()
        parent_df = table_to_predict.parental_table.df.sort_values(["route_id", "timestamp"]).copy()
        features = table_to_predict.features

        last_known = parent_df.groupby("route_id").tail(1).copy()
        pred_frames = []

        for horizon in horizons:
            preds = horizons[horizon].predict(last_known[features])

            temp = last_known[["route_id", "timestamp"]].copy()
            temp["timestamp"] = temp["timestamp"] + pd.Timedelta(minutes=horizon * 30)
            temp[table_to_predict.predict_column] = preds
            pred_frames.append(temp)

        predict_df = pd.concat(pred_frames, ignore_index=True)
        table_to_predict.df = df.merge(predict_df, on=["route_id", "timestamp"], how="left")