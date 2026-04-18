import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb

from model_strategies import TestStrategy, SubmissionStrategy
from config import (
    catboost_model_cfg,
    CatboostModelConfiguration,
    app_cfg,
    features_cfg,
    lightgbm_model_cfg,
    LightgbmModelConfiguration,
)
from tables import EmptyTable, FilledTable, DataHandler
from features_adder import FeaturePipeline


class CatboostModelManager:
    def __init__(self, cfg: CatboostModelConfiguration) -> None:
        self.cfg = cfg
        self.strategy = None
        self.horizons = {
            i: CatBoostRegressor(
                verbose=self.cfg.verbose,
                learning_rate=self.cfg.learning_rate,
                depth=self.cfg.depth,
                iterations=self.cfg.iterations,
                random_seed=self.cfg.random_seed,
                cat_features=self.cfg.cat_features,
                devices=self.cfg.devices,
                task_type=self.cfg.task_type,
            )
            for i in self.cfg.horizons
        }

    def train_all_horizons(self, table_to_train: FilledTable, table_to_predict: EmptyTable | None = None) -> None:
        self.strategy.train_all_horizons(self.horizons, table_to_train, table_to_predict)

    def all_horizons_predict(self, table_to_predict: EmptyTable) -> None:
        self.strategy.all_horizons_predict(self.horizons, table_to_predict)

    def choose_strategy(self, mode: int) -> None:
        if mode == 0:
            self.strategy = TestStrategy()
        elif mode == 1:
            self.strategy = SubmissionStrategy()


class LightgbmModelManager:
    def __init__(self, cfg: LightgbmModelConfiguration) -> None:
        self.cfg = cfg
        self.strategy = None
        self.horizons = {
            i: LGBMRegressor(
                objective=self.cfg.objective,
                num_leaves=self.cfg.num_leaves,
                max_depth=self.cfg.max_depth,
                learning_rate=self.cfg.learning_rate,
                n_estimators=self.cfg.n_estimators,
                min_child_samples=self.cfg.min_data_in_leaf,
                feature_fraction=self.cfg.feature_fraction,
                device=self.cfg.device,  # должно быть "cpu" или "gpu" латиницей
                lambda_l1=self.cfg.lambda_l1,
                lambda_l2=self.cfg.lambda_l2,
                verbosity=self.cfg.verbosity,
                random_state=42,
            )
            for i in self.cfg.horizons
        }

    def train_all_horizons(self, table_to_train: FilledTable, table_to_predict: EmptyTable | None = None) -> None:
        self.strategy.train_all_horizons(self.horizons, table_to_train, table_to_predict)

    def all_horizons_predict(self, table_to_predict: EmptyTable) -> None:
        self.strategy.all_horizons_predict(self.horizons, table_to_predict)

    def choose_strategy(self, mode: int) -> None:
        if mode == 0:
            self.strategy = TestStrategy()
        elif mode == 1:
            self.strategy = SubmissionStrategy()


def make_lgbm_safe(table: FilledTable | EmptyTable) -> None:
    """
    Для LightGBM: object -> category.
    """
    categorical_cols = []
    for col in table.features:
        if col in table.df.columns and table.df[col].dtype == "object":
            table.df[col] = table.df[col].astype("category")
            categorical_cols.append(col)
    table.categorical_features = categorical_cols


def build_ensemble_frame(cat_df: pd.DataFrame, lgb_df: pd.DataFrame) -> pd.DataFrame:
    left_cols = ["route_id", "timestamp", "y_pred_cat"]
    right_cols = ["route_id", "timestamp", "y_pred_lgb"]

    if "target_2h" in cat_df.columns:
        left_cols.insert(2, "target_2h")

    merged = cat_df[left_cols].merge(
        lgb_df[right_cols],
        on=["route_id", "timestamp"],
        how="inner",
    )

    if merged.empty and len(cat_df) == len(lgb_df):
        merged = cat_df[left_cols].copy().reset_index(drop=True)
        merged["y_pred_lgb"] = lgb_df["y_pred_lgb"].values

    return merged


def print_metrics(name, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        print(f"\n{name}: нет данных для метрик")
        return

    wape = np.sum(np.abs(y_pred - y_true)) / (np.sum(np.abs(y_true)) + 1e-6)
    bias = np.abs(np.sum(y_pred) / (np.sum(y_true) + 1e-6) - 1)
    score = wape + bias

    print(f"\n{name}")
    print(f"WAPE : {wape:.6f}")
    print(f"Bias : {bias:.6f}")
    print(f"Score: {score:.6f}")


def make_validation_frame(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Делает frame для оценки горизонта:
    y_true_h = target_2h, сдвинутый назад по времени внутри route_id.
    """
    out = df.sort_values(["route_id", "timestamp"]).copy()
    out[f"y_true_h_{horizon}"] = out.groupby("route_id")["target_2h"].shift(-horizon)
    out = out.dropna(subset=[f"y_true_h_{horizon}"]).copy()
    return out


def align_frames(cat_df: pd.DataFrame, lgb_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Подравнивает строки по route_id + timestamp, если вдруг порядок разъехался.
    """
    keys = cat_df[["route_id", "timestamp"]].merge(
        lgb_df[["route_id", "timestamp"]],
        on=["route_id", "timestamp"],
        how="inner",
    )

    if keys.empty:
        return cat_df.iloc[0:0].copy(), lgb_df.iloc[0:0].copy()

    cat_aligned = cat_df.merge(keys, on=["route_id", "timestamp"], how="inner")
    lgb_aligned = lgb_df.merge(keys, on=["route_id", "timestamp"], how="inner")
    return cat_aligned, lgb_aligned


def main():
    train_window = getattr(app_cfg, "train_window", 14)
    data_handler = DataHandler()
    feature_handler = FeaturePipeline()

    catboost_model_manager = CatboostModelManager(cfg=catboost_model_cfg)
    lgbm_model_manager = LightgbmModelManager(cfg=lightgbm_model_cfg)

    catboost_model_manager.choose_strategy(app_cfg.mode)
    lgbm_model_manager.choose_strategy(app_cfg.mode)

    if app_cfg.mode == 0:
        df_to_train, _ = data_handler.get_dataframes_from_file(
            train_dest=app_cfg.train_data_path,
            split_date=app_cfg.split_date_start,
            train_window=train_window
        )
    else:
        df_to_train, df_to_predict = data_handler.get_dataframes_from_file(
            train_dest=app_cfg.train_data_path,
            predict_dest=app_cfg.predict_data_path,
            train_window=train_window
        )

    cat_train_table = FilledTable(
        df=df_to_train.copy(),
        features=features_cfg.features[:],
        target_column="target_2h",
    )
    lgb_train_table = FilledTable(
        df=df_to_train.copy(),
        features=features_cfg.features[:],
        target_column="target_2h",
    )

    if app_cfg.mode == 0:
        cat_predict_table = EmptyTable(
            df=df_to_train.copy(),
            parental_table=cat_train_table,
            features=features_cfg.features[:],
            target_column="target_2h",
            predict_column="y_pred_cat",
        )
        lgb_predict_table = EmptyTable(
            df=df_to_train.copy(),
            parental_table=lgb_train_table,
            features=features_cfg.features[:],
            target_column="target_2h",
            predict_column="y_pred_lgb",
        )
    else:
        cat_predict_table = EmptyTable(
            df=df_to_predict.copy(),
            parental_table=cat_train_table,
            features=features_cfg.features[:],
            target_column="target_2h",
            predict_column="y_pred_cat",
        )
        lgb_predict_table = EmptyTable(
            df=df_to_predict.copy(),
            parental_table=lgb_train_table,
            features=features_cfg.features[:],
            target_column="target_2h",
            predict_column="y_pred_lgb",
        )

    train_steps = [
        "pars_timestamps",
        "route_hour_mean",
        "cyclic_dow",
        "cyclic_hour",
        "flow_speed",
        "route_mean_h",
        "target_2h_lag",
        "target_features",
        "anomaly_flag"
    ]

    feature_handler.apply(
        table=cat_train_table,
        steps=train_steps,
        context={
            "target_features": [
                "route_hour_mean",
                "cos_hour",
                "sin_hour",
                "cos_dow",
                "sin_dow",
                "flow_speed",
            ]
        },
    )
    feature_handler.apply(
        table=lgb_train_table,
        steps=train_steps,
        context={
            "target_features": [
                "route_hour_mean",
                "cos_hour",
                "sin_hour",
                "cos_dow",
                "sin_dow",
                "flow_speed",
            ]
        },
    )

    if app_cfg.mode == 0:
        predict_steps = [
            "pars_timestamps",
            "route_hour_mean",
            "cyclic_dow",
            "cyclic_hour",
            "flow_speed",
            "route_mean_h",
            "target_2h_lag",
        ]
        feature_handler.apply(table=cat_predict_table, steps=predict_steps)
        feature_handler.apply(table=lgb_predict_table, steps=predict_steps)

    make_lgbm_safe(lgb_train_table)
    make_lgbm_safe(lgb_predict_table)

    catboost_model_manager.train_all_horizons(
        table_to_train=cat_train_table,
        table_to_predict=cat_predict_table,
    )
    lgbm_model_manager.train_all_horizons(
        table_to_train=lgb_train_table,
        table_to_predict=lgb_predict_table,
    )

    if app_cfg.mode == 0:
        horizons = sorted(set(catboost_model_cfg.horizons) | set(lightgbm_model_cfg.horizons))
        scoring_frames = []

        for h in horizons:
            cat_eval = make_validation_frame(cat_train_table.df, h)
            lgb_eval = make_validation_frame(lgb_train_table.df, h)

            if len(cat_eval) == 0 or len(lgb_eval) == 0:
                continue

            cat_eval, lgb_eval = align_frames(cat_eval, lgb_eval)
            if len(cat_eval) == 0 or len(lgb_eval) == 0:
                continue

            y_true = cat_eval[f"y_true_h_{h}"].values

            cat_pred = catboost_model_manager.horizons[h].predict(cat_eval[cat_train_table.features])
            lgb_pred = lgbm_model_manager.horizons[h].predict(lgb_eval[lgb_train_table.features])

            frame = pd.DataFrame({
                "route_id": cat_eval["route_id"].values,
                "timestamp": cat_eval["timestamp"].values,
                "horizon": h,
                "target_2h": y_true,
                "y_pred_cat": cat_pred,
                "y_pred_lgb": lgb_pred,
            })
            scoring_frames.append(frame)

        if not scoring_frames:
            print("⚠️ Нет данных для scoring. Проверь split и горизонты.")
            return

        ensemble_df = pd.concat(scoring_frames, ignore_index=True)
        ensemble_df["y_pred"] = 0.5 * ensemble_df["y_pred_cat"] + 0.5 * ensemble_df["y_pred_lgb"]
        ensemble_df["y_pred"] = np.clip(ensemble_df["y_pred"], 0, None)

        print(f"\nMatched rows for scoring: {len(ensemble_df)}")
        print(ensemble_df.head(10))

        print_metrics("🚀 CatBoost", ensemble_df["target_2h"].values, ensemble_df["y_pred_cat"].values)
        print_metrics("🌲 LightGBM", ensemble_df["target_2h"].values, ensemble_df["y_pred_lgb"].values)
        print_metrics("🔥 Ensemble", ensemble_df["target_2h"].values, ensemble_df["y_pred"].values)

        print("\n📊 SAMPLE PREDICTIONS:")
        print(ensemble_df[["route_id", "timestamp", "horizon", "target_2h", "y_pred_cat", "y_pred_lgb", "y_pred"]].head(20))

    else:
        catboost_model_manager.all_horizons_predict(table_to_predict=cat_predict_table)
        lgbm_model_manager.all_horizons_predict(table_to_predict=lgb_predict_table)

        ensemble_df = build_ensemble_frame(cat_predict_table.df, lgb_predict_table.df)
        if ensemble_df.empty:
            print("⚠️ Не удалось собрать ensemble_df. Проверь совпадение route_id и timestamp.")
            return

        ensemble_df["y_pred"] = 0.5 * ensemble_df["y_pred_cat"] + 0.5 * ensemble_df["y_pred_lgb"]
        ensemble_df["y_pred"] = np.clip(ensemble_df["y_pred"], 0, None)

        submission_df = df_to_predict[["id", "route_id", "timestamp"]].merge(
            ensemble_df[["route_id", "timestamp", "y_pred"]],
            on=["route_id", "timestamp"],
            how="left",
        )[["id", "y_pred"]]

        data_handler.write_submission(
            table_to_predict=EmptyTable(
                df=submission_df.copy(),
                parental_table=cat_train_table,
                features=features_cfg.features[:],
                target_column="target_2h",
                predict_column="y_pred",
            ),
            dest=app_cfg.res_data_path,
        )


if __name__ == "__main__":
    main()