import pathlib
from dataclasses import dataclass
import pandas as pd


@dataclass
class LightgbmModelConfiguration:
    objective: str
    num_leaves: int
    max_depth: int
    learning_rate: float
    n_estimators: int
    min_data_in_leaf: int
    feature_fraction: float
    device: str
    lambda_l1: float
    lambda_l2: float
    verbosity: int
    horizons: list[int]


@dataclass
class AppConfiguration:
    mode: int
    split_date_start: pd.Timestamp
    split_date_end: pd.Timestamp
    train_data_path: pathlib.Path
    predict_data_path: pathlib.Path
    res_data_path: pathlib.Path


@dataclass
class FeaturesConfiguration:
    features: list[str]


@dataclass
class CatboostModelConfiguration:
    learning_rate: float
    depth: int
    verbose: int
    iterations: int
    random_seed: int
    cat_features: list[str]
    horizons: list[int]
    task_type: str
    devices: str


catboost_model_cfg = CatboostModelConfiguration(
    learning_rate=0.1,
    depth=6,
    verbose=50,
    iterations=1000,
    random_seed=42,
    cat_features=["route_id", "office_from_id"],
    horizons=[i for i in range(1, 11)],
    task_type="CPU",
    devices="0"
)


lightgbm_model_cfg = LightgbmModelConfiguration(
    objective="regression",
    num_leaves=63,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=1400,
    min_data_in_leaf=20,
    feature_fraction=0.8,
    device="cpu",
    lambda_l1=0.1,
    lambda_l2=0.1,
    verbosity=-1,
    horizons=[i for i in range(1, 11)]
)


app_cfg = AppConfiguration(
    mode=1,
    split_date_start=pd.Timestamp("2025-04-20 11:00:00"),
    split_date_end=pd.Timestamp("2025-04-20 15:30:00"),
    train_data_path=pathlib.Path(__file__).resolve().parents[2] / pathlib.Path("data/train_team_track.parquet"),
    predict_data_path=pathlib.Path(__file__).resolve().parents[2] / pathlib.Path("data/test_team_track.parquet"),
    res_data_path=pathlib.Path(__file__).resolve().parents[2] / pathlib.Path("data/submission.csv"),
)


features_cfg = FeaturesConfiguration(
    features=[
        "office_from_id",
        "route_id",
        "target_2h",
        "route_hour_mean",
        "cos_hour",
        "sin_hour",
        "cos_dow",
        "sin_dow",
        "target_2h_in_1_mean",
        "target_2h_in_2_mean",
        "target_2h_in_3_mean",
        "target_2h_in_4_mean",
        "target_2h_in_5_mean",
        "target_2h_in_6_mean",
        "target_2h_in_7_mean",
        "target_2h_in_8_mean",
        "target_2h_in_9_mean",
        "target_2h_in_10_mean",
        "flow_speed",
        "target_2h_lag_1",
        "target_2h_lag_2",
        "target_2h_lag_3",
        "target_2h_lag_4",
        "target_2h_lag_5",
        "target_2h_lag_6",
        "target_2h_lag_7",
        "target_2h_lag_8",
        "target_2h_lag_9",
        "target_2h_lag_10",
        "target_2h_lag_11",
        "target_2h_lag_12",
        "target_route_hour_mean",
        "target_cos_hour",
        "target_sin_hour",
        "target_cos_dow",
        "target_sin_dow"
    ]
)

