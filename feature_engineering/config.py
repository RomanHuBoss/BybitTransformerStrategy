# config.py

from dataclasses import dataclass, field
from pathlib import Path
import os

ENV_MODE = os.getenv("ENV_MODE", "dev")

@dataclass
class PathConfig:
    base = Path("../")
    raw_csv: str  = base / "historical_data" / "BTC-ETH-COMBINED" / "combined_csv.csv"     # сырой csv, содержащий OCLHV
    features_columns_path: str = base / "artifacts" / "features" / "feature_columns.pkl"   # колонки фич
    features_scaler_path: str = base / "artifacts" / "features" / "feature_scaler.pkl"                  # скейлер фич

@dataclass
class AssetConfig:
    time_frame: int = 30  # в минутах, например, 30m
    window_size: int = 30 # размер свечного оокна в прошлом, относительно которого будет строиться прогноз в будущем
    required_columns: list[str] = field(default_factory=lambda: ['open_time', 'open', 'high', 'low', 'close', 'volume'])

@dataclass
class FeatureEngineeringConfig:
    default_shift: int = 1  # количество свечей, на которое смещаются фичи

@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    assets: AssetConfig = field(default_factory=AssetConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)

CFG = Config()
