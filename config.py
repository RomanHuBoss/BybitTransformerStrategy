# Models/direction/config.py

from dataclasses import dataclass, field
from pathlib import Path
import os

ENV_MODE = os.getenv("ENV_MODE", "dev")

@dataclass
class PathConfig:
    base: Path = Path("Models/direction")
    data_path: Path = Path("data/binance_30m.csv")
    scaler_path: Path = base / "scaler.pkl"
    model_path: Path = base / "model.pth"

@dataclass
class MarketConfig:
    lookahead_bars: int = 10               # сколько свечей смотреть вперёд
    min_tp_percent: float = 1.5            # минимальный процент пробоя от close

@dataclass
class ArchitectureConfig:
    input_size: int = 6                    # OCHLV + объём
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3

@dataclass
class TrainingConfig:
    train_size: float = 0.8
    val_size: float = 0.1
    batch_size: int = 512
    learning_rate: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 10

@dataclass
class PreprocessingConfig:
    drop_columns: list = field(default_factory=lambda: ['label', 'timestamp', 'symbol'])

@dataclass
class DirectionConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    arch: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    pre: PreprocessingConfig = field(default_factory=PreprocessingConfig)

CFG = DirectionConfig()
