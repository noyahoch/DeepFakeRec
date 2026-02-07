from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class ModelConfig:
    pretrained_name: str = "facebook/wav2vec2-xls-r-300m"
    freeze_backbone: bool = True
    num_layers: int = 24
    hidden_dim: int = 1024
    num_classes: int = 2


@dataclass
class DataConfig:
    train_protocol_path: str
    train_audio_dir: str
    eval_protocol_path: str
    eval_audio_dir: str
    sample_rate: int = 16000
    segment_samples: int = 64600
    augment: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 5
    lr: float = 1e-6
    weight_decay: float = 1e-4
    epochs: int = 50
    early_stop_patience: int = 3
    seed: int = 1234
    num_workers: int = 4
    precision: str = "16-mixed"
    grad_clip: float | None = 1.0


@dataclass
class EvalConfig:
    metrics: list[str] = None
    save_scores_path: str | None = None


@dataclass
class LoggingConfig:
    backend: str = "tensorboard"
    run_dir: str = "runs"
    project: str = "audio-deepfake"
    log_every_n_steps: int = 20


@dataclass
class RunConfig:
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
    logging: LoggingConfig


def _coalesce_metrics(metrics: list[str] | None) -> list[str]:
    if metrics is None:
        return ["eer"]
    return metrics


def load_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    model = ModelConfig(**raw.get("model", {}))
    data = DataConfig(**raw["data"])
    train = TrainConfig(**raw.get("train", {}))
    eval_cfg = EvalConfig(**raw.get("eval", {}))
    eval_cfg.metrics = _coalesce_metrics(eval_cfg.metrics)
    logging = LoggingConfig(**raw.get("logging", {}))

    return RunConfig(model=model, data=data, train=train, eval=eval_cfg, logging=logging)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
