from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]


@dataclass
class RunLogger:
    backend: str
    run_dir: str
    project: str
    writer: Any | None = None

    def start(self) -> None:
        if self.backend == "tensorboard":
            if SummaryWriter is None:
                raise RuntimeError("TensorBoard not available. Install torch utils.")
            Path(self.run_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.run_dir)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def log_metrics(logger: RunLogger, metrics: dict[str, float], step: int) -> None:
    if logger.writer is None:
        return
    # TODO(partner): extend for W&B or other backends if needed.
    for k, v in metrics.items():
        logger.writer.add_scalar(k, v, step)


def log_config(logger: RunLogger, cfg: Any) -> None:
    if logger.writer is None:
        return
    # TODO(partner): serialize config to yaml/json for richer experiment tracking.
    logger.writer.add_text("config", str(cfg))
