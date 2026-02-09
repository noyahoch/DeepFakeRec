"""Logging helpers.

With Lightning + WandbLogger the heavy lifting is handled automatically.
This module is kept for any manual logging needs outside the Trainer loop.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import wandb


def log_dict_to_wandb(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dict of scalars to the active W&B run (if any)."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_config_to_wandb(cfg: Any) -> None:
    """Push the full RunConfig to the active W&B run."""
    if wandb.run is not None:
        wandb.config.update(asdict(cfg), allow_val_change=True)
