from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RawBoostConfig:
    # Placeholder parameters; partners can replace with exact paper settings.
    noise_std: float = 0.005
    fir_taps: int = 101


class RawBoostAugment:
    def __init__(self, cfg: RawBoostConfig) -> None:
        self.cfg = cfg

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Placeholder implementation. Partners should replace with full RawBoost:
        white noise + FIR filtering as described in the paper.
        """
        # TODO(partner): implement RawBoost stages (noise injection + FIR filtering)
        # to match the paper's augmentation pipeline.
        if wav.ndim != 1:
            raise AssertionError(f"wav must be 1D, got {wav.shape}")
        noise = torch.randn_like(wav) * self.cfg.noise_std
        return wav + noise
