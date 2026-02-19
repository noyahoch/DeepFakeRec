"""Smoke test — runs the full training pipeline with random data.

Usage:  uv run python smoke_test.py
"""

import torch
from lightning import LightningDataModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.train import DeepfakeLitModule, build_model, train


class DummyDataModule(LightningDataModule):
    """Generates random waveforms + binary labels."""

    def __init__(
        self,
        n_train: int = 40,
        n_val: int = 20,
        segment_samples: int = 64600,
        batch_size: int = 5,
    ):
        super().__init__()
        self.n_train = n_train
        self.n_val = n_val
        self.segment_samples = segment_samples
        self.batch_size = batch_size

    def setup(self, stage=None):
        self._train = TensorDataset(
            torch.randn(self.n_train, self.segment_samples),
            torch.randint(0, 2, (self.n_train,)),
        )
        self._val = TensorDataset(
            torch.randn(self.n_val, self.segment_samples),
            torch.randint(0, 2, (self.n_val,)),
        )

    def _collate(self, batch):
        wavs = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return {"wav": wavs, "label": labels}

    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate,
        )


def main():
    cfg = load_config("run_configs/config.yaml")

    model = build_model(cfg)
    lit = DeepfakeLitModule(
        model,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        loss_weights=cfg.train.loss_weights,
    )
    data = DummyDataModule(batch_size=cfg.train.batch_size)

    logger = WandbLogger(project=cfg.logging.project, save_dir=cfg.logging.run_dir)

    trainer = Trainer(
        max_epochs=3,
        precision="32-true",  # CPU-friendly, no amp
        gradient_clip_val=cfg.train.grad_clip,
        log_every_n_steps=1,
        logger=logger,
        accelerator="cpu",
        enable_progress_bar=True,
    )
    trainer.fit(lit, datamodule=data)
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
