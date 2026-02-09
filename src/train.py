"""Training loop for Audio Deepfake Detection (XLS-R + SLS).

Uses PyTorch Lightning with:
  - Weighted Cross-Entropy loss  [0.1, 0.9]  (paper reference)
  - Adam optimiser               lr=1e-6, wd=1e-4
  - W&B logging
  - Early stopping + best-model checkpointing
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .config import RunConfig, seed_everything
from .dataset import AudioDataModule
from .metrics import compute_metrics
from .model import DeepfakeDetector, SlsClassifier, XlsrBackbone


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class DeepfakeLitModule(LightningModule):
    def __init__(
        self,
        model: DeepfakeDetector,
        lr: float,
        weight_decay: float,
        loss_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model"])

        # Weighted CE – paper uses [0.1, 0.9]
        w = torch.FloatTensor(loss_weights) if loss_weights else None
        self.criterion = nn.CrossEntropyLoss(weight=w)

        self._val_outputs: list[dict[str, torch.Tensor]] = []

    # -- forward --
    def forward(self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(wav, attention_mask=attention_mask)

    # -- shared step --
    def _step(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
        wav = batch["wav"]
        label = batch["label"]
        logits = self(wav)
        loss = self.criterion(logits, label)

        preds = logits.argmax(dim=1)
        acc = (preds == label).float().mean()
        probs = torch.softmax(logits, dim=-1)[:, 1]

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True, on_step=False)

        return {"loss": loss, "probs": probs.detach(), "labels": label.detach()}

    # -- train --
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")["loss"]

    # -- val --
    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        out = self._step(batch, "val")
        self._val_outputs.append(out)

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            return
        probs = torch.cat([o["probs"] for o in self._val_outputs]).cpu().numpy()
        labels = torch.cat([o["labels"] for o in self._val_outputs]).cpu().numpy()
        metrics = compute_metrics(labels.astype(np.int32), probs.astype(np.float32))
        self.log("val/eer", metrics.eer, prog_bar=True)
        self._val_outputs.clear()

    # -- optimiser --
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_model(cfg: RunConfig) -> DeepfakeDetector:
    backbone = XlsrBackbone(
        model_name=cfg.model.pretrained_name,
        freeze=cfg.model.freeze_backbone,
        num_layers=cfg.model.num_layers,
    )
    classifier = SlsClassifier(
        num_layers=cfg.model.num_layers,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=cfg.model.num_classes,
    )
    return DeepfakeDetector(backbone=backbone, classifier=classifier)


def build_datamodule(cfg: RunConfig) -> AudioDataModule:
    return AudioDataModule(
        train_protocol_path=cfg.data.train_protocol_path,
        train_audio_dir=cfg.data.train_audio_dir,
        eval_protocol_path=cfg.data.eval_protocol_path,
        eval_audio_dir=cfg.data.eval_audio_dir,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        sample_rate=cfg.data.sample_rate,
        segment_samples=cfg.data.segment_samples,
        augment=cfg.data.augment,
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def train(cfg: RunConfig) -> None:
    seed_everything(cfg.train.seed)

    model = build_model(cfg)
    lit = DeepfakeLitModule(
        model,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        loss_weights=cfg.train.loss_weights,
    )
    data = build_datamodule(cfg)

    logger = WandbLogger(
        project=cfg.logging.project,
        save_dir=cfg.logging.run_dir,
    )

    callbacks = [
        EarlyStopping(
            monitor="train/loss",
            patience=cfg.train.early_stop_patience,
            mode="min",
        ),
        ModelCheckpoint(
            dirpath=cfg.train.checkpoint_dir,
            filename="best-{epoch}-{val/eer:.4f}",
            monitor="val/eer",
            save_top_k=1,
            mode="min",
        ),
    ]

    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.grad_clip,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(lit, datamodule=data)
