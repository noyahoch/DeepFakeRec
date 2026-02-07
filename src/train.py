from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from lightning import LightningModule, Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except Exception:  # pragma: no cover
    LightningModule = nn.Module  # type: ignore[misc,assignment]
    Trainer = object  # type: ignore[assignment]
    EarlyStopping = object  # type: ignore[assignment]
    ModelCheckpoint = object  # type: ignore[assignment]

from .config import RunConfig, seed_everything
from .dataset import AudioDataModule
from .logging import RunLogger, log_metrics, log_config
from .metrics import compute_metrics
from .model import DeepfakeDetector, SlsClassifier, XlsrBackbone


class DeepfakeLitModule(LightningModule):
    def __init__(self, model: DeepfakeDetector, lr: float, weight_decay: float) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self._val_outputs: list[dict[str, torch.Tensor]] = []
        self._test_outputs: list[dict[str, torch.Tensor]] = []
        self.save_hyperparameters(ignore=["model"])

    def forward(self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(wav, attention_mask=attention_mask)

    def _step(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
        wav = batch["wav"]
        label = batch["label"]
        logits = self(wav)
        loss = F.cross_entropy(logits, label)

        # Save scores for EER
        probs = torch.softmax(logits, dim=-1)[:, 1]
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return {"loss": loss, "probs": probs.detach(), "labels": label.detach()}

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        out = self._step(batch, "train")
        return out["loss"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, torch.Tensor]:
        out = self._step(batch, "val")
        self._val_outputs.append(out)
        return out

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, torch.Tensor]:
        out = self._step(batch, "test")
        self._test_outputs.append(out)
        return out

    def _epoch_end(self, outputs: list[dict[str, torch.Tensor]], stage: str) -> None:
        if not outputs:
            return
        probs = torch.cat([o["probs"] for o in outputs]).cpu().numpy()
        labels = torch.cat([o["labels"] for o in outputs]).cpu().numpy()
        metrics = compute_metrics(labels.astype(np.int32), probs.astype(np.float32))
        self.log(f"{stage}_eer", metrics.eer, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self._epoch_end(self._val_outputs, "val")
        self._val_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self._epoch_end(self._test_outputs, "test")
        self._test_outputs.clear()

    def configure_optimizers(self):
        # TODO: add scheduler if desired (paper does not specify).
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


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


def train(cfg: RunConfig) -> None:
    seed_everything(cfg.train.seed)
    model = build_model(cfg)
    lit = DeepfakeLitModule(model, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    data = build_datamodule(cfg)

    logger = RunLogger(
        backend=cfg.logging.backend,
        run_dir=cfg.logging.run_dir,
        project=cfg.logging.project,
    )
    logger.start()
    log_config(logger, cfg)

    callbacks = [
        EarlyStopping(monitor="train_loss", patience=cfg.train.early_stop_patience, mode="min"),
        ModelCheckpoint(monitor="train_loss", save_top_k=1, mode="min"),
    ]

    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        precision=cfg.train.precision,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        callbacks=callbacks,
        gradient_clip_val=cfg.train.grad_clip,
    )
    trainer.fit(lit, datamodule=data)
    logger.close()
