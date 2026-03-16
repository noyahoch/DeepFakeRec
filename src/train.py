"""Training loop for Audio Deepfake Detection (XLS-R + SLS).

Uses PyTorch Lightning with:
  - Weighted Cross-Entropy loss  [0.1, 0.9]  (paper reference)
  - Adam optimiser               lr=1e-6, wd=1e-4
  - W&B logging
  - Early stopping + best-model checkpointing
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F

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
        log_train_eer: bool = True,
        save_val_scores_path: str | None = None,
        eval_names: list[str] | None = None,
        eval_phase_filter: str | None = None,
        eval_phase_column: int = 7,
        eval_protocol_path: str | None = None,
        eval_protocol_paths: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_train_eer = log_train_eer
        self.save_val_scores_path = save_val_scores_path
        self.eval_names = eval_names or ["eval"]
        self.eval_phase_filter = eval_phase_filter
        self.eval_phase_column = eval_phase_column
        self.eval_protocol_path = eval_protocol_path
        self.eval_protocol_paths = eval_protocol_paths  # one per val dataloader (primary + extras)
        self.save_hyperparameters(ignore=["model"])

        # Weighted CE – paper uses [0.1, 0.9]
        w = torch.FloatTensor(loss_weights) if loss_weights else None
        self.criterion = nn.CrossEntropyLoss(weight=w)

        self._val_outputs_by_dl: dict[int, list[dict]] = {}
        self._train_outputs: list[dict[str, torch.Tensor]] = []

    # -- forward --
    def forward(
        self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.model(wav, attention_mask=attention_mask)

    # -- shared step --
    def _step(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
        wav = batch["wav"]
        label = batch["label"]
        out = self(wav)  # log-softmax output (matches reference)
        loss = self.criterion(out, label)

        preds = out.argmax(dim=1)
        acc = (preds == label).float().mean()
        scores = out[:, 1]  # log-prob of bonafide class

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True, on_step=False)

        return {"loss": loss, "probs": scores.detach(), "labels": label.detach()}

    def _log_eer_from_outputs(
        self,
        outputs: list[dict],
        metric_key: str,
        save_path: str | None = None,
        protocol_path_override: str | None = None,
    ) -> None:
        """Concatenate step outputs, compute EER, log it, optionally save (file_id, score), and clear the list."""
        if not outputs:
            return
        probs = torch.cat([o["probs"] for o in outputs]).cpu().numpy()
        labels = torch.cat([o["labels"] for o in outputs]).cpu().numpy()
        file_ids: list[str] = []
        for o in outputs:
            fids = o.get("file_id")
            if fids is not None:
                file_ids.extend(fids)

        # Optionally restrict EER to one phase (e.g. "eval") to match organisers' reported metric
        eer_probs, eer_labels = probs, labels
        protocol_path = protocol_path_override or self.eval_protocol_path
        if (
            self.eval_phase_filter is not None
            and protocol_path is not None
            and Path(protocol_path).exists()
            and len(file_ids) == len(probs)
        ):
            file_id_to_phase: dict[str, str] = {}
            with open(Path(protocol_path), encoding="utf-8") as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) > self.eval_phase_column:
                        file_id_to_phase[tokens[1]] = tokens[self.eval_phase_column]
            if file_id_to_phase and any(fid in file_id_to_phase for fid in file_ids):
                mask = np.array(
                    [
                        file_id_to_phase.get(fid) == self.eval_phase_filter
                        for fid in file_ids
                    ],
                    dtype=bool,
                )
                n_before, n_after = len(labels), int(np.sum(mask))
                if n_after > 0:
                    eer_probs = probs[mask]
                    eer_labels = labels[mask]
                    self.print(
                        f"Val EER on phase={self.eval_phase_filter!r} only: {n_after} trials (from {n_before} total)"
                    )

        metrics = compute_metrics(
            eer_labels.astype(np.int32), eer_probs.astype(np.float32)
        )
        self.log(metric_key, metrics.eer, prog_bar=True)
        if save_path and len(file_ids) == len(probs):
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                for fid, s in zip(file_ids, probs):
                    f.write(f"{fid} {s}\n")
        elif save_path:
            self.print(
                f"Warning: not saving val scores (file_id len {len(file_ids)} != probs len {len(probs)})"
            )
        outputs.clear()

    # -- train --
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        out = self._step(batch, "train")
        if self.log_train_eer:
            self._train_outputs.append({"probs": out["probs"], "labels": out["labels"]})
        return out["loss"]

    def on_train_epoch_end(self) -> None:
        if not self.log_train_eer:
            self._train_outputs.clear()
            return
        self._log_eer_from_outputs(self._train_outputs, "train/eer")

    # -- val --
    def validation_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        out = self._step(batch, "val")
        meta = batch.get("meta")
        if isinstance(meta, dict) and "file_id" in meta:
            out["file_id"] = list(meta["file_id"])
        else:
            out["file_id"] = []
        if dataloader_idx not in self._val_outputs_by_dl:
            self._val_outputs_by_dl[dataloader_idx] = []
        self._val_outputs_by_dl[dataloader_idx].append(out)

    def on_validation_epoch_end(self) -> None:
        for dl_idx in sorted(self._val_outputs_by_dl.keys()):
            name = self.eval_names[dl_idx] if dl_idx < len(self.eval_names) else f"eval_{dl_idx}"
            metric_key = "val/eer" if dl_idx == 0 and name == "eval" else f"val/eer_{name}"
            save_path = None
            if dl_idx == 0 and self.save_val_scores_path:
                save_path = self.save_val_scores_path
            protocol_path = None
            if self.eval_protocol_paths and dl_idx < len(self.eval_protocol_paths):
                protocol_path = self.eval_protocol_paths[dl_idx]
            self._log_eer_from_outputs(
                self._val_outputs_by_dl[dl_idx],
                metric_key,
                save_path=save_path,
                protocol_path_override=protocol_path,
            )
        self._val_outputs_by_dl.clear()

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
        layer_aggregation=cfg.model.layer_aggregation,
    )
    return DeepfakeDetector(backbone=backbone, classifier=classifier)


def build_datamodule(cfg: RunConfig) -> AudioDataModule:
    return AudioDataModule(
        train_protocol_path=cfg.data.train_protocol_path,
        train_audio_dir=cfg.data.train_audio_dir,
        eval_protocol_path=cfg.data.eval_protocol_path,
        eval_audio_dir=cfg.data.eval_audio_dir,
        batch_size=cfg.train.batch_size,
        eval_batch_size=cfg.eval.batch_size,
        num_workers=cfg.train.num_workers,
        sample_rate=cfg.data.sample_rate,
        segment_samples=cfg.data.segment_samples,
        eval_extra=getattr(cfg.data, "eval_extra", None),
        eval_key_path=getattr(cfg.data, "eval_key_path", None),
        eval_extra_every_n_epochs=getattr(cfg.train, "eval_extra_every_n_epochs", 1),
        eval_extra_max_trials=getattr(cfg.data, "eval_extra_max_trials", None),
        eval_max_trials=getattr(cfg.data, "eval_max_trials", None),
        use_rawboost=getattr(cfg.data, "use_rawboost", True),
        rawboost_cfg=getattr(cfg.data, "rawboost", None),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _login_wandb_from_file(path: str) -> None:
    """Read W&B API key from a file and log in (e.g. api_keys/yoni.txt)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Wandb key file not found: {p}")
    wandb.login(key=p.read_text().strip())


def _run_path(base: str, run_name: str | None) -> Path:
    """Return base as Path, or base/run_name if run_name is not None."""
    path = Path(base)
    if run_name is not None:
        path = path / run_name
    return path


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def train(
    cfg: RunConfig,
    run_name: str | None = None,
    wandb_key_path: str | None = None,
) -> None:
    seed_everything(cfg.train.seed)

    run_dir = _run_path(cfg.logging.run_dir, run_name)
    checkpoint_save_dir = _run_path(cfg.train.checkpoint_save_dir, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model: {n_trainable:,} trainable / {n_total:,} total params "
        f"(backbone frozen={getattr(cfg.model, 'freeze_backbone', True)})"
    )
    data = build_datamodule(cfg)
    eval_protocol_paths = [cfg.data.eval_protocol_path]
    if getattr(cfg.data, "eval_extra", None):
        for extra in cfg.data.eval_extra:
            p = extra.get("protocol_path") or extra.get("eval_protocol_path")
            if p:
                eval_protocol_paths.append(p)
    lit = DeepfakeLitModule(
        model,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        loss_weights=cfg.train.loss_weights,
        log_train_eer=cfg.train.log_train_eer,
        save_val_scores_path=getattr(cfg.train, "save_val_scores_path", None),
        eval_names=data.eval_names,
        eval_phase_filter=getattr(cfg.data, "eval_phase_filter", None),
        eval_phase_column=getattr(cfg.data, "eval_phase_column", 7),
        eval_protocol_path=cfg.data.eval_protocol_path,
        eval_protocol_paths=eval_protocol_paths,
    )

    if not cfg.logging.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    elif wandb_key_path is not None:
        _login_wandb_from_file(wandb_key_path)
    logger = WandbLogger(
        project=cfg.logging.project,
        save_dir=str(run_dir),
        name=run_name,
    )

    early_monitor = getattr(cfg.train, "early_stop_monitor", None) or "train/loss"
    ckpt_monitor = getattr(cfg.train, "checkpoint_monitor", None) or "val/eer"
    callbacks = []
    # So val_dataloader() can return extra only every N epochs.
    if getattr(cfg.train, "eval_extra_every_n_epochs", 1) > 1:

        class _SetValEpoch(Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                if trainer.datamodule is not None and hasattr(
                    trainer.datamodule, "set_validation_epoch"
                ):
                    trainer.datamodule.set_validation_epoch(trainer.current_epoch)

        callbacks.append(_SetValEpoch())
    callbacks.append(
        EarlyStopping(
            monitor=early_monitor,
            patience=cfg.train.early_stop_patience,
            mode="min",
        ),
    )
    if not cfg.train.eval_only:
        ckpt_filename_metric = ckpt_monitor.replace("/", "-")
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_save_dir),
                filename=f"best-{{epoch}}-{{{ckpt_filename_metric}:.4f}}",
                monitor=ckpt_monitor,
                save_top_k=1,
                save_last=True,
                mode="min",
            ),
        )

    ckpt_path = cfg.train.checkpoint_path if cfg.train.checkpoint_path else None
    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.grad_clip,
        check_val_every_n_epoch=cfg.train.val_every_n_epochs,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
    )

    if cfg.train.eval_only:
        trainer.validate(lit, datamodule=data, ckpt_path=ckpt_path)
    else:
        trainer.fit(lit, datamodule=data, ckpt_path=ckpt_path)
        # Run validation once at the end of training (using the last checkpoint, not necessarily the best)
        trainer.validate(lit, datamodule=data)
