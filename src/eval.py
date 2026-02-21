"""Evaluation / inference script."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import RunConfig
from .dataset import AudioDataset
from .metrics import compute_metrics
from .model import DeepfakeDetector, SlsClassifier, XlsrBackbone


def load_checkpoint(cfg: RunConfig, ckpt_path: str) -> DeepfakeDetector:
    model = DeepfakeDetector(
        backbone=XlsrBackbone(
            model_name=cfg.model.pretrained_name,
            freeze=False,
            num_layers=cfg.model.num_layers,
        ),
        classifier=SlsClassifier(
            num_layers=cfg.model.num_layers,
            hidden_dim=cfg.model.hidden_dim,
            num_classes=cfg.model.num_classes,
        ),
    )
    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def evaluate(cfg: RunConfig, ckpt_path: str) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(cfg, ckpt_path).to(device)

    dataset = AudioDataset(
        protocol_path=cfg.data.eval_protocol_path,
        audio_dir=cfg.data.eval_audio_dir,
        sample_rate=cfg.data.sample_rate,
        segment_samples=cfg.data.segment_samples,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    all_probs = []
    all_labels = []
    for batch in loader:
        wav = batch["wav"].to(device)
        labels = batch["label"]
        logits = model(wav)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_score = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    metrics = compute_metrics(y_true.astype(np.int32), y_score.astype(np.float32))

    if cfg.eval.save_scores_path:
        with open(cfg.eval.save_scores_path, "w") as f:
            for s in y_score:
                f.write(f"{s}\n")
        print(f"Scores saved to {cfg.eval.save_scores_path}")

    return {"eer": metrics.eer}
