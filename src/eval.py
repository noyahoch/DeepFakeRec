"""Evaluation / inference script."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from pathlib import Path

from .config import RunConfig
from .dataset import AudioDataset
from .logging import log_dict_to_wandb
from .metrics import compute_metrics
from .model import DeepfakeDetector
from .train import build_model


def load_checkpoint(cfg: RunConfig, ckpt_path: str) -> DeepfakeDetector:
    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = state.get("state_dict", state)
    # Lightning saves LightningModule state_dict: keys are "model.backbone.xxx", "model.classifier.xxx".
    # build_model() returns the raw DeepfakeDetector (keys "backbone.xxx", "classifier.xxx"). Strip prefix.
    if state_dict and next(iter(state_dict.keys())).startswith("model."):
        state_dict = {(k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)
    num_missing = len(result.missing_keys)
    num_unexpected = len(result.unexpected_keys)
    if num_missing > 10:
        print(
            f"WARNING: {num_missing} model parameters had no checkpoint match (missing_keys). "
            "Weights may not have loaded — eval will use random/constant output. "
            "Check that the checkpoint is from this codebase (LightningModule with self.model)."
        )
    else:
        print(f"Checkpoint loaded: {len(state_dict) - num_unexpected} keys matched, {num_missing} missing, {num_unexpected} unexpected.")
    model.eval()
    return model


@torch.no_grad()
def evaluate(cfg: RunConfig, ckpt_path: str, run_name: str | None = None) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project,
            name=run_name or "eval",
            config={"ckpt": ckpt_path, "eval_protocol": cfg.data.eval_protocol_path},
        )

    print("Loading checkpoint and model...")
    model = load_checkpoint(cfg, ckpt_path).to(device)
    print("Model loaded. Building eval dataset...")

    eval_max_trials = getattr(cfg.data, "eval_max_trials", None)
    if eval_max_trials is not None:
        print(f"Eval: limiting to first {eval_max_trials} trials (quick test).")
    eval_key_path = getattr(cfg.data, "eval_key_path", None)
    eval_phase_filter = getattr(cfg.data, "eval_phase_filter", None)
    eval_phase_column = getattr(cfg.data, "eval_phase_column", 7)
    phase_filter = (
        (eval_phase_column, eval_phase_filter)
        if eval_phase_filter is not None
        else None
    )
    if phase_filter is not None:
        print(f"Eval: loading only phase={eval_phase_filter!r} trials (column {eval_phase_column}).")
    dataset = AudioDataset(
        protocol_path=cfg.data.eval_protocol_path,
        audio_dir=cfg.data.eval_audio_dir,
        sample_rate=cfg.data.sample_rate,
        segment_samples=cfg.data.segment_samples,
        training=False,
        max_trials=eval_max_trials,
        protocol_key_path=eval_key_path,
        phase_filter=phase_filter,
    )
    n_trials = len(dataset)
    print(f"Eval set: {n_trials} trials. Creating DataLoader...")
    loader = DataLoader(
        dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    all_scores = []
    all_labels = []
    all_file_ids = []
    for batch in tqdm(loader, desc="Eval", total=(n_trials + cfg.eval.batch_size - 1) // cfg.eval.batch_size):
        wav = batch["wav"].to(device)
        labels = batch["label"]
        out = model(wav)  # log-softmax: out[:,1] = log P(bonafide), higher = more bonafide
        scores = out[:, 1]
        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        # DataLoader collates meta as dict of lists: {"file_id": [...], "speaker_id": [...], "key": [...]}
        meta = batch["meta"]
        if isinstance(meta, dict):
            all_file_ids.extend(meta["file_id"])
        else:
            for m in meta:
                all_file_ids.append(m["file_id"])

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)

    # Convention: labels bonafide=1 spoof=0; score should be "higher = more bonafide" (official evaluator).
    # If EER(-score) < EER(score), polarity is flipped — fix pipeline (labels / score column), do not rely on auto-flip.
    bonafide_mean = np.mean(y_score[y_true == 1])
    spoof_mean = np.mean(y_score[y_true == 0])
    n_bonafide, n_spoof = int(np.sum(y_true == 1)), int(np.sum(y_true == 0))
    if n_bonafide == 0 or n_spoof == 0:
        print(f"Warning: only one class in eval set (bonafide={n_bonafide}, spoof={n_spoof}). EER undefined (NaN).")
    eer_raw = compute_metrics(y_true.astype(np.int32), y_score.astype(np.float32)).eer
    eer_neg = compute_metrics(y_true.astype(np.int32), (-y_score).astype(np.float32)).eer
    did_flip = bonafide_mean < spoof_mean
    if did_flip:
        print(
            "POLARITY WARNING: EER(-score) < EER(score) — score polarity appears reversed.\n"
            "  Check: (1) Dataset labels bonafide=1 / spoof=0, (2) Training used same convention, "
            "(3) Score = out[:,1] = log P(bonafide); official format expects higher = bonafide.\n"
            f"  EER(score)={eer_raw:.4f}, EER(-score)={eer_neg:.4f}. Fix the pipeline; for this run we apply -score so the saved file has correct polarity."
        )
        y_score = -y_score
    else:
        print(f"Score polarity OK (higher = more bonafide). EER(score)={eer_raw:.4f}, EER(-score)={eer_neg:.4f}.")

    metrics = compute_metrics(y_true.astype(np.int32), y_score.astype(np.float32))

    score_std = float(np.std(y_score))
    n_unique = len(np.unique(np.round(y_score, decimals=5)))
    print(f"Score stats: min={y_score.min():.4f}, max={y_score.max():.4f}, mean={y_score.mean():.4f}, std={score_std:.6f}, n_unique≈{n_unique}")
    if score_std < 1e-5 or n_unique <= 2:
        print(
            "WARNING: Scores are effectively constant — checkpoint may not have loaded, or model/data bug. "
            "Check for 'Checkpoint loaded: ... keys matched' above."
        )

    if cfg.eval.save_scores_path:
        out_path = Path(cfg.eval.save_scores_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        to_save = y_score
        with open(out_path, "w") as f:
            for fid, s in zip(all_file_ids, to_save):
                f.write(f"{fid} {s}\n")
        print(f"Scores saved to {cfg.eval.save_scores_path} (raw (higher=bonafide), file_id score per line)")

    results = {"eer": metrics.eer}
    if cfg.logging.use_wandb:
        log_dict_to_wandb({"eval/eer": results["eer"]})
    return results
