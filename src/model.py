from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from transformers import Wav2Vec2Model


def _assert_rank(t: torch.Tensor, rank: int, name: str) -> None:
    if t.ndim != rank:
        raise AssertionError(f"{name} must be rank {rank}, got shape {tuple(t.shape)}")


class XlsrBackbone(nn.Module):
    def __init__(self, model_name: str, freeze: bool = False, num_layers: int = 24):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.num_layers = num_layers
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(
        self, wav: torch.Tensor, *, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        _assert_rank(wav, 2, "wav")
        if wav.dtype != torch.float32:
            wav = wav.float()
        out = self.model(
            wav,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = out.hidden_states
        if hidden_states is None:
            raise RuntimeError("Wav2Vec2Model did not return hidden_states.")

        # Wav2Vec2Model returns len = num_hidden_layers + 1.
        layer_states = hidden_states[1:]
        if len(layer_states) < self.num_layers:
            raise AssertionError(
                f"Expected at least {self.num_layers} hidden layers, got {len(layer_states)}."
            )
        layer_states = layer_states[: self.num_layers]

        # Stack to (L, B, T', C)
        H = torch.stack(layer_states, dim=0)
        _assert_rank(H, 4, "H")
        return H


class SlsClassifier(nn.Module):
    def __init__(self, num_layers: int = 24, hidden_dim: int = 1024, num_classes: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.weight_fc = nn.Linear(hidden_dim, 1, bias=True)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=True)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (L, B, T', C)
        _assert_rank(H, 4, "H")
        L, B, T, C = H.shape
        if L != self.num_layers:
            raise AssertionError(f"Expected {self.num_layers} layers, got {L}")
        if C != self.hidden_dim:
            raise AssertionError(f"Expected hidden_dim {self.hidden_dim}, got {C}")
        if T <= 0:
            raise AssertionError("T' must be > 0")

        # avgpool over time -> (L, B, 1, C)
        H_avg = H.mean(dim=2, keepdim=True)

        # FC per layer -> (L, B, 1, 1)
        H_fc = self.weight_fc(H_avg.reshape(L * B, C)).view(L, B, 1, 1)
        alpha = torch.sigmoid(H_fc)

        # Weighted sum across layers -> (B, T', C)
        H_weighted = (alpha * H).sum(dim=0)

        # Temporal maxpool -> (B, C)
        pooled = H_weighted.max(dim=1).values

        logits = self.classifier(pooled)
        _assert_rank(logits, 2, "logits")
        return logits


class DeepfakeDetector(nn.Module):
    def __init__(self, backbone: XlsrBackbone, classifier: SlsClassifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(
        self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        H = self.backbone(wav, attention_mask=attention_mask)
        return self.classifier(H)


def _smoke_test_backbone(model_name: str = "facebook/wav2vec2-xls-r-300m") -> None:
    """
    Smoke test: load XLS-R backbone and run a short forward pass.
    This validates that the checkpoint can be resolved and hidden states are returned.
    """
    backbone = XlsrBackbone(model_name=model_name, freeze=True, num_layers=24)
    wav = torch.zeros(1, 64600, dtype=torch.float32)
    with torch.no_grad():
        H = backbone(wav)
    if H.shape[0] != 24 or H.shape[-1] != 1024:
        raise AssertionError(f"Unexpected backbone output shape: {tuple(H.shape)}")
    print(f"Backbone loaded: {model_name}")
    print(f"H shape: {tuple(H.shape)}")


def _smoke_test_classifier() -> None:
    """
    Smoke test: run SLS classifier with a synthetic hidden-state tensor.
    """
    classifier = SlsClassifier(num_layers=24, hidden_dim=1024, num_classes=2)
    H = torch.zeros(24, 2, 10, 1024, dtype=torch.float32)
    with torch.no_grad():
        logits = classifier(H)
    if logits.shape != (2, 2):
        raise AssertionError(f"Unexpected classifier output shape: {tuple(logits.shape)}")
    print(f"Classifier output shape: {tuple(logits.shape)}")


def _smoke_test_full_model(model_name: str = "facebook/wav2vec2-xls-r-300m") -> None:
    """
    Smoke test: end-to-end forward pass (backbone + SLS classifier).
    """
    backbone = XlsrBackbone(model_name=model_name, freeze=True, num_layers=24)
    classifier = SlsClassifier(num_layers=24, hidden_dim=1024, num_classes=2)
    model = DeepfakeDetector(backbone=backbone, classifier=classifier)
    wav = torch.zeros(1, 64600, dtype=torch.float32)
    with torch.no_grad():
        logits = model(wav)
    if logits.shape != (1, 2):
        raise AssertionError(f"Unexpected model output shape: {tuple(logits.shape)}")
    print(f"Full model output shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    # Note: This requires access to the pretrained checkpoint (e.g., via Hugging Face).
    # If running offline, download the checkpoint beforehand or set TRANSFORMERS_CACHE.
    _smoke_test_backbone()
    _smoke_test_classifier()
    _smoke_test_full_model()
