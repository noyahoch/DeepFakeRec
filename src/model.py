"""XLS-R backbone (HuggingFace) + SLS classifier for audio deepfake detection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# With segment_samples=64600 at 16 kHz, XLS-R produces T'=201 frames.
# After MaxPool2d(3,3): floor((201-3)/3)+1=67, floor((1024-3)/3)+1=341 → 67*341=22847.
_FLAT_DIM_AFTER_POOL = 22847


def _assert_rank(t: torch.Tensor, rank: int, name: str) -> None:
    if t.ndim != rank:
        raise AssertionError(f"{name} must be rank {rank}, got shape {tuple(t.shape)}")


class XlsrBackbone(nn.Module):
    """XLS-R backbone via HuggingFace Wav2Vec2Model."""

    def __init__(self, model_name: str, freeze: bool = False, num_layers: int = 24):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.num_layers = num_layers
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(
        self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None
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
        layer_states = hidden_states[1:]
        if len(layer_states) < self.num_layers:
            raise AssertionError(
                f"Expected at least {self.num_layers} hidden layers, got {len(layer_states)}."
            )
        layer_states = layer_states[: self.num_layers]
        H = torch.stack(layer_states, dim=0)
        _assert_rank(H, 4, "H")
        return H


class SlsClassifier(nn.Module):
    """SLS classifier: layer-weighted sum → BN → SELU → MaxPool → Linear → LogSoftmax."""

    def __init__(
        self,
        num_layers: int = 24,
        hidden_dim: int = 1024,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.weight_fc = nn.Linear(hidden_dim, 1, bias=True)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc1 = nn.Linear(_FLAT_DIM_AFTER_POOL, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        _assert_rank(H, 4, "H")
        L, B, T, C = H.shape
        if L != self.num_layers:
            raise AssertionError(f"Expected {self.num_layers} layers, got {L}")
        if C != self.hidden_dim:
            raise AssertionError(f"Expected hidden_dim {self.hidden_dim}, got {C}")
        H_avg = H.mean(dim=2, keepdim=True)
        H_fc = self.weight_fc(H_avg.reshape(L * B, C)).view(L, B, 1, 1)
        alpha = torch.sigmoid(H_fc)
        H_weighted = (alpha *   H).sum(dim=0)
        x = H_weighted.unsqueeze(1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc3(x)
        x = self.selu(x)
        x = self.logsoftmax(x)
        return x


class DeepfakeDetector(nn.Module):
    """Backbone + SLS classifier."""

    def __init__(self, backbone: nn.Module, classifier: SlsClassifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(
        self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        H = self.backbone(wav, attention_mask=attention_mask)
        return self.classifier(H)
