from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class SSLModel(nn.Module):
    def __init__(self, cp_path: str, device: torch.device):
        super(SSLModel, self).__init__()
        self.model = Wav2Vec2Model.from_pretrained(cp_path)
        self.device = device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data: torch.Tensor):
        if (
            next(self.model.parameters()).device != input_data.device
            or next(self.model.parameters()).dtype != input_data.dtype
        ):
            self.model.to(input_data.device, dtype=input_data.dtype)
        self.model.train()
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
        out = self.model(
            input_values=input_tmp,
            output_hidden_states=True,
            return_dict=True,
        )
        emb = out.last_hidden_state
        hidden_states = out.hidden_states
        if hidden_states is None:
            raise RuntimeError("Wav2Vec2Model did not return hidden states.")
        # fairseq layer_results stores each layer as time-major tensor (T, B, C).
        # Keep the same access pattern used by the original model code.
        layerresult = [(h.transpose(0, 1),) for h in hidden_states[1:]]
        return emb, layerresult


def getAttenF(layerResult):
    poollayerResult = []
    fullf = []
    for layer in layerResult:
        layery = layer[0].transpose(0, 1).transpose(1, 2)
        layery = F.adaptive_avg_pool1d(layery, 1)
        layery = layery.transpose(1, 2)
        poollayerResult.append(layery)
        x = layer[0].transpose(0, 1)
        x = x.view(x.size(0), -1, x.size(1), x.size(2))
        fullf.append(x)

    layery = torch.cat(poollayerResult, dim=1)
    fullfeature = torch.cat(fullf, dim=1)
    return layery, fullfeature


class DeepfakeDetector(nn.Module):
    def __init__(self, model_path: str, device: torch.device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(model_path, self.device)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        y0, fullfeature = getAttenF(layerResult)
        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)
        fullfeature = fullfeature.unsqueeze(dim=1)

        x = self.first_bn(fullfeature)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc3(x)
        x = self.selu(x)
        output = self.logsoftmax(x)
        return output
