from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import librosa
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from lightning import LightningDataModule
except Exception:  # pragma: no cover
    LightningDataModule = object  # type: ignore[misc,assignment]

from .rawboost import RawBoostAugment, RawBoostConfig


class AudioSample(TypedDict):
    wav: torch.Tensor
    label: int
    meta: dict[str, Any]


@dataclass
class ProtocolEntry:
    audio_path: str
    label: int
    meta: dict[str, Any]


def parse_protocol(path: str) -> list[ProtocolEntry]:
    """
    Parse ASVspoof 2019 LA protocol file and return a list of entries.

    Protocol format (space-separated): speaker_id file_id - - key
    where key is "bonafide" (label 0) or "spoof" (label 1).
    Returns entries with audio_path = file_id + ".flac" (relative to audio_dir).
    """
    entries: list[ProtocolEntry] = []
    protocol_path = Path(path)
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {path}")

    with open(protocol_path, encoding="utf-8") as f:
        for line in f.readlines():
            if not line:
                continue
            speaker_id, file_id, _, _, key = line.split()
            if not key.lower() in ("bonafide", "spoof"):
                raise ValueError(f"Invalid key: {key}, must be 'bonafide' or 'spoof'")
            label = 0 if key.lower() == "bonafide" else 1
            audio_path = f"{file_id}.flac"
            entries.append(
                ProtocolEntry(
                    audio_path=audio_path,
                    label=label,
                    meta={"speaker_id": speaker_id, "file_id": file_id, "key": key},
                )
            )
    return entries


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    """
    Load audio file and return waveform tensor (T,) as float32.
    Resamples to sample_rate if needed (e.g. with librosa).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    y, _ = librosa.load(path, sr=sample_rate, mono=True, dtype="float32")
    return torch.from_numpy(y)


def crop_or_pad(wav: torch.Tensor, segment_samples: int) -> torch.Tensor:
    """
    Ensure fixed-length waveform. If too long, crop; if too short, pad or repeat.
    """
    if wav.ndim != 1:
        raise AssertionError(f"wav must be 1D, got {wav.shape}")
    if wav.numel() == segment_samples:
        return wav
    if wav.numel() > segment_samples:
        return wav[:segment_samples]
    # Simple repeat-pad to reach segment length.
    reps = (segment_samples + wav.numel() - 1) // wav.numel()
    padded = wav.repeat(reps)[:segment_samples]
    return padded


class AudioDataset(Dataset[AudioSample]):
    def __init__(
        self,
        protocol_path: str,
        audio_dir: str,
        sample_rate: int = 16000,
        segment_samples: int = 64600,
        augment: bool = False,
        rawboost_cfg: RawBoostConfig | None = None,
    ) -> None:
        self.protocol = parse_protocol(protocol_path)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.augment = augment
        self.rawboost = (
            RawBoostAugment(rawboost_cfg or RawBoostConfig()) if augment else None
        )

    def __len__(self) -> int:
        return len(self.protocol)

    def __getitem__(self, idx: int) -> AudioSample:
        entry = self.protocol[idx]
        full_path = os.path.join(self.audio_dir, entry.audio_path)
        wav = load_audio(full_path, self.sample_rate)
        wav = crop_or_pad(wav, self.segment_samples)
        if self.rawboost is not None:
            wav = self.rawboost(wav)
        return {"wav": wav, "label": entry.label, "meta": entry.meta}


class AudioDataModule(LightningDataModule):
    def __init__(
        self,
        train_protocol_path: str,
        train_audio_dir: str,
        eval_protocol_path: str,
        eval_audio_dir: str,
        batch_size: int = 5,
        num_workers: int = 4,
        sample_rate: int = 16000,
        segment_samples: int = 64600,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.train_protocol_path = train_protocol_path
        self.train_audio_dir = train_audio_dir
        self.eval_protocol_path = eval_protocol_path
        self.eval_audio_dir = eval_audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.augment = augment
        self._train_ds: AudioDataset | None = None
        self._eval_ds: AudioDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self._train_ds = AudioDataset(
                self.train_protocol_path,
                self.train_audio_dir,
                sample_rate=self.sample_rate,
                segment_samples=self.segment_samples,
                augment=self.augment,
            )
        if stage in (None, "fit", "validate", "test"):
            self._eval_ds = AudioDataset(
                self.eval_protocol_path,
                self.eval_audio_dir,
                sample_rate=self.sample_rate,
                segment_samples=self.segment_samples,
                augment=False,
            )

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("DataModule not set up. Call setup('fit') first.")
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self._eval_ds is None:
            raise RuntimeError("DataModule not set up. Call setup('fit') first.")
        return DataLoader(
            self._eval_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self._eval_ds is None:
            raise RuntimeError("DataModule not set up. Call setup('test') first.")
        return DataLoader(
            self._eval_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
