from __future__ import annotations

import os
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from lightning import LightningDataModule
except Exception:  # pragma: no cover
    LightningDataModule = object  # type: ignore[misc,assignment]

from .config import RawboostSSIConfig
from .rawboost_ssi import ssi_additive_noise


class AudioSample(TypedDict):
    wav: torch.Tensor
    label: int
    meta: dict[str, Any]


@dataclass
class ProtocolEntry:
    audio_path: str
    label: int
    meta: dict[str, Any]


def _protocol_audio_path(file_id: str) -> str:
    """Keep an explicit filename extension; otherwise default to .flac."""
    return file_id if Path(file_id).suffix else f"{file_id}.flac"


def parse_protocol(
    path: str,
    key_file_path: str | None = None,
    phase_filter: tuple[int, str] | None = None,
) -> list[ProtocolEntry]:
    """
    Parse ASVspoof LA protocol file and return a list of entries.

    Supports three formats:
    - File-id-only (reference style, one file_id per line): e.g. ASVspoof2021.LA.cm.eval.trl.txt
      Requires key_file_path to a key file (e.g. trial_metadata.txt) for labels.
    - ASVspoof 2019 style: speaker_id file_id - [attack_id] key
      e.g. LA_0039 LA_E_2834763 - A11 spoof
    - ASVspoof 2021 style: speaker_id file_id codec ... attack_id key notrim subset
      e.g. LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval

    In multi-column formats, the second column is file_id and one column is "bonafide" or "spoof".
    Label: bonafide -> 1, spoof -> 0. audio_path = file_id + ".flac".

    If phase_filter is (column_index, value), only rows where that column equals value are kept
    (e.g. (7, "eval") for LA 2021 eval phase only).
    """
    entries: list[ProtocolEntry] = []
    protocol_path = Path(path)
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {path}")

    with open(protocol_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        return entries

    first_tokens = lines[0].split()
    if len(first_tokens) == 1:
        # File-id-only format (reference ASVspoof2021.LA.cm.eval.trl.txt)
        if not key_file_path or not Path(key_file_path).exists():
            raise FileNotFoundError(
                f"Protocol is file-id-only; key_file_path must point to key file (e.g. trial_metadata.txt): {key_file_path!r}"
            )
        key_entries = parse_protocol(key_file_path, phase_filter=phase_filter)
        key_by_id = {e.meta["file_id"]: (e.label, e.meta.get("key", "bonafide" if e.label == 1 else "spoof")) for e in key_entries}
        for line in lines:
            file_id = line.strip()
            if not file_id:
                continue
            if file_id not in key_by_id:
                continue
            label, key = key_by_id[file_id]
            entries.append(
                ProtocolEntry(
                    audio_path=_protocol_audio_path(file_id),
                    label=label,
                    meta={"speaker_id": "", "file_id": file_id, "key": key},
                )
            )
        return entries

    # Multi-column format
    for line in lines:
        tokens = line.split()
        if len(tokens) < 2:
            raise ValueError(f"Protocol line has too few columns: {line!r}")
        if phase_filter is not None and (
            len(tokens) <= phase_filter[0] or tokens[phase_filter[0]] != phase_filter[1]
        ):
            continue
        speaker_id, file_id = tokens[0], tokens[1]
        key = None
        for t in tokens:
            if t.lower() in ("bonafide", "spoof"):
                key = t
                break
        if key is None:
            raise ValueError(f"No 'bonafide' or 'spoof' label in protocol line: {line!r}")
        label = 1 if key.lower() == "bonafide" else 0
        entries.append(
            ProtocolEntry(
                audio_path=_protocol_audio_path(file_id),
                label=label,
                meta={"speaker_id": speaker_id, "file_id": file_id, "key": key},
            )
        )
    return entries


def load_audio(path: str, sample_rate: int = 16000) -> torch.Tensor:
    """
    Load audio with librosa only (same as SLS). Returns waveform (T,) float32 at sample_rate.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        y, _ = librosa.load(
            path,
            sr=sample_rate,
            mono=True,
            dtype="float32",
            res_type="kaiser_best",
        )
    return torch.from_numpy(y)


def truncate_or_pad(wav: torch.Tensor, segment_samples: int) -> torch.Tensor:
    """
    Fixed-length waveform. Same as SLS data_utils_SSL.pad():
    - long: take the first segment_samples (prefix truncation, as in SLS).
    - short: repeat to fill, then trim to segment_samples.
    Order in pipeline matches SLS: augment (RawBoost algo 3) then truncate/pad.
    """
    if wav.ndim != 1:
        raise AssertionError(f"wav must be 1D, got {wav.shape}")

    n = wav.numel()
    if n == segment_samples:
        return wav

    if n > segment_samples:
        return wav[:segment_samples].clone()

    reps = (segment_samples + n - 1) // n
    return wav.repeat(reps)[:segment_samples]


class AudioDataset(Dataset[AudioSample]):
    """Train: RawBoost algo 3 (SSI) then truncate/pad (same order as SLS). Eval: truncate/pad only."""

    def __init__(
        self,
        protocol_path: str,
        audio_dir: str,
        sample_rate: int = 16000,
        segment_samples: int = 64600,
        training: bool = False,
        max_trials: int | None = None,
        protocol_key_path: str | None = None,
        phase_filter: tuple[int, str] | None = None,
        rawboost_cfg: RawboostSSIConfig | None = None,
    ) -> None:
        protocol = parse_protocol(
            protocol_path,
            key_file_path=protocol_key_path,
            phase_filter=phase_filter,
        )
        if max_trials is not None:
            rng = random.Random(42)
            protocol = list(protocol)
            rng.shuffle(protocol)
            self.protocol = protocol[:max_trials]
        else:
            self.protocol = protocol
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.training = training
        self.rawboost_cfg = rawboost_cfg

    def __len__(self) -> int:
        return len(self.protocol)

    def __getitem__(self, idx: int) -> AudioSample:
        entry = self.protocol[idx]
        full_path = os.path.join(self.audio_dir, entry.audio_path)
        wav = load_audio(full_path, self.sample_rate)
        if self.training:
            if self.rawboost_cfg is not None:
                wav_np = ssi_additive_noise(
                    wav.numpy(), self.sample_rate, **asdict(self.rawboost_cfg)
                )
            else:
                wav_np = ssi_additive_noise(wav.numpy(), self.sample_rate)
            wav = torch.from_numpy(wav_np)
        wav = truncate_or_pad(wav, self.segment_samples)
        return {"wav": wav, "label": entry.label, "meta": entry.meta}


class AudioDataModule(LightningDataModule):
    def __init__(
        self,
        train_protocol_path: str,
        train_audio_dir: str,
        eval_protocol_path: str,
        eval_audio_dir: str,
        batch_size: int = 5,
        eval_batch_size: int = 32,
        num_workers: int = 4,
        sample_rate: int = 16000,
        segment_samples: int = 64600,
        eval_extra: list[dict] | None = None,
        eval_key_path: str | None = None,
        eval_extra_every_n_epochs: int = 1,
        eval_extra_max_trials: int | None = None,
        eval_max_trials: int | None = None,
        rawboost_cfg: RawboostSSIConfig | None = None,
    ) -> None:
        super().__init__()
        self.train_protocol_path = train_protocol_path
        self.train_audio_dir = train_audio_dir
        self.eval_protocol_path = eval_protocol_path
        self.eval_audio_dir = eval_audio_dir
        self.eval_key_path = eval_key_path
        self.eval_extra_every_n_epochs = eval_extra_every_n_epochs
        self.eval_extra_max_trials = eval_extra_max_trials
        self.eval_max_trials = eval_max_trials
        # -1 so validate-only runs extra; during fit the callback sets current epoch.
        self._validation_epoch = -1
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.eval_extra = eval_extra or []
        self.rawboost_cfg = rawboost_cfg
        self._train_ds: AudioDataset | None = None
        self._eval_ds: AudioDataset | None = None
        self._eval_extra_ds: list[AudioDataset] = []

    @property
    def eval_names(self) -> list[str]:
        """Names for each val dataloader: primary 'eval' plus names from eval_extra."""
        names = ["eval"]
        for i, item in enumerate(self.eval_extra):
            names.append(str(item.get("name", f"extra_{i}")))
        return names

    def set_validation_epoch(self, epoch: int) -> None:
        """Set by a callback so val_dataloader() can return extra only every N epochs."""
        self._validation_epoch = epoch

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self._train_ds = AudioDataset(
                self.train_protocol_path,
                self.train_audio_dir,
                sample_rate=self.sample_rate,
                segment_samples=self.segment_samples,
                training=True,
                rawboost_cfg=self.rawboost_cfg,
            )
        if stage in (None, "fit", "validate", "test"):
            self._eval_ds = AudioDataset(
                self.eval_protocol_path,
                self.eval_audio_dir,
                sample_rate=self.sample_rate,
                segment_samples=self.segment_samples,
                training=False,
                protocol_key_path=self.eval_key_path,
                max_trials=self.eval_max_trials,
            )
            self._eval_extra_ds = []
            for item in self.eval_extra:
                proto = item.get("protocol_path") or item.get("eval_protocol_path")
                audio = item.get("audio_dir") or item.get("eval_audio_dir")
                key_path = item.get("key_path") or item.get("eval_key_path")
                max_trials = item.get("max_trials", self.eval_extra_max_trials)
                if proto and audio:
                    self._eval_extra_ds.append(
                        AudioDataset(
                            proto,
                            audio,
                            sample_rate=self.sample_rate,
                            segment_samples=self.segment_samples,
                            training=False,
                            protocol_key_path=key_path,
                            max_trials=max_trials,
                        )
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

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        if self._eval_ds is None:
            raise RuntimeError("DataModule not set up. Call setup('fit') first.")
        primary = DataLoader(
            self._eval_ds,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        if not self._eval_extra_ds:
            return primary
        # Run extra eval only every eval_extra_every_n_epochs (e.g. 5) to save time.
        if self.eval_extra_every_n_epochs > 1:
            if (self._validation_epoch + 1) % self.eval_extra_every_n_epochs != 0:
                return primary
        extra_loaders = [
            DataLoader(
                ds,
                batch_size=self.eval_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            for ds in self._eval_extra_ds
        ]
        return [primary] + extra_loaders

    def test_dataloader(self) -> DataLoader:
        if self._eval_ds is None:
            raise RuntimeError("DataModule not set up. Call setup('test') first.")
        return DataLoader(
            self._eval_ds,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
