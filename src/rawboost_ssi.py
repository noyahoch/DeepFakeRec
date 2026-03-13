"""SSI (algo 3) augmentation only — ported from SLS RawBoost. Same as SLS with algo=3 + center crop."""

from __future__ import annotations

import numpy as np
from scipy import signal


def _rand_range(x1: float, x2: float, integer: bool) -> float | int:
    y = np.random.uniform(low=x1, high=x2, size=(1,)).item()
    return int(y) if integer else float(y)


def _norm_wav(x: np.ndarray, always: bool) -> np.ndarray:
    if always or np.amax(np.abs(x)) > 1:
        x = x / np.amax(np.abs(x))
    return x


def _gen_notch_coeffs(
    n_bands: int,
    min_f: float,
    max_f: float,
    min_bw: float,
    max_bw: float,
    min_coeff: float,
    max_coeff: float,
    min_g: float,
    max_g: float,
    fs: int,
) -> np.ndarray:
    b = np.array([1.0])
    for _ in range(n_bands):
        fc = _rand_range(min_f, max_f, False)
        bw = _rand_range(min_bw, max_bw, False)
        c = _rand_range(min_coeff, max_coeff, True)
        if c % 2 == 0:
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(
            signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b
        )
    G = _rand_range(min_g, max_g, False)
    _, h = signal.freqz(b, 1, fs=fs)
    b = (10 ** (G / 20)) * b / np.amax(np.abs(h))
    return b


def _filter_fir(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), mode="constant")
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N / 2) : int(y.shape[0] - N / 2)]
    return y


# SLS defaults (main.py --SNRmin 10 --SNRmax 40 --nBands 5 etc.)
SSI_SNRMIN = 10
SSI_SNRMAX = 40
SSI_NBANDS = 5
SSI_MIN_F = 20
SSI_MAX_F = 8000
SSI_MIN_BW = 100
SSI_MAX_BW = 1000
SSI_MIN_COEFF = 10
SSI_MAX_COEFF = 100
SSI_MIN_G = 0
SSI_MAX_G = 0


def ssi_additive_noise(
    x: np.ndarray,
    fs: int,
    snr_min: int = SSI_SNRMIN,
    snr_max: int = SSI_SNRMAX,
    n_bands: int = SSI_NBANDS,
    min_f: float = SSI_MIN_F,
    max_f: float = SSI_MAX_F,
    min_bw: float = SSI_MIN_BW,
    max_bw: float = SSI_MAX_BW,
    min_coeff: float = SSI_MIN_COEFF,
    max_coeff: float = SSI_MAX_COEFF,
    min_g: float = SSI_MIN_G,
    max_g: float = SSI_MAX_G,
) -> np.ndarray:
    """Stationary signal-independent noise (SLS algo 3). Same as RawBoost.SSI_additive_noise."""
    noise = np.random.normal(0, 1, x.shape[0]).astype(np.float64)
    b = _gen_notch_coeffs(
        n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs
    )
    noise = _filter_fir(noise, b)
    noise = _norm_wav(noise, True)
    snr = _rand_range(snr_min, snr_max, False)
    noise = (
        noise
        / np.linalg.norm(noise, 2)
        * np.linalg.norm(x, 2)
        / (10.0 ** (0.05 * snr))
    )
    return (x + noise).astype(np.float32)
