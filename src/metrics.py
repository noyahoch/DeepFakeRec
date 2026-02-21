from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_curve


def eer_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER) from labels and scores.
    y_true: {0,1} labels, y_score: probability or score for class 1.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)


def eer_score_interp(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    d = fpr - fnr

    # Find segment where sign changes (crossing)
    i = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if len(i) == 0:
        # Fallback to closest point if no crossing due to discreteness
        j = np.argmin(np.abs(d))
        return float((fpr[j] + fnr[j]) / 2.0)

    i = i[0]
    # Linear interpolation between i and i+1
    x0, x1 = d[i], d[i + 1]
    w = x0 / (x0 - x1)  # in [0,1] if it crosses
    eer = fpr[i] + w * (fpr[i + 1] - fpr[i])
    return float(eer)


@dataclass
class MetricsResult:
    eer: float


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    interp: bool = True,
) -> MetricsResult:
    """interp=False uses a faster, non-interpolated EER (e.g. for train logging)."""
    eer = eer_score_interp(y_true, y_score) if interp else eer_score(y_true, y_score)
    return MetricsResult(eer=eer)
