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


@dataclass
class MetricsResult:
    eer: float


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> MetricsResult:
    return MetricsResult(eer=eer_score(y_true, y_score))
