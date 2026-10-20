"""Task-independent temporal splitting and classifier metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score


def temporal_split(frame: pd.DataFrame, train_end: str, validation_month: str, test_month: str):
    if not train_end < validation_month < test_month:
        raise ValueError("Require train_end < validation_month < test_month.")
    for value in (train_end, validation_month, test_month):
        if not pd.Series([value]).str.fullmatch(r"\d{4}(0[1-9]|1[0-2])").all():
            raise ValueError("Months must use YYYYMM.")
    return (frame.loc[frame.month.le(train_end)].copy(), frame.loc[frame.month.eq(validation_month)].copy(),
            frame.loc[frame.month.eq(test_month)].copy())


def metrics(y, probabilities, k: int = 20) -> dict:
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if not len(y):
        return {"rows": 0, "reason": "No independently reviewed labels available."}
    predicted = probabilities >= 0.5
    indices = np.argsort(-probabilities, kind="stable")[:min(k, len(y))]
    both = len(np.unique(y)) == 2
    return {"rows": len(y), "positives": int(y.sum()),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "pr_auc_average_precision": float(average_precision_score(y, probabilities)) if both else None,
            "roc_auc": float(roc_auc_score(y, probabilities)) if both else None,
            "precision_at_k": float(y[indices].mean()), "k": len(indices),
            "threshold": 0.5, "single_class": not both}
