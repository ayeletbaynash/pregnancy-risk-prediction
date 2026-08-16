"""Evaluation metrics emphasizing reliable low-risk identification."""
import numpy as np
import pandas as pd

def bottom_percentile_risk(test_df, percentile=40):
    labeled = test_df[test_df["unified_risk"].notna()].copy()
    if labeled.empty:
        return float("nan")
    if percentile <= 0 or percentile > 100:
        raise ValueError("percentile must be in the range (0, 100].")

    labeled = labeled.sort_values("predicted_prob", ascending=True)
    y_true = labeled["unified_risk"].astype(float).to_numpy()
    k = int(np.ceil(len(y_true) * percentile / 100.0))
    k = min(max(k, 1), len(y_true))
    return float(np.sum(y_true[:k]) / k)


def bottom_percentile_positive_capture_rate(test_df, percentile=40):
    labeled = test_df[test_df["unified_risk"].notna()].copy()
    if labeled.empty:
        return float("nan")
    if percentile <= 0 or percentile > 100:
        raise ValueError("percentile must be in the range (0, 100].")

    labeled = labeled.sort_values("predicted_prob", ascending=True)
    y_true = labeled["unified_risk"].astype(float).to_numpy()
    total_positive = np.sum(y_true == 1.0)
    if total_positive == 0:
        return float("nan")
    k = int(np.ceil(len(y_true) * percentile / 100.0))
    k = min(max(k, 1), len(y_true))
    bottom_positive = np.sum(y_true[:k] == 1.0)
    return float(bottom_positive / total_positive)


def average_auc(all_metrics):
    aucs = [values.get("auc") for values in all_metrics.values() if values.get("auc") is not None]
    return float(np.mean(aucs)) if aucs else float("nan")


def average_bottom_40_risk(all_metrics):
    risks = [
        values.get("bottom_40_risk")
        for values in all_metrics.values()
        if values.get("bottom_40_risk") is not None and np.isfinite(values.get("bottom_40_risk"))
    ]
    return float(np.mean(risks)) if risks else float("nan")


def average_bottom_40_positive_capture_rate(all_metrics):
    capture_rates = [
        values.get("bottom_40_positive_capture_rate")
        for values in all_metrics.values()
        if values.get("bottom_40_positive_capture_rate") is not None
        and np.isfinite(values.get("bottom_40_positive_capture_rate"))
    ]
    return float(np.mean(capture_rates)) if capture_rates else float("nan")
