# xgb_model.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    accuracy_score,
)

@dataclass
class XGBPropConfig:
    """
    Config for a binary prop model with XGBoost.

    This is the 'untrained model' equivalent: it specifies how
    the trees will look once you call xgb.train.
    """
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    seed: int = 42
    scale_pos_weight: float = 1.0

    # training-related
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50


def build_xgb_params(cfg: XGBPropConfig) -> Dict:

    return {
        "objective": "binary:logistic",
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "eval_metric": ["logloss", "auc"],
        "seed": cfg.seed,
        "scale_pos_weight": cfg.scale_pos_weight,
    }


def calculate_ece(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray]:

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    # columns: count, avg_conf, avg_acc
    bin_table = np.zeros((n_bins, 3), dtype=float)

    n = len(y_true)
    for i in range(n_bins):
        mask = bin_indices == i
        count = mask.sum()
        if count == 0:
            continue

        conf = float(y_pred_proba[mask].mean())
        acc = float(y_true[mask].mean())

        ece += (count / n) * abs(conf - acc)
        bin_table[i, 0] = count
        bin_table[i, 1] = conf
        bin_table[i, 2] = acc

    return float(ece), bin_table


def calculate_pce(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> float:

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    mask_pos = y_pred_proba >= 0.5
    if mask_pos.sum() == 0:
        return 0.0

    y_pos = y_true[mask_pos]
    p_pos = y_pred_proba[mask_pos]

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(p_pos, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    pce = 0.0
    total = len(y_pos)

    for i in range(n_bins):
        mask = bin_indices == i
        count = mask.sum()
        if count == 0:
            continue

        conf = float(p_pos[mask].mean())
        acc = float(y_pos[mask].mean())
        pce += (count / total) * abs(conf - acc)

    return float(pce)


def evaluate_xgb_binary(
    booster: xgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10,
) -> Dict:

    y_test = np.asarray(y_test)
    dtest = xgb.DMatrix(X_test)

    y_pred_proba = booster.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        "log_loss": float(log_loss(y_test, y_pred_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }

    ece, _ = calculate_ece(y_test, y_pred_proba, n_bins=n_bins)
    pce = calculate_pce(y_test, y_pred_proba, n_bins=n_bins)

    metrics["ece"] = float(ece)
    metrics["pce"] = float(pce)
    metrics["y_pred_proba"] = y_pred_proba  # keep for plots if needed

    return metrics