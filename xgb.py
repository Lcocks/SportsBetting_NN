import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import xgboost as xgb


# ============================================================
# 1. TRAIN CONFIG
# ============================================================

@dataclass
class XGBTrainConfig:
    """
    Configuration for training an XGBoost binary classifier
    on flattened (N, T, E) sequence data.

    This is roughly analogous to TrainConfig for LSTM/TFT, but
    uses tree/boosting hyperparameters instead of epochs/batch size.
    """
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0

    eval_metric: str = "logloss"  # can change to "auc" if you want
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = False  # XGBoost training verbosity

    # If you later want early stopping, you can add:
    # early_stopping_rounds: int | None = None


# ============================================================
# 2. FLATTEN SEQUENCE DATA
# ============================================================

def flatten_sequences(X: Any) -> np.ndarray:
    """
    Flatten sequence data from shape (N, T, E) to (N, T*E)
    for use with XGBoost / sklearn models.

    Accepts either numpy arrays or torch tensors.
    """
    if "torch" in str(type(X)):
        # torch.Tensor → numpy
        X = X.detach().cpu().numpy()

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, E), got shape {X.shape}")

    N, T, E = X.shape
    return X.reshape(N, T * E)


# ============================================================
# 3. MODEL FACTORY
# ============================================================

def build_xgb_model(cfg: XGBTrainConfig) -> xgb.XGBClassifier:
    """
    Build an unfitted XGBClassifier using the training config.

    Note: input_size (T*E) is not needed by XGBClassifier, so we
    don't require it as an argument like in the LSTM/TFT factories.
    """
    model = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=cfg.reg_alpha,
        objective="binary:logistic",
        eval_metric=cfg.eval_metric,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
        use_label_encoder=False,
    )
    return model


# ============================================================
# 4. TRAINING HELPER: XGBoost CLASSIFIER
# ============================================================

def train_xgb_classifier(
    X: Any,
    y: Any,
    lengths: Any,    # kept for API symmetry with LSTM/TFT, but not used
    cfg: XGBTrainConfig = XGBTrainConfig(),
) -> Dict[str, Any]:
    """
    Train an XGBoost binary classifier on flattened sequence data.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Input sequences, shape (N, T, E).
    y : np.ndarray or torch.Tensor
        Binary labels (0/1), shape (N,).
    lengths : np.ndarray or torch.Tensor
        Sequence lengths, shape (N,). Not used here, but kept
        for consistency with other model training functions.
    cfg : XGBTrainConfig
        Hyperparameters and training config for XGBoost.

    Returns
    -------
    result : dict
        {
          "model": fitted XGBClassifier,
          "history": List[dict] with per-iteration training metric
        }
    """
    # Convert inputs
    X_flat = flatten_sequences(X)  # (N, T*E)

    if "torch" in str(type(y)):
        y_np = y.detach().cpu().numpy().astype(np.float32)
    else:
        y_np = np.asarray(y, dtype=np.float32)

    # Build model
    model = build_xgb_model(cfg)

    # XGBoost can log eval metrics each boosting round via eval_set
    eval_set = [(X_flat, y_np)]
    model.fit(
        X_flat,
        y_np,
        eval_set=eval_set,
        verbose=cfg.verbose,
    )

    # Extract eval history (e.g., logloss per iteration)
    evals_result = model.evals_result()
    # evals_result is a dict like: {"validation_0": {"logloss": [...]}}

    history: List[Dict[str, float]] = []
    if evals_result and "validation_0" in evals_result:
        metric_name, values = next(iter(evals_result["validation_0"].items()))
        for i, v in enumerate(values):
            history.append({"iteration": i + 1, metric_name: float(v)})

    return {
        "model": model,
        "history": history,
    }