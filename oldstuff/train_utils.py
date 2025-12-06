import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Literal, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# For XGBoost models
try:
    import xgboost as xgb
except ImportError:
    xgb = None  # handle gracefully if not installed

# Import your LSTM builder from lstm.py
from lstm import build_lstm_model


# ============================================================
# 1. TRAINING CONFIG
# ============================================================

@dataclass
class TrainConfig:
    n_epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    val_split: float = 0.2     # fraction of data used for validation
    shuffle: bool = True
    verbose: bool = True


# ============================================================
# 2. DATA HELPERS: NUMPY -> DATALOADERS
# ============================================================

def make_torch_loaders_from_numpy(
    X: np.ndarray,
    y: np.ndarray,
    lengths: Optional[np.ndarray],
    cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split (X, y, lengths) into train/val and wrap them as DataLoaders.
    X: (N, T, E)
    y: (N,)
    lengths: (N,) or None (for models that don't use lengths, e.g. some TFT impls)
    """

    N = X.shape[0]
    indices = np.random.permutation(N)
    split = int((1.0 - cfg.val_split) * N)
    train_idx = indices[:split]
    val_idx = indices[split:]

    def _slice(arr, idx):
        return arr[idx] if arr is not None else None

    X_train = _slice(X, train_idx)
    y_train = _slice(y, train_idx)
    lengths_train = _slice(lengths, train_idx)

    X_val = _slice(X, val_idx)
    y_val = _slice(y, val_idx)
    lengths_val = _slice(lengths, val_idx)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    if lengths_train is not None:
        lengths_train_t = torch.tensor(lengths_train, dtype=torch.long)
        lengths_val_t = torch.tensor(lengths_val, dtype=torch.long)

        train_ds = TensorDataset(X_train_t, y_train_t, lengths_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t, lengths_val_t)
    else:
        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


# ============================================================
# 3. GENERIC TORCH SEQUENCE TRAINER (LSTM / TFT)
# ============================================================

def train_torch_sequence_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    lengths: Optional[np.ndarray],
    cfg: Optional[TrainConfig] = None,
    task_type: Literal["regression", "binary"] = "regression",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Generic training loop for sequence models that take either:
      - model(x, lengths)  if lengths is not None
      - model(x)           if lengths is None

    Parameters
    ----------
    model : nn.Module
        Your PyTorch model (LSTM, TFT, etc.).
    X, y, lengths : numpy arrays
    cfg : TrainConfig
    task_type : "regression" or "binary"
    device : torch.device or None

    Returns
    -------
    history : dict
        Contains train/val loss (and accuracy for binary) per epoch.
    """

    if cfg is None:
        cfg = TrainConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_torch_loaders_from_numpy(X, y, lengths, cfg)

    model = model.to(device)

    if task_type == "regression":
        criterion = nn.MSELoss()
    elif task_type == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("task_type must be 'regression' or 'binary'")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    def _forward_batch(X_batch, lengths_batch):
        if lengths_batch is not None:
            return model(X_batch, lengths_batch)
        else:
            return model(X_batch)

    for epoch in range(1, cfg.n_epochs + 1):
        # -----------------------------
        # Train
        # -----------------------------
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_train = 0

        for batch in train_loader:
            if lengths is not None:
                X_batch, y_batch, lengths_batch = batch
                lengths_batch = lengths_batch.to(device)
            else:
                X_batch, y_batch = batch
                lengths_batch = None

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = _forward_batch(X_batch, lengths_batch).squeeze(-1)

            if task_type == "binary":
                loss = criterion(outputs, y_batch)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                running_correct += (preds == y_batch).sum().item()
                total_train += y_batch.numel()
            else:  # regression
                loss = criterion(outputs, y_batch)
                total_train += X_batch.size(0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / total_train
        history["train_loss"].append(train_loss)

        if task_type == "binary":
            train_acc = running_correct / total_train
            history["train_acc"].append(train_acc)
        else:
            history["train_acc"].append(None)

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                if lengths is not None:
                    X_batch, y_batch, lengths_batch = batch
                    lengths_batch = lengths_batch.to(device)
                else:
                    X_batch, y_batch = batch
                    lengths_batch = None

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = _forward_batch(X_batch, lengths_batch).squeeze(-1)
                loss = criterion(outputs, y_batch)
                val_running_loss += loss.item() * X_batch.size(0)

                if task_type == "binary":
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    val_running_correct += (preds == y_batch).sum().item()
                    total_val += y_batch.numel()
                else:
                    total_val += X_batch.size(0)

        val_loss = val_running_loss / total_val
        history["val_loss"].append(val_loss)

        if task_type == "binary":
            val_acc = val_running_correct / total_val
            history["val_acc"].append(val_acc)
        else:
            history["val_acc"].append(None)

        if cfg.verbose:
            if task_type == "binary":
                print(
                    f"Epoch {epoch:02d} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}"
                )
            else:
                print(
                    f"Epoch {epoch:02d} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )

    return {
        "model": model,
        "history": history,
        "config": cfg,
    }


# ============================================================
# 4. LSTM-SPECIFIC CONVENIENCE WRAPPERS
# ============================================================

def train_lstm_regression(
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    hidden_size: int = 128,
    cfg: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Build and train an LSTM regression model (StatFromScratch) using your lstm.py.
    """
    input_size = X.shape[-1]
    model = build_lstm_model(
        input_size=input_size,
        hidden_size=hidden_size,
        binary=False,
    )
    return train_torch_sequence_model(
        model=model,
        X=X,
        y=y,
        lengths=lengths,
        cfg=cfg,
        task_type="regression",
        device=device,
    )


def train_lstm_binary(
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    hidden_size: int = 128,
    cfg: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Build and train an LSTM binary classifier (StatFromScratchBinary) using your lstm.py.
    """
    input_size = X.shape[-1]
    model = build_lstm_model(
        input_size=input_size,
        hidden_size=hidden_size,
        binary=True,
    )
    return train_torch_sequence_model(
        model=model,
        X=X,
        y=y,
        lengths=lengths,
        cfg=cfg,
        task_type="binary",
        device=device,
    )


# ============================================================
# 5. TFT TRAINING WRAPPER (USES SAME CORE TRAINER)
# ============================================================

def train_tft_regression(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    lengths: Optional[np.ndarray],
    cfg: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train a TFT (or any other sequence model) for regression.
    Expects model signature:
        model(x, lengths)  or  model(x)
    """
    return train_torch_sequence_model(
        model=model,
        X=X,
        y=y,
        lengths=lengths,
        cfg=cfg,
        task_type="regression",
        device=device,
    )


def train_tft_binary(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    lengths: Optional[np.ndarray],
    cfg: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train a TFT (or any other sequence model) for binary classification.
    """
    return train_torch_sequence_model(
        model=model,
        X=X,
        y=y,
        lengths=lengths,
        cfg=cfg,
        task_type="binary",
        device=device,
    )


# ============================================================
# 6. XGBOOST TRAINERS
# ============================================================

def train_xgboost_regression(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    xgb_params: Optional[Dict] = None,
) -> Any:
    """
    Train an XGBoost regressor.
    X_* can be numpy arrays or pandas DataFrames.
    """
    if xgb is None:
        raise ImportError("xgboost is not installed. Please `pip install xgboost`.")

    if xgb_params is None:
        xgb_params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
        )

    model = xgb.XGBRegressor(**xgb_params)

    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_train, y_train), (X_val, y_val)]

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )

    return model


def train_xgboost_binary(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    xgb_params: Optional[Dict] = None,
) -> Any:
    """
    Train an XGBoost binary classifier.
    y_* should be 0/1 labels.
    """
    if xgb is None:
        raise ImportError("xgboost is not installed. Please `pip install xgboost`.")

    if xgb_params is None:
        xgb_params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            tree_method="hist",
        )

    model = xgb.XGBClassifier(**xgb_params)

    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_train, y_train), (X_val, y_val)]

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )

    return model