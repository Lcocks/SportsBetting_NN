import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib

from sklearn.metrics import roc_auc_score

from lstm import build_lstm_model, train_dual_head_classifier, TrainConfig as LSTMTrainConfig
from tft import build_tft_model, train_tft_classifier, TrainConfig as TFTTrainConfig
from xgb import XGBTrainConfig, train_xgb_classifier, flatten_sequences

from data_prep import add_over_under_label, prepare_receiving_sequences
from metrics import compute_ece, compute_pace
from player_utils import predict_player_over_prob

N_PAST_GAMES = 5
LSTM_HIDDEN  = 128
TFT_D_MODEL  = 128

os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

def make_base_tag(yard_type: str, stat_col: str, line_value: float) -> str:
    # e.g. "receiving_yds_line_37.5_past5"
    return f"{yard_type.lower()}_{stat_col.lower()}_line_{line_value:.1f}_past{N_PAST_GAMES}"


def lstm_paths(base_tag: str):
    return (
        os.path.join("models",  f"lstm_{base_tag}.pt"),
        os.path.join("metrics", f"lstm_{base_tag}_metrics.json"),
    )


def tft_paths(base_tag: str):
    return (
        os.path.join("models",  f"tft_{base_tag}.pt"),
        os.path.join("metrics", f"tft_{base_tag}_metrics.json"),
    )


def xgb_paths(base_tag: str):
    return (
        os.path.join("models",  f"xgb_{base_tag}.pkl"),
        os.path.join("metrics", f"xgb_{base_tag}_metrics.json"),
    )

def build_sequences_for_prop(train_df, test_df, stat_col: str, line_value: float):
    # Copy so we don't mutate original dfs in cache
    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_df = add_over_under_label(train_df, stat_col, line_value=line_value, new_col="over_label")
    test_df  = add_over_under_label(test_df,  stat_col, line_value=line_value, new_col="over_label")

    X_train, y_train, lengths_train, _ = prepare_receiving_sequences(
        train_df, n_past_games=N_PAST_GAMES, target_col="over_label"
    )
    X_test, y_test, lengths_test, _ = prepare_receiving_sequences(
        test_df, n_past_games=N_PAST_GAMES, target_col="over_label"
    )

    return X_train, y_train, lengths_train, X_test, y_test, lengths_test



def train_or_load_lstm(
    base_tag: str,
    X_train,
    y_train,
    lengths_train,
    X_test,
    y_test,
    lengths_test,
    stat_col: str,
    line_value: float,
):
    model_path, metrics_path = lstm_paths(base_tag)

    # If both model + metrics exist, load and return
    if os.path.exists(model_path) and os.path.exists(metrics_path):
        state = torch.load(model_path, map_location="cpu")
        # infer input_size from X_train
        input_size = X_train.shape[-1]
        model = build_lstm_model(input_size=input_size, hidden_size=LSTM_HIDDEN)
        model.load_state_dict(state)
        model.eval()

        with open(metrics_path, "r") as f:
            metrics_payload = json.load(f)
        return model, metrics_payload["test_metrics"]

    # Otherwise, train
    cfg = LSTMTrainConfig(
        n_epochs=10,
        batch_size=64,
        lr=1e-3,
        device="auto",
        verbose=True,
    )

    train_result = train_dual_head_classifier(
        X=X_train,
        y=y_train,
        lengths=lengths_train,
        hidden_size=LSTM_HIDDEN,
        cfg=cfg,
    )
    model = train_result["model"]
    history = train_result["history"]

    # Evaluate on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    len_test_t = torch.tensor(lengths_test, dtype=torch.long).to(device)

    with torch.no_grad():
        _, logits_test = model(X_test_t, len_test_t)
        probs_test = torch.sigmoid(logits_test).cpu().numpy()

    y_true_test = np.asarray(y_test)

    auc   = roc_auc_score(y_true_test, probs_test)
    ece   = compute_ece(y_true_test, probs_test)
    pace2 = compute_pace(y_true_test, probs_test, L=2)

    test_metrics = {
        "auc": float(auc),
        "ece": float(ece),
        "pace2": float(pace2),
        "n_test": int(len(y_true_test)),
    }

    # Save model to CPU
    model_cpu = model.to("cpu")
    torch.save(model_cpu.state_dict(), model_path)

    # Save metrics
    metrics_payload = {
        "timestamp": datetime.now().isoformat(),
        "stat_col": stat_col,
        "line_value": line_value,
        "n_past_games": N_PAST_GAMES,
        "hidden_size": LSTM_HIDDEN,
        "train_cfg": {
            "n_epochs": cfg.n_epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "device": cfg.device,
        },
        "train_history": history,
        "test_metrics": test_metrics,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    return model_cpu, test_metrics


def train_or_load_tft(
    base_tag: str,
    X_train,
    y_train,
    lengths_train,
    X_test,
    y_test,
    lengths_test,
    stat_col: str,
    line_value: float,
):
    model_path, metrics_path = tft_paths(base_tag)

    if os.path.exists(model_path) and os.path.exists(metrics_path):
        state = torch.load(model_path, map_location="cpu")
        input_size = X_train.shape[-1]
        model = build_tft_model(
            input_size=input_size,
            d_model=TFT_D_MODEL,
            n_heads=4,
            num_layers=2,
            dropout=0.1,
        )
        model.load_state_dict(state)
        model.eval()

        with open(metrics_path, "r") as f:
            metrics_payload = json.load(f)
        return model, metrics_payload["test_metrics"]

    cfg = TFTTrainConfig(
        n_epochs=10,
        batch_size=64,
        lr=1e-3,
        device="auto",
        verbose=True,
    )

    train_result = train_tft_classifier(
        X=X_train,
        y=y_train,
        lengths=lengths_train,
        d_model=TFT_D_MODEL,
        n_heads=4,
        num_layers=2,
        dropout=0.1,
        cfg=cfg,
    )

    model = train_result["model"]
    history = train_result["history"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    len_test_t = torch.tensor(lengths_test, dtype=torch.long).to(device)

    with torch.no_grad():
        _, logits_test = model(X_test_t, len_test_t)
        probs_test = torch.sigmoid(logits_test).cpu().numpy()

    y_true_test = np.asarray(y_test)

    auc   = roc_auc_score(y_true_test, probs_test)
    ece   = compute_ece(y_true_test, probs_test)
    pace2 = compute_pace(y_true_test, probs_test, L=2)

    test_metrics = {
        "auc": float(auc),
        "ece": float(ece),
        "pace2": float(pace2),
        "n_test": int(len(y_true_test)),
    }

    model_cpu = model.to("cpu")
    torch.save(model_cpu.state_dict(), model_path)

    metrics_payload = {
        "timestamp": datetime.now().isoformat(),
        "stat_col": stat_col,
        "line_value": line_value,
        "n_past_games": N_PAST_GAMES,
        "d_model": TFT_D_MODEL,
        "train_cfg": {
            "n_epochs": cfg.n_epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "device": cfg.device,
        },
        "train_history": history,
        "test_metrics": test_metrics,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    return model_cpu, test_metrics


# ============================================================
# TRAIN OR LOAD: XGBOOST
# ============================================================

def train_or_load_xgb(
    base_tag: str,
    X_train,
    y_train,
    lengths_train,
    X_test,
    y_test,
    lengths_test,
    stat_col: str,
    line_value: float,
):
    model_path, metrics_path = xgb_paths(base_tag)

    if os.path.exists(model_path) and os.path.exists(metrics_path):
        model = joblib.load(model_path)
        with open(metrics_path, "r") as f:
            metrics_payload = json.load(f)
        return model, metrics_payload["test_metrics"]

    xgb_cfg = XGBTrainConfig(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        verbose=False,
    )

    xgb_result = train_xgb_classifier(
        X=X_train,
        y=y_train,
        lengths=lengths_train,
        cfg=xgb_cfg,
    )
    model = xgb_result["model"]
    history = xgb_result["history"]

    X_test_flat = flatten_sequences(X_test)
    y_true_test = np.asarray(y_test)
    probs_test = model.predict_proba(X_test_flat)[:, 1]

    auc   = roc_auc_score(y_true_test, probs_test)
    ece   = compute_ece(y_true_test, probs_test)
    pace2 = compute_pace(y_true_test, probs_test, L=2)

    test_metrics = {
        "auc": float(auc),
        "ece": float(ece),
        "pace2": float(pace2),
        "n_test": int(len(y_true_test)),
    }

    joblib.dump(model, model_path)

    metrics_payload = {
        "timestamp": datetime.now().isoformat(),
        "stat_col": stat_col,
        "line_value": line_value,
        "n_past_games": N_PAST_GAMES,
        "train_cfg": xgb_cfg.__dict__,
        "train_history": history,
        "test_metrics": test_metrics,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    return model, test_metrics