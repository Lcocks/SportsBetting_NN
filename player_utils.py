import torch
import numpy as np
from typing import Literal, Union

from data_prep import add_over_under_label, prepare_receiving_sequences

ModelType = Literal["lstm", "tft", "xgboost"]


def build_player_sequence(
    df,
    player_name: str,
    stat_col: str,
    line_value: float,
    n_past_games: int,
):
    """
    Filter df to a single player, add over/under label, and build
    (X_p, y_p, lengths_p, meta_p) just like during training.
    """
    # Filter to this player & sort by date
    player_df = df[df["display_name"] == player_name].copy()
    if player_df.empty:
        raise ValueError(f"No rows found for player_name='{player_name}'")

    player_df = player_df.sort_values("date")

    # Add over/under label (consistent with training pipeline)
    player_df = add_over_under_label(
        df=player_df,
        stat_col=stat_col,
        line_value=line_value,
        new_col="over_label",
    )

    # Build sequences
    X_p, y_p, lengths_p, meta_p = prepare_receiving_sequences(
        player_df,
        n_past_games=n_past_games,
        target_col="over_label",
    )

    if len(X_p) == 0:
        raise ValueError(
            f"Not enough games for {player_name} to build a sequence of length {n_past_games}"
        )

    return X_p, y_p, lengths_p, meta_p


def predict_player_over_prob(
    model,
    df,
    player_name: str,
    stat_col: str,
    line_value: float,
    n_past_games: int,
    model_type: ModelType = "lstm",         # <--- NEW
    device: Union[str, torch.device] = "auto",
) -> float:
    """
    Generic player prop predictor that works for:
      - model_type="lstm"   : dual-head LSTM (PyTorch)
      - model_type="tft"    : dual-head TFT (PyTorch)
      - model_type="xgboost": XGBoost / sklearn-like model with predict_proba

    Returns:
      prob_over : float in [0, 1]
    """

    # 1) Build player's sequences
    X_p, y_p, lengths_p, meta_p = build_player_sequence(
        df=df,
        player_name=player_name,
        stat_col=stat_col,
        line_value=line_value,
        n_past_games=n_past_games,
    )

    # Most recent sequence
    X_last_np = X_p[-1:]           # shape (1, T, E)
    lengths_last_np = lengths_p[-1:]  # shape (1,)

    model_type = model_type.lower()

    # 2) PyTorch sequence models: LSTM & TFT
    if model_type in ("lstm", "tft"):
        # Resolve device
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        X_last = torch.tensor(X_last_np, dtype=torch.float32).to(device)
        len_last = torch.tensor(lengths_last_np, dtype=torch.long).to(device)

        model = model.to(device)
        model.eval()

        with torch.no_grad():
            out = model(X_last, len_last)

            # both DualHeadStatModel and DualHeadTFTModel return (y_reg, logits)
            y_reg, logits = out
            prob_over = torch.sigmoid(logits).item()

        return prob_over

    # 3) XGBoost / sklearn-style models
    elif model_type == "xgboost":
        # Flatten (1, T, E) -> (1, T*E)
        X_flat = X_last_np.reshape(1, -1)

        # Assumes model has predict_proba and class 1 = "over"
        proba = model.predict_proba(X_flat)
        prob_over = float(proba[0, 1])
        return prob_over

    else:
        raise ValueError(f"Unknown model_type='{model_type}'. Use 'lstm', 'tft', or 'xgboost'.")