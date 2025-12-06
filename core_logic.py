# core_logic.py
import os
from typing import List, Dict, Tuple

import pandas as pd

from oldstuff.model_manager import (
    make_base_tag,
    build_sequences_for_prop,
    train_or_load_lstm,
    train_or_load_tft,
    train_or_load_xgb,
    N_PAST_GAMES,
)
from player_utils import predict_player_over_prob


# ============================================================
# DATA SOURCES (pure, no Streamlit)
# ============================================================

DATA_DIR = "data"  # folder where your *_2019_2023.csv etc. live

DATA_CONFIG = {
    "Receiving": {
        "prefix": "receiving",
        "default_stat_col": "YDS",
    },
    "Rushing": {
        "prefix": "rushing",
        "default_stat_col": "YDS",
    },
    "Passing": {
        "prefix": "passing",
        "default_stat_col": "YDS",
    },
    "Defensive": {
        "prefix": "defensive",
        "default_stat_col": "YDS",
    },
    "Fumbles": {
        "prefix": "fumbles",
        "default_stat_col": "YDS",
    },
    "Interceptions": {
        "prefix": "interceptions",
        "default_stat_col": "YDS",
    },
    "Kicking": {
        "prefix": "kicking",
        "default_stat_col": "YDS",
    },
    "Kick Returns": {
        "prefix": "kickreturns",
        "default_stat_col": "YDS",
    },
    "Punting": {
        "prefix": "punting",
        "default_stat_col": "YDS",
    },
    "Punt Returns": {
        "prefix": "puntreturn",
        "default_stat_col": "YDS",
    },
}


def load_data(
    yard_type: str,
    data_dir: str = DATA_DIR,
    data_config: Dict[str, Dict] = DATA_CONFIG,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pure data loader for functional testing.

    Returns:
        train_df, test_df, full_df
    """
    cfg = data_config[yard_type]
    prefix = cfg["prefix"]

    train_path = os.path.join(data_dir, f"{prefix}_2019_2023.csv")
    test_path = os.path.join(data_dir, f"{prefix}_24tocurrent.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    return train_df, test_df, full_df


def get_model_for_prop(
    yard_type: str,
    stat_col: str,
    line_value: float,
    model_family: str,       # "LSTM" | "TFT" | "XGBoost"
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Core logic for training/loading a model for a single player prop.
    This is UI-agnostic and easy to unit test.
    """
    base_tag = make_base_tag(yard_type, stat_col, line_value)

    X_train, y_train, lengths_train, X_test, y_test, lengths_test = build_sequences_for_prop(
        train_df, test_df, stat_col, line_value
    )

    model_family = model_family.upper()
    if model_family == "LSTM":
        model, metrics = train_or_load_lstm(
            base_tag,
            X_train, y_train, lengths_train,
            X_test,  y_test,  lengths_test,
            stat_col, line_value,
        )
    elif model_family == "TFT":
        model, metrics = train_or_load_tft(
            base_tag,
            X_train, y_train, lengths_train,
            X_test,  y_test,  lengths_test,
            stat_col, line_value,
        )
    elif model_family == "XGBOOST":
        model, metrics = train_or_load_xgb(
            base_tag,
            X_train, y_train, lengths_train,
            X_test,  y_test,  lengths_test,
            stat_col, line_value,
        )
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    return model, metrics


def compute_parlay_prob(
    parlay_legs: List[Dict],
    yard_type: str,
    parlay_model_choice: str,  # "LSTM" | "TFT" | "XGBoost"
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> Tuple[float, List[Tuple[Dict, float]]]:
    """
    Core logic for multi-leg parlay probability.

    parlay_legs: list of dicts with keys:
        - player
        - stat_col
        - line_value
    """
    model_cache: Dict[Tuple[str, float], object] = {}
    leg_probs: List[Tuple[Dict, float]] = []

    for leg in parlay_legs:
        player = leg["player"]
        stat_col = leg["stat_col"]
        line_value = leg["line_value"]

        prop_key = (stat_col, line_value)

        # train/load once per (stat, line)
        if prop_key not in model_cache:
            model, _ = get_model_for_prop(
                yard_type=yard_type,
                stat_col=stat_col,
                line_value=line_value,
                model_family=parlay_model_choice,
                train_df=train_df,
                test_df=test_df,
            )
            model_cache[prop_key] = model
        else:
            model = model_cache[prop_key]

        model_type_str = parlay_model_choice.lower()  # "lstm" / "tft" / "xgboost"

        p_leg = predict_player_over_prob(
            model=model,
            df=full_df,
            player_name=player,
            stat_col=stat_col,
            line_value=line_value,
            n_past_games=N_PAST_GAMES,
            model_type=model_type_str,
            device="cpu",
        )

        leg_probs.append((leg, p_leg))

    parlay_prob = 1.0
    for _, p in leg_probs:
        parlay_prob *= p

    return parlay_prob, leg_probs