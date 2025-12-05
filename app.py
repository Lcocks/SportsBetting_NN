import os

import streamlit as st
import pandas as pd

from model_manager import (
    make_base_tag,
    build_sequences_for_prop,
    train_or_load_lstm,
    train_or_load_tft,
    train_or_load_xgb,
    N_PAST_GAMES,   # constant defined in model_manager.py
)

from player_utils import predict_player_over_prob


# ============================================================
# DATA SOURCES (MATCHES YOUR FILENAME SCHEME)
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


@st.cache_data
def load_data(yard_type: str):
    """
    Load train/test/full dataframes for a given yard type,
    based on the filename pattern:
        <prefix>_2019_2023.csv
        <prefix>_24tocurrent.csv
    """
    cfg = DATA_CONFIG[yard_type]
    prefix = cfg["prefix"]

    train_path = os.path.join(DATA_DIR, f"{prefix}_2019_2023.csv")
    test_path  = os.path.join(DATA_DIR, f"{prefix}_24tocurrent.csv")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    full_df  = pd.concat([train_df, test_df], ignore_index=True)

    return train_df, test_df, full_df


# ============================================================
# HELPERS FOR MULTI-PROP PARLAYS
# ============================================================

def get_model_for_prop(
    yard_type: str,
    stat_col: str,
    line_value: float,
    model_family: str,       # "LSTM" | "TFT" | "XGBoost"
    train_df,
    test_df,
):
    """
    Train or load the chosen family (LSTM/TFT/XGB) for a given
    (yard_type, stat_col, line_value) prop.
    """
    base_tag = make_base_tag(yard_type, stat_col, line_value)

    X_train, y_train, lengths_train, X_test, y_test, lengths_test = build_sequences_for_prop(
        train_df, test_df, stat_col, line_value
    )

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
    else:  # "XGBoost"
        model, metrics = train_or_load_xgb(
            base_tag,
            X_train, y_train, lengths_train,
            X_test,  y_test,  lengths_test,
            stat_col, line_value,
        )

    return model, metrics


def compute_parlay_prob(
    parlay_legs,
    yard_type: str,
    parlay_model_choice: str,  # "LSTM" | "TFT" | "XGBoost"
    train_df,
    test_df,
    full_df,
):
    """
    parlay_legs: list of dicts with keys:
        - player
        - stat_col
        - line_value

    All legs share the same yard_type, but can have different stat_col + line.
    """
    # Cache models per (stat_col, line_value) so we don't retrain or reload twice
    model_cache = {}
    leg_probs = []

    for leg in parlay_legs:
        player     = leg["player"]
        stat_col   = leg["stat_col"]
        line_value = leg["line_value"]

        prop_key = (stat_col, line_value)

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
    for _, p_leg in leg_probs:
        parlay_prob *= p_leg

    return parlay_prob, leg_probs


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🏈 Player Prop Model Comparator + Parlay Builder")

st.markdown(
    """
This app:

- Trains or loads **three models** (LSTM, TFT-style Transformer, XGBoost)  
  for **each unique prop**: `(yard type, stat, line)`
- Gives **single-leg** probabilities from all three models
- Lets you build **multi-prop parlays**, training/loading models per prop and
  multiplying leg probabilities to get `P(parlay hits)`.
"""
)

# ---------------- YARD TYPE ----------------
yard_type = st.selectbox("Yard type (dataset)", list(DATA_CONFIG.keys()))
train_df, test_df, full_df = load_data(yard_type)
default_stat = DATA_CONFIG[yard_type]["default_stat_col"]

# ---------------- AVAILABLE NUMERIC STATS ----------------
numeric_cols = full_df.select_dtypes(include=["number"]).columns.tolist()
if default_stat in numeric_cols:
    default_idx = numeric_cols.index(default_stat)
else:
    default_idx = 0

all_players = sorted(full_df["display_name"].unique())
default_player = "George Kittle" if "George Kittle" in all_players else all_players[0]

# ============================================================
# SINGLE-LEG CONFIG
# ============================================================

st.markdown("## 🔹 Single-Leg Config (for model comparison)")

single_player = st.selectbox(
    "Single-leg player (for comparing models)",
    all_players,
    index=all_players.index(default_player),
)

single_stat_col = st.selectbox(
    "Single-leg stat column",
    numeric_cols,
    index=default_idx,
)

single_line_value = st.number_input(
    "Single-leg line (threshold)",
    min_value=0.0,
    max_value=500.0,
    value=37.5,
    step=0.5,
)

# ============================================================
# PARLAY CONFIG (TWO LEGS, MULTI-PROP)
# ============================================================

st.markdown("## 🎯 Multi-Leg Parlay Config (2 legs, can be different props)")

parlay_model_choice = st.selectbox(
    "Model family to use for parlay probabilities",
    ["LSTM", "TFT", "XGBoost"],
)

st.markdown("### Leg 1")
parlay_player_1 = st.selectbox(
    "Player (Leg 1)",
    all_players,
    key="parlay_p1",
)
parlay_stat_1 = st.selectbox(
    "Stat (Leg 1)",
    numeric_cols,
    key="parlay_s1",
)
parlay_line_1 = st.number_input(
    "Line (Leg 1)",
    min_value=0.0,
    max_value=500.0,
    value=single_line_value,
    step=0.5,
    key="parlay_l1",
)

st.markdown("### Leg 2")
parlay_player_2 = st.selectbox(
    "Player (Leg 2)",
    all_players,
    key="parlay_p2",
)
parlay_stat_2 = st.selectbox(
    "Stat (Leg 2)",
    numeric_cols,
    key="parlay_s2",
)
parlay_line_2 = st.number_input(
    "Line (Leg 2)",
    min_value=0.0,
    max_value=500.0,
    value=single_line_value,
    step=0.5,
    key="parlay_l2",
)

st.markdown(
    f"""
**Current selection**

- Dataset / yard type: `{yard_type}`  
- Single-leg: `{single_player}` – `{single_stat_col} > {single_line_value}`  
- Parlay model: `{parlay_model_choice}`  
- Parlay Leg 1: `{parlay_player_1}` – `{parlay_stat_1} > {parlay_line_1}`  
- Parlay Leg 2: `{parlay_player_2}` – `{parlay_stat_2} > {parlay_line_2}`  
"""
)

# ============================================================
# MAIN ACTION
# ============================================================

if st.button("Train / Load Models and Compute"):
    with st.spinner("Building sequences and training/loading models..."):
        try:
            # ------------------------------------------------------------
            # 1) SINGLE-LEG: train/load all 3 model families for that prop
            # ------------------------------------------------------------
            # single prop tag
            single_base_tag = make_base_tag(yard_type, single_stat_col, single_line_value)

            # shared sequences for this single-leg prop
            X_train_s, y_train_s, lengths_train_s, X_test_s, y_test_s, lengths_test_s = build_sequences_for_prop(
                train_df, test_df, single_stat_col, single_line_value
            )

            # LSTM
            lstm_model_s, lstm_metrics_s = train_or_load_lstm(
                single_base_tag,
                X_train_s, y_train_s, lengths_train_s,
                X_test_s, y_test_s, lengths_test_s,
                single_stat_col, single_line_value,
            )

            # TFT
            tft_model_s, tft_metrics_s = train_or_load_tft(
                single_base_tag,
                X_train_s, y_train_s, lengths_train_s,
                X_test_s, y_test_s, lengths_test_s,
                single_stat_col, single_line_value,
            )

            # XGBoost
            xgb_model_s, xgb_metrics_s = train_or_load_xgb(
                single_base_tag,
                X_train_s, y_train_s, lengths_train_s,
                X_test_s, y_test_s, lengths_test_s,
                single_stat_col, single_line_value,
            )

            # single-leg probabilities for that prop and player
            prob_lstm = predict_player_over_prob(
                model=lstm_model_s,
                df=full_df,
                player_name=single_player,
                stat_col=single_stat_col,
                line_value=single_line_value,
                n_past_games=N_PAST_GAMES,
                model_type="lstm",
                device="cpu",
            )

            prob_tft = predict_player_over_prob(
                model=tft_model_s,
                df=full_df,
                player_name=single_player,
                stat_col=single_stat_col,
                line_value=single_line_value,
                n_past_games=N_PAST_GAMES,
                model_type="tft",
                device="cpu",
            )

            prob_xgb = predict_player_over_prob(
                model=xgb_model_s,
                df=full_df,
                player_name=single_player,
                stat_col=single_stat_col,
                line_value=single_line_value,
                n_past_games=N_PAST_GAMES,
                model_type="xgboost",
            )

            # ------------------------------------------------------------
            # 2) PARLAY: multi-prop, using chosen model family
            # ------------------------------------------------------------
            parlay_legs = [
                {
                    "player": parlay_player_1,
                    "stat_col": parlay_stat_1,
                    "line_value": parlay_line_1,
                },
                {
                    "player": parlay_player_2,
                    "stat_col": parlay_stat_2,
                    "line_value": parlay_line_2,
                },
            ]

            parlay_prob, leg_probs = compute_parlay_prob(
                parlay_legs=parlay_legs,
                yard_type=yard_type,
                parlay_model_choice=parlay_model_choice,
                train_df=train_df,
                test_df=test_df,
                full_df=full_df,
            )

        except Exception as e:
            st.error(f"Error during training/loading or prediction:\n\n{e}")
        else:
            st.success("Done! Models are trained/loaded and probabilities computed.")

            # ---------------- SINGLE-LEG PROBABILITIES ----------------
            st.subheader(f"🔮 Single-Leg Probabilities for {single_player}")

            st.markdown(
                f"""
- **LSTM**:    P({single_stat_col} > {single_line_value}) = **{prob_lstm:.3f}**  
- **TFT**:     P({single_stat_col} > {single_line_value}) = **{prob_tft:.3f}**  
- **XGBoost**: P({single_stat_col} > {single_line_value}) = **{prob_xgb:.3f}**  
"""
            )

            # ---------------- SINGLE-LEG TEST METRICS ----------------
            st.subheader("📊 Test Metrics for Single-Leg Prop (2019–2023 train → 2024+ test)")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### LSTM")
                st.json(lstm_metrics_s)

            with col2:
                st.markdown("### TFT")
                st.json(tft_metrics_s)

            with col3:
                st.markdown("### XGBoost")
                st.json(xgb_metrics_s)

            # ---------------- MULTI-PROP PARLAY RESULT ----------------
            st.subheader("🎯 Multi-Prop Parlay Result")

            st.markdown("#### Leg probabilities")
            for leg, p_leg in leg_probs:
                st.write(
                    f"- **{leg['player']}**: P({leg['stat_col']} > {leg['line_value']}) "
                    f"= `{p_leg:.3f}`"
                )

            st.markdown("#### Parlay probability")
            st.write(
                f"Using **{parlay_model_choice}**, the probability that "
                f"**ALL** legs hit is:\n\n"
                f"👉 **P(parlay hits) = {parlay_prob:.3f}**"
            )