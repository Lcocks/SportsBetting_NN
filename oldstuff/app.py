# app.py
import os

import streamlit as st
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

# ✅ import the testable logic
from core_logic import (
    DATA_CONFIG,
    load_data as load_data_core,
    get_model_for_prop,
    compute_parlay_prob,
)

# ============================================================
# STREAMLIT-WRAPPED DATA LOADER (adds caching)
# ============================================================

@st.cache_data
def load_data(yard_type: str):
    # just wraps the pure function for caching
    return load_data_core(yard_type)


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
            single_base_tag = make_base_tag(yard_type, single_stat_col, single_line_value)

            X_train_s, y_train_s, lengths_train_s, X_test_s, y_test_s, lengths_test_s = build_sequences_for_prop(
                train_df, test_df, single_stat_col, single_line_value
            )

            # LSTM
            lstm_model_s, lstm_metrics_s = train_or_load_lstm(
                single_base_tag,
                X_train_s, y_train_s, lengths_train_s,
                X_test_s,  y_test_s,  lengths_test_s,
                single_stat_col, single_line_value,
            )

            # TFT
            tft_model_s, tft_metrics_s = train_or_load_tft(
                single_base_tag,
                X_train_s, y_train_s, lengths_train_s,
                X_test_s,  y_test_s,  lengths_test_s,
                single_stat_col, single_line_value,
            )

            # XGBoost
            xgb_model_s, xgb_metrics_s = train_or_load_xgb(
                single_base_tag,
                X_train_s, y_train_s, lengths_train_s,
                X_test_s,  y_test_s,  lengths_test_s,
                single_stat_col, single_line_value,
            )

            # --- single-leg probabilities ---
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
            # 2) PARLAY: use the core `compute_parlay_prob` function
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