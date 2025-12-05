import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


# ============================================================
# 1. BASIC HELPERS
# ============================================================

def _ensure_datetime(df: pd.DataFrame, time_col: str = "date") -> pd.DataFrame:
    """
    Ensure the time_col is a proper datetime.
    """
    df = df.copy()
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col])
    return df


def _add_home_away_flag(df: pd.DataFrame, col: str = "home_away") -> pd.DataFrame:
    """
    Map home/away to a numeric flag: away=0, home=1
    """
    df = df.copy()
    if col in df.columns:
        df["home_away_flag"] = df[col].map({"away": 0, "home": 1}).astype("float32")
    else:
        df["home_away_flag"] = 0.0
    return df


# ============================================================
# 2. SEQUENCE PREPARATION (FOR LSTM / TFT)
# ============================================================

def prepare_sequences_from_long_df(
    df: pd.DataFrame,
    group_col: str = "athlete_id",
    time_col: str = "date",
    target_col: str = "YDS",
    n_past_games: int = 5,
    feature_cols: Optional[List[str]] = None,
    min_games_per_entity: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:

    df = df.copy()

    df = _ensure_datetime(df, time_col=time_col)
    df = _add_home_away_flag(df, col="home_away")

    if feature_cols is None:
        default_feature_candidates = [
            "REC", "YDS", "AVG", "TD", "LONG", "TGTS", "home_away_flag"
        ]
        feature_cols = [c for c in default_feature_candidates if c in df.columns]

        if not feature_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c != target_col]

    if min_games_per_entity is None:
        min_games_per_entity = n_past_games + 1

    X_list = []
    y_list = []
    lengths_list = []
    meta_rows = []

    for entity_id, g in df.groupby(group_col):
        g = g.sort_values(time_col)

        if len(g) < min_games_per_entity:
            continue

        feats = g[feature_cols].to_numpy(dtype=np.float32)  
        targets = g[target_col].to_numpy(dtype=np.float32)

        num_games = len(g)

        for t in range(n_past_games, num_games):
            X_seq = feats[t - n_past_games:t, :]  
            y_target = targets[t]

            X_list.append(X_seq)
            y_list.append(y_target)
            lengths_list.append(n_past_games)

            meta_rows.append({
                group_col: entity_id,
                "target_index_in_group": t,
                "target_game_id": g.iloc[t]["game_id"] if "game_id" in g.columns else None,
                "target_date": g.iloc[t][time_col],
                "target_team": g.iloc[t]["team"] if "team" in g.columns else None,
                "target_opponent": g.iloc[t]["opposing_team"] if "opposing_team" in g.columns else None,
                "target_display_name": g.iloc[t]["display_name"] if "display_name" in g.columns else None,
                "target_position": g.iloc[t]["position"] if "position" in g.columns else None,
            })

    if not X_list:
        raise ValueError(
            "No sequences were created. "
            "Check n_past_games/min_games_per_entity or that the dataframe has enough games per entity."
        )

    X = np.stack(X_list, axis=0)  
    y = np.array(y_list, dtype=np.float32)  
    lengths = np.array(lengths_list, dtype=np.int64)

    meta = pd.DataFrame(meta_rows)

    return X, y, lengths, meta


def prepare_receiving_sequences(
    df: pd.DataFrame,
    n_past_games: int = 5,
    target_col: str = "YDS",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:

    return prepare_sequences_from_long_df(
        df=df,
        group_col="athlete_id",
        time_col="date",
        target_col=target_col,
        n_past_games=n_past_games,
        feature_cols=feature_cols,
    )


# ============================================================
# 3. SEQUENCE → TABULAR (FOR XGBOOST / OTHER MODELS)
# ============================================================

def sequences_to_tabular(
    X: np.ndarray,
    y: np.ndarray,
    meta: Optional[pd.DataFrame] = None,
    flatten: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:

    N, T, E = X.shape

    if flatten:
        X_flat = X.reshape(N, T * E)  

        col_names = [f"f_t{t}_e{e}" for t in range(T) for e in range(E)]
        df_features = pd.DataFrame(X_flat, columns=col_names)
    else:
        raise NotImplementedError("Only flatten=True is implemented for now.")

    if meta is not None:
        meta_reset = meta.reset_index(drop=True)
        df_tabular = pd.concat([meta_reset, df_features], axis=1)
    else:
        df_tabular = df_features

    return df_tabular, y


def add_over_under_label(df, stat_col, line_col=None, line_value=None, new_col="over_label"):

    if line_col is None and line_value is None:
        raise ValueError("You must provide either line_col or line_value.")

    if line_col is not None and line_col not in df.columns:
        raise ValueError(f"line_col '{line_col}' not found in DataFrame.")

    if line_col is not None:
        df[new_col] = (df[stat_col] > df[line_col]).astype(int)

    else:
        df[new_col] = (df[stat_col] > line_value).astype(int)

    return df