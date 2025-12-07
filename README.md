# SportsBetting_NN

This repository contains machine learning models for predicting NFL player prop outcomes using LSTM, TFT (Temporal Fusion Transformer), and XGBoost algorithms. The models analyze historical player performance data to predict the probability of players hitting over/under lines for various statistics and in multi-leg parlays.

## Equal Contributors

-   **Rheyan Ampoyo** (@[rheyoampoyo](https://github.com/rheyoampoyo)) 

## Requirements
```
pandas
numpy
scikit-learn
matplotlib
torch
xgboost
requests
concurrent.futures
functools
dataclasses
dash
jupyterlab
jupyter-dash
plotly
torch
torchvision
torchaudio
xgboost
tqdm
requests
json
seaborn
```

# Files

## Data Format

Under the `data/` folder:

  CSV files with columns: `game_id`, `date`, `season`, `team`, `home_away`, `opposing_team`, `athlete_id`, `display_name`, `position`, stat columns (e.g., `YDS`, `REC`, `TD`)

### apicodes.ipynb

Automated ESPN api via nntrn Github repository scraper that fetches game schedules, play-by-play data, rosters, and player statistics for NFL seasons with parallel processing and intelligent caching.

  Execute notebook sections in order:
- **Section 1-2**: `get_nfl_games()` fetches game schedules for single/multiple years with parallel threading, filters out TBD matchups and games >14 days in future
- **Section 3-4**: `get_game_plays()` and `get_game_roster_from_plays()` extract play-by-play data and infer active rosters from game participants
- **Section 5-6**: `get_game_player_stats()` fetches boxscore stats with athlete position lookup using LRU cache (5000 athletes) to minimize API calls
- **Section 7-8**: `get_season_player_stats_optimized()` parallelizes game stat collection with 20 workers, rate limiting (0.05s delay), retry logic (3 attempts), and progress tracking every 10 games
- **Section 9**: `split_stats_by_category()` splits combined stats into 10 category-specific DataFrames (receiving, rushing, passing, defensive, kicking, punting, fumbles, interceptions, kick/punt returns)
- **Section 10**: Full 2019-2023 season data collection (~1,300 games) and save to `data/` directory as 10 separate CSVs
- **Section 11**: Holdout dataset collection for 2024-current season validation

Pipeline reduces collection time from 20-40 minutes to 2-4 minutes via parallelization. Position data fetched with caching to avoid redundant API calls. Future games and TBD matchups automatically filtered.

#### Output Files within `data/`

Training: `defensive_2019_2023.csv`, `receiving_2019_2023.csv`, `rushing_2019_2023.csv`, `passing_2019_2023.csv`, `fumbles_2019_2023.csv`, `interceptions_2019_2023.csv`, `kickreturns_2019_2023.csv`, `puntreturn_2019_2023.csv`, `kicking_2019_2023.csv`, `punting_2019_2023.csv`

Holdout: Same files with `_24tocurrent.csv` suffix for model validation




## Repository Structure
```
├── data/                      # Historical NFL player statistics datasets
├── models/                    # Saved trained model files (.pt, .json)
├── metrics/                   # Model performance metrics (JSON format)
├── plots/                     # Visualization outputs from test.ipynb
├── figures/                   # Additional charts and analysis visualizations
├── ablation_results/          # Results from ablation studies (results and plots) 
├── trained_models/            # Production-ready trained models
├── testing/                   # Test scripts and validation code
├── PDFS/                      # Documentation and research papers
├── oldstuff/                  # Archived/deprecated code
├── test.ipynb                 # **Main testing notebook** - Primary interface
├── Ablation_Studies.ipynb     # **Secondary testing notebook** - Ablation study experiments
├── core_logic.py              # Core functions for parlay probability computation
├── data_prep.py               # Data preprocessing and sequence preparation
├── lstm.py                    # LSTM model implementation
├── tft.py                     # Temporal Fusion Transformer implementation
├── xgb.py                     # XGBoost model implementation
├── metrics.py                 # Evaluation metrics (ECE, PaCE, AUC)
├── player_utils.py            # Player-specific prediction utilities
├── requirements.txt           # Python package dependencies
├── LICENSE                    # Repository license
└── README.md                  # Introduction to repo
```

## Key Features

- **Multiple Model Architectures**: Compare LSTM, TFT, and XGBoost for different prediction scenarios
- **Multi-Leg Parlay Support**: Calculate probabilities for parlays with multiple player props
- **Comprehensive Metrics**: Evaluate models using AUC, ECE (Expected Calibration Error), and PaCE (Parlay Calibration Error)
- **Flexible Statistics**: Support for receiving, rushing, passing, defensive, and special teams statistics
- **Ablation Studies**: Systematic analysis of hyperparameter (model and data based) effects on model performance

## Getting Started

### Main Usage

The primary interface is `test.ipynb`, a Jupyter notebook that provides:

1. **Single Model Training & Evaluation**
2. **Single-Leg Probability Predictions**
3. **Multi-Leg Parlay Probability Calculations**

#### Quick Start Example
```python
# Configure your prediction
yard_type = "receiving"  # Options: receiving, passing, rushing, defensive, etc.
    """this is set at the beginning of LSTM model type (first model type) and is reset within EACH MODEL to follow a new naming convetion. See core_logic.py"""
STAT_COL = "YDS"         # The statistic to predict (e.g., YDS, REC, TD, etc.)
LINE_VALUE = 80          # The over/under threshold
N_PAST_GAMES = 5         # Number of historical games to use
HIDDEN_SIZE = 128 # number of hidden layers
D_MODEL = 128 # dimension of the tft model

# Train a model and make predictions using test.ipynb
```

## Supported Statistics

### Receiving
- **REC** – Receptions
- **YDS** – Receiving Yards
- **AVG** – Avg Yards/Reception
- **TD** – Receiving TDs
- **LONG** – Longest Reception
- **TGTS** – Targets

### Rushing
- **CAR** – Carries (Rushing Attempts)
- **YDS** – Rushing Yards
- **AVG** – Avg Yards/Carry
- **TD** – Rushing TDs
- **LONG** – Longest Rush

### Passing
- **C_ATT** – Completions/Attempts
- **YDS** – Passing Yards
- **AVG** – Avg Yards/Attempt
- **TD** – Passing TDs
- **INT** – Interceptions Thrown
- **SACKS** – Times Sacked
- **QBR** – QB Rating (ESPN)
- **RTG** – Passer Rating (NFL)

### Defensive
- **TOT** – Total Tackles
- **SOLO** – Solo Tackles
- **SACKS** – Sacks
- **TFL** – Tackles for Loss
- **PD** – Passes Defended
- **QB_HTS** – QB Hits
- **TD** – Defensive TDs

### Other Categories
- **Fumbles**: FUM, LOST, REC
- **Interceptions**: INT, YDS, TD
- **Kicking**: FG, PCT, LONG, XP, PTS
- **Kick Returns**: NO, YDS, AVG, LONG, TD
- **Punt Returns**: NO, YDS, AVG, LONG, TD
- **Punting**: NO, YDS, AVG, TB, IN_20, LONG

## Model Configuration

### LSTM Configuration
```python
HIDDEN_SIZE = 128    # Number of hidden units
N_PAST_GAMES = 5     # Sequence length
n_epochs = 10
batch_size = 64
lr = 1e-3
```

### TFT Configuration
```python
D_MODEL = 128        # Model dimension
n_heads = 4          # Attention heads
num_layers = 2       # Transformer layers
dropout = 0.1
lr = 1e-3

```

### XGBoost Configuration
```python
n_estimators = 300
max_depth = 4
learning_rate = 0.05
subsample = 0.9
colsample_bytree = 0.9
```

## Evaluation Metrics

- **AUC (Area Under ROC Curve)**: Overall classification performance
- **ECE (Expected Calibration Error)**: Measures probability calibration quality
- **PaCE-2 (Parlay Calibration Error)**: Evaluates 2-leg parlay predictions

## Ablation Studies

Run systematic experiments to find optimal hyperparameters:
```python
# See Ablation_Studies.ipynb for m (1) model hyperparameter (2) N_PAST_GAMES analysis (3) compared raw sigmoid outputs, Platt scaling, and isotonic regression calibration for XGBoost receiving yards.
# Results saved to ablation_results/
```

## Multi-Leg Parlay Example
```python
parlay_legs = [
    {
        "player": "George Kittle",
        "stat_col": "YDS",
        "line_value": 55.5,
    },
    {
        "player": "Brandon Aiyuk",
        "stat_col": "TGTS",
        "line_value": 2,
    },
]

parlay_prob, leg_probs = compute_parlay_prob(
    parlay_legs=parlay_legs,
    yard_type="Receiving",
    parlay_model_choice="LSTM",  # or "TFT" or "XGBoost"
    train_df=train_df,
    test_df=test_df,
    full_df=full_df,
)
```

## Output

- **Models**: Saved to `models/` directory as `.pt` (PyTorch) or `.json` (XGBoost) files
- **Metrics**: Performance metrics saved to `metrics/` as JSON files
- **Plots**: Visualizations saved to `plots/` directory
- **Ablation Results**: Study results in `ablation_results/`

## Project Workflow

1. **Data Preparation**: Load and preprocess historical player statistics using `testing/apicodes_recovered.ipynb`
2. **Model Training**: Train LSTM, TFT, or XGBoost models using `test.ipynb`
3. **Evaluation**: Assess model performance using AUC, ECE, and PaCE metrics
4. **Prediction**: Generate single-leg or multi-leg parlay probabilities
5. **Ablation**: Optimize hyperparameters using systematic experiments

## Notes

- **Single Yard Type Per Parlay**: Multi-leg parlays must use the same yard type (e.g., all receiving stats)
- **Model Retraining**: If a leg configuration hasn't been trained, the model will be trained automatically
- **Sequence Length**: N_PAST_GAMES typically ranges from 3-8 games

## Contact

    - epx8hh@virginia.edu



