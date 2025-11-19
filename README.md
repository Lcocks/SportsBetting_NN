# SportsBetting_NN

Predictive model for leg parlay betting of NFL games.

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


## DataTransfandXGBOOST.ipynb

End-to-end machine learning pipeline for predicting NFL player props (Over/Under betting lines) using XGBoost with Expected Calibration Error (ECE) evaluation. The system trains separate models for receiving, rushing, and passing props using player-level rolling averages and historical performance features. This is the code for creating the baseline XGBoost model to compare with the LSTM model. Each cell has separate steps for actions like data transformations, model building, model running, testing and plotting. 

  Execute the notebook sections sequentially:
- **Section 1**: Loads 10 datasets (receiving, rushing, passing, defensive, etc.) from `data/` directory
- **Section 2**: Defines ECE calculation functions for model calibration evaluation (Walsh & Joshi, 2024)
- **Section 3**: Feature engineering with rolling averages (3/5/8 games), season stats, and prop hit rates
- **Section 4**: Single model training test on 2000 receiving records to validate pipeline
- **Section 5**: Inference function for predicting probabilities given player name, stat, and threshold
- **Section 6**: Configuration of 8 props to train (e.g., Receiving YDS Over 50/65/75)
- **Section 7**: Multi-prop training loop that trains all models and saves to `models/` directory
- **Section 8**: Summary report showing best calibrated (lowest ECE) and discriminative (highest AUC) models
- **Section 9**: Example predictions on top players from the dataset

Models achieve AUC > 0.60 validation threshold with calibration curves saved to `plots/`. Final models output probability distributions for betting decisions with confidence scores and ECE metrics.

## lstm.ipynb

NFL Player Prop LSTM with Kelly Criterion & Parlay Analysis

These are the class and functions used to train and load datasets for a specified parlay and output a probability

After running the first code cell to define the functions, there are example usages under "Single Leg Parlay" and "Multi Leg Parlay" specifing a certain player prop statistic for a player and will output the probability of that player hitting that target.

Under the "Implement Kelly ROI" section there is code to sample parlay legs from the test datasets to output the ECE, PCE and Flat ROI per parlay to serve as a metric for the model's performance as well as the example usage code at the last code cell.

  Execute notebook sections sequentially:
- **Section 1**: Configuration and data loading - `StatSeqConfig` sets window length (5 games), batch size (64), hidden size (128), epochs (20), learning rate (1e-3). `load_stat_df()` loads training (2019-2023) and holdout (2024-current) data for any stat category
- **Section 2**: Dataset creation - `StatSequenceDataset` and `make_train_test_sequences()` build sliding windows of past N games to predict next game outcome, split by season (≤2023 train, >2023 test)
- **Section 3**: Custom LSTM - `LSTMCell` implements forget/input/output gates from scratch with Xavier initialization and forget gate bias=1.0. `LSTMSequence` handles variable-length sequences with masking
- **Section 4**: Model architectures - `StatFromScratch` for regression, `StatFromScratchBinary` for Over/Under classification with BCEWithLogitsLoss and gradient clipping (max_norm=5.0)
- **Section 5**: `create_binary_model_from_dataset()` generates binary labels (stat ≥ threshold), filters features, creates train/test DataLoaders with custom collate function for variable lengths
- **Section 6**: `train_binary_model()` trains with Adam optimizer, reports train/test loss and accuracy per epoch
- **Section 7**: Single-leg example - Train passing yards model (Patrick Mahomes Over 305.5), predict probability using `predict_over_probability()` with last 5 games
- **Section 8**: Multi-leg parlay - Train separate models for passing (Mahomes) and rushing (Pacheco), compute joint probability with `parlay_model_prob()` by multiplying individual leg probabilities
- **Section 9**: Kelly Criterion implementation - `kelly_fraction_even()` computes optimal bet sizing (f* = 2p - 1 for even-money odds), `sample_parlays()` generates M random L-leg parlays from test set
- **Section 10**: Calibration metrics - `expected_calibration_error()` (ECE) bins single-leg predictions, `parlay_calibration_error()` (PCE) measures |predicted_prob - actual_outcome| for parlays
- **Section 11**: ROI calculation - `compute_kelly_and_flat_roi()` compares flat betting (equal stakes) vs Kelly sizing on sampled parlays
- **Section 12**: Full experiment pipeline - `run_parlay_experiment()` combines ECE, PCE, flat/Kelly ROI into single summary report with 1000 sampled 2-leg parlays

Model predicts Over/Under probabilities for any player prop, combines multiple legs into parlay probability, and evaluates profitability with Kelly optimal bet sizing.

### Key Metrics
- **ECE**: Single-leg calibration error (lower = better probability estimates)
- **PCE**: Parlay calibration error (measures multi-leg prediction quality)
- **Kelly ROI**: Expected return with optimal bet sizing
- **Flat ROI**: Baseline return with equal-weight betting
