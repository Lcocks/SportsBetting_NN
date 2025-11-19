# SportsBetting_NN
Predictive model for leg parlay betting of NFL games.

## Requirements
```
xgboost
pandas
numpy
scikit-learn
matplotlib
torch
```
## Data Format

Under the `data/` folder:

CSV files with columns: `game_id`, `date`, `season`, `team`, `home_away`, `opposing_team`, `athlete_id`, `display_name`, `position`, stat columns (e.g., `YDS`, `REC`, `TD`)

# Files

### apicodes.ipynb

These are the functions used to grab the player prop and nfl game data from the espn api via nntrn Github repository

### DataTransfandXGBOOST.ipynb

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
- **Section 9**: Example predictions on top players from the dataset****

Models achieve AUC > 0.60 validation threshold with calibration curves saved to `plots/`. Final models output probability distributions for betting decisions with confidence scores and ECE metrics.

### lstm.ipynb

These are the class and functions used to train and load datasets for a specified parlay and output a probability

After running the first code cell to define the functions, there are example usages under "Single Leg Parlay" and "Multi Leg Parlay" specifing a certain player prop statistic for a player and will output the probability of that player hitting that target

Under the "Implement Kelly ROI" section there is code to sample parlay legs from the test datasets to output the ECE, PCE and Flat ROI per parlay to serve as a metric for the model's performance as well as the example usage code at the last code cell

