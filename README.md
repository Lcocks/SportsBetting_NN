# SportsBetting_NN
Predictive model for leg parlay betting of NFL games.

# Files

### apicodes.ipynb

These are the functions used to grab the player prop and nfl game data from the espn api via nntrn Github repository


### lstm.ipynb

These are the class and functions used to train and load datasets for a specified parlay and output a probability

After running the first code cell to define the functions, there are example usages under "Single Leg Parlay" and "Multi Leg Parlay" specifing a certain player prop statistic for a player and will output the probability of that player hitting that target

Under the "Implement Kelly ROI" section there is code to sample parlay legs from the test datasets to output the ECE, PCE and Flat ROI per parlay to serve as a metric for the model's performance as well as the example usage code at the last code cell

### DataTransfandXGBOOST.ipynb

This is the code for creating the baseline XGBoost model to compare with the LSTM model