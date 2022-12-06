'''
Place to build functions and/or test out the program pieces and try joining them together.
'''
import pandas as pd

import accountperformance
import indicators
import visualization
import datamanipulation
import datetime as dt
import MLcomponents

# Prep dailydf
dailydf = datamanipulation.retrieve()
dailydf = datamanipulation.timeformatter(dailydf)
dailydf = indicators.all_indicators(dailydf)

# Split into original comparison set, training, validation, and testing sets
original, training, validation, testing = MLcomponents.data_processor(dailydf)

# Select fold 5 to use (previously determined by iterating through)
x_train = training[8][0]
y_train = training[8][1]
x_val = validation[8][0]
y_val = validation[8][1]

# Get testing data out
x_test = testing[0]
y_test = testing[1]

# Assemble full dataset for training
x_train = pd.concat([x_train, x_val])
y_train = pd.concat([y_train, y_val])

# Normalize x_values with StandardScaler
x_train, x_val, x_test = MLcomponents.SSnormalize(x_train, x_val, x_test)

# Clean up y values with y_cleaner
y_train, y_val, y_test = MLcomponents.y_cleaner(y_train, y_val, y_test)

# Use ML model to make prediction
# y_pred = MLcomponents.RF_prediction(x_train, y_train, x_test, n_estimators=140, maxdepth=10)
# y_pred, history = MLcomponents.ANN_prediction(x_train, y_train, x_val, y_val, x_test)
y_pred = MLcomponents.KNN_prediction(x_train, y_train, x_test, n_neighbors=7)

# Take predicted values and reattach them to previous data for evaluation
idealresults, MLresults = MLcomponents.assemble_results(y_pred, y_test, original)

# Evaluate prediction quality by confusion matrix
MLcomponents.evaluate_confusion(idealresults, MLresults)

# Evaluate return metrics
idealacctdf, MLacctdf = MLcomponents.evaluate_returns(idealresults, MLresults)

# Plot account value over time
visualization.account_comparison_plot(idealacctdf, MLacctdf, showideal=False)