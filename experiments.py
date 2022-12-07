'''
Place to put the project experiments.
'''

#Making changes
import pandas as pd
import numpy as np
import indicators
import datamanipulation
import MLcomponents
import visualization
# import Validate_Analyze
import sklearn
import matplotlib as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Global variables and constants
min_splits = 2
max_splits = 10
fold = 5
min_neighbors = 1
max_neighbors = 10

dailydf = datamanipulation.retrieve()
weeklydf = datamanipulation.retrieve(timeframe='weekly')
monthlydf = datamanipulation.retrieve(timeframe='monthly')
"""
Function runs our knn method
"""

def find_knn_parameters():
#################################
# This section iterates through 1 - 10 neighbors with 10-fold cross-validation.
# It was found that 9NN was best for 10-fold returns. 7NN was best for accuracy and 2nd best for returns.
# We chose 7NN because accuracy would seem to be a better measure because returns are based not only
# on accuracy, but what happens after the buy/sell.
#################################

    return_matrix = pd.DataFrame(np.zeros((max_splits + 1)))
    acc_matrix = pd.DataFrame(np.zeros((max_splits + 1)))

    n_splits = 10
    max_return = 0
    max_n_splits = 0

    # Prep daily data
    dailydf = datamanipulation.retrieve()
    dailydf = datamanipulation.timeformatter(dailydf)
    dailydf = indicators.all_indicators(dailydf)

    print("Iterate 1-10 Neighbors for 10 Splits")

    fold = n_splits - 1

    # Split into original comparison set, training, validation, and testing sets
    original, training, validation, testing = MLcomponents.data_processor(dailydf, n_splits)

    # Iterate through n_neighbors to create matrix
    for n_neighbors in range(1, max_neighbors + 1):
        print("\n\nNeighbors: ", n_neighbors, "\n")
        idealresults, MLresults = knn_helper(original, training, validation, testing, fold, n_neighbors)

        # Evaluate prediction quality by confusion matrix
        acc, p, r, f1 = MLcomponents.evaluate_confusion(idealresults, MLresults)

        # Evaluate return metrics
        idealacctdf, MLacctdf = MLcomponents.evaluate_returns(idealresults, MLresults)

        if MLacctdf.iloc[-1, :]['account value'] - 10000 > max_return:
            max_return = MLacctdf.iloc[-1, :]['account value'] - 10000
            max_n_splits = n_neighbors
            # max_fold = fold

        return_matrix.iloc[n_neighbors] = MLacctdf.iloc[-1, :]['account value'] - 10000
        acc_matrix.iloc[n_neighbors] = acc

    print("Matrix of n_neighbors vs Returns")
    print(return_matrix.round(2))
    print("\nMatrix of n_neighbors vs Accuracy")
    print(acc_matrix.round(3))
    print("\nMax return and number of splits and fold: ", max_return, max_n_splits)

#################################
# End of n_neighbors evaluation
#################################


def knn_experiment():
    #################################
    # This section uses 10 splits and 7NN and all indicators
    ################################
    # Prep daily data
    n_splits = 10
    fold = n_splits - 1
    ideal_neighbors = 7

    dailydf = datamanipulation.retrieve()
    dailydf = datamanipulation.timeformatter(dailydf)
    dailydf = indicators.all_indicators(dailydf)

    print("All Indicators")

    # Split into original comparison set, training, validation, and testing sets
    original, training, validation, testing = MLcomponents.data_processor(dailydf, n_splits)

    idealresults, MLresults = knn_helper(original, training, validation, testing, fold, ideal_neighbors)

    # Evaluate prediction quality by confusion matrix
    acc, p, r, f1 = MLcomponents.evaluate_confusion(idealresults, MLresults)

    # Evaluate return metrics
    idealacctdf, MLacctdf = MLcomponents.evaluate_returns(idealresults, MLresults)

    print()
    #################################
    # End of optimized parameters evaluation
    #################################

    #################################
    # This section uses 10 splits and 7NN and NO indicators.
    ################################
    # Prep daily data
    n_splits = 10
    fold = n_splits - 1
    ideal_neighbors = 7

    dailydf = datamanipulation.retrieve()
    dailydf = datamanipulation.timeformatter(dailydf)

    print("No Indicators")

    # Split into original comparison set, training, validation, and testing sets
    original, training, validation, testing = MLcomponents.data_processor(dailydf, n_splits)

    # Create versions with no indicators
    # stripped_original = original[['open', 'close', 'high', 'low', 'volume']]
    # stripped_training = training[['open', 'close', 'high', 'low', 'volume']]
    # stripped_validation = validation[['open', 'close', 'high', 'low', 'volume']]
    # stripped_testing = testing[['open', 'close', 'high', 'low', 'volume']]

    idealresults, MLresults = knn_helper(original, training, validation, testing, fold, ideal_neighbors)

    # Evaluate prediction quality by confusion matrix
    acc, p, r, f1 = MLcomponents.evaluate_confusion(idealresults, MLresults)

    # Evaluate return metrics
    idealacctdf, MLacctdf = MLcomponents.evaluate_returns(idealresults, MLresults)


"""
    Helper function to clean up knn_experiment
    :param original
    :param training set to use for training
    :param validation set to use for validation
    :param testing set to use for testing
    :param fold fold number to use for training and validation
    :param n_neighbors number of neighbors to use for knn. Default is 2
    :return idealresults results from ideal case
    :return MLresults results from the chosen ML method
"""
def knn_helper(original, training, validation, testing, fold, n_neighbors=2):
    x_train = training[fold][0]
    y_train = training[fold][1]
    x_val = validation[fold][0]
    y_val = validation[fold][1]

    # Get testing data out
    x_test = testing[0]
    y_test = testing[1]

    # Normalize x_values with StandardScaler
    x_train, x_val, x_test = MLcomponents.SSnormalize(x_train, x_val, x_test)

    # Clean up y values with y_cleaner
    y_train, y_val, y_test = MLcomponents.y_cleaner(y_train, y_val, y_test)

    # Use ML model to make prediction
    y_pred = MLcomponents.KNN_prediction(x_train, y_train, x_test, n_neighbors)

    # Take predicted values and reattach them to previous data for evaluation
    idealresults, MLresults = MLcomponents.assemble_results(y_pred, y_test, original)

    return idealresults, MLresults