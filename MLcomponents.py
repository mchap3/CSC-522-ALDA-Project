'''
Place for putting together machine learning functions/analyses.
'''

import numpy as np
import pandas as pd
import datamanipulation
import math
import sklearn
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import sys
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
import accountperformance
import matplotlib.pyplot as plt

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def cont_trend_label(df, calc='close', w=0.05):
    """
    Labels class as [1, -1] (up/down) based on continuous trend
    algorithm from https://www.mdpi.com/1099-4300/22/10/1162/htm
    :param df: dataframe with stock price data
    :param calc: data attribute for trend calculations
    (default: 'close')
    :param w: proportion threshold parameter (default: 0.05)
    :return: dataframe with Class label added
    """
    # initialize
    X = df[calc].to_numpy()
    FP = X[0]   # first price
    xH = X[0]   # highest price
    HT = -1     # time of highest price
    xL = X[0]   # lowest price
    LT = -1     # time of lowest price
    Cid = 0     # current trend label
    FP_N = 0    # initial high/low price
    y = np.zeros(len(X))    # array to store labels

    # find first high/low
    for i in range(len(X)):
        if X[i] > FP + X[0] * w:
            xH, HT, FP_N, Cid = X[i], i, i, 1
            break
        if X[i] < FP - X[0] * w:
            xL, LT, FP_N, Cid = X[i], i, i, -1
            break

    # process labels
    for i in range(FP_N + 1, len(X)):
        if Cid > 0:
            if X[i] > xH:
                xH, HT = X[i], i
            if X[i] < xH - xH * w and LT <= HT:
                for j in range(len(y)):

                    if j > LT and j <= HT:

                        y[j] = 1
                xL, LT, Cid = X[i], i, -1
        if Cid < 0:
            if X[i] < xL:
                xL, LT = X[i], i
            if X[i] > xL + xL * w and HT <= LT:
                for j in range(len(y)):

                    if j > HT and j <= LT:

                        y[j] = -1
                xH, HT, Cid = X[i], i, 1

    # process any remaining points at end
    for j in range(1, len(y)):
        if y[j] == 0:
            y[j] = Cid

    # add labels to dataframe
    df['class'] = y

    return df


def five_day_centroid(data):
    '''
    Runs a 5-day moving average of the Centroid and then classifies as 1 (buy) or -1 (sell)
    if the 5-day average changed by more than the threshold.
    :param data: stock data to work with
    :return: dataframe with columns for five-day average, buy, sell, and Buy_Sell appended
    '''

    data = datamanipulation.centroid(data)
    minimum_delta = .25
    num_rolling_days = 3
    data['Rolling5'] = data['centroid'].rolling(num_rolling_days).mean()
    data.Rolling5 = data.Rolling5.shift(-1 * num_rolling_days)
    data['Rolling5_Buy'] = data.Rolling5 > (data.Rolling5.shift() + minimum_delta)
    data['Rolling5_Sell'] = data.Rolling5 < (data.Rolling5.shift() - minimum_delta)
    data['class'] = data.Rolling5_Buy * 1 + data.Rolling5_Sell * (-1)

    # Drop all rows with NaN
    data = data.dropna()

    # Reset row numbers
    data = data.reset_index(drop=True)

    # Remove unneeded columns
    data = data.drop('Rolling5_Buy', axis = 1)
    data = data.drop('Rolling5_Sell', axis = 1)

    data.loc[0, 'class'] = 1
    for current in data.loc[data['class'] == 0].index:
        if current != 0:
            data.loc[current, 'class'] = data.loc[current - 1, 'class']

    return data


def data_processor(data):
    """
    Handles all of the data splitting to feed into models.
    :param data: full dataset to be split
    :return: original dataset, training data, validation data, and testing data
    """
    # Assign the target values and split
    modeldata = cont_trend_label(data, w=0.008)
    modeldata.set_index('date', inplace=True)
    modeldata = modeldata.dropna()

    # Capture the original data for comparison later.
    original = modeldata.get(['open', 'high', 'low', 'close', 'volume'])
    original = original[original.index >= '2017-01-01']

    x = modeldata.iloc[:, 0:(modeldata.shape[1] - 1)]
    y = modeldata.iloc[:, (modeldata.shape[1] - 1)]
    x_test = x[x.index >= '2017-01-01']
    y_test = y[y.index >= '2017-01-01']
    x = x[x.index < '2017-01-01']
    y = y[y.index < '2017-01-01']
    tscv = TimeSeriesSplit(n_splits=10)
    training = []
    validation = []
    testing = [x_test, y_test]

    # Split values with time series cross validation.
    for train_index, test_index in tscv.split(x):
        trainentry = []
        valentry = []
        x_train, x_val = x.iloc[train_index], x.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        trainentry.append(x_train)
        trainentry.append(y_train)
        valentry.append(x_val)
        valentry.append(y_val)
        training.append(trainentry)
        validation.append(valentry)
    return original, training, validation, testing


def SSnormalize(x_train, x_val, x_test):
    """
    Normalizes everything with StandardScaler
    :param x_train: training data
    :param x_val: validation data
    :param x_test: testing data
    :return: returns normalized versions of each of the inputs
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return x_train, x_val, x_test


def PTnormalize(x_train, x_val, x_test):
    """
    Normalizes everything with Yeo-Johnson method PowerTransform
    :param x_train: training data
    :param x_val: validation data
    :param x_test: testing data
    :return: returns normalized versions of each of the inputs
    """
    scaler = PowerTransformer(method='yeo-johnson')
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return x_train, x_val, x_test


def y_cleaner(y_train, y_val, y_test):
    """
    Takes all of the y columns and replaces -1s with 0s for full compatibility with all models
    :param y_train: training data
    :param y_val: validation data
    :param y_test: testing data
    :return: 0s and 1s for sell and buy, respectively, instead of -1s and 1s
    """
    y_train = y_train.replace(to_replace=-1, value=0)
    y_val = y_val.replace(to_replace=-1, value=0)
    y_test = y_test.replace(to_replace=-1, value=0)

    return y_train, y_val, y_test


def KNN_prediction(x_train, y_train, x_test, n_neighbors=7):
    """
    Builds KNN model with training data and returns a prediction array.
    :param x_train: training data input
    :param y_train: training data target
    :param x_test: testing data input
    :param n_neighbors: number of neighbors to be passed to the KNN model
    :return: prediction results as dataframe
    """
    model = KNN(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)
    results = pd.DataFrame(model.predict(x_test), columns=['Predicted Class'])

    return results


def RF_prediction(x_train, y_train, x_test, n_estimators=140, maxdepth=10):
    """
    Builds Random Forest classification model with training data and returns a prediction array.
    :param x_train: training data input
    :param y_train: training data target
    :param x_test: testing data input
    :param n_estimators: parameter for random forest
    :return: prediction results as dataframe
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=maxdepth)
    model.fit(x_train, y_train)
    results = pd.DataFrame(model.predict(x_test), columns=['Predicted Class'])

    return results


def ANN_prediction(x_train, y_train, x_val, y_val, x_test, hidden=100, dropout1=0.1,
                   dropout2=0.5, epochs=12, es=True, verbose=1):
    """
    ANN model with optimized parameters as defaults to predict buy/sell points.
    :param x_train:
    :param y_train:
    :param x_val:
    :param y_val:
    :param x_test:
    :return: prediction and model history for evaluation
    """
    input_neurons = x_train.shape[1]
    hidden1 = input_neurons * 2
    output_neurons = 1

    # Set variables
    epochs = epochs
    hidden_act = 'relu'
    out_act = 'sigmoid'
    loss = 'binary_crossentropy'
    optim = 'adam'
    metrics = 'accuracy'
    batch_size = int(x_train.shape[1] / 100)
    # batch_size = 10

    # Set up model
    model = keras.Sequential([keras.layers.Dense(hidden1, activation=hidden_act, input_dim=input_neurons),
                              keras.layers.Activation('relu'),
                              keras.layers.Dropout(dropout1),
                              keras.layers.Dense(hidden, activation=hidden_act),
                              keras.layers.Activation('relu'),
                              keras.layers.Dense(hidden, activation=hidden_act),
                              keras.layers.Dropout(dropout2),
                              keras.layers.Activation('relu'),
                              keras.layers.Dense(hidden, activation=hidden_act),
                              keras.layers.Activation('relu'),
                              keras.layers.Dense(output_neurons, activation=out_act, name='predictions')])

    model.compile(optimizer=optim, loss=loss, metrics=metrics)
    if es:
        early_stopping = EarlyStopping(patience=4, restore_best_weights=True)
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                  callbacks=[early_stopping], verbose=verbose)
    else:
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                            verbose=verbose)

    results = pd.DataFrame(model.predict(x_test), columns=['Predicted Class'])

    return results, history


def NB_prediction(x_train, y_train, x_test):
    """
    Builds Naive Bayes classification model with training data and returns a prediction array.
    :param x_train: training data input
    :param y_train: training data target
    :param x_test: testing data input
    :return: prediction results as dataframe
    """
    model = GaussianNB(var_smoothing=0.0004641588833612782)
    model.fit(x_train, y_train)
    results = pd.DataFrame(model.predict(x_test), columns=['Predicted Class'])

    return results


def SVM_prediction(x_train, y_train, x_test):
    """
    Builds Naive Bayes classification model with training data and returns a prediction array.
    :param x_train: training data input
    :param y_train: training data target
    :param x_test: testing data input
    :return: prediction results as dataframe
    """
    model = SVC(kernel='linear', C=0.1)
    model.fit(x_train, y_train)
    results = pd.DataFrame(model.predict(x_test), columns=['Predicted Class'])

    return results


def assemble_results(y_pred, y_test, original):
    """
    Takes predicted results for y and original x and y data, uses it all to create two dataframes for further
    use/comparison: ideal results and ML results. Note: original data is only for the test set
    :param y_pred: ML model output
    :param y_test: original y values for comparison
    :param original: original x values for model performance in estimate_returns
    :return: ideal results as dataframe and ML results as dataframe
    """
    # Assign 1s or 0s to output from probabilistic models. Will have no effect on the models that already assign
    # 1s and 0s.
    y_pred.loc[y_pred['Predicted Class'] > 0.5, 'Predicted Class'] = 1
    y_pred.loc[y_pred['Predicted Class'] <= 0.5, 'Predicted Class'] = 0

    # Get dates out of dataframe to have something to index by
    y = pd.DataFrame(y_test)
    dates = y.index.to_frame()
    datedresults = pd.concat([dates.reset_index(drop=True), y_pred.reset_index(drop=True)], axis=1, ignore_index=True)
    datedresults.rename(columns={0: 'date', 1: 'class'}, inplace=True)
    datedresults.set_index('date', inplace=True)

    # Assemble DF with ideal results
    idealresults = pd.concat([original, y], axis=1, ignore_index=True)
    idealresults.columns = ['open', 'high', 'low', 'close', 'volume', 'class']

    # Assemble DF with ML results
    MLresults = pd.concat([original, datedresults], axis=1, ignore_index=False)
    MLresults.columns = ['open', 'high', 'low', 'close', 'volume', 'class']

    return idealresults, MLresults

def evaluate_confusion(idealresults, MLresults):
    """
    Evaluates results/quality of predictions. No outputs/returns, just prints values and shows graphs and such
    :param idealresults: x_test data with the target y values
    :param MLresults: x_test data with the predicted y values
    :return: nothing, prints and graphs
    """
    # Back out arrays of class values from results
    y_test = idealresults['class'].values
    y_pred = MLresults['class'].values

    # Calculate and display confusion matrix
    confusion = sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[1, 0])
    # print('\n')
    # print('Confusion Matrix:')
    # print(confusion)

    # Plot the confusion matrix
    # cmplot = sklearn.metrics.ConfusionMatrixDisplay(confusion, display_labels=['Buy', 'Sell'])
    # cmplot.plot()
    # plt.show()

    # Calculate accuracy, precision, etc
    TP = confusion[0][0]
    TN = confusion[1][1]
    FP = confusion[1][0]
    FN = confusion[0][1]
    
    acc = (TP + TN) / (TP + TN + FP + FN)
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    f1 = (2 * p * r) / (p + r)

    results = pd.DataFrame(columns=['result'])
    results.loc['Accuracy'] = acc
    results.loc['Precision'] = p
    results.loc['Recall'] = r
    results.loc['F1 Score'] = f1

    # Print results
    # print('Accuracy: ' + str(acc))
    # print('Precision: ' + str(p))
    # print('Recall: ' + str(r))
    # print('F1 Score: ' + str(f1))
    # print('\n')

    return results


def evaluate_returns(idealresults, MLresults):
    """
    Generates tables with ROI information for each input. I want to make the MLresults input a dictionary so that
    it concatenates all of the results into one table, but might be unnecessary.
    :param idealresults: original labeled data with open high low close volume
    :param MLresults: model prediction table
    :return: nothing, prints table with results
    """

    # Calculate ideal returns
    idealresults = datamanipulation.mid(idealresults)
    idealresults.reset_index(inplace=True)
    idealreturns, idealacctdf = accountperformance.estimate_returns(idealresults)

    # Calculate ML returns
    MLresults = datamanipulation.mid(MLresults)
    MLresults.reset_index(inplace=True)
    MLreturns, MLacctdf = accountperformance.estimate_returns(MLresults)

    # Print Results
    # print('Ideal Return Comparison: ')
    # print(idealreturns)
    # print('\n')
    # print('ML Return Comparison: ')
    # print(MLreturns)

    return idealreturns, MLreturns, idealacctdf, MLacctdf

