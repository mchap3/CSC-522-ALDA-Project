'''
Place for putting together machine learning functions/analyses.
'''
import numpy as np
import pandas as pd
import Validate_Analyze
import indicators
from sklearn.neighbors import KNeighborsClassifier
import math

def cont_trend_label(df, calc='close', w=0.008):
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
    df['Class'] = y

    return df


def moving_centroid(data):
    '''
    Runs a 5-day moving average of the Centroid and then classifies as 1 (buy) or -1 (sell)
    if the 5-day average changed by more than the threshold.
    :param data: stock data to work with
    :return: dataframe with columns for five-day average, buy, sell, and Class appended
    '''

    # data = centroid(data)
    # delta_values = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
    # rolling_days = {3, 4, 5, 6, 7, 8, 9}
    minimum_delta = .7
    num_rolling_days = 4
    data['Rolling'] = data['centroid'].rolling(num_rolling_days).mean()
    # Shift all data by num_rolling_days because there won't be an average for those first days
    data.Rolling = data.Rolling.shift(-1 * num_rolling_days - 1)
    # Rolling_Buy is true if the shifted average is more than the average at the current date plus the delta
    data['Rolling_Buy'] = data.Rolling > (data.Rolling.shift() + minimum_delta)
    # Rolling_Sell is true if the shifted average is less than the average at the current date minus the delta
    data['Rolling_Sell'] = data.Rolling < (data.Rolling.shift() - minimum_delta)
    data['Class'] = data.Rolling_Buy * 1 + data.Rolling_Sell * (-1)
    # print(data['date'])
    # Drop all rows with NaN
    data = data.dropna()
    # Reset row numbers
    data = data.reset_index(drop=True)
    # Remove unneeded columns
    # data = data.drop('date', axis=1)
    data = data.drop('Rolling_Buy', axis=1)
    data = data.drop('Rolling_Sell', axis=1)

    # Set day one to "buy"
    data.loc[0, 'Class'] = 1
    # Set all holds to the previous buy/sell
    for current in data.loc[data['Class'] == 0].index:
        if current != 0:
            data.loc[current, 'Class'] = data.loc[current - 1, 'Class']
    # print(data)
    # # Set all -1s to 0s
    # for current in data.loc[data['Class'] == -1].index:
    #     if current != 0:
    #         data.loc[current, 'Class'] = 0

    # print(data[['date', 'Rolling', 'Rolling_Buy', 'Rolling_Sell', 'Class']])

    return data


def do_knn(train, train_idx, test, test_idx, y, k):
    minimum = 1

    # for k in range(1, 11):
    # create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    # train KNN classifier
    knn.fit(train, y[train_idx].ravel())

    # test set predictions
    y_pred = knn.predict(test)

    # determine error
    y_size = y_pred.shape[0]
    y_pred = y_pred.reshape(y_size, 1)


    y_test = y[test_idx].copy()
    e = y_test != y_pred
    error = e.sum() / len(test)

    if error < minimum:
        minimum = error
    # print('error:', round(error, 3))

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    # print('error: ', round(error, 2))
    # print(cm)

    return minimum, cm, y_pred;