'''
Contains functions to retrieve data from spreadsheets, put them in dataframes, slice things up, etc.
'''

import pandas as pd
import os
import re
import datetime as dt

def retrieve(timeframe='daily'):
    '''
    :param timeframe: pass the desired timeframe you want to work with (daily, weekly, monthly, 1 min, 5 min, etc.)
    :return: returns a dataframe with the chosen stock information (price, time, volume)
    Note: currently only accommodates the timeframes with one version of the file
    '''
    path = 'Data'

    for (root, dirs, file) in os.walk(path):
        files = file

    for item in files:
        if timeframe in item:
            data = item
    # print(data)
    filepath = 'Data/' + str(data)

    df = pd.read_excel(filepath)
    return df


def timeformatter(df):
    '''
    :param dataframe: dataframe with time and date data
    :return: dataframe with just the date included for daily/monthly/weekly data in the format yyyy/mm/day
    '''

    # df['time and date'] = pd.to_datetime(df['time and date']).dt.date
    # df.rename(columns={'time and date': 'date'}, inplace=True)
    df['date'] = df['time and date'].dt.strftime('%Y-%m-%d')
    df.drop(columns=['time and date'], inplace=True)

    return df

# def knn(data, k = 5, fold = 10):
#     '''
#     Uses knn and k-fold validation
#     :param data: stock data with classification column already appended
#     :param k: the number of nearest neighbors to use for classification
#     :param fold: the number of folds
#     :return: dataframe with column for knn predicted value appended
#     '''
#     import numpy as np
#     from sklearn.neighbors import KNeighborsClassifier
#     from sklearn.model_selection import cross_val_score
#
#     print(list(data.columns))
#
#     # create KNN classifier
#     knn_k = KNeighborsClassifier(n_neighbors = k, metric = 'euclidean')
#
#     # TODO:  create training and testing sets
#
#     # train KNN classifier
#     print(train.shape)
#     knn_k.fit(train.reshape(-1, 1), Y[train_idx])
#
#     # test set predictions
#     y_pred = knn_k.predict(test.reshape(-1, 1))
#
#     # determine error
#     e = Y[test_idx] != y_pred
#     print(e)
#     error = sum(e) / len(test)
#     print("error for k = ", k, ":", np.mean(error))
#
#     return data
