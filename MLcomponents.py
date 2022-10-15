'''
Place for putting together machine learning functions/analyses.
'''

import pandas as pd
import numpy as np
import datamanipulation


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
    df['Class'] = y

    return df
