'''
Place for putting together machine learning functions/analyses.
'''
from scipy.signal import argrelextrema
import numpy as np


def minmax(df, start_date, end_date, calc='close'):
    '''
    Function to find extrema in data and signal potential reversal
    :param df: dataframe with stock data, indicators optional
    :param start_date: first data point
    :param end_date: last data point
    :param calc: points used to determine extrema
    :return: dataframes with the extrema and dates
    '''
    if 'time and date' in df.columns:
        df = datamanipulation.timeformatter(df)
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    data = df.loc[mask]
    # data.set_index('date', inplace=True)

    local_max = argrelextrema(data[calc].values, np.greater)[0]
    local_min = argrelextrema(data[calc].values, np.less)[0]

    highs = data.iloc[local_max, :]
    lows = data.iloc[local_min, :]

    return highs, lows

