'''
Place to calculate any desirable indicators to incorporate into the machine learning bot.
Thinking it might be good to add quite a few... preferably one or two from each class.
From https://www.investopedia.com/articles/active-trading/011815/top-technical-indicators-rookie-traders.asp#:~:text=In%20general%2C%20technical%20indicators%20fit,strength%2C%20volume%2C%20and%20momentum.
    5 categories of indicator:
        trend indicators
        mean reversion
        relative strength
        momentum
        volume
    and each category can be subdivided into leading and lagging:
        leading attempt to predict where price is going
        lagging offer a historical perspective
    a few examples
        1) moving averages: trend indicators, lagging
        2) bollinger bands: mean reversion indicator, lagging
        3) stochastic RSI: relative strength indicator, leading
        4) MACD: momentum indicator, generally lagging, sometimes leading
        5) on-balance volume: volume indicator, leading
'''

import pandas as pd
import datamanipulation

def sma(data, n=50, calc='close'):
    '''
    Calculates simple moving average.
    :param data: dataframe with standard stock data (open, high, low, close, vol, date)
    :param n: number of periods to be included in calculation
    :param calc: data point to calculate sma (close is standard)
    :return: original dataframe with additional column including SMA
    '''
    if 'time and date' in data.columns:
        data = datamanipulation.timeformatter(data)
    colname = 'SMA' + str(n)
    data[colname] = data[calc].rolling(n).mean()

    return data

def ema(data, n=10, calc='close'):
    '''
    Calculates exponentiol moving average.
    :param data: dataframe with standard stock data
    :param n: number of periods
    :param calc: data point to be used in the ema calculation
    :return: dataframe with column appended for the ema
    '''

    if 'time and date' in data.columns:
        data = datamanipulation.timeformatter(data)
    colname = 'EMA' + str(n)
    data[colname] = data[calc].ewm(span=n).mean()

    return data
