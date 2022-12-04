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
import numpy as np


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
    Calculates exponential moving average.
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

def rsi(data, n=14, calc='close'):
    '''
    Calculates relative strength index.
    :param data: standard stock data
    :param n: number of periods
    :param calc: data series to be used in rsi calculation. Close is standard
    :return:
    '''
    # chop up data a bit, handle formatting
    if 'time and date' in data.columns:
        data = datamanipulation.timeformatter(data)
    colname = 'RSI' + str(n)
    reldat = data.loc[:, [calc]]

    # split everything into gain/loss for relative strength calcs
    reldat['diff'] = reldat.diff(1)
    reldat['gain'] = reldat['diff'].clip(lower=0)
    reldat['loss'] = reldat['diff'].clip(upper=0).abs()

    # calculate averages for relative strength. The ewm syntax mimics the exponential smoothing from the wilders moving
    # average
    reldat['avg_gain'] = reldat['gain'].ewm(com=n-1, min_periods=n).mean()
    reldat['avg_loss'] = reldat['loss'].ewm(com=n-1, min_periods=n).mean()

    # final relative strength calcs, append to original data
    reldat['rs'] = reldat['avg_gain'] / reldat['avg_loss']
    reldat['rsi'] = 100 - (100 / (1.0 + reldat['rs']))
    data[colname] = 100 - (100 / (1.0 + reldat['rs']))

    return data


def obv(data, calc='close'):
    '''
    Calculates on balance volume and adds it to the chart.
    :param data: dataframe containing standard stock data (including volume)
    :param calc: data point to base calculations off. Closing price is standard.
    :return: original data frame with on balance volume values appended
    '''
    if 'time and date' in data.columns:
        data = datamanipulation.timeformatter(data)
    colname = 'OBV_' + calc
    reldat = data.loc[:, [calc, 'volume']]

    data[colname] = (np.sign(reldat['close'].diff()) * reldat['volume']).fillna(0).cumsum()
    return data

def macd(data, n=12, m=26, s=9, calc='close'):
    """
    Calculates Moving Average Convergence/Divergence oscillator. Indicates momentum
    as the difference between shorter-term and longer-term moving averages. Also
    calculates difference from signal line (exponential weighted average of MACD)
    :param data: dataframe with stock price data
    :param n: days in shorter timeframe (default: 12)
    :param m: days in longer timeframe (default: 26)
    :param s: days in signal line timeframe (default: 9)
    :param calc: data attribute to calculate MACD (default: 'close')
    :return: dataframe with MACD attributes added
    """
    # calculate fast/slow EMAs
    data = ema(data, n, calc)
    data = ema(data, m, calc)
    data['MACD'] = data[f'EMA{n}'] - data[f'EMA{m}']

    # create signal line from above
    ema(data, s, 'MACD')

    # difference from signal line
    data['MACD_diff'] = data['MACD'] - data[f'EMA{s}']
    data.drop(columns=[f'EMA{n}', f'EMA{m}', f'EMA{s}'], inplace=True)
    return data


def bollinger_bands(data, n=20, m=2):
    """
    Calculates Bollinger Bands as indicators of overbought and oversold levels.
    :param data: dataframe with stock price data
    :param n: days to be included in SMA window (default: 20)
    :param m: standard deviation multiplication factor (default: 2)
    :return: dataframe with upper/lower Bollanger Band attributes added
    """
    boll_dat = data.loc[:, ['high', 'low', 'close']]

    # calculate moving avg of typical price
    boll_dat['TP'] = boll_dat.sum(axis=1) / 3
    boll_dat['TP_SMA'] = boll_dat.loc[:, 'TP'].rolling(n).mean()

    # calculate std and bands
    boll_dat['TP_std'] = boll_dat.loc[:, 'TP'].rolling(n).std()
    data['BOLU'] = boll_dat['TP_SMA'] + boll_dat['TP_std'] * 2
    data['BOLD'] = boll_dat['TP_SMA'] - boll_dat['TP_std'] * 2

    return data


def all_indicators(data):
    """
    Calls the following functions to make a combined df: ema(n = 10), ema(n = 25), ema(n = 50), sma(n = 100),
    sma(n = 200), rsi(n = 3), rsi(n = 14), macd(), bollinger_bands(), obv(), and centroid().
    Will also include, open, high, low, close, and volume from retrieve().
    :param data: dataframe with stock price data
    :return: dataframe with the following attributes and classifications added: ['open', 'high', 'low', 'close',
    'volume', 'date', 'EMA10', 'EMA25', 'EMA50', 'SMA100', 'SMA200', 'RSI3', 'RSI14', 'MACD', 'MACD_diff',
    'BOLU', 'BOLD', 'OBV_close', 'centroid', 'Rolling5']
    """

    data = ema(data, n=10)
    data = ema(data, n=25)
    data = ema(data, n=50)
    data = sma(data, n=100)
    data = sma(data, n=200)
    data = rsi(data, n=3)
    data = rsi(data, n=14)
    data = macd(data)
    data = bollinger_bands(data)
    data = obv(data)

    return data

