'''
Place to calculate any desirable indicators to incorporate into the machine learning bot.
'''

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


def centroid(data):
    '''
    :param data: stock data to work with
    :return: dataframe with centroid appended (mean of open, high, low, and close)
    '''

    data['centroid'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4

    return data


def five_day_centroid(data):
    '''
    Runs a 5-day moving average of the Centroid and then classifies as 1 (buy) or -1 (sell)
    if the 5-day average changed by more than the threshold.
    :param data: stock data to work with
    :return: dataframe with columns for five-day average, buy, sell, and Buy_Sell appended
    '''

    data = centroid(data)
    minimum_delta = .25
    num_rolling_days = 3
    data['Rolling5'] = data['centroid'].rolling(num_rolling_days).mean()
    data.Rolling5 = data.Rolling5.shift(-1 * num_rolling_days)
    data['Rolling5_Buy'] = data.Rolling5 > (data.Rolling5.shift() + minimum_delta)
    data['Rolling5_Sell'] = data.Rolling5 < (data.Rolling5.shift() - minimum_delta)
    data['Buy_Sell'] = data.Rolling5_Buy * 1 + data.Rolling5_Sell * (-1)

    # Drop all rows with NaN
    data = data.dropna()
    # Reset row numbers
    data = data.reset_index(drop=True)
    # Remove unneeded columns
    data = data.drop('Rolling5_Buy', axis = 1)
    data = data.drop('Rolling5_Sell', axis = 1)

    for current in data.loc[data['Buy_Sell'] == 0].index:
        if current != 0:
            data.loc[current, 'Buy_Sell'] = data.loc[current - 1, 'Buy_Sell']

    return data

def simple_mov_avg(df, n=5, calc='close'):
    """
    Calculates simple moving average over user-defined timeframe
    :param df: dataframe with closing price attribute
    :param n: number of days in averaging timeframe
    :return: df with SMA column added
    """
    df[f'SMA_{n}'] = df.loc[:, calc].rolling(n).mean()
    return df


def exp_mov_avg(df, n=5, calc='close'):
    """
    Calculates exponentially weighted moving average over user-defined timeframe
    :param df: dataframe with closing price attribute
    :param n: number of days used as averaging span
    :return: df with EMA column added
    """
    df[f'EMA_{n}'] = df.loc[:, calc].ewm(span=n, adjust=False).mean()
    return df


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
    data = exp_mov_avg(data, n, calc)
    data = exp_mov_avg(data, m, calc)
    data['MACD'] = data[f'EMA_{n}'] - data[f'EMA_{m}']

    # create signal line from above
    exp_mov_avg(data, s, 'MACD')

    # difference from signal line
    data['MACD_diff'] = data['MACD'] - data[f'EMA_{s}']
    data.drop(columns=[f'EMA_{n}', f'EMA_{m}', f'EMA_{s}'], inplace=True)
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
    sma(n = 200), rsi(n = 3), rsi(n = 14), macd(), bollinger_bands(), obv(), centroid(), and moving_centroid().
    sma(n = 200), rsi(n = 3), rsi(n = 14), macd(), bollinger_bands(), obv(), and centroid().
    Will also include, open, high, low, close, and volume from retrieve().

    :param data: dataframe with stock price data
    :return: dataframe with the following attributes and classifications added: ['open', 'high', 'low', 'close',
    'volume', 'date', 'EMA10', 'EMA25', 'EMA50', 'SMA100', 'SMA200', 'RSI3', 'SMA14', 'MACD', 'MACD_diff',
    'BOLU', 'BOLD', 'OBV_close', 'centroid', 'Rolling5', and 'Buy_Sell']
    'BOLU', 'BOLD', 'OBV_close', 'centroid', 'Rolling5']
    """

    data = ema(data, n=10)
    data = ema(data, n=25)
    data = ema(data, n=50)
    data = sma(data, n=100)
    data = sma(data, n=200)
    data = rsi(data, n=3)
    data = sma(data, n=14)
    data = macd(data)
    data = bollinger_bands(data)
    data = obv(data)
    data = centroid(data)
    data = five_day_centroid(data)

    return data

