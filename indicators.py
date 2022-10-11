'''
Place to calculate any desirable indicators to incorporate into the machine learning bot.
'''


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
    data.drop(columns=[f'EMA_{n}', f'EMA_{m}'], inplace=True)
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
