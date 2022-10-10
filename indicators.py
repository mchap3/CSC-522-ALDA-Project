'''
Place to calculate any desirable indicators to incorporate into the machine learning bot.
'''


def simple_mov_avg(df, n=5):
    """
    Calculates simple moving average over user-defined timeframe

    :param df: dataframe with closing price attribute
    :param n: number of days in averaging timeframe
    :return: series of average values with same length as df
    """
    return df.loc[:, 'close'].rolling(n).mean()


def exp_mov_avg(df, n=5):
    """
    Calculates exponentially weighted moving average over user-defined timeframe

    :param df: dataframe with closing price attribute
    :param n: number of days used as averaging span
    :return: series of average values with same length as df
    """
    return df.loc[:, 'close'].ewm(span=n, adjust=False).mean()


def macd(df, n, m):
    """
    Calculates Moving Average Convergence/Divergence oscillator. Indicates momentum
    as the difference between shorter-term and longer-term moving averages

    :param df: dataframe with closing price attribute
    :param n: days in shorter timeframe
    :param m: days in longer timeframe
    :return: series of MACD values with same length as df
    """
    ema_n = exp_mov_avg(df, n)
    ema_m = exp_mov_avg(df, m)
    return ema_n - ema_m
