'''
Place to calculate any desirable indicators to incorporate into the machine learning bot.
'''

from datamanipulation import *
import numpy as np
import matplotlib.pyplot as plt

def centroid(data):
    '''
    :param data: stock data to work with
    :return: dataframe with centroid appended (mean of open, high, low, and close)
    '''

    data['centroid'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4

    return data

def five_Day_Centroid(data):
    '''
    Created a function five_Day_Centroid, in indicators.py that runs a 5-day
    moving average of the Centroid and then classifies as 1 (buy) or -1 (sell)
    if the 5-day average changed by more than ten cents and 0 if it didn't.
    :param data: stock data to work with
    :return: dataframe with columns for five-day average, buy, sell, and Buy_Sell appended
    '''

    data = centroid(data)
    minimum_delta = .15
    num_rolling_days = 3
    data['Rolling5'] = data['centroid'].rolling(num_rolling_days).mean()
    data.Rolling5 = data.Rolling5.shift(-1 * num_rolling_days)
    data['Rolling5_Buy'] = data.Rolling5 > (data.Rolling5.shift() + minimum_delta)
    data['Rolling5_Sell'] = data.Rolling5 < (data.Rolling5.shift() - minimum_delta)
    data['Buy_Sell'] = data.Rolling5_Buy * 1 + data.Rolling5_Sell * (-1)

    for current in data['Buy_Sell']:
        if current == 0:
            data.loc['Buy_Sell':current] = data.loc['Buy_Sell':current - 1]

    print(data[['centroid', 'Rolling5', 'Rolling5_Buy', 'Rolling5_Sell', 'Buy_Sell']].head(15))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(data['Rolling5'][1000:1250])
    ax2.plot(data['Buy_Sell'][1000:1250], c='r')
    ax1.set_ylabel('SPY Price')
    ax2.set_ylabel('Class')
    ax2.set_yticks(range(-1, 2, 1))
    plt.show()

    return data

five_Day_Centroid(retrieve())


