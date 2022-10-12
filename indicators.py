'''
Place to calculate any desirable indicators to incorporate into the machine learning bot.
'''

import pandas as pd
import datamanipulation
import numpy as np
from datamanipulation import *

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
    minimum_delta = 0.1
    data['Rolling5'] = data['centroid'].rolling(5).mean()
    data['Rolling5_Buy'] = data.Rolling5 > (data.Rolling5.shift() + minimum_delta)
    data['Rolling5_Sell'] = data.Rolling5 < (data.Rolling5.shift() - minimum_delta)
    data['Buy_Sell'] = data.Rolling5_Buy * 1 + data.Rolling5_Sell * (-1)

    #print(data[['centroid', 'Rolling5', 'Rolling5_Buy', 'Rolling5_Sell', 'Buy_Sell']].head(15))

    return data

#five_Day_Centroid(retrieve())


