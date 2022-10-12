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

def five_Day_Centroid():
    daily = retrieve()
    daily = centroid(daily)
    daily['Rolling5'] = daily['centroid'].rolling(5).mean()
    daily['Rolling5_Buy'] = daily.Rolling5 > (daily.Rolling5.shift() + 0.1)
    daily['Rolling5_Sell'] = daily.Rolling5 < (daily.Rolling5.shift() - 0.1)
    daily['Buy_Sell'] = daily.Rolling5_Buy * 1 + daily.Rolling5_Sell * (-1)

    print(daily[['centroid', 'Rolling5', 'Rolling5_Buy', 'Rolling5_Sell', 'Buy_Sell']].head(15))

    return daily

five_Day_Centroid()


