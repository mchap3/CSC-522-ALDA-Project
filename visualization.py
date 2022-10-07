'''
Contains tools/functions to visualize data, chart elements, and some of the ML outputs.
'''
import pandas as pd

import datamanipulation
import mplfinance as mpl
from matplotlib import pyplot as plt

dailydf = datamanipulation.retrieve()
weeklydf = datamanipulation.retrieve(timeframe='weekly')
monthlydf = datamanipulation.retrieve(timeframe='monthly')

dailydf = datamanipulation.timeformatter(dailydf)

def candlestick_chart(df, start_date, end_date, indicators=None):
    '''
    :param start_date: pass desired start date of stock data in format 'yyyy-mm-dd'
    :param end_date: same as above, but for end date
    :return: generates a plot, does not return any variables
    '''

    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    data = df.loc[mask]
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    mpl.plot(data,
             type='candle',
             title='SPY Price',
             style='yahoo')
    # print(data)


candlestick_chart(dailydf, start_date='2008-01-01', end_date='2010-01-01')
