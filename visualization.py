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

# dailydf = datamanipulation.timeformatter(dailydf)

def candlestick_chart(df, start_date, end_date, indicators=None):
    '''
    Plotting tool to see candlestick chart. Not sure if indicators can easily be added with mplfinance
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



def line_plot(df, start_date, end_date, calc='close', indicators=None):
    '''
    Simple line plot of stock data.
    :param data: stock dataframe
    :param start_date: first time series point displayed
    :param end_date: last time series point displayed
    :param calc: points chosen to plot
    :param indicators: list of indicators to pass, will have to update for this functionality
    :return: doesn't return anything, just generates the plot
    '''
    if 'time and date' in df.columns:
        df = datamanipulation.timeformatter(df)
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    data = df.loc[mask]
    data.set_index('date', inplace=True)

    data.plot(kind='line', y=calc)
    plt.show()

