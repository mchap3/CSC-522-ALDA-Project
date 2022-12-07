'''
Contains tools/functions to visualize data, chart elements, and some of the ML outputs.
'''
import pandas as pd
import numpy as np
import indicators
import datamanipulation
# import mplfinance as mpl
from matplotlib import pyplot as plt
import MLcomponents


# dailydf = datamanipulation.retrieve()
# weeklydf = datamanipulation.retrieve(timeframe='weekly')
# monthlydf = datamanipulation.retrieve(timeframe='monthly')
#
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



def line_plot(df, start_date, end_date, calc='centroid', indicator=None):
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
    # data = indicators.ema(data, n=10, calc=calc)

    # classdata = MLcomponents.minmax(data, start_date, end_date, calc='EMA10')


    # classdata.set_index('date', inplace=True)


    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(data[calc])
    ax2.plot(data['Class'], c='r')
    ax1.set_ylabel('SPY Price')
    ax2.set_ylabel('Class')
    ax2.set_yticks(range(-1, 2, 1))
    plt.xticks(rotation=70, ha='right')
    # if indicator == 'EMA3':
    #     plt.data['EMA3']

    plt.show()

    # data.plot(kind='line', y='centroid')
    # plt.show()

def misc_plotter(df, y, start_date, end_date):
    '''
    Function to plot anything you might be interested in vs. time.
    :param df: dataframe with standard stock data
    :param y: item to be plotted. For example - 'close', 'EMA10', 'RSI14', etc.
    :param start_date: first time series point
    :param end_date: last time series point
    :param calc: points chosen to plot
    :return: no return, generates a plot
    '''

    if 'time and date' in df.columns:
        df = datamanipulation.timeformatter(df)
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    data = df.loc[mask]
    data.set_index('date', inplace=True)

    data.plot(kind='line', y=y)
    plt.show()

def account_comparison_plot(idealdf, MLdf, showideal=True):
    if showideal is True:
        plt.plot(idealdf['account value'], label='Ideal')
    plt.plot(MLdf['account value'], label='ML')
    plt.title('Account Growth Comparison Over Time')
    plt.xlabel('Date')
    plt.ylabel('Account Value ($)')
    plt.legend()
    plt.show()
