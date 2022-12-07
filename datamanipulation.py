'''
Contains functions to retrieve data from spreadsheets, put them in dataframes, slice things up, etc.
'''

import pandas as pd
import os


def retrieve(timeframe='daily'):
    '''
    :param timeframe: pass the desired timeframe you want to work with (daily, weekly, monthly, 1 min, 5 min, etc.)
    :return: returns a dataframe with the chosen stock information (price, time, volume)
    Note: currently only accommodates the timeframes with one version of the file
    '''
    path = 'Data'

    for (root, dirs, file) in os.walk(path):
        files = file

    for item in files:
        if timeframe in item:
            data = item
    # print(data)
    filepath = 'Data/' + str(data)

    df = pd.read_excel(filepath)
    return df


def timeformatter(df):
    '''
    :param dataframe: dataframe with time and date data
    :return: dataframe with just the date included for daily/monthly/weekly data in the format yyyy/mm/day
    '''

    # df['time and date'] = pd.to_datetime(df['time and date']).dt.date
    # df.rename(columns={'time and date': 'date'}, inplace=True)
    df['date'] = df['time and date'].dt.strftime('%Y-%m-%d')
    df.drop(columns=['time and date'], inplace=True)

    return df


def mid(data):
    '''
    :param data: stock data to work with
    :return: dataframe with midpoint appended (mean of high and low)
    '''

    data['mid'] = (data['high'] + data['low'])/2

    return data

def center(data):
    '''
    :param data: stock data to work with
    :return: dataframe with center appended (mean of open and close)
    '''

    data['center'] = (data['open'] + data ['close']) / 2

    return data


def centroid(data):
    '''
    :param data: stock data to work with
    :return: dataframe with centroid appended (mean of open, high, low, and close)
    '''

    data['centroid'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4

    return data