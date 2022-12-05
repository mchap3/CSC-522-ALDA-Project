'''
A file for backtesting results, analyzing our data outputs, etc.
'''


import datamanipulation
import indicators
# import MLcomponents
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



def maxdrawdown_long(data):
    '''
    Calculates max drawdown for a trading period if you bought shares at midpoint start. Excludes last day, as the trade
    would be exited by midpoint if the mdd occurs on that day.
    :param data: dataframe with stock data
    :return: max drawdown
    '''
    data = data.iloc[:-1, :]
    inds = data.index.values.tolist()
    window = len(inds)
    rollingmax = data['high'].rolling(window, min_periods=1).max()
    dailydown = data['low'] / rollingmax - 1
    mddframe = dailydown.rolling(window, min_periods=1).min()
    mddlong = round(mddframe.min() * 100, 2)

    return mddlong

def maxdrawdown_short(data):
    '''
    Same as the mdd_long, except opposite for a short sale. I.e. borrow shares and sell at a peak, buy back cheaper
    at a trough
    :param data: same as long
    :return: same as long
    '''

    # print(data['date'])
    data = data.iloc[:-1, :]
    inds = data.index.values.tolist()
    window = len(inds)
    rollingmin = data['low'].rolling(window, min_periods=1).min()
    # print('Rolling Min:\n')
    # print(rollingmin)
    # print('\n')
    dailyup = data['high'] / rollingmin - 1
    # print('DailyUp:\n')
    # print(dailyup)
    # print('\n')
    mddframe = dailyup.rolling(window, min_periods=1).max()
    # print('MDD Frame:\n')
    # print(mddframe)
    # print('\n')
    mddshort = round(mddframe.max() * 100, 2) * -1
    # print('MDD Short:\n')
    # print(mddshort)
    # print('\n')

    return mddshort


def active_return(data):
    '''
    Support function for 'estimate_returns.' Handles the return for active trading over period available and adds it
    to summary dictionary in estimate_returns. If there is only one day in a trade, assume that it is detected around
    mid price and immediately dumps current position and initiates opposite.
    :param data: Same data received by estimate_returns
    :return: list that basically serves as the dictionary entry for the active trading return summary, with one
    additional entry that gives the available capital when entering each trade.
    '''
    initial_capital = 10000
    # First thing is to split the classed dataframe into trades
    data['trade'] = data['class'].ne(data['class'].shift()).cumsum()
    maxindex = max(data.index.values.tolist())
    df = data.groupby('trade')
    trades = []
    for name, value in df:
        trades.append(value)

    # Now basically run the same things as the base case in estimate returns here, keep everything in a list of lists
    tradesummaries = []
    acctdata = []
    i = 1

    # Iterate over each trade and determine results for each of them
    for trade in trades:

        entry = []
        indices = trade.index.values.tolist()

        # Handle special cases first, mostly arising from a single day being assigned a particular class
        if len(indices) == 1:
            # Handle the first case of there being a single trade, but more data to follow
            if indices[0] < maxindex:
                trade.loc[indices[0] + 1, :] = data.loc[indices[0] + 1, :]
                indices = trade.index.values.tolist()
                # print(trade)
            # Then handle the case where the index value would exceed the available data if looking for a place to exit
            # the trade. This isn't a great way to handle, but it basically negates the trade.
            if indices[0] == maxindex:
                # idx = indices[0]
                # trade.loc[indices[0] + 1] = trade.loc[indices[0]]
                # trade.reset_index(inplace=True)
                # trade.rename(index={0: idx, 1: idx + 1})
                # indices = trade.index.values.tolist()
                break

        # print(trade.to_string())

        # Determine class to do calcs for long vs short
        cl = trade.loc[[indices[0]], 'class']

        if cl.values[0] == 1.0:
            buy = True
        else:
            buy = False

        # Perform buy-side trade calculations
        if buy:
            buyprice = data.loc[indices[0], 'mid']
            nshares = math.floor(initial_capital / buyprice)
            buyval = round(buyprice * nshares, 2)
            cashbal = initial_capital - buyval
            sellprice = data.loc[indices[len(indices) - 1], 'mid']
            sellval = round(nshares * sellprice, 2)
            realreturn = round(sellval - buyval, 2)
            pctreturn = round(((sellval - buyval) / buyval) * 100, 2)
            mdd = maxdrawdown_long(trade)
            final_capital = round(initial_capital + realreturn, 2)
            entry.append(initial_capital)
            entry.append(final_capital)
            entry.append(realreturn)
            entry.append(pctreturn)
            entry.append(mdd)
            trade['initial value'] = initial_capital
            trade['account value'] = (trade['close'] * nshares) + cashbal
            trade.loc[trade.index.values[len(trade.index.values) - 1], 'account value'] = final_capital
            trade['cash balance'] = cashbal
            trade['nshares'] = nshares
            # print(trade.to_string())
            acctdataentry = trade.get(['date', 'account value', 'trade'])
            initial_capital = final_capital

        # Perform sell-side trade calculations
        if not buy:
            sellprice = data.loc[indices[0], 'mid']
            nshares = math.floor((initial_capital / 2) / sellprice)
            sellval = round(nshares * sellprice, 2)
            cashbal = initial_capital - sellval
            buyprice = data.loc[indices[len(indices) - 1], 'mid']
            buyval = round(nshares * buyprice, 2)
            realreturn = round(sellval-buyval, 2)
            pctreturn = round(((sellval - buyval) / buyval) * 100, 2)
            mdd = maxdrawdown_short(trade)
            final_capital = round(initial_capital + realreturn, 2)
            entry.append(initial_capital)
            entry.append(final_capital)
            entry.append(realreturn)
            entry.append(pctreturn)
            entry.append(mdd)
            trade['initial value'] = initial_capital
            trade['change'] = trade['close'].diff()
            trade.loc[indices[0], 'change'] = trade.loc[indices[0], 'close'] - trade.loc[indices[0], 'mid']
            trade['P/L'] = trade.change * nshares * -1
            trade['account value'] = (trade['close'] * nshares) + cashbal
            trade['cash balance'] = cashbal
            trade['nshares'] = nshares

            # Calculate next account value for short trade
            for i in range(0, len(trade.index.values)):
                if i == 0:
                    trade.loc[indices[i], 'actual account value'] = trade.loc[indices[i], 'account value'] + \
                                                                    trade.loc[indices[i], 'P/L']
                elif (i > 0) and (i < (len(trade.index.values) - 1)):
                    trade.loc[indices[i], 'actual account value'] = trade.loc[indices[i-1], 'actual account value'] + \
                                                                    trade.loc[indices[i], 'P/L']
                else:
                    trade.loc[indices[i], 'actual account value'] = final_capital

            acctdataentry = trade.get(['date', 'actual account value', 'trade'])
            acctdataentry.rename(columns={'actual account value': 'account value'}, inplace=True)
            initial_capital = final_capital

        acctdata.append(acctdataentry)
        tradesummaries.append(entry)
        i += 1

    return tradesummaries, acctdata


def estimate_returns(data):
    '''
    Calculates return over the provided time frame based on the switches between buy and sell.
    Assumptions:
        1) Initial capital is $10,000
        2) Whenever the buy/sell signal flips, entire position is liquidated, and the full dollar amount available goes
        to the new short or long position.
        3) No fees (pretty commmon these days)
        4) The bot probably can't perfectly time the top or bottom, so entries will be at the midpoint of the candle
        for any day a trade is taken.
        5) A position is initiated at the very beginning of the time frame based on the class there.
        6) If shorting, only use 50% capital to be able to escape the trade if it goes against.
    :param data: Stock data with ohlc, center/mid/centroid, and Class data. Only works if the entire date range
    provided is classed.
    :return: new dataframe with performance metrics (total return, avg return per trade, max gain, max drawdown, etc.)
    '''

    initial_capital = 10000
    summary = {}
    summary['summary'] = ['total return ($)', 'return (%)', 'max drawdown (%)', 'avg. gain per trade (%)']
    # Calculate the simple buy and hold return over the chosen time period, add to dict.
    # print(data.to_string())
    inds = data.index.values.tolist()
    simplebuy = data.loc[inds[0], 'mid']
    nshares = math.floor(initial_capital / simplebuy)
    simplesell = data.loc[inds[len(inds) - 1], 'mid']
    buyval = nshares * simplebuy
    sellval = nshares * simplesell
    realreturn = round(sellval - initial_capital, 2)
    pctreturn = round(((sellval - initial_capital) / initial_capital) * 100, 2)
    mdd = maxdrawdown_long(data)
    summary['buy and hold'] = [realreturn, pctreturn, mdd, 'NA']

    # Call active_return to get data for each trade, aggregate that into a timeframe stat here
    activedata, acctdata = active_return(data)
    # print(activedata)
    realreturn = round(activedata[-1][1] - activedata[0][0], 2)
    pctreturn = round(((activedata[-1][1] - activedata[0][0]) / activedata[0][0]) * 100, 2)
    mddlist = []
    for item in activedata:
        mddlist.append(item[-1])
    mdd = min(mddlist)
    gains = 0
    for item in activedata:
        gains += item[3]
    avg_gain = round(gains/len(activedata), 2)
    keyname = 'active'
    summary[keyname] = [realreturn, pctreturn, mdd, avg_gain]

    summarydf = pd.DataFrame.from_dict(summary)
    acctvaldf = pd.concat(acctdata)
    acctvaldf.set_index('date', inplace=True)


    return summarydf, acctvaldf


# cont = estimate_returns(MLcomponents.cont_trend_label(data, w=0.1), method='cont_trend_label')
# mov = estimate_returns(MLcomponents.five_day_centroid(data), method='five_day_centroid')
# cont['active: five_day_centroid'] = mov['active: five_day_centroid']
# cont.rename(columns={'summary': 'summary: w=0.1'}, inplace=True)
# cont.set_index('summary: w=0.1', inplace=True)
# print(cont.to_string())
# print(mov)
#


# print(summary1.to_string())
# summaries = estimate_returns(data, w=0.02)
# print(results)


