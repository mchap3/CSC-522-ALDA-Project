'''
A file for backtesting results, analyzing our data outputs, etc.
'''


import datamanipulation
import indicators
import MLcomponents
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

dailydf = datamanipulation.retrieve()
dailydf = datamanipulation.centroid(dailydf)
dailydf = datamanipulation.center(dailydf)
dailydf = datamanipulation.mid(dailydf)
dailydf = datamanipulation.timeformatter(dailydf)

start_date = '2002-01-01'
end_date = '2012-01-01'
mask = (dailydf['date'] >= start_date) & (dailydf['date'] <= end_date)
data = dailydf.loc[mask]
# print(data.to_string())
# check = MLcomponents.cont_trend_label(data, w=0.02)
# print(check.to_string())
# data = MLcomponents.cont_trend_label(data, w=0.02)
# data = MLcomponents.five_day_centroid(data)


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
    data['trade'] = data['Class'].ne(data['Class'].shift()).cumsum()
    df = data.groupby('trade')
    trades = []
    for name, value in df:
        trades.append(value)

    # Now basically run the same things as the base case in estimate returns here, keep everything in a list of lists
    tradesummaries = []
    i = 1
    for trade in trades:
        # print('Trade: ' + str(i))
        entry = []
        indices = trade.index.values.tolist()
        # print(indices)
        # cl = trade.loc[[indices[0]], 'Class']
        # if cl.values[0] == 1.0:
        #     buy = True
        #     print('Buy')
        # else:
        #     buy = False
        #     print('Sell')
        maxindex = max(data.index.values.tolist())
        # if len(indices) == 1 & (indices[0] + 1) < maxindex:
        if len(indices) == 1:
            trade.loc[indices[0] + 1] = data.loc[indices[0] + 1]
            indices = trade.index.values.tolist()
        # print(indices)

        # Determine class to do calcs for long vs short
        cl = trade.loc[[indices[0]], 'Class']
        if cl.values[0] == 1.0:
            buy = True
            # print('Buy')
        else:
            buy = False
            # print('Sell')

        if buy:
            buyprice = data.loc[indices[0], 'mid']
            # print('Entry Price: ' + str(buyprice))
            nshares = math.floor(initial_capital / buyprice)
            # print('Shares Traded: ' + str(nshares))
            buyval = round(buyprice * nshares, 2)
            # print('Trade Entry Cost: ' + str(buyval))
            sellprice = data.loc[indices[len(indices) - 1], 'mid']
            # print('Exit Price: ', str(sellprice))
            sellval = round(nshares * sellprice, 2)
            # print('Trade Closing Value: ' + str(sellval))
            realreturn = round(sellval - buyval, 2)
            # print('Realized Return: ', str(realreturn))
            pctreturn = round(((sellval - buyval) / buyval) * 100, 2)
            # print('Percentage Return: ', str(pctreturn))
            mdd = maxdrawdown_long(trade)
            # print('Max Drawdown: ', str(mdd))
            final_capital = round(initial_capital + realreturn, 2)
            entry.append(initial_capital)
            entry.append(final_capital)
            entry.append(realreturn)
            entry.append(pctreturn)
            entry.append(mdd)
            initial_capital = final_capital
            # print('Capital Remaining: ', str(initial_capital))
        if not buy:
            sellprice = data.loc[indices[0], 'mid']
            # print('Entry Price: ' + str(sellprice))
            nshares = math.floor((initial_capital / 2) / sellprice)
            # print('Shares Traded: ' + str(nshares))
            sellval = round(nshares * sellprice, 2)
            # print('Trade Entry Cost: ' + str(sellval))
            buyprice = data.loc[indices[len(indices) - 1], 'mid']
            # print('Exit Price: ', str(buyprice))
            buyval = round(nshares * buyprice, 2)
            # print('Trade Closing Value: ' + str(buyval))
            realreturn = round(sellval-buyval, 2)
            # print('Realized Return: ', str(realreturn))
            pctreturn = round(((sellval - buyval) / buyval) * 100, 2)
            # print('Percentage Return: ', str(pctreturn))
            mdd = maxdrawdown_short(trade)
            # print('Max Drawdown: ', str(mdd))
            final_capital = round(initial_capital + realreturn, 2)
            entry.append(initial_capital)
            entry.append(final_capital)
            entry.append(realreturn)
            entry.append(pctreturn)
            entry.append(mdd)
            initial_capital = final_capital
            # print('Capital Remaining: ', str(initial_capital))
        tradesummaries.append(entry)
        i += 1
    return tradesummaries


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
    activedata = active_return(data)
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

    return summarydf


# cont = estimate_returns(MLcomponents.cont_trend_label(data, w=0.1), method='cont_trend_label')
# mov = estimate_returns(MLcomponents.five_day_centroid(data), method='five_day_centroid')
# cont['active: five_day_centroid'] = mov['active: five_day_centroid']
# cont.rename(columns={'summary': 'summary: w=0.1'}, inplace=True)
# cont.set_index('summary: w=0.1', inplace=True)
# print(cont.to_string())
# print(mov)
#
start1 = '2002-01-01'
end1 = '2012-01-01'
start2 = '2007-01-01'
end2 = '2017-01-01'
start3 = '2011-01-01'
end3 = '2021-01-01'
mask1 = (dailydf['date'] >= start1) & (dailydf['date'] <= end1)
mask2 = (dailydf['date'] >= start2) & (dailydf['date'] <= end2)
mask3 = (dailydf['date'] >= start3) & (dailydf['date'] <= end3)
data1 = dailydf.loc[mask1]
data2 = dailydf.loc[mask2]
data3 = dailydf.loc[mask3]

results = []
for num in range(5, 50, 1):
    entry = []
    w = num / 1000
    entry.append(w)
    data1 = MLcomponents.cont_trend_label(data1, w=w)
    summaries = estimate_returns(data1)
    entry.append(summaries.iloc[0, 2])
    results.append(entry)

summary1 = pd.DataFrame(results, columns=['w parameter', 'real return 2002-2012'])

results = []
for num in range(5, 50, 1):
    entry = []
    w = num / 1000
    entry.append(w)
    data2 = MLcomponents.cont_trend_label(data2, w=w)
    summaries = estimate_returns(data2)
    entry.append(summaries.iloc[0, 2])
    results.append(entry)

summary2 = pd.DataFrame(results, columns=['w parameter', 'real return 2007-2017'])

results = []
for num in range(5, 50, 1):
    entry = []
    w = num / 1000
    entry.append(w)
    data3 = MLcomponents.cont_trend_label(data3, w=w)
    summaries = estimate_returns(data3)
    entry.append(summaries.iloc[0, 2])
    results.append(entry)

summary3 = pd.DataFrame(results, columns=['w parameter', 'real return 2011-2021'])

summary1['real return 2007-2017'] = summary2['real return 2007-2017']
summary1['real return 2011-2021'] = summary3['real return 2011-2021']
summary1.set_index('w parameter', inplace=True)

fig, ax1 = plt.subplots()
ax1.plot(summary1['real return 2002-2012'])
ax1.plot(summary1['real return 2007-2017'])
ax1.plot(summary1['real return 2011-2021'])
ax1.set_ylabel('Real Return ($)')
ax1.set_xlabel('w')
plt.title('Real Decade Stock Returns vs. w Parameter')
plt.legend(['2002-2012', '2007-2017', '2011-2021'])
plt.show()

# print(summary1.to_string())
# summaries = estimate_returns(data, w=0.02)
# print(results)


