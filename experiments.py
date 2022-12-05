"""
Place to collect and summarize experiments/optimization trials for models and such.
"""
import datamanipulation
import pandas as pd
from matplotlib import pyplot as plt
import MLcomponents
import accountperformance


# 1) Initial labelling evaluation: 5-day centroid vs cont_trend_label
def label_comparison():
    """
    Compares peformance between 5-day centroid labeling scheme we developed and the cont_trend label scheme from
    literature. Specifie a 10-year period from the beginning of the data set, and tries three values for the weighting
    parameter reported in the paper we found. Parameter values = w, 2*w, and 0.5*w.
    :return: no return, prints a table with the final results
    """
    # Pull in data and fix time
    dailydf = datamanipulation.retrieve()
    dailydf = datamanipulation.timeformatter(dailydf)
    dailydf = datamanipulation.mid(dailydf)

    # Get test data
    range = dailydf['date'] < '2012-01-01'
    dailydf = dailydf.loc[range]

    # Assign labels and evaluate returns for each method
    centroid_returns = accountperformance.estimate_returns(MLcomponents.five_day_centroid(dailydf))[0]
    centroid_returns.rename(columns={'active': '5-Day'}, inplace=True)
    default_trend_returns = accountperformance.estimate_returns(MLcomponents.cont_trend_label(dailydf, calc='close',
                                                                                              w=0.05))[0]
    hi_w_trend_returns = accountperformance.estimate_returns(MLcomponents.cont_trend_label(dailydf, calc='close',
                                                                                           w=0.1))[0]
    lo_w_trend_returns = accountperformance.estimate_returns(MLcomponents.cont_trend_label(dailydf, calc='close',
                                                                                           w=0.025))[0]

    # Compile results
    results = centroid_returns
    results['Trend (w=0.1)'] = hi_w_trend_returns['active']
    results['Trend (w=0.05)'] = default_trend_returns['active']
    results['Trend (w=0.025)'] = lo_w_trend_returns['active']

    print(results.to_string())


# 2) Optimize cont_trend_label for best returns by varying w parameter
def trend_optimization():
    # Pull in data and fix time
    dailydf = datamanipulation.retrieve()
    dailydf = datamanipulation.timeformatter(dailydf)
    dailydf = datamanipulation.mid(dailydf)

    # Set up masks, split into 3 10 year periods
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

    # Run w loop for set 1
    results = []
    for num in range(5, 50, 1):
        entry = []
        w = num / 1000
        entry.append(w)
        data1 = MLcomponents.cont_trend_label(data1, w=w)
        summaries = accountperformance.estimate_returns(data1)[0]
        entry.append(summaries.iloc[0, 2])
        results.append(entry)

    summary1 = pd.DataFrame(results, columns=['w parameter', 'real return 2002-2012'])

    # Run w loop for set 2
    results = []
    for num in range(5, 50, 1):
        entry = []
        w = num / 1000
        entry.append(w)
        data2 = MLcomponents.cont_trend_label(data2, w=w)
        summaries = accountperformance.estimate_returns(data2)[0]
        entry.append(summaries.iloc[0, 2])
        results.append(entry)

    summary2 = pd.DataFrame(results, columns=['w parameter', 'real return 2007-2017'])

    # Run w loop for set 3
    results = []
    for num in range(5, 50, 1):
        entry = []
        w = num / 1000
        entry.append(w)
        data3 = MLcomponents.cont_trend_label(data3, w=w)
        summaries = accountperformance.estimate_returns(data3)[0]
        entry.append(summaries.iloc[0, 2])
        results.append(entry)

    summary3 = pd.DataFrame(results, columns=['w parameter', 'real return 2011-2021'])

    # Add all the data to the first dataframe, set w as index
    summary1['real return 2007-2017'] = summary2['real return 2007-2017']
    summary1['real return 2011-2021'] = summary3['real return 2011-2021']
    summary1.set_index('w parameter', inplace=True)

    # Plot results
    fig, ax1 = plt.subplots()
    ax1.plot(summary1['real return 2002-2012'])
    ax1.plot(summary1['real return 2007-2017'])
    ax1.plot(summary1['real return 2011-2021'])
    ax1.set_ylabel('Real Return ($)')
    ax1.set_xlabel('w')
    plt.title('Real Decade Stock Returns vs. w Parameter')
    plt.legend(['2002-2012', '2007-2017', '2011-2021'])
    plt.show()

trend_optimization()