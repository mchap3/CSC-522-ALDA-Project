"""
Place to collect and summarize experiments/optimization trials for models and such.
"""
import datamanipulation
import pandas as pd
import matplotlib as plt
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


def trend_optimization():
    # Pull in data and fix time
    dailydf = datamanipulation.retrieve()
    dailydf = datamanipulation.timeformatter(dailydf)

    # # Split into 3 ten-year periods
    # range1 = dailydf['date'] < '2012-09-16'
    # range2 = (dailydf['date'] >= '2017-09-16') & (dailydf['date'] < '2017-09-16')