'''
Contains tools/functions to visualize data, chart elements, and some of the ML outputs.
'''
import pandas as pd
import numpy as np
import datamanipulation
import mplfinance as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
cmap_cv = plt.cm.coolwarm

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


def plot_cv_indices(cv, n_splits, X, y, date_col=None):
    """
    Create a sample plot for indices of a cross-validation object.
    Function modified from
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
    """

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=10, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Formatting
    yticklabels = list(range(n_splits))

    if date_col is not None:
        tick_locations = ax.get_xticks()
        tick_dates = [" "] + date_col.iloc[list(tick_locations[1:-1])].astype(str).tolist() + [" "]

        tick_locations_str = [str(int(i)) for i in tick_locations]
        new_labels = ['\n\n'.join(x) for x in zip(list(tick_locations_str), tick_dates)]
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(new_labels)

    ax.set(yticks=np.arange(n_splits) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits + 0.2, -.2])
    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
              ['Testing set', 'Training set'], loc=(1.02, .8))
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
