'''
Place for putting together machine learning functions/analyses.
'''

from indicators import *
from sklearn.neighbors import KNeighborsClassifier

def cont_trend_label(df, calc='close', w=0.05):
    """
    Labels class as [1, -1] (up/down) based on continuous trend
    algorithm from https://www.mdpi.com/1099-4300/22/10/1162/htm

    :param df: dataframe with stock price data
    :param calc: data attribute for trend calculations
    (default: 'close')
    :param w: proportion threshold parameter (default: 0.05)
    :return: dataframe with Class label added
    """
    # initialize
    X = df[calc].to_numpy()
    FP = X[0]   # first price
    xH = X[0]   # highest price
    HT = -1     # time of highest price
    xL = X[0]   # lowest price
    LT = -1     # time of lowest price
    Cid = 0     # current trend label
    FP_N = 0    # initial high/low price
    y = np.zeros(len(X))    # array to store labels

    # find first high/low
    for i in range(len(X)):
        if X[i] > FP + X[0] * w:
            xH, HT, FP_N, Cid = X[i], i, i, 1
            break
        if X[i] < FP - X[0] * w:
            xL, LT, FP_N, Cid = X[i], i, i, -1
            break

    # process labels
    for i in range(FP_N + 1, len(X)):
        if Cid > 0:
            if X[i] > xH:
                xH, HT = X[i], i
            if X[i] < xH - xH * w and LT <= HT:
                for j in range(len(y)):
                    if j > LT and j <= HT:
                        y[j] = 1
                xL, LT, Cid = X[i], i, -1
        if Cid < 0:
            if X[i] < xL:
                xL, LT = X[i], i
            if X[i] > xL + xL * w and HT <= LT:
                for j in range(len(y)):
                    if j > HT and j <= LT:
                        y[j] = -1
                xH, HT, Cid = X[i], i, 1

    # process any remaining points at end
    for j in range(1, len(y)):
        if y[j] == 0:
            y[j] = Cid

    # add labels to dataframe
    df['Class'] = y

    return df


def five_day_centroid(data):
    '''
    Runs a 5-day moving average of the Centroid and then classifies as 1 (buy) or -1 (sell)
    if the 5-day average changed by more than the threshold.
    :param data: stock data to work with
    :return: dataframe with columns for five-day average, buy, sell, and Buy_Sell appended
    '''

    data = centroid(data)
    minimum_delta = .25
    num_rolling_days = 3
    data['Rolling5'] = data['centroid'].rolling(num_rolling_days).mean()
    data.Rolling5 = data.Rolling5.shift(-1 * num_rolling_days - 1)
    data['Rolling5_Buy'] = data.Rolling5 > (data.Rolling5.shift() + minimum_delta)
    data['Rolling5_Sell'] = data.Rolling5 < (data.Rolling5.shift() - minimum_delta)
    data['Buy_Sell'] = data.Rolling5_Buy * 1 + data.Rolling5_Sell * (-1)

    # Drop all rows with NaN
    data = data.dropna()
    # Reset row numbers
    data = data.reset_index(drop=True)
    # Remove unneeded columns
    data = data.drop('date', axis=1)
    data = data.drop('Rolling5_Buy', axis=1)
    data = data.drop('Rolling5_Sell', axis=1)

    for current in data.loc[data['Buy_Sell'] == 0].index:
        if current != 0:
            data.loc[current, 'Buy_Sell'] = data.loc[current - 1, 'Buy_Sell']

    return data

def split_data(data, n_splits=5):
    '''
    :param data: stock data to work with
    :return: tscv
    '''
    from sklearn.model_selection import TimeSeriesSplit
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits)

    return tscv


def do_knn(train, train_idx, test, test_idx, y, k=3):

    max = 0

    for k in range(1, 11):
        # create KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

        # train KNN classifier
        knn.fit(train, y[train_idx].ravel())

        # test set predictions
        y_pred = knn.predict(test)

        # determine error
        y_pred = y_pred.reshape(805, 1)
        e = y[test_idx] != y_pred

        error = e.sum() / len(test)
        if error > max:
            max = error
            max_k = k
        print('error:', round(error, 3))

    print('max:', round(max, 3), 'k = ', max_k)