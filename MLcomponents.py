'''
Place for putting together machine learning functions/analyses.
'''
from scipy.signal import argrelextrema
import numpy as np
import pandas as pd
import datamanipulation
import indicators
import math
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import sys
import tensorflow as tf
from tensorflow import keras

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

#
# def minmax(df, start_date, end_date, calc='close'):
#     '''
#     Function to find extrema in data and signal potential reversal. Labels data with 'buy', 'sell', or 'hold' classes.
#     :param df: dataframe with stock data, indicators optional
#     :param start_date: first data point
#     :param end_date: last data point
#     :param calc: points used to determine extrema
#     :return: dataframes with the extrema and dates
#     '''
#     # Check date, get date range
#     if 'time and date' in df.columns:
#         df = datamanipulation.timeformatter(df)
#     mask = (df['date'] >= start_date) & (df['date'] <= end_date)
#     data = df.loc[mask]
#     # data.set_index('date', inplace=True)
#
#     if calc == 'EMA10':
#         data['EMA10'] = data['EMA10'].shift()
#     inds = data.index.values.tolist()
#     print(inds)
#     # Use scipy kit to find relative extrema
#     local_max = argrelextrema(data['high'].values, np.greater)[0]
#     local_min = argrelextrema(data['low'].values, np.less)[0]
#
#     # Extract high and low values
#     highs = data.iloc[local_max, :]
#     lows = data.iloc[local_min, :]
#
#     # Extract indices
#     highlist = highs.index.values.tolist()
#     lowlist = lows.index.values.tolist()
#
#     # Use indices to get surrounding values
#     truhis = []
#     for ind in highlist:
#         # entry = []
#         indlist = list(range(ind - 1, ind))
#         newlist = []
#         for item in indlist:
#             if item in inds:
#                 newlist.append(item)
#         print(newlist)
#         localvals = data.loc[newlist]
#         # try:
#         #     indlist = range(ind-3, ind+3)
#         #     localvals = data.iloc[indlist]
#         # except:
#         #     localvals = data.iloc[[ind]]
#         # truhi = localvals['high'].max()
#         index = localvals['high'].idxmax()
#         # y = 'sell'
#         # entry.append(index)
#         # entry.append(truhi)
#         # entry.append(y)
#         truhis.append(index)
#
#     trulos = []
#     for ind in lowlist:
#         # entry = []
#         indlist = list(range(ind - 1, ind))
#         newlist = []
#         for item in indlist:
#             if item in inds:
#                 newlist.append(item)
#         print(newlist)
#         localvals = data.loc[newlist]
#         # try:
#         #     indlist = list(range(ind - 3, ind + 3))
#         #     localvals = data.iloc[indlist]
#         # except:
#         #     localvals = data.iloc[[ind]]
#         # trulo = localvals['low'].min()
#         index = localvals['low'].idxmin()
#         # y = 'buy'
#         trulos.append(index)
#         # entry.append(trulo)
#         # entry.append(y)
#         # trulos.append(entry)
#
#     # highclassdf = pd.DataFrame(truhis, columns=['ind', 'label'])
#     # highclassdf.set_index('ind', inplace=True)
#     # lowclassdf = pd.DataFrame(trulos, columns=['ind', 'label'])
#     # lowclassdf.set_index('ind', inplace=True)
#
#     data.loc[truhis, 'y'] = -1
#     data.loc[trulos, 'y'] = 1
#     data.fillna(method='ffill', inplace=True)
#
#     return data


    # print('Highs:')
    # print(highclassdf)
    # print('\nLows:')
    # print(lowclassdf)

    # finaldf = pd.concat([df, highclassdf, lowclassdf], axis='columns')
    # print(finaldf.to_string())

    # return highs, lows

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

    data = datamanipulation.centroid(data)
    minimum_delta = .25
    num_rolling_days = 3
    data['Rolling5'] = data['centroid'].rolling(num_rolling_days).mean()
    data.Rolling5 = data.Rolling5.shift(-1 * num_rolling_days)
    data['Rolling5_Buy'] = data.Rolling5 > (data.Rolling5.shift() + minimum_delta)
    data['Rolling5_Sell'] = data.Rolling5 < (data.Rolling5.shift() - minimum_delta)
    data['Class'] = data.Rolling5_Buy * 1 + data.Rolling5_Sell * (-1)

    # Drop all rows with NaN
    data = data.dropna()
    # Reset row numbers
    data = data.reset_index(drop=True)
    # Remove unneeded columns
    data = data.drop('Rolling5_Buy', axis = 1)
    data = data.drop('Rolling5_Sell', axis = 1)

    for current in data.loc[data['Class'] == 0].index:
        if current != 0:
            data.loc[current, 'Class'] = data.loc[current - 1, 'Class']

    return data

# No time series
def random_forest_shufflesplit(data):
    '''
    Random forest classifier for stock data.
    :param data: dataframe with desired indicators, stock data, etc
    :return:
    '''

    # First step is to divide the data into training, validation, and test sets. Validation will serve as the test set
    # while training the model, test will be the final performance evaluator
    indices = data.index.values.tolist()
    n_entries = len(indices)
    testn = math.ceil(0.8 * n_entries)
    testinds = indices[testn:]
    modelinds = indices[:testn]
    testdf = data.iloc[testinds, :]

    # Now we will go through with the training/testing as normal with what is left. Assign the target values and split
    modeldata = data.iloc[modelinds, :]
    modeldata = cont_trend_label(modeldata, w=0.01)
    modeldata.set_index('date', inplace=True)
    modeldata = modeldata.dropna()
    # print(modeldata.shape)
    x = modeldata.iloc[:, 0:(modeldata.shape[1] - 2)]
    y = modeldata.iloc[:, (modeldata.shape[1] - 1)]
    # xinds = x.index.values.tolist()
    # n = len(xinds)
    # testn = math.ceil(0.75 * n)
    # testinds = xinds[testn:]
    # traininds = xinds[:testn]
    # x = x.reindex()
    # y = y.reindex()
    # x_train = x.iloc[traininds, 0:18]
    # x_test = x.iloc[testinds, 0:18]
    # y_train = y.iloc[traininds, 3]
    # y_test = y.iloc[testinds, 3]
    # to_compare = y_test
    # print(to_compare)
    # x_train.set_index('date', inplace=True)
    # x_test.set_index('date', inplace=True)
    # y_train.set_index('date', inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)



    # Standardize data
    scale = StandardScaler()
    x_train = scale.fit_transform(x_train)
    x_test = scale.transform(x_test)

    # Apply model and predict
    model = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=10)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    # print(predict)
    # modeldata['y prediction'] = predict
    # print(modeldata.iloc[:, 18:19].to_string())
    # newdf = pd.DataFrame()
    # newdf['target'] = to_compare['close']
    # newdf['prediction'] = predict
    # to_compare['close prediction'] = predict
    # print(newdf.to_string())
    # print(predict)

    # Calculate and display metrics
    mse = round(metrics.mean_squared_error(y_test, prediction), 3)
    accuracy = metrics.accuracy_score(y_test, prediction)
    print('Mean Squared Error: ' + str(mse))
    print('Accuracy: ' + str(round(accuracy * 100, 2)) + '%')

    print(x_test)
    print(y_test)
    print(prediction)
    resultdf = pd.DataFrame(data=[x_test, y_test, prediction], columns=['Input Data', 'Class', 'Predicted Class'])
    print(resultdf.to_string())


def random_forest_timesplit(data):
    '''
    Random forest classifier for stock data.
    :param data: dataframe with desired indicators, stock data, etc
    :return:
    '''

    # Assign the target values and split
    modeldata = cont_trend_label(data, w=0.01)
    modeldata.set_index('date', inplace=True)
    modeldata = modeldata.dropna()
    # print(modeldata.shape)
    x = modeldata.iloc[:, 0:(modeldata.shape[1] - 2)]
    y = modeldata.iloc[:, (modeldata.shape[1] - 1)]
    tscv = TimeSeriesSplit(n_splits=10, max_train_size=(math.ceil(0.6 * modeldata.shape[1])))
    n_splits = 10,
    for train_index, test_index in tscv.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Standardize data
    # scale = StandardScaler()
    # x_train = scale.fit_transform(x_train)
    # x_test = scale.transform(x_test)

    # Apply model and predict
    model = RandomForestClassifier()
    # n_estimators = 500, random_state = 42, max_depth = 10
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    # print(predict)
    # modeldata['y prediction'] = predict
    # print(modeldata.iloc[:, 18:19].to_string())
    # newdf = pd.DataFrame()
    # newdf['target'] = to_compare['close']
    # newdf['prediction'] = predict
    # to_compare['close prediction'] = predict
    # print(newdf.to_string())
    # print(predict)

    # Calculate and display metrics
    mse = round(metrics.mean_squared_error(y_test, prediction), 3)
    accuracy = metrics.accuracy_score(y_test, prediction)
    print('Mean Squared Error: ' + str(mse))
    print('Accuracy: ' + str(round(accuracy * 100, 2)) + '%')

    # print(x_test)
    # print(y_test)
    # print(prediction)
    # resultdf = pd.DataFrame(data=[x_test, y_test, prediction], columns=['Input Data', 'Class', 'Predicted Class'])
    # print(resultdf.to_string())


def data_processor(data):
    # Assign the target values and split
    modeldata = cont_trend_label(data, w=0.01)
    modeldata.set_index('date', inplace=True)
    modeldata = modeldata.dropna()
    # print(modeldata.shape)
    x = modeldata.iloc[:, 0:(modeldata.shape[1] - 1)]
    y = modeldata.iloc[:, (modeldata.shape[1] - 1)]
    # print(modeldata.shape[0])
    tscv = TimeSeriesSplit(n_splits=2, max_train_size=(math.ceil(0.6 * modeldata.shape[0])))
    n_splits = 10,
    training = []
    testing = []
    for train_index, test_index in tscv.split(x):
        # print(train_index)
        # print(test_index)
        trainentry = []
        testentry = []
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        trainentry.append(x_train)
        trainentry.append(y_train)
        testentry.append(x_test)
        testentry.append(y_test)
        training.append(trainentry)
        testing.append(testentry)
    return training, testing

# dailydf = datamanipulation.retrieve()
# dailydf = datamanipulation.timeformatter(dailydf)
# print('Random Forest Classifier Results, No Indicators: ')
# random_forest_timesplit(dailydf)
# print('Random Forest Classifier Results, All Indicators: ')
# dailydf = datamanipulation.retrieve()
# dailydf = indicators.all_indicators(dailydf)
# # print(dailydf)
# random_forest_timesplit(dailydf)

dailydf = datamanipulation.retrieve()
dailydf = datamanipulation.timeformatter(dailydf)
dailydf = indicators.all_indicators(dailydf)
training, testing = data_processor(dailydf)
samptrax = training[0][0]
samptray = training[0][1]
samptesx = testing[0][0]
samptesy = testing[0][1]

scaler = StandardScaler()
x_train = scaler.fit_transform(samptrax)
x_test = scaler.transform(samptesx)
y_train = samptray.replace(to_replace=-1, value=0)
y_test = samptesy.replace(to_replace=-1, value=0)
print(y_train)
print(x_train.shape)
input_neurons = x_train.shape[1]
hidden1 = input_neurons * 2
output_neurons = 1

# Set variables
epochs = 250
hidden_act = 'relu'
out_act = 'sigmoid'
loss = 'binary_crossentropy'
optim = 'adam'
metrics = 'accuracy'
batch_size = 10

# Set up model
inputs = keras.Input(shape=(input_neurons,), name='data')
x1 = keras.layers.Dense(hidden1, activation=hidden_act, name='hidden')(inputs)
# x1 = keras.layers.LSTM(32, input_shape=(1, x_train.shape[1]), activation = 'relu', return_sequences=False)(inputs)
x2 = keras.layers.Dropout(.75)(x1)
x3 = keras.layers.Dense(hidden1/4)(x2)
output = keras.layers.Dense(output_neurons, activation=out_act, name='predictions')(x3)

model = keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer=optim, loss=loss, metrics=metrics)
model.fit(x_train, y_train, epochs=epochs)

results = pd.DataFrame(model.predict(x_test), columns=['Predicted Class'])
results.loc[results['Predicted Class'] > 0.5, 'Predicted Class'] = 1
results.loc[results['Predicted Class'] <= 0.5, 'Predicted Class'] = 0
y = pd.DataFrame(y_test)
y.reset_index(inplace=True)
results = pd.concat([y, results], axis=1, ignore_index=True)


def match(x):
    if x[1] == x[2]:
        return 'match'
    else:
        return 'mismatch'


results['Check'] = results.apply(lambda x: match(x), axis=1)
print(results['Check'].value_counts().to_string())
# print('loss, accuracy: ', results)
# print('Training