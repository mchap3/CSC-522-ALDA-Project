'''
Place to build functions and/or test out the program pieces and try joining them together.
'''
import pandas as pd
import numpy as np
import indicators
import datamanipulation
import MLcomponents
import visualization
import Validate_Analyze
import sklearn
import matplotlib as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

dailydf = datamanipulation.retrieve()
weeklydf = datamanipulation.retrieve(timeframe='weekly')
monthlydf = datamanipulation.retrieve(timeframe='monthly')

# Pre-processing
# Get data
data = indicators.all_indicators(datamanipulation.retrieve())
X = data.copy()
data = MLcomponents.cont_trend_label(data)
whole_y = np.array(data['Class'])

# Name the day column
X.index.name = 'day'
# Copy date column for later
date = X['date']
# Get classification via moving_centroid()
# X = moving_centroid(X)
X = MLcomponents.cont_trend_label(X)
# Drop all rows with NaN
X = X.dropna()
# Reset row numbers
X = X.reset_index(drop=True)

y = X['Class']
y = np.array(y)

# Separate post Jan. 1, 2017 for validation of returns
val_start = '2017-01-01'
val_end = '2022-09-16'
val_set = datamanipulation.date_mask(X, val_start, val_end)
# val_set = val_set.reset_index(drop=True)
val_set = val_set.drop('Class', axis=1)
val_set_copy = val_set.copy()
val_set.drop('date', axis=1, inplace=True)
val_set_idx = val_set.index.values.tolist()
val_set_idx = np.array(val_set_idx)

# Normalize val_set
# Get column headers
col_names = val_set.columns
# Normalize val_set
min_max = MinMaxScaler()
val_set = min_max.fit_transform(val_set)
# Create df
val_set = pd.DataFrame(val_set, columns=col_names)

# Training/testing set
X_start = '2002-09-15'
X_end = '2016-12-31'
X = datamanipulation.date_mask(X, X_start, X_end)
X_copy = X.copy()

first_y = X[['Class']]
first_y = np.array(first_y)
X.drop('Class', axis=1, inplace=True)
X.drop('date', axis=1, inplace=True)

# Normalize X
# Get column headers
col_names = X.columns
# Normalize X
min_max = MinMaxScaler()
X = min_max.fit_transform(X)
# Create df
X = pd.DataFrame(X, columns=col_names)

# Add 'date' column back
# X['date'] = date
X_copy = X
# Convert X to numpy
X = X.to_numpy()

max_k = 5
n_splits = 5

# Split into time series
tscv = datamanipulation.split_data(X, n_splits)

# visualization.plot_cv_indices(tscv, 10, X, first_y)
# plt.pyplot.show()

matrix = np.zeros((max_k - 1, n_splits))
matrix2 = np.zeros((max_k - 1, n_splits))
min_train_idx = np.array([])
fold_idx = np.array([])

for k in range(1, max_k):
    # print('k = ', k)
    total_error = 0
    min_acc = 1
    min_k = 0
    min_fold = 0
    min = 1

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], first_y[test_idx]

        train = np.array(X)[train_idx]
        test = np.array(X)[test_idx]

        # print("Fold #", fold)
        min, cm, y_pred = MLcomponents.do_knn(train, train_idx, test, test_idx, first_y, k)

        if min < min_acc:
            min_acc = min
            min_train_idx = train_idx
            min_fold = fold

        total_error += min
        matrix[k - 1, fold] = min

        y = np.array(y)
        min, cm, y_pred = MLcomponents.do_knn(train, train_idx, val_set, val_set_idx, y, k)
        total_error += min

        y_pred = pd.DataFrame(y_pred)
        est_ret = val_set_copy[['open', 'high', 'low', 'close']].copy()

        est_ret.reset_index(inplace=True)
        datamanipulation.mid(est_ret)
        est_ret['Class'] = y_pred.copy()

        # print('Trade based on knn and kfcv.')
        # print(est_ret.head())
        if est_ret.iloc[-1, -1] != est_ret.iloc[-2, -1]:
            est_ret = est_ret.iloc[:-1, :]
        # print(Validate_Analyze.estimate_returns(est_ret))
        # print(cm)

    # print(round(total_error, 2))

matrix = pd.DataFrame(matrix)
matrix['mean'] = matrix.mean(axis=1)
matrix['stdev'] = matrix.std(axis=1)
print(matrix.to_string())

# k = 10 had the highest average accuracy of ~85%
# Training on the first 15 years and testing on the last 5 years.
best_k = 10
data = indicators.all_indicators(datamanipulation.retrieve())

train = datamanipulation.date_mask(data, X_start, X_end)
train.drop('date', axis=1, inplace=True)
# Normalize train
# Get column headers
col_names = train.columns
min_max = MinMaxScaler()
train = min_max.fit_transform(train)
# Create df
train = pd.DataFrame(train, columns=col_names)
train_idx = train.index.values.tolist()

test = datamanipulation.date_mask(data, val_start, val_end)
test.drop('date', axis=1, inplace=True)
test_idx = test.index.values.tolist()

min, cm, y_pred = MLcomponents.do_knn(train, train_idx, test, test_idx, whole_y, best_k)
y_pred = pd.DataFrame(y_pred)
est_ret = test[['open', 'high', 'low', 'close']].copy()
est_ret.reset_index(inplace=True)
datamanipulation.mid(est_ret)
est_ret['Class'] = y_pred.copy()

print('Trade based on knn and kfcv.')
# print(est_ret.head())
if est_ret.iloc[-1, -1] != est_ret.iloc[-2, -1]:
    est_ret = est_ret.iloc[:-1, :]
print(Validate_Analyze.estimate_returns(est_ret))
print(cm)