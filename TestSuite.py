'''
Place to build functions and/or test out the program pieces and try joining them together.
'''
from indicators import *
from datamanipulation import *
from MLcomponents import *
from visualization import *
import sklearn
import matplotlib as plt

dailydf = retrieve()
weeklydf = retrieve(timeframe='weekly')
monthlydf = retrieve(timeframe='monthly')

X = all_indicators(retrieve())
X.index.name = 'day'
X = five_day_centroid(X)

y = X[['Buy_Sell']]
y = np.array(y)
# y.index.name = 'day'
X = X.drop('Buy_Sell', axis=1)
X = np.array(X)
tscv = split_data(X)

plot_cv_indices(tscv, 5, X, y)
# plt.pyplot.show()

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train = np.array(X)[train_idx]
    test = np.array(X)[test_idx]

    print("Fold #", fold)
    do_knn(train, train_idx, test, test_idx, y)

