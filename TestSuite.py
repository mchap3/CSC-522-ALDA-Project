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
y.index.name = 'day'
tscv = split_data(X)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

plot_cv_indices(tscv,5, X, y)
plt.pyplot.show()


