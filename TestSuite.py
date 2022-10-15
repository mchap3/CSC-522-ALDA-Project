'''
Place to build functions and/or test out the program pieces and try joining them together.
'''
from indicators import *
from datamanipulation import *

dailydf = retrieve()
weeklydf = retrieve(timeframe='weekly')
monthlydf = retrieve(timeframe='monthly')

data = five_day_centroid(dailydf)
knn(data, 5)