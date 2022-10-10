'''
Place to build functions and/or test out the program pieces and try joining them together.
'''
import pandas as pd
import indicators
import visualization
import datamanipulation
import datetime as dt

dailydf = datamanipulation.retrieve()
weeklydf = datamanipulation.retrieve(timeframe='weekly')
monthlydf = datamanipulation.retrieve(timeframe='monthly')

print(indicators.ema(dailydf))
