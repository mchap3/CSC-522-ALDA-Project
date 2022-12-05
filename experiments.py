"""
Place to collect and summarize experiments/optimization trials for models and such.
"""
import datamanipulation
import pandas as pd
from matplotlib import pyplot as plt
import MLcomponents
import accountperformance
import indicators
from sklearn.metrics import accuracy_score
import math
import numpy as np
import tensorflow as tf


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
    """
    Plots real stock returns for trading based on labeled classification for a range of values passed as the w parameter
    in the labeling algorithm we found. Compares returns over three different 10 year periods to choose returns.
    :return: Nothing, generates a plot
    """
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
    for num in range(2, 50, 1):
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
    for num in range(2, 50, 1):
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
    for num in range(2, 50, 1):
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

    # Get index of maximum returns, find w value with max return, print table of max returns and w
    row1 = summary1[summary1['real return 2002-2012'] == summary1['real return 2002-2012'].max()]
    row2 = summary1[summary1['real return 2007-2017'] == summary1['real return 2007-2017'].max()]
    row3 = summary1[summary1['real return 2011-2021'] == summary1['real return 2011-2021'].max()]
    maxvals = pd.concat([row1, row2, row3])

    print('The maximum returns correspond to the following w parameter values: ')
    print(maxvals.to_string())


def RF_optimization():
    """
    Code to optimize random forest classification based on folds and n_estimators.
    :return: Nothing, prints tables/graphs to show results
    """
    # Prep dailydf
    dailydf = datamanipulation.retrieve()
    dailydf = datamanipulation.timeformatter(dailydf)
    dailydf = indicators.all_indicators(dailydf)

    # Split into original comparison set, training, validation, and testing sets
    original, training, validation, testing = MLcomponents.data_processor(dailydf)

    # Get testing data out
    x_test = testing[0]
    y_test = testing[1]

    # Initialize dfs to store results
    estimatorsummary = pd.DataFrame()
    depthsummary = pd.DataFrame()

    # List of max depths to test
    depthvals = [10, 25, 50, 100, 250, 500, 1000, None]



    # Iterate over each fold
    for i in range(len(training)):
        x_train = training[i][0]
        y_train = training[i][1]
        x_val = validation[i][0]
        y_val = validation[i][1]

        # Initialize counter for depth indexing
        k = 0

        # Then iterate over each value of n_estimators
        for n in range(10, 260, 10):
            j = (n/10) - 1

            # Normalize x_values with StandardScaler
            x_train, x_val, x_test = MLcomponents.SSnormalize(x_train, x_val, x_test)

            # Clean up y values with y_cleaner
            y_train, y_val, y_test = MLcomponents.y_cleaner(y_train, y_val, y_test)

            # Use ML model to make prediction
            y_pred = MLcomponents.RF_prediction(x_train, y_train, x_val, n_estimators=n)

            # Calculate accuracy
            acc = accuracy_score(y_val, y_pred)

            # Store
            estimatorsummary.loc[i, j] = acc


        # Also run a loop for max depth
        for m in depthvals:

            # Normalize x_values with StandardScaler
            x_train, x_val, x_test = MLcomponents.SSnormalize(x_train, x_val, x_test)

            # Clean up y values with y_cleaner
            y_train, y_val, y_test = MLcomponents.y_cleaner(y_train, y_val, y_test)

            # Use ML model to make prediction
            y_pred = MLcomponents.RF_prediction(x_train, y_train, x_val, maxdepth=m)

            # Calculate accuracy
            acc = accuracy_score(y_val, y_pred)

            # Store and update counter
            depthsummary.loc[i, k] = acc
            k += 1

    # Arrange estimator data for plotting
    colnames = estimatorsummary.columns
    newcols = []
    for col in colnames:
        n = (col + 1) * 10
        newcols.append(n)

    estimatorsummary.columns = newcols
    estimatorsummary = estimatorsummary.transpose()

    print('Estimator Results: ')
    print(estimatorsummary.to_string())

    # Arrange depth data for plotting
    depthsummary.columns = depthvals

    print('Depth Results:')
    print(depthsummary.to_string())

    # Generate plot for n_estimators
    plt.figure()
    for i in range(len(estimatorsummary.columns)):
        fold = i + 1
        label = 'Fold ' + str(fold)
        plt.plot(estimatorsummary[i], label=label)
    plt.title('Performance Varying n_estimators')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Generate plot for max_depth
    plt.figure()
    for col in depthsummary.columns:
        if math.isnan(col):
            plt.plot(depthsummary.iloc[:, -1], label='MD: None')
        else:
            depth = int(col)
            label = 'MD: ' + str(depth)
            plt.plot(depthsummary[col], label=label)
    plt.title('Performance Varying maxdepth')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Looking at the data, folds 5 and 6 consistently appear to be the best training set. The results are a little all
    # over the place and maybe not easy to decipher as these two hyperparameters seem to have less of an impact than
    # the folds. At the peak, lower max depth gives better accuracy. There also seem to be peaks in n_estimator
    # performance around 140ish. We will run further trials to compare maxdepth=10 and n_estimator=140 with fold 6 as
    # the training set against the validation set.

def RF_comparison():
    """
    Does a multi-run comparison between default RF parameters and the potential 'optimized' parameters determined from
    the RF_optimize() function.
    :return: Plot with average results over 10 trials.
    """

    # Set up data as before
    dailydf = datamanipulation.retrieve()
    dailydf = datamanipulation.timeformatter(dailydf)
    dailydf = indicators.all_indicators(dailydf)

    # Split into original comparison set, training, validation, and testing sets
    original, training, validation, testing = MLcomponents.data_processor(dailydf)

    # Get testing data out
    x_test = testing[0]
    y_test = testing[1]

    # Take optimal fold
    x_train = training[5][0]
    y_train = training[5][1]
    x_val = validation[5][0]
    y_val = validation[5][1]

    # Initialize data storage
    defaultdf = pd.DataFrame()
    optimaldf = pd.DataFrame()

    # Gather data for default
    for i in range(1, 11, 1):
        # Normalize x_values with StandardScaler
        x_train, x_val, x_test = MLcomponents.SSnormalize(x_train, x_val, x_test)

        # Clean up y values with y_cleaner
        y_train, y_val, y_test = MLcomponents.y_cleaner(y_train, y_val, y_test)

        # Use ML model to make prediction
        y_pred = MLcomponents.RF_prediction(x_train, y_train, x_val)

        # Calculate accuracy
        acc = accuracy_score(y_val, y_pred)

        defaultdf.loc[i, 'accuracy'] = acc

    # Gather data for "optimized" model
    for i in range(1, 11, 1):
        # Normalize x_values with StandardScaler
        x_train, x_val, x_test = MLcomponents.SSnormalize(x_train, x_val, x_test)

        # Clean up y values with y_cleaner
        y_train, y_val, y_test = MLcomponents.y_cleaner(y_train, y_val, y_test)

        # Use ML model to make prediction
        y_pred = MLcomponents.RF_prediction(x_train, y_train, x_val, n_estimators=140, maxdepth=10)

        # Calculate accuracy
        acc = accuracy_score(y_val, y_pred)

        optimaldf.loc[i, 'accuracy'] = acc

    # Calculate means
    defaultmean = round(defaultdf.mean(), 3)
    optimalmean = round(optimaldf.mean(), 3)

    print('Default Results: ')
    print(defaultdf)
    print('Optimal Results: ')
    print(optimaldf)
    print('\n')
    print('Default parameters: accuracy = ' + str(defaultmean))
    print('\n')
    print('Optimal parameters: accuracy = ' + str(optimalmean))

    # Consistently got marginally better performance with the optimized model than the default. Using those parameters
    # for final model.


def ANN_optimization():
    """
    Optimizing ANN for number of nodes and dropout rates.
    :return: Nothing, prints graphs and tables for viewing
    """

    # Set seed for reproducibility
    seed = 2022
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Initialize everything as in other trials
    dailydf = datamanipulation.retrieve()
    dailydf = datamanipulation.timeformatter(dailydf)
    dailydf = indicators.all_indicators(dailydf)

    # Split into original comparison set, training, validation, and testing sets
    original, training, validation, testing = MLcomponents.data_processor(dailydf)

    # Get testing data out
    x_test = testing[0]
    y_test = testing[1]

    # Take fold with most of the available trainign data
    x_train = training[8][0]
    y_train = training[8][1]
    x_val = validation[8][0]
    y_val = validation[8][1]

    # Normalize x_values with StandardScaler
    x_train, x_val, x_test = MLcomponents.SSnormalize(x_train, x_val, x_test)

    # Clean up y values with y_cleaner
    y_train, y_val, y_test = MLcomponents.y_cleaner(y_train, y_val, y_test)

    # Initialize lists with number of neurons and dropout rates
    hidden_neurons = [25, 50, 100, 175, 250, 500]
    dropout1 = [0.1, 0.25, 0.5, 0.75, 0.9]
    dropout2 = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Initialize dictionary to capture dataframes
    results = {}

    # Iterate through the possible values and run the model
    for i in hidden_neurons:
        summary = pd.DataFrame()
        key = 'n Neurons: ' + str(i)
        for j in dropout1:
            count1 = 0
            for k in dropout2:
                count2 = 0

                # Passing it test data because I have to based on the way I set up the function, but only using
                # training and validation data for tuning.
                y_pred, history = MLcomponents.ANN_prediction(x_train, y_train, x_val, y_val, x_test, hidden=i,
                                                              dropout1=j, dropout2=k)

                maxvalacc = max(history.history['val_accuracy'])

                # acc = accuracy_score(y_test, y_pred)
                summary.loc[('Drop1: ' + str(j)), ('Drop2: ' + str(k))] = maxvalacc
        entry = {key: summary}
        results.update(entry)

    # Display tables of results
    for key, value in results.items():
        print(key)
        print(value)

    # Tables were visually inspected for the highest maximum accuracies. The highest validation accuracy achieved was
    # 88.673%, which occurred at the following combinations of neurons and drop rates:
    # 1) 500 neurons, drop1 = 0.25, drop2 = 0.25
    # 2) 100 neurons, drop1 = 0.1, drop2 = 0.5
    # For further testing, 2) was chosen for the reduction in computational complexity with fewer nodes/neurons

def ann_regularization():
    """
    Function to optimize number of training epochs to prevent overfitting.
    :return: Nothing, prints a graph
    """
    # Set seed for reproducibility
    seed = 2022
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Initialize everything as in other trials
    dailydf = datamanipulation.retrieve()
    dailydf = datamanipulation.timeformatter(dailydf)
    dailydf = indicators.all_indicators(dailydf)

    # Split into original comparison set, training, validation, and testing sets
    original, training, validation, testing = MLcomponents.data_processor(dailydf)

    # Get testing data out
    x_test = testing[0]
    y_test = testing[1]

    # Take fold with most of the available trainign data
    x_train = training[8][0]
    y_train = training[8][1]
    x_val = validation[8][0]
    y_val = validation[8][1]

    # Normalize x_values with StandardScaler
    x_train, x_val, x_test = MLcomponents.SSnormalize(x_train, x_val, x_test)

    # Clean up y values with y_cleaner
    y_train, y_val, y_test = MLcomponents.y_cleaner(y_train, y_val, y_test)

    y_pred, history = MLcomponents.ANN_prediction(x_train, y_train, x_val, y_val, x_test, hidden=100,
                                                      dropout1=0.1, dropout2=0.5, epochs=50)

    trainacc = history.history['accuracy']
    valacc = history.history['val_accuracy']
    trainloss = history.history['loss']
    valloss = history.history['val_loss']
    epochs = [i for i in range(1, 51, 1)]

    # Plot results
    plt.figure()
    plt.title('Training and Validation Accuracy vs. # of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epochs, trainacc, label='Training')
    plt.plot(epochs, valacc, label='Validation')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title('Training and Validation Loss vs. # of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss: Binary Crossentropy')
    plt.plot(epochs, trainloss, label='Training')
    plt.plot(epochs, valloss, label='Validation')
    plt.legend()
    plt.tight_layout()
    plt.show()

ann_regularization()