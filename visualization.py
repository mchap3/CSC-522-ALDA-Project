'''
Contains tools/functions to visualize data, chart elements, and some of the ML outputs.
'''
import pandas as pd
import numpy as np
import datamanipulation
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import indicators
import MLcomponents
import accountperformance



def account_comparison_plot(idealdf, MLdfs, showideal=False):
    """
    Plotting tool to compare model account values over time against each other and/or the ideal case.
    :param idealdf: dataframe with ideal returns information
    :param MLdfs: dictionary of model dataframes. The dictionary key is used to assign the label for the model's line,
           and the values are pulled from the associated value (dataframe)
    :param showideal: toggle the ideal display. It is nice to not see it, because the ideal performance dwarfs all
           the models.
    :return: no return, generates a plot with everything passed in the MLdf dictionary on it
    """
    if not showideal:
        MLdfs.pop('Ideal')

    fig, ax = plt.subplots()

    plt.title('Account Growth Comparison Over Time')
    plt.xlabel('Date')
    plt.ylabel('Account Value ($)')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    for key, df in MLdfs.items():
        df.reset_index(inplace=True)
        df['date'] = df['date'].astype('datetime64[ns]')
        ax.plot(df['date'], df['account value'], label=key)
    plt.legend()
    plt.tight_layout()
    plt.show()


def project_summary(add_indicators=True, plotideal=False):
    """
    Code to be executed in StockPredictor.py to put together all the result summaries and plots.
    :param models: pass a list of the models you would like to see results for
    :param add_indicators: decide whether you would like to see results with indicators or not
    :param plotideal: decide whether to include the ideal results on the accountperformance plots. False is sort of a
           good idea because it dwarfs performance of all the models
    :return: nothing, prints a summary table and generates plots
    """
    # Prep dailydf
    dailydf = datamanipulation.retrieve()
    dailydf = datamanipulation.timeformatter(dailydf)
    if add_indicators:
        dailydf = indicators.all_indicators(dailydf)

    # Split into original comparison set, training, validation, and testing sets
    original, training, validation, testing = MLcomponents.data_processor(dailydf)

    # Select training folds for final model performance. Training fold 8 was chosen for final runs for most models, as
    # it left a solid gap between training and testing data. KNN was an exception to this rule, as the prices are part
    # of the 'neighbor' features, and with too large a gap, the prices move far away, and all points have a large
    # distance, which undermined model performance.
    x_train = training[8][0]
    y_train = training[8][1]
    x_val = validation[8][0]
    y_val = validation[8][1]

    x_train_KNN = training[9][0]
    y_train_KNN = training[9][1]
    x_val_KNN = validation[9][0]
    y_val_KNN = validation[9][1]

    # Get testing data out
    x_test = testing[0]
    y_test = testing[1]

    # Assemble full dataset for training if you so choose
    # x_train = pd.concat([x_train, x_val])
    # y_train = pd.concat([y_train, y_val])

    # Normalize x_values with StandardScaler
    x_train, x_val, x_test = MLcomponents.SSnormalize(x_train, x_val, x_test)
    x_train_KNN, x_val_KNN, x_test_KNN = MLcomponents.SSnormalize(x_train_KNN, x_val_KNN, x_test)

    # Clean up y values with y_cleaner
    y_train, y_val, y_test = MLcomponents.y_cleaner(y_train, y_val, y_test)
    y_train_KNN, y_val_KNN, y_test_KNN = MLcomponents.y_cleaner(y_train_KNN, y_val_KNN, y_test)

    # Use ML model to make prediction then take predicted values and reattach them to previous data for evaluation.
    # Calculate confusion metrics and account details for each model, one at a time, in case there are any issues with
    # how the model handles the data.
    # Prediction quality is evaluated by confusion matrix. Feel free to go into this code and uncomment the part that
    # plots the confusion matrices if you would like to see them. For the purposes of the final code, it was not
    # necessary to display.

    # RF
    RF_pred = MLcomponents.RF_prediction(x_train, y_train, x_test)
    idealresults, RFresults = MLcomponents.assemble_results(RF_pred, y_test, original)
    RFmetrics = MLcomponents.evaluate_confusion(idealresults, RFresults)
    idealreturns, RFreturns, idealacctdf, RFacctdf = MLcomponents.evaluate_returns(idealresults, RFresults)

    # ANN
    ANN_pred, history = MLcomponents.ANN_prediction(x_train, y_train, x_val, y_val, x_test, verbose=0)
    ANNideal, ANNresults = MLcomponents.assemble_results(ANN_pred, y_test, original)
    ANNmetrics = MLcomponents.evaluate_confusion(ANNideal, ANNresults)
    _, ANNreturns, _, ANNacctdf = MLcomponents.evaluate_returns(ANNideal, ANNresults)

    # KNN
    KNN_pred = MLcomponents.KNN_prediction(x_train_KNN, y_train_KNN, x_test)
    KNNideal, KNNresults = MLcomponents.assemble_results(KNN_pred, y_test, original)
    KNNmetrics = MLcomponents.evaluate_confusion(KNNideal, KNNresults)
    _, KNNreturns, _, KNNacctdf = MLcomponents.evaluate_returns(KNNideal, KNNresults)

    # SVM
    SVM_pred = MLcomponents.SVM_prediction(x_train, y_train, x_test)
    SVMideal, SVMresults = MLcomponents.assemble_results(SVM_pred, y_test, original)
    SVMmetrics = MLcomponents.evaluate_confusion(SVMideal, SVMresults)
    _, SVMreturns, _, SVMacctdf = MLcomponents.evaluate_returns(SVMideal, SVMresults)

    # NB
    NB_pred = MLcomponents.NB_prediction(x_train, y_train, x_test)
    NBideal, NBresults = MLcomponents.assemble_results(NB_pred, y_test, original)
    NBmetrics = MLcomponents.evaluate_confusion(NBideal, NBresults)
    _, NBreturns, _, NBacctdf = MLcomponents.evaluate_returns(NBideal, NBresults)

    # Assemble confusion metrics into nice summary table
    metrics_summary = pd.concat([RFmetrics, ANNmetrics, KNNmetrics, SVMmetrics, NBmetrics], axis=1)
    metrics_summary.columns = ['RF', 'ANN', 'KNN', 'SVM', 'NB']
    print('\n')
    print('\n')
    print('Confusion Metrics:')
    print(metrics_summary)

    # Assemble return metrics into nice summary table
    returns_summary = pd.concat([idealreturns, RFreturns['active'], ANNreturns['active'], KNNreturns['active'],
                                 SVMreturns['active'], NBreturns['active']], axis=1)
    returns_summary.columns = ['summary', 'buy and hold', 'ideal', 'RF', 'ANN', 'KNN', 'SVM', 'NB']
    returns_summary.set_index('summary', inplace=True)
    print('\n')
    print('\n')
    print('Return Metrics:')
    print(returns_summary.to_string())

    # Assemble account value dictionaries to feed to the plotter function
    acctperformance = {'Ideal': idealacctdf, 'RF': RFacctdf, 'ANN': ANNacctdf, 'KNN': KNNacctdf,
                       'SVM': SVMacctdf, 'NB': NBacctdf}

    # Plot account value over time
    account_comparison_plot(idealacctdf, acctperformance, showideal=plotideal)