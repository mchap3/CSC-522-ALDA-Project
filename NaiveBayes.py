import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datamanipulation
import indicators
import MLcomponents
import accountperformance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, PowerTransformer


def process_data(labeler='cont', all_indicators=True):
    """
    Creates and returns a dataframe of all market data with or without
    indicators and class labels using the passed labeler method.
    :param labeler: 'cont' or '5-day' labeling method
    :param all_indicators: True/False
    :return: dataframe with specified data
    """
    # download data
    data = datamanipulation.retrieve()
    data = datamanipulation.timeformatter(data)
    data['date'] = pd.to_datetime(data.loc[:, 'date'])
    data.set_index('date', inplace=True)
    # calculate indicators
    if all_indicators:
        data = indicators.all_indicators(data)
    data.dropna(inplace=True)
    # label data
    if labeler not in ('cont', '5-day'):
        print("Data processing error: incorrect labeler")
        exit(1)
    # continuous trend labeling
    if labeler == 'cont':
        data = MLcomponents.cont_trend_label(data, w=0.008)
    # 5-day trend labeling
    if labeler == '5-day':
        data_labeled = MLcomponents.five_day_centroid(data)
        data_labeled.set_index(data.iloc[:-3, :].index, inplace=True)
        data = data_labeled.drop(columns=['Rolling5', 'centroid'])
        data.rename(columns={'Buy_Sell': 'class'}, inplace=True)

    return data


def split_data(data):
    """
    Split data into training and testing sets with testing set starting at
    2017-01-01. Attribute sets (X) and labels (y) are separated for each.
    :param data: labeled dataframe
    :return: split data as training/testing sets
    """
    # split into training/testing sets
    training = data['2003-07-01':'2016-12-31']
    testing = data['2017-01-01':]

    # split into X and y sets
    X_train = training.drop(columns='class')
    y_train = training['class']
    X_test = testing.drop(columns='class')
    y_test = testing['class']

    return (X_train, y_train, X_test, y_test)


def tune_NB(pipe, X_train, y_train):
    """
    Find optimal smoothing parameter for Naive Bayes classifier using
    accuracy as the metric for optimization.
    :param pipe: scikit Pipeline
    :param X_train: training set attribute data
    :param y_train: training set label data
    :return: results as a GridSearchCV object
    """
    # define CV method and classifier
    tscv = TimeSeriesSplit(n_splits=10)
    # initialize parameter search values
    parameters = {'gnb__var_smoothing': np.logspace(0, -11, num=100)}
    # specify scoring metrics
    scoring = ['accuracy', 'f1']
    # use GridSearchCV to determine best parameters
    search = GridSearchCV(estimator=pipe, param_grid=parameters, cv=tscv, scoring=scoring, refit='accuracy', n_jobs=-1)
    # grid search to tune var_smoothing
    search.fit(X_train, y_train)

    return search


def predict_NB(pipe, X_train, y_train, X_test):
    """
    Make predictions using the passed optimized model.
    :param pipe: scikit Pipeline
    :param X_train: training set attribute data
    :param y_train: training set label data
    :param X_test: testing set attribute data
    :return: ndarray with prediction values
    """
    # train with whole training set
    pipe.fit(X_train, y_train)
    # predict values for test set
    y_pred = pipe.predict(X_test)
    return y_pred


def metrics_NB(y_test, y_pred):
    """
    Prints confusion matrix metrics from classification predictions.
    :param y_test: ground-truth labels
    :param y_pred: predicted labels
    """
    # create confusion matrix and display
    conf_mat = confusion_matrix(y_test, y_pred, labels=[1, -1])
    # normalize confusion matrix
    conf_mat_std = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    print("Confusion Matrix:\n", conf_mat)
    # print("Confusion Matrix Percentages:\n", conf_mat_std)
    for item in (zip(['Accuracy', 'Precision', 'Recall', 'F1 Score'], [accuracy_score(y_test, y_pred),
                                                                       precision_score(y_test, y_pred),
                                                                       recall_score(y_test, y_pred),
                                                                       f1_score(y_test, y_pred)])):
        print("%s: %.3f" % item)
    disp = ConfusionMatrixDisplay(conf_mat, display_labels=['Buy', 'Sell'])
    disp.plot()
    plt.show()


def get_returns_NB(X_test, y_pred):
    """
    Prints estimated market returns using classification predictions.
    :param X_test: testing set attribute data
    :param y_pred: predicted labels
    """
    # estimate returns
    est_ret = X_test[['open', 'high', 'low', 'close']].copy()
    est_ret = datamanipulation.mid(est_ret)
    est_ret.reset_index(inplace=True)
    est_ret['class'] = y_pred.copy()
    returns = accountperformance.estimate_returns(est_ret)
    print(returns[0], "\n\n")


def compare_transformations_NB(labels='cont'):
    """
    Run experiments to compare which type of data pre-processing transformation
    works best with NB classifier. Each tranformation is optimized for smoothing
    parameter and optimal model is trained and tested.
    :param labels: 'cont' or '5-day' labeling method
    """
    data = process_data(labeler=labels, all_indicators=True)
    X_train, y_train, X_test, y_test = split_data(data)
    pipes = [Pipeline(steps=[('gnb', GaussianNB())]),
             Pipeline(steps=[('trans', StandardScaler()),('gnb', GaussianNB())]),
             Pipeline(steps=[('trans', PowerTransformer(method='yeo-johnson')),('gnb', GaussianNB())])]
    transformations = ['raw', 'standardized', 'yeo-johnson']
    for i, pipe in enumerate(pipes):
        print("Results for %s data transformation:" % transformations[i])
        gs = tune_NB(pipe, X_train, y_train)
        print(gs.best_params_)
        pipe.set_params(**gs.best_params_)
        y_pred = predict_NB(pipe, X_train, y_train, X_test)
        metrics_NB(y_test, y_pred)
        get_returns_NB(X_test, y_pred)


def run_NB_optimization_experiment():
    """
    Run experiment to determine optimal data preprocessing techniques and
    smoothing parameter for NB classification. Labeling methods are compared.
    Prints optimal parameters, confusion matrix metrics, adn summary results tables.
    """
    print("Running NB optimization experiment:\n")
    print("Using continuous trend labeling...")
    compare_transformations_NB()
    print("Using 5-day threshold labeling...")
    compare_transformations_NB(labels='5-day')


def run_NB_final_experiment():
    """
    Run experiment for results of optimized model (yeo-johnson power
    transformation with smoothing parameter = 0.0005) predictions on testing set
    labeled with cont_trend_method. Prints summary tables for confusion
    matrix and return metrics.
    """
    print("Running NB optimized model comparison:\n")
    print("Using all indicators...")
    data = process_data(labeler='cont', all_indicators=True)
    X_train, y_train, X_test, y_test = split_data(data)
    pipe = Pipeline(steps=[('trans', PowerTransformer(method='yeo-johnson')),
                           ('gnb', GaussianNB(var_smoothing=0.0004641588833612782))])
    y_pred = predict_NB(pipe, X_train, y_train, X_test)
    metrics_NB(y_test, y_pred)
    get_returns_NB(X_test, y_pred)

    print("Using no indicators...")
    data = process_data(labeler='cont', all_indicators=False)
    X_train, y_train, X_test, y_test = split_data(data)
    pipe = Pipeline(steps=[('trans', PowerTransformer(method='yeo-johnson')),
                           ('gnb', GaussianNB(var_smoothing=0.0004641588833612782))])
    y_pred = predict_NB(pipe, X_train, y_train, X_test)
    metrics_NB(y_test, y_pred)
    get_returns_NB(X_test, y_pred)

if __name__ == "__main__":
    run_NB_optimization_experiment()
    run_NB_final_experiment()
