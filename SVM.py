import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datamanipulation
import indicators
import MLcomponents
import accountperformance
from NaiveBayes import *
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, PowerTransformer


def tune_SVM_kernel(pipe, X_train, y_train):
    """
    Determine optimal kernel to use for SVM classifier. Kernels tested:
    linear, polynomial, RBF, sigmoid. Each tested with TimeSeriesCV under
    default kernel parameter conditions.
    :param pipe: scikit Pipeline
    :param X_train: training set attribute data
    :param y_train: training set label data
    :return: results as dataframe and GridSearchCV object
    """
    # define CV method and classifier
    tscv = TimeSeriesSplit(n_splits=5)
    # initialize parameter search values
    parameters = {'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    # specify scoring metrics
    scoring = ['accuracy', 'f1']
    # use GridSearchCV to determine best parameters
    search = GridSearchCV(estimator=pipe, param_grid=parameters, cv=tscv, scoring=scoring, refit='accuracy', n_jobs=-1)
    # grid search to tune SVM kernel
    search.fit(X_train, y_train)
    # return grid search results
    results = pd.DataFrame(search.cv_results_)

    return results, search


def tune_SVM_linear(pipe, X_train, y_train):
    """
    Determine optimal C parameter for linear kernel SVM classifier.
    :param pipe: scikit Pipeline
    :param X_train: training set attribute data
    :param y_train: training set label data
    :return: results as dataframe and GridSearchCV object
    """
    # define CV method and classifier
    tscv = TimeSeriesSplit(n_splits=10)
    # initialize parameter search values
    parameters = [{'svc__kernel': ['linear'], 'svc__C': [0.1, 1, 10, 100]}]
    # specify scoring metrics
    scoring = ['accuracy', 'f1']
    # use GridSearchCV to determine best parameters
    search = GridSearchCV(estimator=pipe, param_grid=parameters, cv=tscv, scoring=scoring, refit='accuracy', n_jobs=-1)
    # grid search to tune C parameter
    search.fit(X_train, y_train)
    # return grid search results
    results = pd.DataFrame(search.cv_results_)

    return results, search


def predict_SVM(pipe, X_train, y_train, X_test):
    """
    Make predictions using the passed optimized SVM classifier.
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


def metrics_SVM(y_test, y_pred):
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


def get_returns_SVM(X_test, y_pred):
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


def compare_transformations_SVM(labels='cont'):
    """
    Run experiments to compare which type of data pre-processing transformation
    works best with SVM classifier. Each transformation is optimized for kernel
    selection and linear kernel hyperparameter.
    :param labels: 'cont' or '5-day' labeling method
    """
    data = process_data(labeler=labels, all_indicators=True)
    X_train, y_train, X_test, y_test = split_data(data)
    pipes = [Pipeline(steps=[('trans', StandardScaler()),('svc', SVC())]),
             Pipeline(steps=[('trans', PowerTransformer(method='yeo-johnson')),('svc', SVC())])]
    transformations = ['standardized', 'yeo-johnson']
    for i, pipe in enumerate(pipes):
        results, gs = tune_SVM_kernel(pipe, X_train, y_train)
        print("Optimizations for %s data transformation:" % transformations[i])
        print(gs.best_params_)
        print(results)
        results, gs = tune_SVM_linear(pipe, X_train, y_train)
        print(gs.best_params_)
        print(results)


def compare_best_SVM(labels='cont', indicators=True):
    """
    Run experiments to compare which type of data pre-processing transformation
    works best with SVM classifier. The optimized linear kernel SVM classifier
    is trained and tested and compared for each transformation method.
    :param labels: 'cont' or '5-day' labeling method
    """
    data = process_data(labeler=labels, all_indicators=indicators)
    X_train, y_train, X_test, y_test = split_data(data)
    pipes = [Pipeline(steps=[('trans', StandardScaler()), ('svc', SVC())])]#,
             # Pipeline(steps=[('trans', PowerTransformer(method='yeo-johnson')), ('svc', SVC())])]
    transformations = ['standardized', 'yeo-johnson']
    best_params = [0.1, 0.1]
    for i, pipe in enumerate(pipes):
        pipe.set_params(svc__kernel='linear', svc__C=best_params[i])
        print("Best SVM for %s data transformation:" % transformations[i])
        y_pred = predict_SVM(pipe, X_train, y_train, X_test)
        metrics_SVM(y_test, y_pred)
        get_returns_SVM(X_test, y_pred)


def run_SVM_optimization_experiment():
    """
    Run experiment to determine optimal kernel and linear kernel parameters
    for SVM classification. Data preprocessing techniques (standardization and
    labeling methods) are compared. Prints optimal paramaters and summary
    results table.
    """
    print("Using continuous trend labeling...")
    compare_transformations_SVM()
    print("Using 5-day threshold labeling...")
    compare_transformations_SVM(labels='5-day')


def run_SVM_final_experiment():
    """
    Run experiment for results of optimized model (linear kernel
    with C=0.1) predictions on testing set. Compare results using
    all indicators and no indicators. Prints summary tables for confusion
    matrix and return metrics.
    """
    print("Using all indicators...")
    compare_best_SVM()
    print("Using no indicators...")
    compare_best_SVM(indicators=False)

if __name__ == "__main__":
    run_SVM_optimization_experiment()
    run_SVM_final_experiment()
