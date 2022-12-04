import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datamanipulation
import indicators
import MLcomponents
import Validate_Analyze
from NaiveBayes import *
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, PowerTransformer


def tune_SVM_kernel(pipe, X_train, y_train):
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
    # define CV method and classifier
    tscv = TimeSeriesSplit(n_splits=5)
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
    # train with whole training set
    pipe.fit(X_train, y_train)
    # predict values for test set
    y_pred = pipe.predict(X_test)
    return y_pred


def metrics_SVM(y_test, y_pred):
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
    # estimate returns
    est_ret = X_test[['open', 'high', 'low', 'close']].copy()
    est_ret = datamanipulation.mid(est_ret)
    est_ret.reset_index(inplace=True)
    est_ret['Class'] = y_pred.copy()
    returns = Validate_Analyze.estimate_returns(est_ret)
    print(returns, "\n\n")


def compare_transformations_SVM(labels='cont'):
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


def compare_best_SVM(labels='cont'):
    data = process_data(labeler=labels, all_indicators=True)
    X_train, y_train, X_test, y_test = split_data(data)
    pipes = [Pipeline(steps=[('trans', StandardScaler()), ('svc', SVC())]),
             Pipeline(steps=[('trans', PowerTransformer(method='yeo-johnson')), ('svc', SVC())])]
    transformations = ['standardized', 'yeo-johnson']
    for i, pipe in enumerate(pipes):
        pipe.set_params(svc__kernel='linear')
        print("Best SVM for %s data transformation:" % transformations[i])
        y_pred = predict_SVM(pipe, X_train, y_train, X_test)
        metrics_SVM(y_test, y_pred)
        get_returns_SVM(X_test, y_pred)


if __name__ == "__main__":
    print("Using continuous trend labeling...")
    # compare_transformations_SVM()
    compare_best_SVM()
    print("Using 5-day threshold labeling...")
    # compare_transformations_SVM(labels='5-day')
    compare_best_SVM(labels='5-day')