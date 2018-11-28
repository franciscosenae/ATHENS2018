from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sklearn as sk
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


def plot_contours(X, y, clf, ax=None):
    if ax is None:
        fig, sub = plt.subplots(1, 1)
        ax, = fig.get_axes()

    X0, X1 = X[X.columns[0]], X[X.columns[1]]

    # Make meshgrid
    h = .02
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot contours
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = 1-Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)

    # Plot points
    ax.scatter(X0, X1, c=1-y, cmap=plt.cm.coolwarm, s=100, edgecolors='k')

    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    return ax


def plot_roc(X, y, clf):
    scores = clf.decision_function(X)
    fpr, tpr, thresholds = roc_curve(y, scores)
    return plt.plot(fpr, tpr)


def plot_confusion_matrix(y_true, y_predicted):
    """Plots both the absolute and relative confusion matrix"""
    confusion_matrix = sk.metrics.confusion_matrix(
        y_true, y_predicted)
    # print(confusion_matrix)
    fig, sub = plt.subplots(1, 2)
    g = sns.heatmap(
        confusion_matrix / confusion_matrix.sum(axis=1)[:, None],
        annot=True,
        ax=sub[0],
        cmap='Greens')
    g.set_ylabel('True')
    g.set_xlabel('Predicted')
    g.set_title('Relative values')
    g = sns.heatmap(
        confusion_matrix,
        annot=True,
        ax=sub[1],
        cmap='Greens')
    g.set_ylabel('True')
    g.set_xlabel('Predicted')
    g.set_title('Absolute values')
    return g


def results_on_years(clf, FEATURES_USED, years=[2016, 2017]):
    """Wrapper to present the results on the 2016 and 2017 datasets"""

    if len(FEATURES_USED) == 2:
        fig, sub = plt.subplots(1, 2)

    for i, year in enumerate(years):

        _df = pd.read_csv(f'data/processed/{year}.csv')
        X = _df[FEATURES_USED]
        y = _df.label == 'brushing'

        print(f'Metrics on the {year} data:')
        print_metrics(y, clf.predict(X))

        if len(FEATURES_USED) == 2:
            ax = plot_contours(X, y, clf, ax=sub[i])
            ax.set_title(year)

        plot_confusion_matrix(y, clf.predict(X))


def print_metrics(y_true, y_predicted):
    metrics = {
        'Accuracy': sk.metrics.accuracy_score(y_true, y_predicted),
        'Precision': sk.metrics.precision_score(y_true, y_predicted),
        'Recall': sk.metrics.recall_score(y_true, y_predicted)
    }
    pprint(metrics)
