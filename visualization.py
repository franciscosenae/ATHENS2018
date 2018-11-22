import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sklearn as sk
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

def _make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def _plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_contours(X, y, clf, ax=None):
    if ax is None:
        fig, sub = plt.subplots(1, 1)
        ax, = fig.get_axes()

    X0, X1 = X[X.columns[0]], X[X.columns[1]]
    xx, yy = _make_meshgrid(X0, X1)

    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    _plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    return ax


def plot_roc(X, y, clf):
    scores = clf.decision_function(X)
    fpr, tpr, thresholds = roc_curve(y, scores)
    return plt.plot(fpr, tpr)


def plot_confusion_matrix(X, y, clf):
    confusion_matrix = sk.metrics.confusion_matrix(y, clf.predict(X))
    df_cm = pd.DataFrame(
        confusion_matrix,
        index = X.columns,
        columns = X.columns)
    return sns.heatmap(
        df_cm,
        annot=True)


def main():
    df = pd.read_csv('data/processed/2018.csv')

    X = df[['std', 'gy']]
    y = df.label == 'brushing'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    clf = svm.SVC(kernel='linear')
    # clf = svm.SVC(gamma='scale', kernel='rbf')
    clf.fit(X_train, y_train)

    fig, sub = plt.subplots(1, 1)
    ax, = fig.get_axes()

    X0, X1 = X[X.columns[0]], X[X.columns[1]]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.show()



if __name__ == '__main__':
    main()
