"""Visualizations of classification results."""

import itertools

import matplotlib.pyplot as plt
import numpy as np

from ..preprocessing import PCA


def pca_label_plot(x, y_true, y_pred, ax=None, pca=None):
    """Reduce data to 2 dimensions and plot predicted labels colored according
    to whether they were correctly classified.

    Parameters
    ----------
    x: array-like
        Feature data matrix.
    y_true: array-like
        Vector of true target labels.
    y_pred: array-like
        Vector of predicted target labels.
    ax: matplotlib axis, optional
        The axis on which to plot the data.
    pca: PCA, optional
        The PCA object used to reduce the data. If not specified, a new PCA
        object will be fitted on the provided data.

    Returns
    -------
    ax: matplotlib axis
        The current matplotlib axis.
    pca: PCA
        The PCA object used to reduce the data.
    """
    if pca is None:
        pca = PCA()
        pca.fit(x)
    elif not isinstance(pca, PCA) or not pca._fitted:
        raise TypeError("Parameter 'pca' must be a fitted PCA object.")

    if ax is None:
        ax = plt.gca()

    x = pca.transform(x, dim=2)

    labels = np.unique(np.concatenate((y_true, y_pred)))
    for label_true, label_pred in itertools.product(labels, labels):
        marker = "$" + str(label_pred) + "$"
        color = "g" if label_pred == label_true else "r"
        idx = (y_true == label_true) & (y_pred == label_pred)
        ax.scatter(x[idx, 0], x[idx, 1], c=color, marker=marker)

    ax.set_xlabel("1st principal axis")
    ax.set_ylabel("2nd principal axis")

    return ax, pca
