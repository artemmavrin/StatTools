"""Visualizations of classification results."""

import itertools

import matplotlib.pyplot as plt
import numpy as np

from ..preprocessing import PCA


def pca_label_plot(x, y_true, y_pred, ax=None, pca=None, marker_dict=None):
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
    marker_dict: dict, optional
        Dictionary of how to transform labels into markers for the scatter plot.

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
    elif not isinstance(pca, PCA) or not pca.fitted:
        raise TypeError("Parameter 'pca' must be a fitted PCA object.")

    if ax is None:
        ax = plt.gca()

    x = pca.transform(x, dim=2)

    if marker_dict is None:
        marker_dict = dict()

    labels = np.unique(np.concatenate((y_true, y_pred)))
    for label_true, label_pred in itertools.product(labels, labels):
        if label_pred in marker_dict:
            marker = marker_dict[label_pred]
        else:
            marker = "$" + str(label_pred) + "$"
        color = "green" if label_pred == label_true else "red"
        zorder = 1 if label_pred == label_true else 2
        idx = (y_true == label_true) & (y_pred == label_pred)
        ax.scatter(x[idx, 0], x[idx, 1], c=color, marker=marker, zorder=zorder)

    ax.set_xlabel("1st Principal Axis")
    ax.set_ylabel("2nd Principal Axis")

    return ax, pca
