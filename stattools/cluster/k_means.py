"""Implementation of the k-means clustering algorithm.

See Section 14.3.6 of Hastie, Tibshirani, & Friedman (2009).

References
----------
Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of
    Statistical Learning: Data Mining, Inference, and Prediction (Second).
    Springer 2009
"""

import itertools
import numbers

import numpy as np

from ..generic import Predictor
from ..utils import validate_samples


class ClusterKMeans(Predictor):
    """Partition data into K clusters based on minimizing the
    within-cluster-sum-of-squares loss function.

    Properties
    ----------
    k : int
        The number of clusters.
    centers : numpy.ndarray
        The centers of each cluster.
    """
    k: int = None
    centers: np.ndarray = None

    def __init__(self, k):
        """Initialize a ClusterKMeans object.

        Parameters
        ----------
        k : int
            The number of clusters.
        """
        if isinstance(k, numbers.Integral) and int(k) > 1:
            self.k = int(k)
        else:
            raise ValueError("Parameter 'k' must be an integer greater than 1.")

    def fit(self, x, tol=1e-5, iterations=None, repeats=5, random_state=None):
        """Fit the K-means cluster model to data.

        Parameters
        ----------
        x : array-like
            Feature matrix to determine the clustering.
        tol : float, optional
            Positive tolerance for algorithm convergence.
            If the Euclidean distance between the centers in successive
            iterations is less than `tol`, the algorithm stops.
        iterations : int, optional
            Maximum number of iterations to perform. If None, no maximum number
            is imposed.
        repeats : int, optional
            Number of times to repeat the algorithm with different initial
            center assignments.
        random_state : int or numpy.random.RandomState object, optional
            A valid initializer for a numpy.random.RandomState object.

        Returns
        -------
        This ClusterKMeans instance.
        """
        # Validate parameters
        x = validate_samples(x, n_dim=2)
        if len(x) <= self.k:
            print("There must be more observations than number of clusters.")
        if isinstance(tol, numbers.Real) and float(tol) > 0:
            tol = float(tol)
        else:
            raise ValueError("Parameter 'tol' must be a positive float.")
        if iterations is None:
            pass
        elif isinstance(iterations, numbers.Integral) and int(iterations) > 0:
            iterations = int(iterations)
        else:
            raise ValueError("Parameter 'iterations' must be a positive int.")
        if isinstance(repeats, numbers.Integral) and int(repeats) > 0:
            repeats = int(repeats)
        else:
            raise ValueError("Parameter 'repeats' must be a positive integer.")

        # Seed the RNG
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        # n = number of observations, p = number of features
        n, p = x.shape

        # Initialize array of cluster centers
        centers = np.empty(shape=(repeats, self.k, p), dtype=np.float_)

        # Initialize array of within-cluster-sum-of-squares losses
        loss = np.empty(shape=repeats)

        for i in range(repeats):
            # Initialize random initial cluster centers by choosing k
            # observations with replacement from the feature matrix
            ind = random_state.choice(n, size=self.k, replace=False)
            # Run the k-means algorithms starting at the initial cluster centers
            centers[i], clusters = _fit_clusters(x, centers=x.take(ind, axis=0),
                                                 tol=tol, iterations=iterations)
            # Evaluate the clusters by their within-cluster-sum-of-squares
            loss[i] = _within_cluster_sum_of_squares(x, clusters=clusters,
                                                     centers=centers[i])

        # Choose the cluster center assignment that minimizes the loss
        self.centers = centers[np.argmin(loss)]

        self.fitted = True
        return self

    def predict(self, x):
        """Assign each observation to one of K clusters.

        Parameters
        ----------
        x : array-like
            Feature matrix of shape (n, p) to cluster.

        Returns
        -------
        clusters : numpy.ndarray
            Array of shape (n, ) in which the i-th entry is the index of the
            cluster of the i-th observation x[i, :].
        """
        # Check if the model is fitted
        if not self.fitted:
            raise self.unfitted_exception

        # Validate the feature matrix
        x = validate_samples(x, n_dim=2)

        # Return cluster assignments
        return _assign_clusters(x, self.centers)


def _within_cluster_sum_of_squares(x: np.ndarray, clusters, centers):
    """Compute the within-cluster-sum-of-squares loss function.

    Parameters
    ----------
    x : numpy.ndarray
        Feature matrix of shape (n, p) to cluster.
    clusters : numpy.ndarray
        Array of shape (n, ) in which the i-th entry is the index of the cluster
        of the i-th observation x[i, :].
    centers : numpy.ndarray
        Array of shape (k, p) of the centers of each of the k clusters.

    Returns
    -------
    The within-cluster-sum-of-squares loss (cf. equation (14.31) in Hastie,
    Tibshirani, & Friedman (2009)).
    """
    loss = 0.0
    for i, center in enumerate(centers):
        n = np.sum(clusters == i)
        loss += n * np.sum(np.linalg.norm(x[clusters == i] - center, axis=1))
    return loss


def _assign_clusters(x: np.ndarray, centers: np.ndarray):
    """Given cluster centers, assign data points to clusters.

    Parameters
    ----------
    x : numpy.ndarray
        Feature matrix of shape (n, p) to cluster.
    centers : numpy.ndarray
        Array of shape (k, p) of the centers of each of the k clusters.

    Returns
    -------
    clusters : numpy.ndarray
        Array of shape (n, ) in which the i-th entry is the index of the cluster
        of the i-th observation x[i, :].
    """
    clusters = np.empty(shape=len(x), dtype=np.int_)
    for i, obs in enumerate(x):
        clusters[i] = np.argmin(np.linalg.norm(obs - centers, axis=1))
    return clusters


def _fit_clusters(x: np.ndarray, centers: np.ndarray, tol, iterations):
    """Perform the k-means clustering algorithm.

    Parameters
    ----------
    x : numpy.ndarray
        Feature matrix of shape (n, p) to cluster.
    centers : numpy.ndarray
        Array of shape (k, p) of the initial centers of each of the k clusters.
    tol : float
        Positive tolerance for algorithm convergence. If the Euclidean distance
        between the centers in successive iterations is less than `tol`, the
        algorithm stops early.
    iterations : int or None
        Maximum number of iterations to perform. If None, no maximum number is
        imposed.
    """
    # n = number of observations, p = number of features
    n, p = x.shape

    # Number of clusters
    k = len(centers)

    counter = itertools.count() if iterations is None else range(iterations)
    for _ in counter:
        # Assign each observation to a cluster
        clusters = _assign_clusters(x, centers)

        # Save the current centers for later
        centers_old = centers

        # Get new centers as the means of the currently assigned clusters
        centers = np.zeros(shape=(k, p), dtype=np.float_)
        for j in range(k):
            centers[j, :] = np.mean(x[clusters == j, :], axis=0)

        # Check for convergence
        if np.linalg.norm(centers - centers_old) < tol:
            break

    return centers, clusters
