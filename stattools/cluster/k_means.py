"""Implementation of the k-means clustering algorithm.

See Section 14.3.6 of Hastie, Tibshirani, & Friedman (2009) and Section 9.1 of
Bishop (2006).

References
----------
Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of
    Statistical Learning: Data Mining, Inference, and Prediction (Second).
    Springer 2009
Christopher Bishop. Pattern Recognition and Machine Learning. Springer 2006.
"""

import itertools

import numpy as np

from ..generic import Predictor
from ..utils.validation import validate_bool
from ..utils.validation import validate_float
from ..utils.validation import validate_int
from ..utils.validation import validate_samples


class KMeansCluster(Predictor):
    """Partition data into K clusters based on minimizing the
    within-cluster-sum-of-squares loss function.

    Properties
    ----------
    k : int
        The number of clusters.
    centers : numpy.ndarray
        The centers of each cluster.
    standardize : bool
        Indicate whether the input data should be standardized.
    """
    k: int = None
    standardize: bool = None

    # Cluster centers in terms of possibly standardized units
    _centers: np.ndarray = None

    # Vectors of means and standard deviations of the feature matrix columns
    _x_mean: np.ndarray = None
    _x_std: np.ndarray = None

    def __init__(self, k, standardize=True):
        """Initialize a ClusterKMeans object.

        Parameters
        ----------
        k : int
            The number of clusters.
        standardize : bool
            Indicate whether the input data should be standardized.
        """
        # Validate parameters
        self.k = validate_int(k, "k", minimum=2)
        self.standardize = validate_bool(standardize, "standardize")

    @property
    def centers(self):
        """Get the centers of each cluster."""
        # Check if the model is fitted
        if not self.fitted:
            raise self.unfitted_exception
        # Undo standardization if necessary
        if self.standardize:
            return self._x_mean + self._x_std * self._centers
        else:
            return self._centers

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
            center assignments. This can decrease the chance of finding only
            non-global minima of the loss function
        random_state : int or numpy.random.RandomState object, optional
            A valid initializer for a numpy.random.RandomState object.

        Returns
        -------
        This KMeansCluster instance.
        """
        # Validate parameters
        x, tol, iterations, repeats = _validate_fit_params(x, self.k, tol,
                                                           iterations, repeats)

        # Seed the RNG
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        # n = number of observations, p = number of features
        n, p = x.shape

        # Initialize array of cluster centers
        centers = np.empty(shape=(repeats, self.k, p), dtype=np.float_)

        # Initialize array of within-cluster-sum-of-squares losses
        loss = np.empty(shape=repeats)

        # Standardize data if necessary
        if self.standardize:
            self._x_mean = x.mean(axis=0)
            self._x_std = x.std(axis=0, ddof=0)
            if np.any(self._x_std == 0):
                self._x_std[self._x_std == 0] = 1.0
            x = (x - self._x_mean) / self._x_std

        # Repeat the k-means clustering algorithm to decrease the chance of
        # finding only non-global minima of the loss function
        for i in range(repeats):
            # Initialize random starting cluster centers by choosing k
            # observations without replacement from the feature matrix
            ind = random_state.choice(n, size=self.k, replace=False)
            centers_ = x.take(ind, axis=0)

            if iterations is None:
                counter = itertools.count()
            else:
                counter = range(iterations)

            # Perform the k-means clustering algorithm (an EM algorithm)
            for _ in counter:
                # Save the current centers for later
                centers_old = centers_

                # (E step) Assign each observation to a cluster
                clusters = _assign_clusters(x, centers_)

                # (M step) Get new centers as the means of the currently
                # assigned clusters
                centers_ = np.zeros(shape=(self.k, p), dtype=np.float_)
                for j in range(self.k):
                    centers_[j, :] = np.mean(x[clusters == j, :], axis=0)

                # Check for convergence
                if np.linalg.norm(centers_ - centers_old) < tol:
                    break

            # Save the resulting centers and compute final cluster assignments
            centers[i] = centers_
            clusters = _assign_clusters(x, centers_)

            # Evaluate the clusters by their within-cluster-sum-of-squares
            loss[i] = _k_means_loss(x, clusters=clusters, centers=centers_)

        # Choose the cluster center assignment that minimizes the loss
        self._centers = centers[np.argmin(loss)]

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

        # Standardize data if necessary
        if self.standardize:
            x = (x - self._x_mean) / self._x_std

        # Return cluster assignments
        return _assign_clusters(x, self._centers)


def _k_means_loss(x: np.ndarray, clusters, centers):
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


def _validate_fit_params(x, k, tol, iterations, repeats, ):
    """Validate the parameters for KMeansCluster.fit().

    Parameters
    ----------
    See the parameter descriptions for GaussianMixture.fit().

    Returns
    -------
    The updated parameters.
    """
    x = validate_samples(x, n_dim=2)
    if len(x) <= k:
        print("There must be more observations than number of clusters.")

    tol = validate_float(tol, "tol", positive=True)

    if iterations is not None:
        iterations = validate_int(iterations, "iterations", minimum=1)

    repeats = validate_int(repeats, "repeats", minimum=1)

    return x, tol, iterations, repeats
