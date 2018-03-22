"""Linear regression with the elastic net penalty."""

import numbers
import warnings

import matplotlib.pyplot as plt
import numpy as np

from .linear import LinearRegression
from ..generic import Regressor


def _soft_threshold(a, b):
    """Soft-threshold operator."""
    return np.sign(a) * (np.abs(a) - b) * (np.abs(a) > b)


def _enet_loss(x, y, coef, lam, alpha):
    """Loss function for the elastic net."""
    mse = np.mean((y - x.dot(coef)) ** 2)
    l1 = np.sum(np.abs(coef))
    l2 = np.sum(coef ** 2)
    return 0.5 * mse + lam * (alpha * l1 + 0.5 * (1 - alpha) * l2)


def _enet_cd(x, y, coef0, lam, alpha, tol, max_iter, random, seed, callback):
    """Approximate the elastic net estimator for the linear model coefficients
    by coordinate descent.

    References
    ----------
    Jerome Friedman, Trevor Hastie, and Robert Tibshirani. "Regularization Paths
        for Generalized Linear Models via Coordinate Descent". Journal of
        Statistical Software Vol. 33, No. 1, 2010, pp. 1--22. PMCID: PMC2929880
    """
    # Validate `tol` and `max_iter`
    if not isinstance(tol, numbers.Real) or int(max_iter) <= 0:
        raise ValueError("Parameter 'tol' must be a positive float.")
    if not isinstance(max_iter, numbers.Integral) or int(max_iter) <= 0:
        raise ValueError("Parameter 'max_iter' must be a positive integer.")
    tol = float(tol)
    max_iter = int(max_iter)

    # Number of explanatory variables (AKA predictors)
    p = x.shape[1]

    # Initialize a random number generator
    rng = np.random.RandomState(seed)

    # Initialize the coefficient vector
    coef = np.copy(coef0)
    if callback is not None:
        callback(coef)

    # Coordinate descent algorithm
    i = 0
    while True:
        i += 1
        coef_max = 0.0
        coef_update_max = 0.0

        # Cycle through coefficients OR choose coefficients at random
        for j in range(p):
            if random:
                j = rng.randint(p)

            # Previous coefficient at index j
            coef_j_prev = coef[j]

            # Compute the update
            temp = coef[j] + np.mean((y - x.dot(coef)) * x[:, j])
            coef[j] = _soft_threshold(temp, lam * alpha) / 1 + lam * (1 - alpha)
            if callback is not None:
                callback(coef)

            if np.abs(coef[j]) > coef_max:
                coef_max = np.abs(coef[j])
            if np.abs(coef[j] - coef_j_prev) > coef_update_max:
                coef_update_max = np.abs(coef[j] - coef_j_prev)

        # Check for stopping criteria
        if coef_max == 0 or coef_update_max / coef_max < tol or i >= max_iter:
            break

    return coef, i


def _enet_path(x, y, coef0, lam_min, lam_max, n_lam, alpha, tol, max_iter,
               random, seed):
    lambdas = np.geomspace(lam_max, lam_min, n_lam)
    p = x.shape[1]
    path = np.empty((n_lam, p))
    for i, lam in enumerate(lambdas):
        coef0, *_ = _enet_cd(x=x, y=y, coef0=coef0, lam=lam, alpha=alpha,
                             tol=tol, max_iter=max_iter, random=random,
                             seed=seed, callback=None)
        path[i, :] = coef0

    return path, lambdas


class ElasticNet(LinearRegression, Regressor):

    def __init__(self, lam=0.1, alpha=1):
        # Validate `lam`
        if not isinstance(lam, numbers.Real) or float(lam) <= 0:
            raise ValueError("Parameter 'lam' must be a positive float.")

        # Validate 'alpha'
        if (not isinstance(alpha, numbers.Real) or float(alpha) < 0 or
                float(alpha) > 1):
            raise ValueError("Parameter 'alpha' must be a float in [0, 1].")

        self.lam = float(lam)
        self.alpha = float(alpha)
        super(ElasticNet, self).__init__(standardize=True, fit_intercept=True)

    def fit(self, x, y, tol=1e-4, max_iter=1000, random=False, seed=None,
            callback=None, warm_start=True, verbose=False):
        """Fit the elastic net model using coordinate descent.

        Parameters
        ----------
        x : array-like, shape (n, p)
            The explanatory variable matrix (AKA feature matrix or design
            matrix). Columns of `x` correspond to different explanatory
            variables; rows of `x` correspond to different observations of the
            explanatory variables (i.e., n=number of observations, p=number of
            explanatory variables). If `x` is a scalar or one-dimensional array,
            then it is interpreted as a single explanatory variable (i.e., a
            matrix of shape (n, 1)).
        y : array-like, shape (n,)
            The response variable vector (AKA target vector).
        tol : float, optional
            Convergence tolerance.
        max_iter : int, optional
            Number of iterations of coordinate descent to perform.
        random : bool, optional
            If True, the coordinate along which to maximize is selected randomly
            in each iteration. Otherwise, the coordinates are cycled through in
            order.
        seed : int, optional
            Seed for a NumPy RandomState object. Used if `random` is True;
            otherwise ignored.
        callback : callable, optional
            Optional function of the standardized coefficients to call during
            every iteration of the coordinate descent algorithm.
        warm_start : bool, optional
            If True, initialize the coefficient with a previously computed
            coefficient vector (if available).
        verbose : bool, optional
            If True, print some fitting convergence results to stdout.

        Returns
        -------
        self : ElasticNet
            This ElasticNet instance.
        """
        # Validate explanatory and response variables
        x = self._preprocess_x(x=x)
        y = self._preprocess_y(y=y, x=x)

        # Initialize the coefficients
        if warm_start and self.fitted:
            if len(self._coef) == self._p:
                coef0 = self._coef
            else:
                warnings.warn("Cannot perform warm start: existing coefficient "
                              "estimate has an incompatible length.")
                coef0 = np.zeros(self._p)
        else:
            coef0 = np.zeros(self._p)

        # Coordinate descent
        self._coef, n_iter = _enet_cd(x=x, y=y, coef0=coef0, lam=self.lam,
                                      alpha=self.alpha, tol=tol,
                                      max_iter=max_iter, random=random,
                                      seed=seed, callback=callback)

        if verbose:
            # Final value of the loss function of the standardized data
            loss = _enet_loss(x=x, y=y, coef=self._coef,
                              lam=self.lam, alpha=self.alpha)

            # Summarize fit
            print(f"Fitted elastic net (λ: {self.lam}, α: {self.alpha}) with "
                  f"{len(y)} observations, {self._p} predictors")
            print(f"\tNumber of iterations: {n_iter} (tolerance: {tol})")
            print(f"\tFinal (standardized) elastic net loss: {loss:.5f}")

        self.fitted = True
        return self

    def path(self, x, y, lam_min, lam_max, n_lam=50, tol=1e-4, max_iter=1000,
             random=False, seed=None):
        """Return a regularization path for the coefficients accross a grid of
        lambda values.

        Parameters
        ----------
        x : array-like, shape (n, p)
            The explanatory variable matrix.
        y : array-like, shape (n,)
            The response variable vector.
        lam_min : float
            The minimum lambda.
        lam_max : float
            The maximum lambda.
        n_lam : int, optional
            Number of lambdas to consider.
        tol : float, optional
            Convergence tolerance.
        max_iter : int, optional
            Number of iterations of coordinate descent to perform.
        random : bool, optional
            If True, the coordinate along which to maximize is selected randomly
            in each iteration. Otherwise, the coordinates are cycled through in
            order.
        seed : int, optional
            Seed for a NumPy RandomState object. Used if `random` is True;
            otherwise ignored.

        Returns
        -------
        coefs : numpy.ndarray
            Matrix of coefficient paths. Each column is the path of one of the
            coefficients from the biggest to the smallest lambda. Note: these
            are coefficients for the standardized explanatory and response
            variables.
        lambdas : numpy.ndarray
            Discrete lambda values from the biggest to the smallest, spaced
            logarithmically.
        """
        # Validate explanatory and response variables
        x = self._preprocess_x(x=x)
        y = self._preprocess_y(y=y, x=x)

        # Initialize the coefficients
        coef0 = np.zeros(self._p)

        # Return the path
        return _enet_path(x=x, y=y, coef0=coef0, lam_min=lam_min,
                          lam_max=lam_max, n_lam=n_lam, alpha=self.alpha,
                          tol=tol, max_iter=max_iter, random=random, seed=seed)

    def path_plot(self, x, y, lam_min, lam_max, n_lam=50, tol=1e-4,
                  max_iter=1000, random=False, seed=None, ax=None, **kwargs):
        """Draw the regularization path for the elastic net model."""
        path, lambdas = self.path(x=x, y=y, lam_min=lam_min, lam_max=lam_max,
                                  n_lam=n_lam, tol=tol, max_iter=max_iter,
                                  random=random, seed=seed)
        if ax is None:
            ax = plt.gca()

        for i in range(self._p):
            ax.plot(-np.log(lambdas), path[:, i], **kwargs)

        ax.set(title="Regularization Path")
        ax.set(xlabel="$-\log(\lambda)$", ylabel="Standardized Coefficient")
        ax.set(xlim=(-np.log(lam_max), -np.log(lam_min)))
