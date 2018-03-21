"""Linear regression with the elastic net penalty."""

import numbers
from itertools import cycle

import numpy as np

from .linear import LinearRegression
from ..generic import Regressor


def _soft_threshold(a, b):
    """Soft-threshold operator."""
    return np.sign(a) * (np.abs(a) - b) * (np.abs(a) > b)


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

    def fit(self, x, y, iterations=1000, callback=None):
        """Fit the elastic net model using cyclical coordinate descent.

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
        iterations : int, optional
            Number of iterations of coordinate descent to perform.
        callback : callable, optional
            Optional function of the standardized coefficients to call during
            every iteration of the coordinate descent algorithm.

        Returns
        -------
        self : ElasticNet
            This ElasticNet instance.
        """
        # Validate explanatory and response variables
        x = self._preprocess_x(x=x)
        y = self._preprocess_y(y=y, x=x)

        # Validate `iterations`
        if not isinstance(iterations, numbers.Integral) or int(iterations) <= 0:
            raise ValueError("Number of iterations must be a positive integer.")
        iterations = int(iterations)

        # Initialize coefficients for the standardized linear model
        self._coef = np.zeros(self._p)
        if callback is not None:
            callback(self._coef)

        # Cyclical coordinate descent
        for j, _ in zip(cycle(range(self._p)), range(iterations)):
            r = y - x.dot(self._coef)
            a = self._coef[j] + np.mean(r * x[:, j])
            b = self.lam * self.alpha
            c = 1 + self.lam * (1 - self.alpha)
            self._coef[j] = _soft_threshold(a, b) / c
            if callback is not None:
                callback(self._coef)

        self.fitted = True
        return self
