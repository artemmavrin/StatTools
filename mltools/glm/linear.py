"""Variants of the linear model."""

import abc

import numpy as np

from .glm import GLM
from ..generic import Regressor
from ..optimization import Optimizer


class MSELoss(object):
    """Mean squared error loss function for linear regression.

    Minimizing this loss function is equivalent to maximizing the likelihood
    function of the linear regression model.
    """

    def __init__(self, x, y):
        """Initialize with the training data.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable.
        y : array-like, shape (n, )
            Response variable.
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.n = len(self.x)

    def __call__(self, coef):
        """Compute the mean squared error loss for the training data."""
        residuals = self.x.dot(coef) - self.y
        return residuals.dot(residuals) / self.n

    def grad(self, coef):
        """Compute the gradient of the mean squared error loss."""
        return 2 * self.x.T.dot(self.x.dot(coef) - self.y) / self.n

    def hess(self, _):
        """Compute the Hessian of the mean squared error loss."""
        return 2 * self.x.T.dot(self.x) / self.n


class LinearModel(GLM, Regressor, metaclass=abc.ABCMeta):
    """Abstract base class for linear models."""

    # The link function for linear models is the identity function (which is
    # of course its own inverse)
    _inv_link = staticmethod(lambda x: x)

    def __init__(self, standardize=True, fit_intercept=True):
        """Initialize a LinearModel object.

        Parameters
        ----------
        standardize : bool, optional
            Indicates whether the explanatory and response variables should be
            centered to have mean 0 and scaled to have variance 1.
        fit_intercept : bool, optional
            Indicates whether the model should fit an intercept term.
        """
        self.standardize = standardize
        self.fit_intercept = fit_intercept

    def predict(self, x):
        """Predict the response variable corresponding to the explanatory
        variable.

        Parameters
        ----------
        x : array-like, shape (n, p)
            The explanatory variable.
        """
        return self.estimate(x)


class LinearRegression(LinearModel):
    """Classical linear regression via least squares/maximum likelihood
    estimation.
    """

    def fit(self, x, y, solver=None, **kwargs):
        """Fit the linear regression model via least squares.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variables.
        y : array-like, shape (n,)
            Response variable.
        solver : None or Optimizer, optional
            Specify how to estimate the linear regression model coefficients.
            None:
                Ordinary least squares estimation. This is basically a wrapper
                for numpy.linalg.lstsq().
            Optimizer instance:
                Specify an Optimizer to minimize the MSE loss function.
        kwargs : dict, optional
            If `solver` is an Optimizer, these are additional keyword arguments
            for its optimize() method. Otherwise, these are ignored.

        Returns
        -------
        This LinearRegression instance.
        """
        # Validate input
        x = self._preprocess_x(x=x)
        y = self._preprocess_y(y=y, x=x)

        if solver is None:
            # Fit the model by least squares
            self._coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        elif isinstance(solver, Optimizer):
            # Minimize the mean squared error loss function
            loss = MSELoss(x, y)
            coef0 = np.zeros(x.shape[1])
            self._coef = solver.optimize(x0=coef0, func=loss, **kwargs)
        else:
            raise ValueError(f"Unknown value for parameter 'solver': {solver}")

        self.fitted = True
        return self
