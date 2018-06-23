"""Variants of the linear model.

References
----------
Seber, George A. F. and Lee, Alan J. (2003) Linear regression analysis. Second
    edition. Wiley Series in Probability and Statistics. xvi+557.
    doi:10.1002/9780471722199
"""

import abc

import numpy as np

from .glm import GLM
from ..generic import Regressor
from ..optimization import Optimizer
from ..optimization.gradient_descent import validate_gd_params


class MSELoss(object):
    """Mean squared error loss function for linear regression:
        L(b) = 0.5 * sum((y - x.dot(b)) ** 2) / n,
    where n is the number of observations.

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
        """Mean squared error loss for the training data."""
        return 0.5 * np.sum((self.y - self.x.dot(coef)) ** 2) / self.n

    def grad(self, coef):
        """Gradient of the mean squared error loss."""
        return self.x.T.dot(self.x.dot(coef) - self.y) / self.n

    def hess(self, _):
        """Hessian of the mean squared error loss."""
        return self.x.T.dot(self.x) / self.n


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

    def fit(self, x, y, names=None, solver="qr", **kwargs):
        """Fit the linear regression model via least squares.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Design matrix consisting of n observations of p explanatory
            variables.
        y : array-like, shape (n,)
            Response vector.
        names : list, optional
            List of feature names corresponding to the columns of `x`.
        solver : None or str or stattools.optimization.Optimizer, optional
            Specify how to estimate the linear regression model coefficients.
            Acceptable values:
                None or "qr" (default):
                    Use the QR factorization of the design matrix.
                "lstsq":
                    This is basically a wrapper for numpy.linalg.lstsq().
                "gd":
                    Use gradient descent to minimize the mean squared error.
                    Acceptable keyword arguments (kwargs):
                    rate, momentum, nesterov, anneal, iterations
                    See mltools.optimization.GradientDescent for descriptions.
                mltools.optimization.Optimizer instance:
                    Specify an optimizer to minimize the MSE loss function.
        kwargs : dict, optional
            If `solver` is "gd", these specify gradient descent parameters rate,
            momentum, nesterov, anneal, and iterations. See
            mltools.optimization.GradientDescent for descriptions.
            If `solver` is an mltools.optimization.Optimizer, these are keyword
            arguments for its optimize() method.

        Returns
        -------
        This LinearRegression instance.
        """
        # Validate input
        x = self._preprocess_features(x=x, names=names)
        y = self._preprocess_response(y=y, x=x)

        if solver is None or solver == "qr":
            # Fit the model using the QR factorization of the design matrix
            q, r = np.linalg.qr(x, mode="reduced")
            self._coef = np.linalg.solve(r, q.T.dot(y))
        elif solver == "lstsq":
            # Fit the model by solving the least squares problem directly
            self._coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        elif solver == "gd":
            # Fit the model by gradient descent
            gd_params = {"rate": 0.1,
                         "momentum": 0.0,
                         "nesterov": False,
                         "anneal": np.inf,
                         "iterations": 1000}
            gd_params.update(kwargs)
            self._coef = _fit_lr_gd(x, y, **gd_params)
        elif isinstance(solver, Optimizer):
            # Minimize the mean squared error loss function
            loss = MSELoss(x, y)
            coef0 = np.zeros(x.shape[1])
            self._coef = solver.optimize(x0=coef0, func=loss, **kwargs)
        else:
            raise ValueError(f"Unknown value for parameter 'solver': {solver}")

        self.fitted = True
        return self


def _fit_lr_gd(x, y, rate, momentum, nesterov, anneal, iterations):
    """Fit a linear regression model using gradient descent.

    This is an alternative to mltools.optimization.GradientDescent that
    eliminates function call overhead.

    Parameters
    ----------
    x : numpy.ndarray
        Design matrix
    y : numpy.ndarray
        Response vector
    rate: float, optional
        Step size/learning rate. Must be positive.
    momentum: float, optional
        Momentum parameter. Must be positive.
    nesterov: bool
        If True, the update rule is Nesterov's accelerated gradient descent.
        If False, the update rule is vanilla gradient descent with momentum.
    anneal: float, optional
        Factor determining the annealing schedule of the learning rate. Must
        be positive. Smaller values lead to faster shrinking of the learning
        rate over time.
    iterations: int, optional
        Number of iterations of the algorithm to perform. Must be positive.
    """
    # Validate parameters
    rate, momentum, nesterov, anneal, iterations \
        = validate_gd_params(rate, momentum, nesterov, anneal, iterations)

    n, p = x.shape

    # Initialize coefficient vector and update vector
    coef = np.zeros(p)
    u = np.zeros(p)

    if nesterov:
        # Nesterov's accelerated gradient descent
        for t in range(iterations):
            step = rate / (1 + t / anneal)
            u_prev = u
            u = momentum * u - step * x.T.dot(x.dot(coef) - y) / n
            coef = coef - momentum * u_prev + (1 + momentum) * u
    elif momentum > 0:
        # Gradient descent with momentum
        for t in range(iterations):
            step = rate / (1 + t / anneal)
            u = momentum * u - step * x.T.dot(x.dot(coef) - y) / n
            coef = coef + u
    else:
        # Vanilla gradient descent
        for t in range(iterations):
            step = rate / (1 + t / anneal)
            coef = coef - step * x.T.dot(x.dot(coef) - y) / n
    return coef
