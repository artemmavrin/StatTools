"""Variants of the linear model."""

import abc
import itertools
import numbers

import numpy as np

from .glm import GLM
from ..generic import Regressor
from ..optimization import Optimizer
from ..utils import validate_data
from ..visualization import func_plot


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
            Indicates whether the module should fit an intercept term.
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


class PolynomialRegression(LinearRegression):
    """Polynomial regression (this is really a special case of a linear
    regression model).
    """

    # Degree of the polynomial model.
    deg: int = None

    def __init__(self, deg, standardize=True):
        """Initialize a PolynomialRegression instance.

        Parameters
        ----------
        deg : int
            Degree of the polynomial model.
        standardize : bool, optional
            Indicates whether the explanatory and response variables should be
            centered to have mean 0 and scaled to have variance 1.
        """
        # Validate the degree
        if not isinstance(deg, numbers.Integral) or deg < 1:
            raise ValueError("'deg' must be a positive integer.")
        self.deg = int(deg)

        super(PolynomialRegression, self).__init__(standardize=standardize,
                                                   fit_intercept=True)

    def fit(self, x, y, solver=None, **kwargs):
        """Fit the polynomial regression model.

        Parameters
        ----------
        x : array-like, shape (n,)
            Explanatory variable.
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
        This PolynomialRegression instance is returned.
        """
        # Convert `x` to a Vandermonde matrix.
        x = validate_data(x, max_ndim=1)
        x = np.vander(x, N=(self.deg + 1), increasing=True)[:, 1:]

        # Fit the model
        return super(PolynomialRegression, self).fit(x=x, y=y, solver=solver,
                                                     **kwargs)

    def estimate(self, x):
        """Return the model's estimate for the given input data.

        Parameters
        ----------
        x : array-like, shape (n, )
            Explanatory variable.

        Returns
        -------
        The polynomial model estimate.
        """
        # Convert `x` to a Vandermonde matrix.
        x = validate_data(x, max_ndim=1)
        x = np.vander(x, N=(self.deg + 1), increasing=True)[:, 1:]

        # Return the model estimate
        return super(PolynomialRegression, self).estimate(x=x)

    def fit_plot(self, x_min=None, x_max=None, num=500, ax=None, **kwargs):
        """Plot the polynomial regression curve.

        Parameters
        ----------
        x_min : float, optional
            Smallest explanatory variable observation. If not provided, grabs
            the smallest x value from the given axes.
        x_max : float, optional
            Biggest explanatory variable observation. If not provided, grabs the
            biggest x value from the given axes.
        num : int, optional
            Number of points to plot.
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the plot.
        kwargs : dict, optional
            Additional keyword arguments to pass to plot()

        Returns
        -------
        The matplotlib.axes.Axes object on which the plot was drawn.
        """
        return func_plot(func=self.predict, x_min=x_min, x_max=x_max, num=num,
                         ax=ax, **kwargs)

    def poly_str(self, precision=3, tex=True):
        """Get a string representation of the estimated polynomial model.

        Parameters
        ----------
        precision : int, optional
            Number of decimal places of the coefficients to print.
        tex : bool, optional
            Indicate whether to use TeX-style polynomial representations
            (e.g., "$2 x^{2}$") vs Python-style polynomial representations
            (e.g., "2 * x ** 2")
        """
        if tex:
            s = "$y ="
        else:
            s = "y ="

        for i, c in enumerate(itertools.chain([self.intercept], self.coef)):
            if i == 0:
                s += f" {c:.{precision}f}"
            elif i == 1:
                if tex:
                    s += f" {'+' if c >= 0 else '-'} {abs(c):.{precision}f} x"
                else:
                    s += f" {'+' if c >= 0 else '-'} {abs(c):.{precision}f} * x"
            else:
                if tex:
                    s += f" {'+' if c >= 0 else '-'} "
                    s += f"{abs(c):.{precision}f} x^{{{i}}}"
                else:
                    s += f" {'+' if c >= 0 else '-'} "
                    s += f"{abs(c):.{precision}f} * x ** {i}"

        if tex:
            s += "$"

        return s
