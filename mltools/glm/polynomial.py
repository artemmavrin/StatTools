"""Polynomial regression for fitting a curve through a 2D scatter plot."""

import numbers

import numpy as np

from .glm import GLM
from ..generic import Regressor
from ..utils import validate_data
from ..visualization import func_plot


class PolynomialRegression(GLM, Regressor):
    """Polynomial regression. This is really a special case of the linear
    regression model, but we just use numpy.polyfit to avoid dealing with
    Vandermonde matrices.
    """

    # Degree of the polynomial model.
    deg: int = None

    # Polynomial function corresponding to the coefficients of the model
    poly: np.poly1d = None

    # Although it isn't used in this implementation, the link function for
    # polynomial regression is the identity function (since polynomial
    # regression is a special case of linear regression)
    _inv_link = staticmethod(lambda x: x)

    def __init__(self, deg):
        """Initialize a PolynomialRegression instance.

        Parameters
        ----------
        deg : int
            Degree of the polynomial model.
        """
        self.standardize = False
        self.fit_intercept = True

        # Validate the degree
        if not isinstance(deg, numbers.Integral) or deg < 1:
            raise ValueError("'deg' must be a positive integer.")
        self.deg = int(deg)

    def fit(self, x, y):
        """Fit the polynomial regression model.

        Parameters
        ----------
        x : array-like, shape (n,)
            Explanatory variable.
        y : array-like, shape (n,)
            Response variable.

        Returns
        -------
        This PolynomialRegression instance is returned.
        """
        # Validate input
        x = validate_data(x, max_ndim=1)
        y = self._preprocess_y(y=y, x=x)

        # Compute the least squares polynomial coefficients
        c = np.polyfit(x=x, y=y, deg=self.deg)
        self.poly = np.poly1d(c)
        self._coef = np.flipud(c)
        self.fitted = True
        return self

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
        # Check whether the model is fitted
        if not self.fitted:
            raise self.unfitted_exception()

        # Validate input
        x = validate_data(x, max_ndim=1)

        return self.poly(x)

    def predict(self, x):
        """Predict the response variable corresponding to the explanatory
        variable.

        Parameters
        ----------
        x : array-like, shape (n, p)
            The explanatory variable.
        """
        return self.estimate(x)

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

        i = 0
        for c in self._coef:
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
            i += 1

        if tex:
            s += "$"

        return s
