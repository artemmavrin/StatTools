"""Polynomial regression for fitting a curve through a 2D scatter plot."""

import itertools
import numbers

import numpy as np

from .linear import LinearRegression
from ..utils import validate_data
from ..visualization import func_plot


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
        # Convert `x` to a Vandermonde matrix.
        x = validate_data(x, max_ndim=1)
        x = np.vander(x, N=(self.deg + 1), increasing=True)[:, 1:]
        
        # Fit the model
        return super(PolynomialRegression, self).fit(x=x, y=y)

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
