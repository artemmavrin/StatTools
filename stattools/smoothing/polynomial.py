"""Fit a polynomial curve through a scatterplot."""

import itertools

import numpy as np

from .smoothing import ScatterplotSmoother
from ..glm.linear import LinearRegression
from ..utils.validation import validate_samples, validate_int


class PolynomialSmoother(LinearRegression, ScatterplotSmoother):
    """Polynomial smoothing (this is a special case of linear regression)."""

    # Degree of the polynomial model.
    deg: int = None

    def __init__(self, deg, standardize=True):
        """Initialize a PolynomialSmoother instance.

        Parameters
        ----------
        deg : int
            Degree of the polynomial model.
        standardize : bool, optional
            Indicates whether the explanatory and response variables should be
            centered to have mean 0 and scaled to have variance 1.
        """
        self.deg = validate_int(deg, "deg", minimum=1)
        super(PolynomialSmoother, self).__init__(standardize=standardize,
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
        x = validate_samples(x, n_dim=1)
        x = np.vander(x, N=(self.deg + 1), increasing=True)[:, 1:]

        # Fit the model
        return super(PolynomialSmoother, self).fit(x=x, y=y, solver=solver,
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
        x = validate_samples(x, n_dim=1)
        x = np.vander(x, N=(self.deg + 1), increasing=True)[:, 1:]

        # Return the model estimate
        return super(PolynomialSmoother, self).estimate(x=x)

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
