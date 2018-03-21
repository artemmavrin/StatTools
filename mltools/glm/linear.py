"""Classical linear regression using least squares estimation."""

import numpy as np

from .glm import GLM
from ..generic import Regressor


class LinearRegression(GLM, Regressor):
    """Linear regression via least squares/maximum likelihood estimation."""

    # The link function for linear regression is the identity function (which is
    # of course its own inverse)
    _inv_link = staticmethod(lambda x: x)

    def __init__(self, standardize=True, fit_intercept=True):
        """Initialize a LinearRegression object.

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

    def fit(self, x, y):
        """Fit the linear regression model via least squares.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variables.
        y : array-like, shape (n,)
            Response variable.

        Returns
        -------
        This LinearRegression instance.
        """
        # Validate input
        x = self._preprocess_x(x=x)
        y = self._preprocess_y(y=y, x=x)

        # Fit the model by least squares
        self._coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        self.fitted = True
        return self

    def predict(self, x):
        """Predict the response variable corresponding to the explanatory
        variable.

        Parameters
        ----------
        x : array-like, shape (n, p)
            The explanatory variable.
        """
        return self.estimate(x)
