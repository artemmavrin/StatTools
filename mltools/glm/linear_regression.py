"""Defines the LinearRegression class."""

import numpy as np

from .generalized_linear_model import GeneralizedLinearModel
from mltools.regression import Regressor


class LinearRegression(GeneralizedLinearModel, Regressor):
    """Linear regression via least squares estimation."""

    # The link function for linear regression is the identity function (which is
    # of course its own inverse).
    _inv_link = staticmethod(lambda x: x)

    def fit(self, x, y):
        """Train the linear regression model.

        Parameters
        ----------
        x: array-like
            Feature matrix
        y: array-like
            Target vector

        Returns
        -------
        This LinearRegression instance is returned.
        """
        x = self._preprocess_features(x, training=True)
        y = self._preprocess_target(y)
        self._weights = np.linalg.lstsq(x, y)[0]
        self._fitted = True
        return self

    def predict(self, x):
        """Predict numeric values of the targets corresponding to the data."""
        return self.estimate(x)
