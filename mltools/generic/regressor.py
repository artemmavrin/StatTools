"""Defines the Regressor abstract base class."""

import abc


class Regressor(metaclass=abc.ABCMeta):
    """Abstract base class for regressors."""

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the regressor."""
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """Predict numeric values from input feature data."""
        pass
