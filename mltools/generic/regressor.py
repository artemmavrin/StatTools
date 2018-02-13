"""Defines the Regressor abstract base class."""

import abc

from .fittable import Fittable


class Regressor(Fittable, metaclass=abc.ABCMeta):
    """Abstract base class for regressors."""

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """Predict numeric values from input feature data."""
        pass
