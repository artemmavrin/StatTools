"""Machine learning algorithms implemented in Python."""

import abc

import numpy as np


class Classifier(metaclass=abc.ABCMeta):
    """Abstract base class for classifiers."""

    # Distinct classes---usually to be determined during model fitting
    _classes = None

    def _preprocess_classes(self, target):
        """Extract distinct classes from a target vector.

        This also converts the target vector to numeric indices pointing to the
        class in the `_classes` attribute.
        """
        self._classes, target = np.unique(target, return_inverse=True)
        return target

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the classifier."""
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """Predict class labels from input feature data."""
        pass


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
