"""Implements the generic Fittable mixin class."""

import abc


class UnfittedModelException(Exception):
    """Exception raised when a model is used before being fitted."""

    def __init__(self, model):
        message = f"This {model.__class__.__name__} object is not fitted."
        super(UnfittedModelException, self).__init__(message)


class Fittable(metaclass=abc.ABCMeta):
    """Abstract mixin class for objects implementing a fit() method."""
    # Indicator for whether this object has been fitted.
    _fitted = False

    @abc.abstractmethod
    def fit(self, *arg, **kwargs):
        """Fit this object to data."""
        raise NotImplementedError()

    def is_fitted(self):
        """Return whether this object has been fitted."""
        return self._fitted

    def unfitted_exception(self):
        """Return an unfitted model exception for this fittable object."""
        return UnfittedModelException(self)
