"""Defines the generic Fittable mixin class."""

import abc


class UnfittedException(Exception):
    """Exception raised when a Fittable object is used before being fitted."""

    def __init__(self, obj, message=None):
        """Initialize the exception.

        Parameters
        ----------
        obj : Fittable
            The object being improperly used before fitting.
        message : str, optional
            Error message.
        """
        if message is None:
            message = f"This {obj.__class__.__name__} object is not fitted."
        super(UnfittedException, self).__init__(message)


class Fittable(metaclass=abc.ABCMeta):
    """Abstract mixin class for objects implementing a fit() method.

    Properties
    ----------
    fitted : bool
        Indicates whether this object has been fitted.
    """

    fitted: bool = False

    @abc.abstractmethod
    def fit(self, *arg, **kwargs):
        """Fit this object to data. This function should return self."""
        return self

    @property
    def unfitted_exception(self) -> UnfittedException:
        """Return an exception to be raised when a post-fitting method is called
        before fitting.

        Parameters
        ----------
        message : str, optional
            Error message.
        """
        return UnfittedException(obj=self)
