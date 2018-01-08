"""Defines miscellaneous exceptions."""


class UnfittedModelException(Exception):
    """Exception raised when a model is used before being fitted."""

    def __init__(self, model):
        message = f"This {model.__class__.__name__} object is not fitted."
        super(UnfittedModelException, self).__init__(message)
