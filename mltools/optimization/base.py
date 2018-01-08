"""Defines the Minimizer abstract base class."""

import abc


class Minimizer(metaclass=abc.ABCMeta):
    """Abstract base class for function minimizers.

    Subclasses should have an `__init__` method which sets the minimization
    algorithm parameters and a `minimize` method that accepts an objective
    function, an initial minimizer guess, and other optional parameters.
    """

    @abc.abstractmethod
    def minimize(self, *args, **kwargs):
        pass
