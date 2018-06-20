"""Defines the Optimizer abstract base class."""

import abc


class Optimizer(metaclass=abc.ABCMeta):
    """Abstract base class for function optimization.

    Subclasses should have an `__init__` method which sets the optimzation
    algorithm parameters and a `optimize` method that accepts an objective
    function, an initial optimizer guess, and other optional parameters.
    """

    @abc.abstractmethod
    def optimize(self, *args, **kwargs):
        pass
