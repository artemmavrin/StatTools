"""Defines the BaseSummary abstract base class."""

import abc


class BaseSummary(object, metaclass=abc.ABCMeta):
    """Abstract base class for summaries of models and other objects."""

    @abc.abstractmethod
    def __str__(self):
        pass
