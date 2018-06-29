"""Defines abstract base classes for classifiers and regressors."""

import abc
import numbers

import numpy as np

from .fit import Fittable
from ..utils import validate_samples
from ..utils import validate_int


class Predictor(Fittable, metaclass=abc.ABCMeta):
    """Abstract base class for both classifiers and regressors."""

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        pass


class Classifier(Predictor, metaclass=abc.ABCMeta):
    """Abstract base class for classifiers.

    Properties
    ----------
    classes : numpy.ndarray
        List of distinct class labels. These will usually be determined during
        model fitting.
    """

    classes: np.ndarray = None

    def _preprocess_classes(self, y, max_classes):
        """Extract distinct classes from a response variable vector.

        This also converts the response variable vector to numeric indices
        pointing to the corresponding class in the `classes` attribute.

        Parameters
        ----------
        y : array-like
            Categorical response variable vector.

        Returns
        -------
        indices : numpy.ndarray
            Indices pointing to the class of each item in `y`.
        """
        # Validate `max_classes`
        if max_classes is not None:
            max_classes = validate_int(max_classes, "max_classes", minimum=2)

        # Extract unique classes, convert response vector to indices.
        self.classes, indices = np.unique(y, return_inverse=True)

        # Doing classification with 1 (or 0?) classes is useless
        if len(self.classes) < 2:
            raise ValueError(
                "Response vector must contain at least two distinct classes.")

        # Make sure we don't have too many classes
        if max_classes is not None:
            if len(self.classes) > max_classes:
                raise ValueError(
                    "Response vector contains too many distinct classes.")

        return indices

    @abc.abstractmethod
    def predict_prob(self, *args, **kwargs):
        """Return estimated probability that the response corresponding to a
        set of features belongs to each possible class.

        This method should return a matrix of shape (n_observations, n_classes).
        """
        raise NotImplementedError()

    def predict(self, *args, **kwargs):
        """Return the estimated class label for each input."""
        p = self.predict_prob(*args, **kwargs)
        return self.classes[np.argmax(p, axis=1)]

    def mcr(self, x, y, *args, **kwargs):
        """Compute the misclassification rate of the model for given values of
        the explanatory and response variables.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variables.
        y : array-like, shape (n,)
            Response variable.
        args : sequence, optional
            Positional arguments to pass to this regressor's predict() method.
        kwargs : dict, optional
            Keyword arguments to pass to this regressor's predict() method.

        Returns
        -------
        mcr : float
            The misclassification rate.
        """
        # Validate input
        x, y = validate_samples(x, y, n_dim=(None, 1), equal_lengths=True)
        return np.mean(y != self.predict(x, *args, **kwargs))


class Regressor(Predictor, metaclass=abc.ABCMeta):
    """Abstract base class for regressors."""

    def mse(self, x, y, *args, **kwargs):
        """Compute the mean squared error of the model for given values of the
        explanatory and response variables.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variables.
        y : array-like, shape (n,)
            Response variable.
        args : sequence, optional
            Positional arguments to pass to this regressor's predict() method.
        kwargs : dict, optional
            Keyword arguments to pass to this regressor's predict() method.

        Returns
        -------
        mse : float
            The mean squared prediction error.
        """
        # Validate input
        x, y = validate_samples(x, y, n_dim=(None, 1), equal_lengths=True)
        return np.mean((y - self.predict(x, *args, **kwargs)) ** 2)

    def mae(self, x, y, *args, **kwargs):
        """Compute the mean absolute error of the model for given values of the
        explanatory and response variables.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variables.
        y : array-like, shape (n,)
            Response variable.
        args : sequence, optional
            Positional arguments to pass to this regressor's predict() method.
        kwargs : dict, optional
            Keyword arguments to pass to this regressor's predict() method.

        Returns
        -------
        mae : float
            The mean absolute prediction error.
        """
        # Validate input
        x, y = validate_samples(x, y, n_dim=(None, 1), equal_lengths=True)
        return np.mean(np.abs(y - self.predict(x, *args, **kwargs)))
