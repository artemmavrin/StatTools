"""Defines abstract base classes for classifiers and regressors."""

import abc

import numpy as np

from .fit import Fittable
from ..utils import validate_samples


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

    def _preprocess_classes(self, y):
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
        self.classes, indices = np.unique(y, return_inverse=True)
        return indices

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


class BinaryClassifier(Classifier):
    """Abstract base class for binary classifiers.

    In this case, the `classes` attribute will be a list of the form [C0, C1],
    where C0 and C1 are the distinct class labels.
    """

    def _preprocess_classes(self, y):
        """Extract distinct classes from a target vector, ensuring that there
        are at most 2 classes.

        Parameters
        ----------
        y : array-like
            Categorical response variable vector.

        Returns
        -------
        indices : numpy.ndarray
            Indices pointing to the class of each item in `y`.
        """
        indices = super(BinaryClassifier, self)._preprocess_classes(y)
        if len(self.classes) != 2:
            raise ValueError(f"This model is a binary classifier; "
                             f"found {len(self.classes)} distinct classes")
        return indices

    @abc.abstractmethod
    def predict_prob(self, *args, **kwargs):
        """Return probability P(y=C1|x) that the data belong to class C1."""
        raise NotImplementedError()

    def predict(self, x, cutoff=0.5, *args, **kwargs):
        """Classify input samples according to their probability estimates.

        Parameters
        ----------
        x : array-like
            Explanatory variable.
        cutoff : float in [0, 1], optional
            If P(y=C1|x)>cutoff, then x is classified as class C1, otherwise C0.
        args : sequence, optional
            Positional arguments to pass to `predict_prob`.
        kwargs : dict, optional
            Keyword arguments to pass to `predict_prob`.

        Returns
        -------
        Vector of predicted class labels.
        """
        prob = self.predict_prob(x, *args, **kwargs)
        return self.classes[list(map(int, np.less(cutoff, prob)))]


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
