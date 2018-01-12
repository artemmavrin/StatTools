"""Defines the abstract GeneralizedLinearModel base class."""

from abc import ABC, abstractmethod

import numpy as np

from ..utils.exceptions import UnfittedModelException


def _add_intercept_column(x):
    """Add a column of 1's before the first column of a matrix."""
    return np.c_[np.ones(np.shape(x)[0]), x]


class GeneralizedLinearModel(ABC):
    """Generalized linear model abstract base class."""

    # Indicates whether the module should fit an intercept term
    intercept = True

    # Weights of the model
    _weights = None

    # Indicates whether the model has been fitted
    _fitted = False

    # Number of columns of compatible feature matrices---to be determined during
    # model fitting
    _n_features = None

    def _preprocess_features(self, x, training=False):
        """Apply necessary validation and preprocessing to a feature matrix.

        Parameters
        ----------
        x : array-like
            Feature matrix
        training : bool, optional
            Indicates whether preprocessing is being done during training

        Returns
        -------
        x : array-like
            Updated feature matrix
        """
        if np.ndim(x) == 1:
            x = np.atleast_2d(x).T
        elif np.ndim(x) != 2:
            raise ValueError("Feature matrix must be 2-dimensional.")
        else:
            x = np.array(x)

        if self.intercept:
            x = _add_intercept_column(x)

        if training:
            self._n_features = x.shape[1]
        else:
            if x.shape[1] != self._n_features:
                raise ValueError(f"Expected {self._n_features} features, "
                                 f"but found {np.shape(x)[1]}")

        return x

    @staticmethod
    def _preprocess_target(y):
        """Apply necessary validation and preprocessing to a target vector.

        Parameters
        ----------
        y : array-like
            Target vector

        Returns
        -------
        y : array-like
            Updated target vector
        """
        if np.ndim(y) != 1:
            raise ValueError("Target vector must be 1-dimensional.")

        return np.asarray(y)

    @staticmethod
    @abstractmethod
    def _inv_link(*args):
        """Inverse link function for the given generalized linear model."""
        pass

    def estimate(self, x):
        """Return the model's estimate for the given input data.

        Parameters
        ----------
        x: array-like
            Feature matrix.

        Returns
        -------
        f(x * w), where f is the model's inverse link function and w is the
        model's weight vector.
        """
        if not self._fitted:
            raise UnfittedModelException(self)
        x = self._preprocess_features(x, training=False)
        return self._inv_link(x.dot(self._weights))
