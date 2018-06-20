"""Defines the abstract DataTransformer base class."""

import abc

import numpy as np

from ..generic import Fittable


class DataTransformer(Fittable, metaclass=abc.ABCMeta):
    """Data transformer abstract base class."""

    # Number of features found in the training data
    _n_features = None

    @abc.abstractmethod
    def transform(self, *args, **kwargs):
        """Transform data."""
        raise NotImplementedError()

    def _preprocess_data(self, x, fitting=False):
        """Apply some necessary validation and preprocessing to input data to
        prepare it for fitting or transformation.

        Parameters
        ----------
        x: array-like
            Data matrix of shape (n_samples, n_features). If this data is one
            dimensional, it is treated as a data matrix of shape (n_samples, 1),
            i.e., one feature column.
        fitting: bool, optional
            Indicates whether preprocessing is being done during fitting.

        Returns
        -------
        x: array-like
            Validated and preprocessed data matrix
        """
        if np.ndim(x) == 1:
            x = np.atleast_2d(x).T
        elif np.ndim(x) == 2:
            x = np.asarray(x)
        else:
            raise ValueError("Input data must be 2-dimensional.")

        if fitting:
            self._n_features = x.shape[1]
        else:
            if x.shape[1] != self._n_features:
                raise ValueError(
                    f"Expected {self._n_features} columns; "
                    f"found {np.shape(x)[1]}.")

        return x


class InvertibleDataTransformer(DataTransformer):
    """Data transformer where transformations can be un-done at least partly."""

    @abc.abstractmethod
    def inv_transform(self, *args):
        """Undo a transformation."""
        raise NotImplementedError()
