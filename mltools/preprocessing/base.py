"""Defines the abstract DataTransformer base class."""

import abc

import numpy as np

from ..utils.exceptions import UnfittedModelException


class DataTransformer(metaclass=abc.ABCMeta):
    """Data transformer abstract base class."""

    # Indicate whether the transformer is fitted
    _fitted = False

    # Number of features found in the training data
    _n_features = None

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the transformer to training data.

        This method should return self to allow transformer.fit().transform().
        One way to do this safely is to call
            return super(subclass, self).fit()
        at the end of the subclass's fit() implementation.
        """
        self._fitted = True
        return self

    @abc.abstractmethod
    def transform(self, *args, **kwargs):
        """Transform data.

        This method should raise UnfittedModelException if the instance's
        `_fitted` flag is False. This can be ensured by calling
            super(Normalizer, self).transform()
        at the start of the subclass's transform() implementation."""
        if not self._fitted:
            raise UnfittedModelException(self)

    def inv_transform(self, *args):
        """(Optional) Undo a transformation."""
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
            Indicates whether preprocessing is being done during training.

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
