"""Defines the Normalizer class."""

import numpy as np

from .base import InvertibleDataTransformer


class Normalizer(InvertibleDataTransformer):
    """Translate and scale data columns to have mean 0 and variance 1."""

    # Vector of means of the feature columns of the training data
    _mean = None

    # Vector of standard deviations of the feature columns of the training data
    _std = None

    # Indices of columns with nonzero variance in the training data
    _idx = None

    def __init__(self, center=True, scale=True, reduce=True, bias=True):
        """Initialize the Normalizer.

        Parameters
        ----------
        center: bool, optional
            If True, columns with nonzero variance will be centered to have mean
            zero.
        scale: bool, optional
            If True, columns with nonzero variance will be scaled to have unit
            variance.
        reduce: bool, optional
            If True, columns with zero variance will be removed from the data
            during transformation.
        bias: bool, optional
            If True, standard deviation is computed without Bessel's correction:
                std = sqrt((1 / n) (x - mean) * (x - mean))
            If False, standard deviation is computed with Bessel's correction:
                std = sqrt((1 / (n - 1)) (x - mean) * (x - mean))
        """
        self.center = center
        self.scale = scale
        self.reduce = reduce
        self.bias = bias

    def fit(self, x):
        """Determine column means and standard deviations for transformation.

        Parameters
        ----------
        x: array-like
            Training data---2d array of shape (n_samples, n_features) whose
            column-wise sample means and sample standard deviations are to be
            computed.

        Returns
        -------
        self: Normalizer
            This Normalizer instance.
        """
        x = self._preprocess_data(x, fitting=True)

        self._mean = np.mean(x, axis=0)
        self._std = np.std(x, axis=0, ddof=0 if self.bias else 1)
        self._idx = np.where(self._std > 0)[0]

        self._fitted = True
        return self

    def transform(self, x):
        """Normalize the data.

        Parameters
        ----------
        x: array-like
            Data matrix---2d array of shape (n_samples, n_features) to be
            normalized.

        Returns
        -------
        z: array-like
            Normalized data matrix.
        """
        if not self.is_fitted():
            raise self.unfitted_exception()

        x = self._preprocess_data(x)
        z = np.asarray(x, dtype=np.float_)

        if self.center:
            z[:, self._idx] -= self._mean[self._idx]

        if self.scale:
            z[:, self._idx] /= self._std[self._idx]

        if self.reduce:
            z = z[:, self._idx]

        return z

    def inv_transform(self, z):
        """Un-normalize the data.

        Parameters
        ----------
        z: array-like
            Data matrix to be un-normalized.
            If `reduce` was False during fitting, `z` must have the same number
            of features (columns) as the training data.
            If `reduce` was True during fitting, `z` must have as many columns
            as there are non-zero-variance columns in the training data.

        Returns
        -------
        x: array-like
            Un-normalized data with as many columns as the training data.
        """
        if not self.is_fitted():
            raise self.unfitted_exception()

        # Validate input
        if np.ndim(z) == 1:
            z = np.atleast_2d(z).T
        elif np.ndim(z) != 2:
            raise ValueError("Data must be 2-dimensional.")
        if self.reduce:
            if np.shape(z)[1] != len(self._idx):
                raise ValueError(
                    f"Expected {len(self._idx)} columns; "
                    f"found {np.shape(z)[1]}.")
        else:
            if np.shape(z)[1] != self._n_features:
                raise ValueError(
                    f"Expected {self._n_features} columns;"
                    f"found {np.shape(z)[1]}.")

        if self.reduce:
            z = np.asarray(z)
            x = np.zeros((np.shape(z)[0], self._n_features))
            j = 0
            for i in range(self._n_features):
                if i in self._idx:
                    x[:, i] = z[:, j]
                    j += 1
                else:
                    x[:, i] = self._mean[i]
        else:
            x = np.asarray(z)

        if self.scale:
            x[:, self._idx] *= self._std[self._idx]

        if self.center:
            x[:, self._idx] += self._mean[self._idx]

        return x
