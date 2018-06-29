"""Defines the PCA (principal component analysis) class."""

import numpy as np

from .base import InvertibleDataTransformer
from ..utils import validate_int


class PCA(InvertibleDataTransformer):
    """Decompose data into principal components."""

    # Eigenvalues of the covariance matrix of the training data, sorted from
    # biggest to smallest
    _evals = None

    # Matrix of eigenvectors of the covariance of the training data, with the
    # ith column being the eigenvector corresponding the ith eigenvalue
    _evecs = None

    def fit(self, x):
        """Fit a PCA model to training data.

        Parameters
        ----------
        x: array-like
            Training feature matrix of shape (n_samples, n_features).

        Returns
        -------
        self: PCA
            This PCA instance.
        """
        x = self._preprocess_data(x, fitting=True)

        # Compute the projection matrix onto the principal component axes
        evals, evecs = np.linalg.eig(np.cov(x.T, ddof=1))
        idx = np.argsort(evals)[::-1]
        self._evals = evals[idx]
        self._evecs = np.atleast_1d(evecs[:, idx])

        self.fitted = True
        return self

    def transform(self, x, dim=None):
        """Reduce a data matrix to a certain number of principal components.

        Parameters
        ----------
        x: array-like
            Feature matrix of shape (n_samples, n_features) to be reduced.
        dim: int or None, optional
            If `dim` is a positive integer, return this many columns. If `dim`
            is None, return all the principal components.

        Returns
        -------
        y: array-like
            Matrix whose columns are principal components.
        """
        if not self.fitted:
            raise self.unfitted_exception

        x = self._preprocess_data(x)

        # Project onto principal component axes
        if dim is None:
            return np.dot(x, self._evecs)
        else:
            dim = validate_int(dim, "dim", minimum=1)
            if self._n_features < dim:
                raise ValueError(f"Parameter 'dim' is too large; "
                                 f"expected at most {self._n_features}")
            return np.dot(x, self._evecs[:, :dim])

    def inv_transform(self, y):
        """Reconstruct the original data from the principal component space.

        Parameters
        ----------
        y: array-like
            Matrix of principal component columns.

        Returns
        -------
        x: array-like
            Matrix with the full number of dimensions.
        """
        if not self.fitted:
            raise self.unfitted_exception

        # Validate input
        if np.ndim(y) == 1:
            y = np.atleast_2d(y).T
        elif np.ndim(y) != 2:
            raise ValueError("Data must be 2-dimensional.")

        dim = np.shape(y)[1]
        if dim > self._n_features:
            raise ValueError(f"Too many columns in input data; "
                             f"expected at most {self._n_features}")

        return np.dot(y, self._evecs[:, :dim].T)
