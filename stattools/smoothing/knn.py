"""k nearest neighbors smoothers."""

import numpy as np

from .smoothing import ScatterplotSmoother
from ..utils.validation import validate_samples, validate_int


class KNNSmoother(ScatterplotSmoother):
    """Naive k nearest neighbors scatterplot smoother implementation."""

    # Training predictors
    x: np.ndarray = None

    # Trainijng response
    y: np.ndarray = None

    def __init__(self, k=10):
        """Initialize the KNNSmoother.

        Parameters
        ----------
        k : int
            Number of neighbors.
        """
        self.k = validate_int(k, "k", minimum=1)

    def fit(self, x, y):
        """Store the training data.

        Parameters
        ----------
        x : array-like, shape (n,)
            Explanatory variable.
        y : array-like, shape (n,)
            Response variable.

        Returns
        -------
        This KNNSmoother instance.
        """
        self.x, self.y = validate_samples(x, y, n_dim=1, equal_lengths=True)
        if self.k > len(x):
            raise ValueError("Sample size too small.")
        return self

    def predict(self, x):
        """Compute averages of training points closest to the points in x.

        Parameters
        ----------
        x : array-like, shape (n, )
            Explanatory variable.

        Returns
        -------
        The kNN smoother prediction.
        """
        x = validate_samples(x, n_dim=1)
        y = np.empty(len(x))

        for i, x0 in enumerate(x):
            idx = np.abs(self.x - x0).argsort()
            y[i] = self.y[idx[:self.k]].mean()
        return y
