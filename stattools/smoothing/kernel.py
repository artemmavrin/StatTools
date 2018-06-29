"""Kernel smoothing."""

import numpy as np
import scipy.stats as st

from .smoothing import ScatterplotSmoother
from ..utils.validation import validate_samples, validate_float

# Standard Gaussian density kernel
kernel_gauss = st.norm(loc=0, scale=1).pdf


def kernel_epanechnikov(t):
    """Epanechnikov kernel."""
    y = np.zeros(np.shape(t))
    idx = (np.abs(t) <= 1)
    y[idx] = (3 / 4) * (1 - t[idx] ** 2)
    return y


def kernel_minvar(t):
    """Minimum variance kernel."""
    y = np.zeros(np.shape(t))
    idx = (np.abs(t) <= 1)
    y[idx] = (3 / 8) * (3 - 5 * t[idx] ** 2)
    return y


class KernelSmoother(ScatterplotSmoother):
    """Kernel smoother."""

    # Training predictors
    x: np.ndarray = None

    # Trainijng response
    y: np.ndarray = None

    def __init__(self, kernel="gauss", bandwidth=1.0):
        """Initialize the KernelSmoother.

        Parameters
        ----------
        kernel : str
            Type of kernel. Can be "gauss", "epanechnikov", or "minvar".
        bandwidth : float
            Smoothing parameter.
        """
        if kernel == "gauss":
            self.kernel = kernel_gauss
        elif kernel == "epanechnikov":
            self.kernel = kernel_epanechnikov
        elif kernel == "minvar":
            self.kernel = kernel_minvar
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}.")

        self.bandwidth = validate_float(bandwidth, "bandwidth", positive=True)

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
        This KernelSmoother instance.
        """
        self.x, self.y = validate_samples(x, y, n_dim=1, equal_lengths=True)
        return self

    def predict(self, x):
        """Compute the kernel smoother estimates.

        Parameters
        ----------
        x : array-like, shape (n, )
            Explanatory variable.

        Returns
        -------
        The kernel smoother prediction.
        """
        x = validate_samples(x, n_dim=1)
        y = np.empty(len(x))

        for i, x0 in enumerate(x):
            d = self.kernel((x0 - self.x) / self.bandwidth)
            y[i] = d.dot(self.y) / d.sum()
        return y
