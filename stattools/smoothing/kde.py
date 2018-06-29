"""Defines a class for non-parametric density function estimation."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from ..generic import Fittable
from ..utils import validate_samples, validate_int, validate_float


class KernelDensityEstimator(Fittable):
    """Class for kernel density estimation.

    Properties
    ----------
    data : numpy.ndarray
        The sample from the density being estimated.
    kernel : callable
        The kernel (non-negative function which integrates to 1) that assigns
        weights to each point in the sample.
    bandwidth : float
        The positive smoothing parameter
    """

    data: np.ndarray = None
    kernel = None
    bandwidth: float = None

    # The estimated density function
    _density = None

    def fit(self, data, kernel=None, bandwidth=None):
        """Basically just validate the parameters and define the density
        estimate.

        Parameters
        ----------
        data : array-like, of shape (n, )
            The sample from the density being estimated.
        kernel : callable, optional
            The kernel (non-negative function which integrates to 1) that
            assigns weights to each point in the sample.
            By default, this is the standard Gaussian density.
        bandwidth : float, optional
            The positive smoothing parameter.
            By default, this is chosen to be Silverman's rule of thumb, which is
            optimal (in the sense of minimizing mean integrated square error) in
            the Gaussian case. Generally you should choose your bandwidth on a
            case-by-case basis (e.g., by cross-validation).

        Returns
        -------
        This KernelDensityEstimator instance.
        """
        self.data = validate_samples(data, n_dim=1)

        if kernel is None:
            self.kernel = st.norm.pdf
        elif callable(kernel):
            self.kernel = np.vectorize(kernel)
        else:
            raise TypeError("Parameter 'kernel' must be a function.")

        if bandwidth is None:
            # Silverman's rule of thumb
            n = len(self.data)
            scale = self.data.std(ddof=1)
            self.bandwidth = 1.06 * scale * (n ** (-1 / 5))
        else:
            self.bandwidth = validate_float(bandwidth, "bandwidth",
                                            positive=True)

        h = self.bandwidth

        def density(x):
            return np.mean(self.kernel((x - data) / h)) / h

        self._density = np.vectorize(density)

        self.fitted = True
        return self

    def __call__(self, x):
        """Evaluate the estimated density at a point (or a list of points)."""
        if not self.fitted:
            raise self.unfitted_exception
        x = validate_samples(x, n_dim=1)
        return self._density(x)

    def plot(self, x_min=None, x_max=None, num=500, ax=None, **kwargs):
        """Plot the estimated density curve.

        Parameters
        ----------
        x_min : float, optional
        x_max : float, optional
            The endpoints of the interval on which to plot.
        num : int, optional
            The number of points to plot
        ax : matplotlib.axes.Axes object, optional
            The axes on which to draw the plot.
        kwargs : dict
            Additional keyword arguments to pass to the underlying matplotlib
            plotting function.

        Returns
        -------
        The matplotlib.axes.Axes on which the plot was drawn.
        """
        if not self.fitted:
            raise self.unfitted_exception

        # Validate all parameters
        if x_min is None:
            x_min = min(self.data)
        else:
            x_min = validate_float(x_min, "x_min")

        if x_max is None:
            x_max = max(self.data)
        else:
            x_max = validate_float(x_max, "x_max")

        num = validate_int(num, "num", minimum=1)

        if ax is None:
            ax = plt.gca()

        x = np.linspace(x_min, x_max, num)
        y = self(x)

        params = dict(label="Kernel density estimate")
        params.update(kwargs)

        ax.plot(x, y, **params)
        ax.set(xlabel="Data", ylabel="Density")
        ax.set(xlim=(x_min, x_max))
        return ax
