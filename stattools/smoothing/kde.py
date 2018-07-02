"""Defines a class for non-parametric density function estimation."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from ..generic import Fittable
from ..utils import validate_sample, validate_int, validate_float, validate_bool


class KernelDensityEstimator(Fittable):
    """Class for kernel density estimation.

    Properties
    ----------
    p : int
        Number of dimensions of the data.
    data : numpy.ndarray
        The sample from the density being estimated.
    kernel : callable
        The kernel (non-negative function which integrates to 1) that assigns
        weights to each point in the sample.
    bandwidth : float
        The positive smoothing parameter
    """
    p: int = None
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
        data : array-like, of shape (n, p)
            The sample from the density being estimated.
        kernel : str or callable, optional
            The kernel (non-negative function which integrates to 1) that
            assigns weights to each point in the sample.
            Possible values:
            *   None
                Standard Gaussian density function
            *   callable
                Custom kernel. No checking is done to ensure that the definition
                of a kernel is satisfied.
        bandwidth : float, optional
            The positive smoothing parameter.
            In the case of univariate data, the default is chosen to be
            Silverman's rule of thumb, which is optimal (in the sense of
            minimizing mean integrated square error) in the Gaussian case.
            Generally you should choose your bandwidth on a case-by-case basis
            (e.g., by cross-validation).
            There is no default bandwidth in the multivariate case.

        Returns
        -------
        This KernelDensityEstimator instance.
        """
        self.data = validate_sample(data, n_dim=2)
        self.p = self.data.shape[1]

        if kernel is None:
            self.kernel = st.norm.pdf
        elif callable(kernel):
            self.kernel = np.vectorize(kernel)
        else:
            raise TypeError("Parameter 'kernel' must be a function.")

        if bandwidth is None:
            if self.p == 1:
                # Silverman's rule of thumb
                n = len(self.data)
                scale = self.data.std(ddof=1)
                self.bandwidth = 1.06 * scale * (n ** (-1 / 5))
            else:
                raise ValueError("Parameter 'bandwidth' must be provided for "
                                 "multivariate kernel density estimation.")
        else:
            self.bandwidth = validate_float(bandwidth, "bandwidth",
                                            positive=True)

        def density(x):
            out = np.empty(shape=(len(x),), dtype=np.float_)
            for i in range(len(x)):
                u = np.linalg.norm(x[i] - self.data, axis=1) / self.bandwidth
                out[i] = np.mean(self.kernel(u)) / (self.bandwidth ** self.p)
            return out

        self._density = density

        self.fitted = True
        return self

    def __call__(self, x):
        """Evaluate the estimated density at a point (or a list of points)."""
        if not self.fitted:
            raise self.unfitted_exception
        x = validate_sample(x, n_dim=2)
        if x.shape[1] != self.p:
            raise ValueError(f"Expected {self.p} columns, found {x.shape[1]}.")
        return self._density(x)

    def plot(self, num=200, fill=True, ax=None, **kwargs):
        """Plot the estimated density in the 1D and 2D cases.

        In the 1D case, the density curve is plotted. In the 2D case, a contour
        plot of the density is drawn.

        Parameters
        ----------
        num : int
            Number of points to sample.
        fill : bool, optional
            Indicates whether to use contourf (True) or contour (False) to plot
            contours in the 2D case.
        ax : matplotlib.axes.Axes, optional
            The matplotlib.axes.Axes on which to plot.
        kwargs : dict
            Additional keyword arguments to pass to either ax.plot() (in the 1D
            case) or ax.contourf() (in the 2D case).

        Returns
        -------
        The matplotlib.axes.Axes on which the density was plotted.
        """
        num = validate_int(num, "num", minimum=1)
        fill = validate_bool(fill, "fill")

        if ax is None:
            ax = plt.gca()

        if self.p == 1:
            x_min, x_max = ax.get_xlim()
            x = np.linspace(x_min, x_max, num)
            plot_kwargs = dict(label="Kernel density estimate")
            plot_kwargs.update(kwargs)
            ax.plot(x, self(x), **plot_kwargs)
            ax.set_xlim(x_min, x_max)
            ax.autoscale(enable=True, axis="y")
            ax.set_xlabel("Data")
            ax.set_ylabel("Density")
        elif self.p == 2:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x = np.linspace(x_min, x_max, num)
            y = np.linspace(y_min, y_max, num)
            xx, yy = np.meshgrid(x, y)
            xy = np.column_stack((xx.ravel(), yy.ravel()))
            density = self(xy).reshape(-1, num)
            if fill:
                ax.contourf(xx, yy, density, **kwargs)
            else:
                ax.contour(xx, yy, density, **kwargs)
        else:
            raise AttributeError(
                "Density plotting is only supported for 1D and 2D models.")

        return ax
