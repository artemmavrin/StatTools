"""Bin smoothers."""

import numbers

import matplotlib.pyplot as plt
import numpy as np

from .smoothing import ScatterplotSmoother
from ..utils.validation import validate_samples


class BinSmoother(ScatterplotSmoother):
    """Fit piecewise horizontal line segments to a scatterplot."""

    n_bins: int = None
    bins: np.ndarray = None
    means: np.ndarray = None

    def __init__(self, bins=5):
        """Initialize a BinSmoother object.

        Parameters
        ----------
        bins : int or array-like of shape (N, ), optional
            If int:
                Number of bins to partition the predictor into.
            If array-like:
                1-D array of bin endpoints.
        """
        if isinstance(bins, numbers.Integral):
            self.n_bins = int(bins)
            if self.n_bins <= 0:
                raise ValueError("Parameter 'bins' cannot be <= 0.")
            self.bins = None
        else:
            if np.ndim(bins) == 1:
                self.bins = np.unique(np.asarray(bins, dtype=float))
                self.bins[0] = -np.inf
                self.bins[-1] = np.inf
                self.n_bins = len(bins) - 1
            else:
                raise ValueError(
                    "Parameter 'bins' must be a positive int or a 1-D array.")

    def fit(self, x, y):
        """Compute the binned means of the response vector.

        Parameters
        ----------
        x : array-like, shape (n,)
            Explanatory variable.
        y : array-like, shape (n,)
            Response variable.

        Returns
        -------
        This BinSmoother instance.
        """
        x, y = validate_samples(x, y, n_dim=1, equal_lengths=True)

        if self.bins is None:
            p = np.linspace(0, 100, num=(self.n_bins + 1))
            self.bins = np.percentile(x, p)
            self.bins[0] = -np.inf
            self.bins[-1] = np.inf

        self.means = np.empty(self.n_bins)

        for i in range(self.n_bins):
            idx = (self.bins[i] <= x) & (x < self.bins[i + 1])
            self.means[i] = np.mean(y[idx])

        return self

    def predict(self, x):
        """Return the model's prediction for the given input data.

        Parameters
        ----------
        x : array-like, shape (n, )
            Explanatory variable.

        Returns
        -------
        The bin smoother prediction.
        """
        x = validate_samples(x, n_dim=1)
        y = np.empty(len(x))
        for i, x_ in enumerate(x):
            for j in range(self.n_bins):
                if (self.bins[j] <= x_) & (x_ < self.bins[j + 1]):
                    y[i] = self.means[j]
        return y

    def fit_plot(self, x_min=None, x_max=None, num=None, ax=None, **kwargs):
        """Plot the bin smoother segments.

        Parameters
        ----------
        x_min : float, optional
            Smallest explanatory variable observation. If not provided, grabs
            the smallest x value from the given axes.
        x_max : float, optional
            Biggest explanatory variable observation. If not provided, grabs the
            biggest x value from the given axes.
        num : int, optional
            Ignored.
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the plot.
        kwargs : dict, optional
            Additional keyword arguments to pass to plot()

        Returns
        -------
        The matplotlib.axes.Axes object on which the plot was drawn.
        """
        if ax is None:
            ax = plt.gca()

        # Get bounds if not provided
        y_min, y_max = ax.get_ylim()
        reset_y = False
        if x_min is None or x_max is None:
            x_min, x_max = ax.get_xlim()
            reset_y = True

        # Plot the segments
        for i in range(self.n_bins):
            x = [max(x_min, self.bins[i]), min(x_max, self.bins[i + 1])]
            y = [self.means[i], self.means[i]]
            p = ax.plot(x, y, **kwargs)
            if "label" in kwargs:
                del kwargs["label"]
            if "color" not in kwargs and "c" not in kwargs:
                kwargs["color"] = p[0].get_color()

        # Set the axes bounds
        ax.set(xlim=(x_min, x_max))
        if reset_y:
            ax.set(ylim=(y_min, y_max))

        return ax
